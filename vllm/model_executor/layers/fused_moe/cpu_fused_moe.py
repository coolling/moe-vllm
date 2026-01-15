# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import weakref
from collections.abc import Callable
from collections import defaultdict
import torch
from torch.nn import functional as F
import time
from collections import defaultdict, deque
import threading
from vllm import _custom_ops as ops
from vllm._custom_ops import cpu_fused_moe, cpu_prepack_moe_weight
from vllm.model_executor.layers.activation import SiluAndMul, SwigluOAIAndMul
from vllm.model_executor.layers.quantization.utils.layer_utils import replace_parameter
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.model_executor.model_loader.default_loader import GLOBAL_HF_WEIGHTS_FILE
from vllm.model_executor.model_loader.weight_utils import load_expert_weight
_CPU_MOE_LAYER_CACHE = {}
_CPU_MOE_ACT = {
    "silu": SiluAndMul(),
    "swigluoai": SwigluOAIAndMul(),
}
# coolling ==========================

def tensors_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    # 检查形状是否相同
    if a.shape != b.shape:
        return False
    # 逐元素比较，全部为 True 则返回 True
    return torch.equal(a, b)  # 最高效！
def is_all_zero(tensor):
    return torch.allclose(tensor, torch.zeros_like(tensor))
import ctypes

def fast_copy(dst: torch.Tensor, src: torch.Tensor):
    """使用ctypes进行快速内存复制"""
    # 确保张量连续
    if not dst.is_contiguous():
        dst = dst.contiguous()
    if not src.is_contiguous():
        src = src.contiguous()
    
    # 获取内存指针和大小
    dst_ptr = dst.data_ptr()
    src_ptr = src.data_ptr()
    nbytes = dst.numel() * dst.element_size()
    
    # 使用memmove进行内存复制
    ctypes.memmove(dst_ptr, src_ptr, nbytes)
class ExpertWeightManager:
    def __init__(
        self,
        num_experts: int,
        w13: torch.Tensor,
        w2: torch.Tensor,
        max_size: int = 2,  # 队列最大容量（预加载层数）
    ):
        self.num_experts = num_experts
        self.weight_file = GLOBAL_HF_WEIGHTS_FILE
        
        # 当前用于计算的全局张量（由 consumer 绑定到队列头部）
        self.global_w13_tensor = torch.empty_like(w13)
        print("init",self.global_w13_tensor.data_ptr())
        self.global_w2_tensor = torch.empty_like(w2)

        # 加载状态跟踪（可选，用于调试）
        self._EXPERT_WEIGHT_LOADED = defaultdict(dict)

        # 缓冲池：预先分配 max_size 个 (w2, w13) buffer
        self.buffer_pool = []
        for _ in range(max_size):
            w2_buf = torch.empty_like(w2)
            w13_buf = torch.empty_like(w13)
            self.buffer_pool.append((w2_buf, w13_buf))

        # 生产者-消费者队列：每个元素为 (layer_id, w2_buf, w13_buf)
        self.queue = deque()  # 使用 deque 更高效
        self.max_size = max_size

        # 线程同步原语
        self.lock = threading.Lock()
        self.not_full = threading.Condition(self.lock)   # 队列未满
        self.not_empty = threading.Condition(self.lock)  # 队列非空

        # 控制加载顺序
        self.layer_id_to_load = 0  # 下一个要加载的 layer_id（按注册顺序）
        self.total_layers = 0      # 由外部设置（或动态探测）

        # 启动后台生产者线程
        self._shutdown = threading.Event()
        self.producer_thread = threading.Thread(
            target=self._producer_worker,
            daemon=True,
            name="MoEWeightProducer"
        )
        self.producer_thread.start()

    def set_total_layers(self, total_layers: int):
        """由外部调用，告知总共有多少 MoE 层（必须在推理开始前设置）"""
        self.total_layers = total_layers

    def _load_expert_into_buffer(self, layer_id: int, expert_id: int, w13_buf: torch.Tensor, w2_buf: torch.Tensor):
        import time
        
        """将单个 expert 的权重加载到 buffer 的对应位置"""
        rank = layer_id  # 假设 layer_id == 推理顺序中的 rank（需与注册顺序一致）

        w13_key = f"model.layers.{rank}.block_sparse_moe.experts.{expert_id}.w13.weight"
        w2_key = f"model.layers.{rank}.block_sparse_moe.experts.{expert_id}.w2.weight"
        start = time.time()
        # 加载 w2
        w2_t=load_expert_weight(self.weight_file, w2_key)
        
        w13_t=load_expert_weight(self.weight_file, w13_key)
        elapsed_ms = (time.time() - start) * 1000
        print(f"[DEBUG1] time1 {elapsed_ms:.2f} ms")
        start = time.time()
        # w2_buf[expert_id].copy_(w2_t)
        # w13_buf[expert_id].copy_(w13_t)
        fast_copy(w2_buf[expert_id], w2_t)
        fast_copy(w13_buf[expert_id], w13_t)
        elapsed_ms = (time.time() - start) * 1000
        print(f"[DEBUG1] time2 {elapsed_ms:.2f} ms")

        self._EXPERT_WEIGHT_LOADED[layer_id][expert_id] = True
        
    def _producer_worker(self):
        """后台生产者线程：循环预加载下一层权重"""
        while not self._shutdown.is_set():
            if self.total_layers == 0:
                time.sleep(0.1)
                continue

            with self.lock:
                # 等待队列未满
                while len(self.queue) >= self.max_size and not self._shutdown.is_set():
                    self.not_full.wait(timeout=1.0)

                if self._shutdown.is_set():
                    break

                # 获取空闲 buffer
                # if not self.buffer_pool:
                #     print("[Producer] Warning: No free buffer available!")
                #     time.sleep(0.05)
                #     continue

                w2_buf, w13_buf = self.buffer_pool.pop()
                target_layer_id = self.layer_id_to_load

                # 加载整层所有 experts
                try:
                    for eid in range(self.num_experts):
                        self._load_expert_into_buffer(target_layer_id, eid, w13_buf, w2_buf)
                    # print(w13_buf,w2_buf)
                except Exception as e:
                    print(f"[Producer] Failed to load layer {target_layer_id}: {e}")
                    # 加载失败，归还 buffer
                    self.buffer_pool.append((w2_buf, w13_buf))
                    time.sleep(0.5)
                    continue

                # 入队
                self.queue.append((target_layer_id, w2_buf, w13_buf))
                print(f"[Producer] Loaded layer {target_layer_id} into queue (size={len(self.queue)})")

                # 更新下一个要加载的 layer_id（循环）
                self.layer_id_to_load = (self.layer_id_to_load + 1) % self.total_layers

                # 通知消费者
                self.not_empty.notify_all()

            # 控制加载节奏（避免 CPU 占满）
            time.sleep(0.01)

    def acquire_weights_for_layer(self,layer, expected_layer_id: int) -> bool:
        """
        消费者调用：等待并绑定队列头部的 buffer 到 global_tensor。
        返回 True 表示成功，False 表示已关闭。
        """
        with self.not_empty:
            while len(self.queue) == 0 and not self._shutdown.is_set():
                self.not_empty.wait(timeout=1.0)

            if self._shutdown.is_set():
                return False

            layer_id, w2_buf, w13_buf = self.queue[0]
            if layer_id != expected_layer_id:
                print(f"[Consumer] Expected layer {expected_layer_id}, but head is {layer_id}")
                return False
            
            return True

    def release_current_layer(self):
        """消费者调用：释放当前层 buffer，归还到池中"""
        with self.lock:
            # if self.queue:
                layer_id, w2_buf, w13_buf = self.queue.popleft()
                self.buffer_pool.append((w2_buf, w13_buf))
 
                self._EXPERT_WEIGHT_LOADED[layer_id].clear()
                self.not_full.notify()  

    def shutdown(self):
        """优雅关闭"""
        self._shutdown.set()
        with self.not_full:
            self.not_full.notify_all()
        with self.not_empty:
            self.not_empty.notify_all()
        self.producer_thread.join(timeout=2)
        
manager=None
#==========================================
def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    gating_output = gating_output.float()
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    num_token = scores.shape[0]
    if e_score_correction_bias is not None:
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores.view(num_token, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        )
    else:
        group_scores = (
            scores.view(num_token, num_expert_group, -1).max(dim=-1).values
        )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)[1]
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights, topk_ids.to(torch.int32)


def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    topk_group: int | None = None,
    num_expert_group: int | None = None,
    custom_routing_function: Callable | None = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_grouped_topk:
        assert topk_group is not None
        assert num_expert_group is not None
        return grouped_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
        )
    elif custom_routing_function is None:
        assert scoring_func == "softmax"
        topk_logit_vals, topk_idx = torch.topk(
            router_logits, k=top_k, dim=-1, sorted=False
        )
        if renormalize:
            topk_vals = torch.softmax(topk_logit_vals, dim=-1)
        else:
            logZ = torch.logsumexp(router_logits, dim=-1, keepdim=True)
            topk_vals = (topk_logit_vals - logZ).exp()
        return topk_vals.to(torch.float32), topk_idx.to(torch.int32)
    else:
        return custom_routing_function(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
        )


class SGLFusedMOE:
    def __init__(self, layer: torch.nn.Module) -> None:
        pass

    def __call__(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ) -> torch.Tensor:
        assert activation == "silu", f"{activation} is not supported."
        assert not apply_router_weight_on_input
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
        )

        torch.ops._C.fused_experts_cpu(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            True,
            False,
            False,
            None,
            None,
            None,
            None,
            None,
            True,
        )
        # print(topk_weights, topk_ids)
        return x


class CPUFusedMOE:
    def __init__(self, layer: torch.nn.Module) -> None:
        # print("coolling:CPUFusedMOE")
        use_grouped_gemm, isa = self.check_grouped_gemm(layer)
        self.isa = isa
        if use_grouped_gemm:
            self.forward_method = self.forward_grouped_gemm
            self.init_moe_grouped_gemm(layer=layer)
        else:
            self.forward_method = self.forward_torch
            self.init_moe_torch(layer=layer)

    def __call__(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ) -> torch.Tensor:
        assert activation in _CPU_MOE_ACT, f"{activation} is not supported."
        assert not apply_router_weight_on_input

        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
        )
        # print(topk_weights, topk_ids)

        return self.forward_method(
            layer,
            x,
            topk_weights,
            topk_ids,
            activation,
            global_num_experts,
        )

    def check_grouped_gemm(
        self,
        layer: torch.nn.Module,
    ) -> tuple[bool, str]:
        if not hasattr(torch.ops._C, "prepack_moe_weight"):
            return False, "none"

        dtype = layer.w13_weight.dtype
        w13_input_size = layer.w13_weight.size(2)
        w13_output_size = layer.w13_weight.size(1)
        w2_input_size = layer.w2_weight.size(2)
        w2_output_size = layer.w2_weight.size(1)

        if not (w13_output_size % 32 == 0 and w2_output_size % 32 == 0):
            return False, "none"

        supports_amx = torch._C._cpu._is_amx_tile_supported()

        if (
            supports_amx
            and dtype == torch.bfloat16
            and w13_input_size % 32 == 0
            and w2_input_size % 32 == 0
        ):
            return True, "amx"

        if supports_amx:
            return False, "none"

        return True, "vec"

    def init_moe_grouped_gemm(
        self,
        layer: torch.nn.Module,
    ) -> None:
        new_w13 = cpu_prepack_moe_weight(layer.w13_weight, self.isa)
        replace_parameter(layer, "w13_weight", new_w13)
        new_w2 = cpu_prepack_moe_weight(layer.w2_weight, self.isa)
        replace_parameter(layer, "w2_weight", new_w2)
    #coolling: load state
    def init_moe_torch(
        self,
        layer: torch.nn.Module,
    ) -> None:
        global manager
        
        use_onednn_mm = ops._supports_onednn and ops.is_onednn_acl_supported()
        num_experts = layer.w13_weight.size(0)
        has_w13_bias = hasattr(layer, "w13_bias")
        has_w2_bias = hasattr(layer, "w2_bias")
        if manager == None:
            manager = ExpertWeightManager(
                num_experts,
                layer.w13_weight,layer.w2_weight
            )
        layer.w13_weight.set_(manager.global_w13_tensor )
        layer.w2_weight.set_(manager.global_w2_tensor )
        layer.gate_up_linear = []
        layer.down_linear = []
        # layer.w2_weight
        for i in range(num_experts):
            weight_loaded=True
            layer_w13_weight = layer.w13_weight[i]
            # print(i,layer_w13_weight)
            layer_w13_bias = layer.w13_bias[i] if has_w13_bias else None
            layer_w2_weight = layer.w2_weight[i]
            layer_w2_bias = layer.w2_bias[i] if has_w2_bias else None
            if is_all_zero(layer_w13_weight) and is_all_zero(layer_w2_weight):
                weight_loaded=False
            # manager.set_load_state(id(layer),i,weight_loaded)
            
                
            if use_onednn_mm:
                gate_up_handle = ops.create_onednn_mm(layer_w13_weight.t(), 32)
                layer.gate_up_linear.append(
                    lambda x, handle=gate_up_handle, bias=layer_w13_bias: ops.onednn_mm(
                        handle, x, bias
                    )
                )
                down_handle = ops.create_onednn_mm(layer_w2_weight.t(), 32)
                layer.down_linear.append(
                    lambda x, handle=down_handle, bias=layer_w2_bias: ops.onednn_mm(
                        handle, x, bias
                    )
                )
            else:
                layer.gate_up_linear.append(
                    lambda x, w=layer_w13_weight, b=layer_w13_bias: F.linear(x, w, b)
                )
                layer.down_linear.append(
                    lambda x, w=layer_w2_weight, b=layer_w2_bias: F.linear(x, w, b)
                )

        if use_onednn_mm:  # remove weight
            layer.w13_weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)

        _CPU_MOE_LAYER_CACHE[id(layer)] = weakref.ref(layer)
        manager.set_total_layers(len(_CPU_MOE_LAYER_CACHE))


    def forward_grouped_gemm(
        self,
        layer: torch.nn.Module,
        input: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int = -1,
    ) -> torch.Tensor:
        
        output = cpu_fused_moe(
            input,
            layer.w13_weight,
            layer.w2_weight,
            getattr(layer, "w13_bias", None),
            getattr(layer, "w2_bias", None),
            topk_weights,
            topk_ids,
            activation,
            self.isa,
        )
        return output

    def forward_torch(
        self,
        layer: torch.nn.Module,
        input: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int = -1,
    ) -> torch.Tensor:
        output = torch.empty_like(input)
        layer_id = id(layer)
        torch.ops.vllm.cpu_fused_moe_torch(
            layer_id,
            output,
            input,
            topk_weights,
            topk_ids,
            activation,
            global_num_experts,
        )

        return output

#coolling: expert compute
def cpu_fused_moe_torch(
    layer_id: int,
    output: torch.Tensor,
    input: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    global_num_experts: int = -1,
) -> None:
    global manager
    # 获取 layer_id 在插入顺序中的 0-based 排名
    keys_in_order = list(_CPU_MOE_LAYER_CACHE.keys())
    rank = keys_in_order.index(layer_id)  # 0-based 
    print("=== cpu_fused_moe_torch 参数详情 ===")
    print(f"layer_id: {layer_id}")
    print(f"rank: {rank}")
    # print(f"global_num_experts: {global_num_experts}")
    # # 可选：打印部分数值（避免太多）
    # print(f"topk_ids (first 10): {topk_ids.flatten()[:10].tolist()}")
    layer = _CPU_MOE_LAYER_CACHE[layer_id]()
    print("====================================\n")
    # print(layer.w13_weight)
    while not manager.acquire_weights_for_layer(layer,rank):
        print("wait load",rank)
        time.sleep(0)
    _, w2_buf, w13_buf=manager.queue[0]
    print("layer.w13_weight.data_ptr()",layer.w13_weight.data_ptr())
    print("manager.global_w13_tensor.data_ptr()",manager.global_w13_tensor.data_ptr())
    len_experts = global_num_experts
    cnts = topk_ids.new_zeros((topk_ids.shape[0], len_experts))
    cnts.scatter_(1, topk_ids.to(torch.int64), 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = topk_ids.view(-1).argsort()
    sorted_tokens = input[idxs // topk_ids.shape[1]]
    tokens_per_expert = tokens_per_expert.cpu().numpy()

    outputs = []
    start_idx = 0
    
    for i, num_tokens in enumerate(tokens_per_expert):
        
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        layer.w2_weight[i]=w2_buf[i]
        layer.w13_weight[i]=w13_buf[i] 
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
        gate_up = layer.gate_up_linear[i](tokens_for_this_expert)  # type: ignore
        gate_up = _CPU_MOE_ACT[activation].forward_native(gate_up)
        expert_out = layer.down_linear[i](gate_up)  # type: ignore
        outputs.append(expert_out)
        start_idx = end_idx
        # manager.unload_expert(layer_id,i)
   
    manager.release_current_layer()
    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
    new_x = torch.empty_like(outs)

    new_x[idxs] = outs
    final_out = (
        new_x.view(*topk_ids.shape, -1)
        .type(topk_weights.dtype)
        .mul_(topk_weights.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )
    output.copy_(final_out)


direct_register_custom_op(
    op_name="cpu_fused_moe_torch",
    op_func=cpu_fused_moe_torch,
    mutates_args=["output"],
)
