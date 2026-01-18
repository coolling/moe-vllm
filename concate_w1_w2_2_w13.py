import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file
import re

def sanitize_filename(name: str) -> str:
    """清理文件名中的非法字符"""
    return re.sub(r'[\\/:*?"<>|]', '_', name)

def concatenate_w1_w3_to_w13(
    base_dir: str,
    rank: int,
    expert_id: int,
    delete_original: bool = False
) -> bool:
    """
    将w1和w3文件拼接成w13文件
    
    Args:
        base_dir: 基础目录（包含experts文件夹）
        rank: 层ID
        expert_id: expert ID
        delete_original: 是否删除原始的w1和w3文件
    
    Returns:
        bool: 是否成功
    """
    # 构建原始文件名
    w1_key = f"model.layers.{rank}.mlp.experts.{expert_id}.up_proj.weight"
    w2_key = f"model.layers.{rank}.mlp.experts.{expert_id}.gate_proj.weight"
    w3_key = f"model.layers.{rank}.mlp.experts.{expert_id}.down_proj.weight"
    
    w13_key = f"model.layers.{rank}.block_sparse_moe.experts.{expert_id}.w13.weight"
    w_key = f"model.layers.{rank}.block_sparse_moe.experts.{expert_id}.w2.weight"
    
    # 转换为安全的文件名
    w1_safe = sanitize_filename(w1_key)
    w2_safe = sanitize_filename(w2_key)
    w3_safe = sanitize_filename(w3_key)
    w13_safe = sanitize_filename(w13_key)
    w_safe = sanitize_filename(w_key)
    # 构建文件路径
    experts_dir = os.path.join(base_dir, "experts")
    w1_file = os.path.join(experts_dir, f"{w1_safe}.safetensors")
    w3_file = os.path.join(experts_dir, f"{w3_safe}.safetensors")
    w13_file = os.path.join(experts_dir, f"{w13_safe}.safetensors")
    w2_file = os.path.join(experts_dir, f"{w2_safe}.safetensors")
    w_file = os.path.join(experts_dir, f"{w_safe}.safetensors")
    # 检查文件是否存在
    if not os.path.exists(w1_file):
        print(f"w1文件不存在: {w1_file}")
        return False
    
    if not os.path.exists(w3_file):
        print(f"w3文件不存在: {w3_file}")
        return False
    
    try:
        # 加载w1权重
        with safe_open(w1_file, framework="pt", device="cpu") as f:
            w1_tensor = f.get_tensor(w1_key)
            print(f"w1形状: {w1_tensor.shape}")
        
        # 加载w3权重
        with safe_open(w3_file, framework="pt", device="cpu") as f:
            w3_tensor = f.get_tensor(w3_key)
            print(f"w3形状: {w3_tensor.shape}")
        
        # 拼接w1和w3 (在dim=0上拼接)
        w13_tensor = torch.cat([w1_tensor, w3_tensor.transpose(0, 1)], dim=0)
        
        # 保存为新的w13文件
        save_file({w13_key: w13_tensor}, w13_file)
        print(f"已创建w13文件: {w13_file}")
        print(f"w13形状: {w13_tensor.shape}")
        # 加载w2权重
        with safe_open(w2_file, framework="pt", device="cpu") as f:
            w2_tensor = f.get_tensor(w2_key)
        save_file({w_key: w2_tensor}, w_file)
        print(f"已创建w13文件: {w_file}")
        print(f"w2形状: {w2_tensor.shape}")
        # 可选：删除原始文件
        if delete_original:
            os.remove(w1_file)
            os.remove(w3_file)
            print(f"已删除原始文件: {w1_file}, {w3_file}")
        
        return True
        
    except Exception as e:
        print(f"拼接w1和w3失败: {e}")
        return False

def process_all_experts(
    base_dir: str,
    num_layers: int,
    num_experts_per_layer: int,
    delete_original: bool = False
):
    """
    处理所有层的所有experts
    
    Args:
        base_dir: 基础目录
        num_layers: MoE层数
        num_experts_per_layer: 每层expert数
        delete_original: 是否删除原始文件
    """
    total_success = 0
    total_failed = 0
    
    for rank in range(num_layers):
        for expert_id in range(num_experts_per_layer):
            print(f"\n处理 layer={rank}, expert={expert_id}")
            
            success = concatenate_w1_w3_to_w13(
                base_dir=base_dir,
                rank=rank,
                expert_id=expert_id,
                delete_original=delete_original
            )
            
            if success:
                total_success += 1
            else:
                total_failed += 1
    
    print(f"\n处理完成:")
    print(f"成功: {total_success}")
    print(f"失败: {total_failed}")

def update_load_function_for_w13(
    weight_file: str,
    weight_name: str,
    output_buffer: torch.Tensor = None
) -> torch.Tensor | bool:
    """
    更新加载函数以支持w13文件
    
    Args:
        weight_file: 主safetensors文件路径
        weight_name: 权重名称，如 "model.layers.0.mlp.experts.0.w13.weight"
        output_buffer: 预分配的输出缓冲区
    
    Returns:
        如果output_buffer为None，返回张量
        如果output_buffer不为None，返回True/False
    """
    import time
    start = time.time()
    
    try:
        # 构建文件路径
        st_file_dir = os.path.dirname(weight_file)
        experts_dir = os.path.join(st_file_dir, "experts")
        safe_name = sanitize_filename(weight_name)
        target_file = os.path.join(experts_dir, f"{safe_name}.safetensors")
        
        # 检查w13文件是否存在
        if os.path.exists(target_file):
            # 直接加载w13文件
            with safe_open(target_file, framework="pt", device="cpu") as f:
                tensor = f.get_tensor(weight_name)
        else:
            # 回退到加载w1和w3然后拼接
            # 提取层ID和expert ID
            import re
            match = re.search(r'layers\.(\d+)\.mlp\.experts\.(\d+)', weight_name)
            if match:
                rank = int(match.group(1))
                expert_id = int(match.group(2))
                
                # 构建w1和w3的文件名
                w1_key = f"model.layers.{rank}.mlp.experts.{expert_id}.up_proj.weight"
                w3_key = f"model.layers.{rank}.mlp.experts.{expert_id}.down_proj.weight"
                
                w1_safe = sanitize_filename(w1_key)
                w3_safe = sanitize_filename(w3_key)
                
                w1_file = os.path.join(experts_dir, f"{w1_safe}.safetensors")
                w3_file = os.path.join(experts_dir, f"{w3_safe}.safetensors")
                
                # 加载w1和w3
                with safe_open(w1_file, framework="pt", device="cpu") as f:
                    w1_tensor = f.get_tensor(w1_key)
                
                with safe_open(w3_file, framework="pt", device="cpu") as f:
                    w3_tensor = f.get_tensor(w3_key)
                
                # 拼接
                tensor = torch.cat([w1_tensor, w3_tensor], dim=0)
            else:
                raise ValueError(f"无法从权重名中提取层ID和expert ID: {weight_name}")
        
        if output_buffer is not None:
            output_buffer.copy_(tensor)
            elapsed_ms = (time.time() - start) * 1000
            print(f"加载 {weight_name} 用时 {elapsed_ms:.2f} ms")
            return True
        else:
            tensor = tensor.detach().clone()
            elapsed_ms = (time.time() - start) * 1000
            print(f"加载 {weight_name} 用时 {elapsed_ms:.2f} ms")
            return tensor
            
    except Exception as e:
        print(f"加载失败：{e}")
        return None if output_buffer is None else False

# 使用示例
if __name__ == "__main__":
    # 1. 预处理：将w1和w3合并为w13
    base_dir = "/mnt/nvme0/home/chenyunling/models/Qwen/Qwen1.5-MoE-A2.7B-Chat"  # 修改为你的模型目录
    num_layers = 24  # 修改为你的MoE层数
    num_experts_per_layer = 60  # 修改为每层expert数
    
    # 处理所有experts（不删除原始文件）
    process_all_experts(
        base_dir=base_dir,
        num_layers=num_layers,
        num_experts_per_layer=num_experts_per_layer,
        delete_original=False  # 第一次运行时设为False，确认无误后再设为True
    )
    
   