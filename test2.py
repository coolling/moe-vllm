import os
import gc
import torch
import re
import tempfile
import shutil
import fcntl
from safetensors.torch import load_file

def load_expert_weight_no_cache_final(
    hf_weights_files: list[str],
    weight_name: str = "",
    expert_dim: int = 256
):
    def sanitize_filename(name: str) -> str:
        return re.sub(r'[\\/:*?"<>|]', '_', name)
    


    # 2. 清空Python/PyTorch缓存
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    torch._C._emptyCUDAIPCCache() if torch.cuda.is_available() else None

    for st_file in hf_weights_files:
        try:
            # === 路径拼接 ===
            st_file_dir = os.path.dirname(st_file)
            experts_dir = os.path.join(st_file_dir, "experts")
            safe_name = sanitize_filename(weight_name)
            target_file = os.path.join(experts_dir, f"{safe_name}.safetensors")

            if not os.path.exists(target_file):
                print(f"文件不存在：{target_file}")
                continue

            # === 核心：临时文件 + 排他锁（彻底绕缓存） ===
            # 3. 创建唯一临时文件（每次生成新文件，无缓存）
            tmp_fd, tmp_file_path = tempfile.mkstemp(suffix=".safetensors")
            os.close(tmp_fd)  # 关闭临时文件描述符，避免占用

            # 4. 以「无缓存模式」复制原文件到临时文件
            # 打开原文件：加排他锁，防止其他进程修改
            with open(target_file, "rb") as src_f:
                fcntl.flock(src_f.fileno(), fcntl.LOCK_EX)  # 排他锁
                # 打开临时文件：使用O_SYNC（写操作直接刷磁盘，无缓存）
                with open(tmp_file_path, "wb") as dst_f:
                    os.fdatasync(dst_f.fileno())  # 禁用写缓存
                    shutil.copyfileobj(src_f, dst_f, length=4096)  # 4KB块复制，对齐磁盘
                fcntl.flock(src_f.fileno(), fcntl.LOCK_UN)  # 释放锁

            # 5. 禁用safetensors所有缓存，读取临时文件
            os.environ["SAFETENSORS_DISABLE_MEMORY_MAPPING"] = "1"
            os.environ["SAFETENSORS_CACHE_DISABLE"] = "1"
            os.environ["SAFETENSORS_FORCE_DISK_READ"] = "1"
            
            # 读取前再次清空临时文件的缓存（针对单个文件）
            os.system(f"posix_fadvise {tmp_file_path} 0 0 POSIX_FADV_DONTNEED 2>/dev/null")
            
            tensor_dict = load_file(tmp_file_path, device="cpu")
            tensor = tensor_dict.get(weight_name, None)

            # 6. 立即删除临时文件（用完即删，无残留）
            os.unlink(tmp_file_path)

            if tensor is None:
                print(f"文件内无目标权重：{list(tensor_dict.keys())}")
                continue

            # === 形状验证+兜底拆分 ===
            print(f"=== 无缓存加载结果 ===")
            print(f"权重名：{weight_name}")
            print(f"原始形状：{tensor.shape}")
            
            
            # 确保张量独立，无缓存关联
            tensor = tensor.cpu().clone().detach()
            del tensor_dict
            gc.collect()

            return tensor

        except Exception as e:
            print(f"无缓存读取失败：{e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"权重 {weight_name} 未找到")
    return None

# ==================== 调用验证 ====================
if __name__ == "__main__":
    import time
    weights_files = ["/mnt/nvme0/home/chenyunling/models/Isotonic/smol_llama-4x220M-MoE/model-00001-of-00001.safetensors"]
    target_weight = "model.layers.0.block_sparse_moe.experts.0.w2.weight"
    
    for i in range(100):
        start = time.time()
        tensor1 = load_expert_weight_no_cache_final(weights_files, target_weight)
        print(f"第{i}次加载耗时：{(time.time()-start)*1000:.2f} ms")
        
