import threading
import io
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
import torch
import os
import re
import tempfile
import shutil
import fcntl
import re
import os
import gc
import tempfile
import fcntl
import struct
import torch
from safetensors.torch import load_file
from safetensors import safe_open
import ctypes
import mmap
def fast_copy(dst: torch.Tensor, src: torch.Tensor):
    """
    使用内存映射进行复制（适合超大张量）
    """
    dst_ptr = dst.data_ptr()
    src_ptr = src.data_ptr()
    nbytes = dst.numel() * dst.element_size()
    # t=torch.empty_like(src)
    # t.copy_(src)
    # 使用memmove进行内存复制
    ctypes.memmove(dst_ptr, src_ptr, nbytes)
    
    return dst
def is_all_zero(tensor):
    return torch.allclose(tensor, torch.zeros_like(tensor))
def load_expert_weight_no_cache_final(
    hf_weights_files: list[str],
    weight_name: str = "",
    expert_dim: int = 256,
    parallel_loading: bool = True,
    max_workers: int = None,
    chunk_size_mb: int = 1
):
    
    def sanitize_filename(name: str) -> str:
        return re.sub(r'[\\/:*?"<>|]', '_', name)
    
    def read_file_chunk_to_memory(file_path, start_pos, chunk_size, worker_id):
        """读取文件的指定块到内存"""
        try:
            with open(file_path, "rb") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # 共享锁，允许多个读取
                try:
                    f.seek(start_pos)
                    data = f.read(chunk_size)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            return worker_id, data, len(data), None
        except Exception as e:
            return worker_id, None, 0, str(e)
    
    def parallel_read_and_parse(file_path, weight_name, max_workers=None, chunk_size_mb=10):
        """并行读取并解析safetensors文件"""
        # 第一步：先读取文件头（单独线程）
        print("读取safetensors文件头...")
        
        # safetensors文件格式：前8字节是header长度，然后是json header，最后是tensor数据
        with open(file_path, "rb") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                # 读取header长度（8字节，little-endian）
                header_len_bytes = f.read(8)
                if len(header_len_bytes) != 8:
                    raise ValueError("文件太小或格式错误")
                
                header_len = struct.unpack("<Q", header_len_bytes)[0]
                
                # 读取header JSON
                header_bytes = f.read(header_len)
                if len(header_bytes) != header_len:
                    raise ValueError("header长度不匹配")
                
                import json
                header = json.loads(header_bytes.decode('utf-8'))
                
                # 获取文件总大小和tensor数据位置
                total_size = os.path.getsize(file_path)
                data_start_pos = 8 + header_len  # 数据开始位置
                
                # 查找目标tensor的信息
                if "__metadata__" in header:
                    metadata = header["__metadata__"]
                    del header["__metadata__"]
                
                tensor_info = None
                for key, info in header.items():
                    if key == weight_name:
                        tensor_info = info
                        break
                
                if not tensor_info:
                    return None, list(header.keys())
                
                # 获取tensor的数据范围
                dtype = tensor_info["dtype"]
                shape = tensor_info["shape"]
                data_offsets = tensor_info["data_offsets"]
                tensor_start = data_start_pos + data_offsets[0]
                tensor_end = data_start_pos + data_offsets[1]
                tensor_size = tensor_end - tensor_start
                
                print(f"目标tensor: {weight_name}")
                print(f"  位置: {tensor_start}-{tensor_end} (大小: {tensor_size} 字节)")
                print(f"  形状: {shape}, 类型: {dtype}")
                
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        
        # 第二步：并行读取tensor数据
        print(f"并行读取tensor数据 ({tensor_size/1024/1024:.2f} MB)...")
        
        # 计算需要的线程数和块大小
        if max_workers is None:
            cpu_count = os.cpu_count() or 4
            max_workers = min(8, cpu_count * 2)
        
        # 计算每个线程读取的块
        chunk_size = 1024 * 1024
        num_chunks = math.ceil(tensor_size / chunk_size)
        
        # 限制线程数不超过块数
        actual_workers = min(max_workers, num_chunks)
        start=time.time()
        # 创建线程池
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures = []
            
            # 提交所有数据块读取任务
            for i in range(num_chunks):
                chunk_start = tensor_start + i * chunk_size
                actual_chunk_size = min(chunk_size, tensor_end - chunk_start)
                
                future = executor.submit(
                    read_file_chunk_to_memory,
                    file_path,
                    chunk_start,
                    actual_chunk_size,
                    i
                )
                futures.append(future)
            
            # 收集所有数据块
            chunks = [None] * num_chunks
            total_read = 0
            errors = []
            
            for future in as_completed(futures):
                worker_id, data, size, error = future.result()
                
                if error:
                    errors.append(f"块 {worker_id}: {error}")
                elif data:
                    chunks[worker_id] = data
                    total_read += size
                    
                    progress = total_read / tensor_size * 100
                    print(f"数据读取进度: {progress:.1f}% ({total_read/1024/1024:.2f}/{tensor_size/1024/1024:.2f} MB)")
            
            if errors:
                print(f"读取错误: {errors}")
                return None, []
        print(f"test：{(time.time()-start)*1000:.2f} ms")
        # 第三步：合并数据并创建tensor
        print("合并数据并创建tensor...")
        
        # # 合并所有数据块
        # combined_data = b''.join(chunks)
        
        # if len(combined_data) != tensor_size:
        #     print(f"数据大小不匹配: 期望 {tensor_size}, 实际 {len(combined_data)}")
        #     return None, []
        
        # # 将字节数据转换为torch tensor
        # import numpy as np
        
        # # 根据dtype创建numpy数组
        # dtype_map = {
        #     "F16": np.float16,
        #     "BF16": np.float16,  # 注意：numpy没有bfloat16，需要特殊处理
        #     "F32": np.float32,
        #     "F64": np.float64,
        #     "I8": np.int8,
        #     "I16": np.int16,
        #     "I32": np.int32,
        #     "I64": np.int64,
        #     "U8": np.uint8,
        #     "BOOL": bool,
        # }
        
        # if dtype not in dtype_map:
        #     print(f"不支持的数据类型: {dtype}")
        #     return None, []
        
        # np_dtype = dtype_map[dtype]
        
        # # 将字节数据转换为numpy数组
        # if dtype == "BF16":
        #     # bfloat16特殊处理：先读为uint16，然后转换
        #     import torch
        #     uint16_data = np.frombuffer(combined_data, dtype=np.uint16)
        #     # 使用torch的bfloat16支持
        #     tensor = torch.frombuffer(combined_data, dtype=torch.bfloat16).reshape(shape)
        # else:
        #     # 其他数据类型
        #     np_array = np.frombuffer(combined_data, dtype=np_dtype).reshape(shape)
        #     tensor = torch.from_numpy(np_array).clone()
        
        # # 确保在CPU上
        # tensor = tensor.cpu().contiguous()
        
        # return tensor, list(header.keys())
    
    

    
    # 查找目标文件
    target_file = None
    st_file = hf_weights_files[0]
    st_file_dir = os.path.dirname(st_file)
    experts_dir = os.path.join(st_file_dir, "experts")
    safe_name = sanitize_filename(weight_name)
    possible_file = os.path.join(experts_dir, f"{safe_name}.safetensors")
    
    
    target_file = possible_file
       
    
    if target_file is None:
        print(f"权重 {weight_name} 未找到")
        return None
    
    # 检查文件大小
    try:
        file_size = os.path.getsize(target_file)
        file_size_mb = file_size / (1024 * 1024)
        print(f"文件大小: {file_size_mb:.2f} MB")
    except:
        file_size_mb = 0
    
   
    
    try:
        print("使用并行内存读取模式...")
        tensor, all_keys = parallel_read_and_parse(
            target_file,
            weight_name,
            max_workers=max_workers,
            chunk_size_mb=chunk_size_mb
        )
       
        
       
        
        print(f"\n=== 加载成功 ===")
        print(f"权重名: {weight_name}")
        print(f"文件: {os.path.basename(target_file)}")
        print(f"形状: {tensor.shape}")
        print(f"类型: {tensor.dtype}")
        # print(f"模式: {'并行内存读取' if use_parallel else '传统文件复制'}")
        print(f"文件大小: {file_size_mb:.2f} MB")
        
        
        return tensor
        
    except Exception as e:
        print(f"加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None
# ==================== 调用验证 ====================
def tensors_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    # 检查形状是否相同
    if a.shape != b.shape:
        return False
    # 逐元素比较，全部为 True 则返回 True
    return torch.equal(a, b)  # 最高效！
def load_expert_weight(
    hf_weights_files: list[str],
    weight_name:str="",):
    
    import time 
    start = time.time()
    def sanitize_filename(name: str) -> str:
        return re.sub(r'[\\/:*?"<>|]', '_', name)

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

         
            os.system(f"posix_fadvise {tmp_file_path} 0 0 POSIX_FADV_DONTNEED 2>/dev/null")
            tensor_dict = load_file(tmp_file_path, device="cpu")
            tensor = tensor_dict.get(weight_name, None)
            os.unlink(tmp_file_path)

            if tensor is None:
                print(f"文件内无目标权重：{list(tensor_dict.keys())}")
                continue

            # === 形状验证+兜底拆分 ===
            # print(f"=== 无缓存加载结果 ===")
            # print(f"权重名：{weight_name}")
            # print(f"原始形状：{tensor.shape}")
            elapsed_ms = (time.time() - start) * 1000
            
            
           
            # print(f"加载时间 in {elapsed_ms:.2f} ms")
            return tensor

        except Exception as e:
            print(f"无缓存读取失败：{e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"权重 {weight_name} 未找到")
    return None

def load_expert_weight_safeopen(
    hf_weights_files: list[str],
    weight_name:str="",):
    import time 
    start = time.time()
    def sanitize_filename(name: str) -> str:
        return re.sub(r'[\\/:*?"<>|]', '_', name)

    for st_file in hf_weights_files:
        try:     
            with safe_open(st_file, framework="pt") as f:
                    full_tensor = f.get_tensor(weight_name)
                    return full_tensor
        except Exception as e:
            print(f"无缓存读取失败：{e}")
            continue
    print(f"权重 {weight_name} 未找到")
    return None
if __name__ == "__main__":
    import time
    weights_files = ["/mnt/nvme0/home/chenyunling/models/Isotonic/smol_llama-4x220M-MoE/model-00001-of-00001.safetensors"]
    target_weight = "model.layers.0.block_sparse_moe.experts.0.w2.weight"
    
    for i in range(1):
        
        # tensor1 = load_expert_weight_no_cache_final(weights_files, target_weight)
        # t=torch.empty_like(tensor1)
        # t.copy_(tensor1)
        # print(f"第{i}次加载耗时1：{(time.time()-start)*1000:.2f} ms")
        # start = time.time()
        start = time.time()
        tensor2 = load_expert_weight_no_cache_final(weights_files, target_weight)
        # print(tensor2)
        print(f"第{i}次加载耗时2：{(time.time()-start)*1000:.2f} ms")
        # t=torch.empty_like(tensor2)
        
        # fast_copy(t,tensor2)
        # is_all_zero(tensor2)
        # t=torch.empty_like(tensor2)
        # t.copy_(tensor2)
        
        # print(t)
        # print("是否正确",tensors_equal(tensor1,tensor2))