import time
import os

def measure_binary_read_speed(filename):
    """测量二进制文件读取速度"""
    start_time = time.time()
    
    # 使用二进制模式读取
    with open(filename, 'rb') as f:
        content = f.read()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 获取文件大小
    file_size = os.path.getsize(filename)
    
    # 计算读取速度
    speed_mb_per_sec = (file_size / (1024 * 1024)) / elapsed_time
    speed_gb_per_sec = speed_mb_per_sec / 1024
    
    return elapsed_time, speed_mb_per_sec, speed_gb_per_sec, file_size

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

def measure_binary_read_speed_multithreaded(filename, chunk_size=65536, num_threads=1):
    """多线程分块读取二进制文件，测量速度"""
    start_time = time.time()
    
    # 获取文件大小
    file_size = os.path.getsize(filename)
    
    # 计算每个线程读取的字节范围
    chunk_per_thread = file_size // num_threads
    ranges = []
    for i in range(num_threads):
        start = i * chunk_per_thread
        end = start + chunk_per_thread - 1 if i < num_threads - 1 else file_size - 1
        ranges.append((start, end))
    
    # 用于存储结果的锁和共享变量
    lock = threading.Lock()
    total_bytes_read = 0
    results = []
    
    def read_chunk(thread_id, start_pos, end_pos):
        """单个线程的读取函数"""
        thread_bytes = 0
        with open(filename, 'rb') as f:
            f.seek(start_pos)
            bytes_to_read = end_pos - start_pos + 1
            
            while bytes_to_read > 0:
                read_size = min(chunk_size, bytes_to_read)
                chunk = f.read(read_size)
                if not chunk:
                    break
                thread_bytes += len(chunk)
                bytes_to_read -= len(chunk)
        
        with lock:
            nonlocal total_bytes_read
            total_bytes_read += thread_bytes
        
        return thread_id, thread_bytes
    
    # 使用线程池执行
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i, (start_pos, end_pos) in enumerate(ranges):
            future = executor.submit(read_chunk, i, start_pos, end_pos)
            futures.append(future)
        
        # 等待所有线程完成
        for future in as_completed(futures):
            thread_id, thread_bytes = future.result()
            results.append((thread_id, thread_bytes))
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 验证读取完整性
    if total_bytes_read != file_size:
        print(f"警告：读取不完整！期望 {file_size} 字节，实际读取 {total_bytes_read} 字节")
    
    # 计算读取速度
    speed_mb_per_sec = (total_bytes_read / (1024 * 1024)) / elapsed_time
    speed_gb_per_sec = speed_mb_per_sec / 1024
    
    return elapsed_time, speed_mb_per_sec, speed_gb_per_sec, total_bytes_read, results

def benchmark_safetensors_file(filename, chunk_sizes=None):
    """对safetensors文件进行完整的基准测试"""
    if chunk_sizes is None:
        chunk_sizes = [4096*4*4*4
                       ]  # 4KB 到 1MB
    
    file_size = os.path.getsize(filename)
    print(f"=== Safetensors文件读取基准测试 ===")
    print(f"文件: {filename}")
    print(f"大小: {file_size:,} 字节")
    print(f"     = {file_size/1024:.2f} KB")
    print(f"     = {file_size/1024/1024:.2f} MB")
    print(f"     = {file_size/1024/1024/1024:.2f} GB")
    print("-" * 60)
    

    
    # 方法2：不同块大小读取
    print("方法2: 不同块大小读取")
    for chunk_size in chunk_sizes:
        print(f"\n  块大小: {chunk_size:,} 字节 ({chunk_size/1024:.1f} KB)")
        elapsed_time, speed_mb_per_sec, speed_gb_per_sec, total_bytes_read, results = measure_binary_read_speed_multithreaded(filename, chunk_size,16)
        print(f"  读取时间: {elapsed_time:.6f} 秒")
        print(f"  读取速度: {speed_mb_per_sec:.2f} MB/秒")
        print(f"  读取速度: {speed_gb_per_sec:.4f} GB/秒")
        print(f"  读取字节: {total_bytes_read:,}")
        print(results)

    # 方法3：使用内存映射（适用于大文件）
    print("\n" + "-" * 60)
    print("方法3: 内存映射读取（适用于超大文件）")
    try:
        start_time = time.time()
        import mmap
        
        with open(filename, 'rb') as f:
            # 创建内存映射
            with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mmapped_file:
                # 访问文件内容（这里只是验证访问）
                _ = mmapped_file[:10000]  # 读取前100字节
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"  映射时间: {elapsed:.6f} 秒")
        print(f"  注意：内存映射延迟加载，实际访问时才会加载数据")
    except Exception as e:
        print(f"  内存映射错误: {e}")

# 使用示例
if __name__ == "__main__":
    # 您的safetensors文件路径
    safetensors_file = '/mnt/nvme0/home/chenyunling/models/Isotonic/smol_llama-4x220M-MoE/experts/model-00001-of-00001.safetensors'
    
    # 检查文件是否存在
    if os.path.exists(safetensors_file):
        benchmark_safetensors_file(safetensors_file)
    else:
        print(f"文件不存在: {safetensors_file}")
        
        # 可以创建一个测试二进制文件
        print("\n创建测试二进制文件...")
        test_file = "test_binary.bin"
        with open(test_file, 'wb') as f:
            # 写入100MB测试数据
            f.write(os.urandom(100 * 1024 * 1024))  # 100MB随机数据
        
        benchmark_safetensors_file(test_file)