import time
import numpy as np
import psutil
import ctypes
from concurrent.futures import ThreadPoolExecutor
import threading

def test_memory_bandwidth_basic(size_gb=1, operations=['read', 'write', 'copy']):
    """
    基础内存带宽测试
    size_gb: 测试数据大小 (GB)
    operations: 要测试的操作 ['read', 'write', 'copy']
    """
    size_bytes = int(size_gb * 1024**3)
    
    print(f"测试内存带宽 - 数据大小: {size_gb} GB ({size_bytes:,} 字节)")
    print(f"系统内存: {psutil.virtual_memory().total // 1024**3} GB")
    print(f"可用内存: {psutil.virtual_memory().available // 1024**3} GB")
    print("-" * 60)
    
    results = {}
    
    # 创建测试数据
    print("准备测试数据...")
    data = np.random.rand(size_bytes // 8).astype(np.float64)  # 使用float64 (8字节)
    
    if 'read' in operations:
        # 测试读取带宽
        print("测试读取带宽...")
        start = time.time()
        
        # 强制读取所有数据
        total = 0
        for i in range(0, len(data), 1024):  # 以1024个元素为步长
            chunk = data[i:i+1024]
            total += chunk.sum()  # 确保数据被实际读取
        
        end = time.time()
        elapsed = end - start
        bandwidth_gb_per_sec = size_gb / elapsed
        bandwidth_gb_per_sec = bandwidth_gb_per_sec if total > 0 else 0
        
        results['read'] = bandwidth_gb_per_sec
        print(f"  读取带宽: {bandwidth_gb_per_sec:.2f} GB/s")
    
    if 'write' in operations:
        # 测试写入带宽
        print("测试写入带宽...")
        start = time.time()
        
        # 创建新数组并写入数据
        new_data = np.empty_like(data)
        new_data[:] = data  # 批量写入
        
        end = time.time()
        elapsed = end - start
        bandwidth_gb_per_sec = size_gb / elapsed
        
        results['write'] = bandwidth_gb_per_sec
        print(f"  写入带宽: {bandwidth_gb_per_sec:.2f} GB/s")
    
    if 'copy' in operations:
        # 测试复制带宽
        print("测试复制带宽...")
        start = time.time()
        
        # 使用np.copy
        copied_data = data.copy()
        
        end = time.time()
        elapsed = end - start
        bandwidth_gb_per_sec = size_gb / elapsed
        
        results['copy'] = bandwidth_gb_per_sec
        print(f"  复制带宽: {bandwidth_gb_per_sec:.2f} GB/s")
    
    return results

test_memory_bandwidth_basic()