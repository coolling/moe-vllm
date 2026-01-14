import torch
import time

def measure_memcpy_16mb():
    """测量16MB内存拷贝的实际时间"""
    # 16 MB数据（float32）
    num_elements = 16 * 1024 * 1024 // 4  # 16 MB / 4字节 = 4,194,304个元素
    print(f"16MB数据包含 {num_elements:,} 个float32元素")
    
    # 创建源数据
    src = torch.randn(num_elements, dtype=torch.float32)
    dst = torch.empty_like(src)
    
    # 预热（让CPU进入稳定状态）
    print("预热中...")
    for _ in range(10):
        dst.copy_(src)
    
    # 实际测量
    print("开始正式测量...")
    num_iterations = 100
    total_time = 0
    
    for i in range(num_iterations):
        start = time.perf_counter()
        dst.copy_(src)
        torch.cuda.synchronize() if dst.is_cuda else None
        end = time.perf_counter()
        total_time += (end - start) * 1000  # 转换为毫秒
        
        # 每10次打印一次进度
        if (i + 1) % 10 == 0:
            avg_time = total_time / (i + 1)
            print(f"第 {i+1:3d} 次: 本次 {((end-start)*1000):.3f} ms, 平均 {avg_time:.3f} ms")
    
    avg_time = total_time / num_iterations
    print(f"\n=== 测量结果 ===")
    print(f"数据大小: 16 MB")
    print(f"迭代次数: {num_iterations}")
    print(f"平均拷贝时间: {avg_time:.3f} ms")
    print(f"平均带宽: {16 / (avg_time / 1000):.2f} GB/s")
    
    return avg_time

# 运行测试
time_ms = measure_memcpy_16mb()