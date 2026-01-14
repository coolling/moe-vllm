# 编译
VLLM_TARGET_DEVICE=cpu python setup.py install
# 运行
cd vllm
source vllm-env/bin/activate
export VLLM_CPU_KVCACHE_SPACE=1
export VLLM_CPU_OMP_THREADS_BIND=0-11
python -m vllm.entrypoints.openai.api_server --model /mnt/nvme0/home/chenyunling/models/Isotonic/smol_llama-4x220M-MoE --trust-request-chat-template --disable-custom-all-reduce --enforce-eager

# 客户端
python chat.py

# 监测
top -b -n 1 -c | grep -i "VLLM::EngineCore"