# 编译
VLLM_TARGET_DEVICE=cpu python setup.py install
# 运行
cd vllm
source vllm-env/bin/activate
export VLLM_CPU_KVCACHE_SPACE=10
export VLLM_CPU_OMP_THREADS_BIND=20-31
python -m vllm.entrypoints.openai.api_server --model /mnt/nvme0/home/chenyunling/models/Isotonic/smol_llama-4x220M-MoE --trust-request-chat-template --disable-custom-all-reduce --enforce-eager
vllm serve --model /mnt/nvme0/home/chenyunling/models/Qwen/Qwen1.5-MoE-A2.7B-Chat --trust-request-chat-template --disable-custom-all-reduce --enforce-eager
vllm serve --model /mnt/nvme0/home/chenyunling/models/mistralai/Mixtral-8x7B-Instruct-v0.1 --trust-request-chat-template --disable-custom-all-reduce --enforce-eager
# 客户端
python chat.py

# 监测
top -b -n 1 -c | grep -i "VLLM::EngineCore"


# 编译
python3.10 -m venv vllm-env

# 3. 激活并安装
source vllm-env/bin/activate
pip install --upgrade pip
pip install -v -r requirements/cpu-build.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -v -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu




# 2. 清理所有构建缓存
rm -rf build/
rm -rf .deps/
rm -rf dist/
rm -rf *.egg-info/
rm -f CMakeCache.txt
rm -rf cmake-build-*/

# 3. 清理 Python 构建文件
find . -name "*.so" -type f -delete
find . -name "*.pyc" -type f -delete
find . -name "__pycache__" -type d -exec rm -rf {} +