# 编译
VLLM_TARGET_DEVICE=cpu python setup.py install
# 运行
cd vllm
source vllm-env/bin/activate
export VLLM_CPU_KVCACHE_SPACE=10
export VLLM_CPU_OMP_THREADS_BIND=45-56
pip install polars

python -m vllm.entrypoints.openai.api_server --model /mnt/nvme0/home/chenyunling/models/Isotonic/smol_llama-4x220M-MoE --trust-request-chat-template --disable-custom-all-reduce --enforce-eager
vllm serve --model /sharenvme/usershome/cyl/test/model/Qwen/Qwen1.5-MoE-A2.7B --trust-request-chat-template --disable-custom-all-reduce --enforce-eager
vllm serve --model /sharenvme/usershome/cyl/test/model/mistralai/Mixtral-8x7B-Instruct-v0.1 --trust-request-chat-template --disable-custom-all-reduce --enforce-eager
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
cp  ./safetensors_reader.cpython-310-x86_64-linux-gnu.so ./vllm-env/bin/



# 2. 清理所有构建缓存
rm -rf build/
rm -rf .deps/
rm -rf dist/
rm -rf *.egg-info/
rm -f CMakeCache.txt
rm -rf cmake-build-*/

# 注意修改适配
self.topk=2 缓存的每层专家数、预测的下层专家加载顺序时算法输出的专家数