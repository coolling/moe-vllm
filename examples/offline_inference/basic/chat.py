# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


def create_parser():
    parser = FlexibleArgumentParser()
    # 添加引擎参数（模型路径、设备等）
    EngineArgs.add_cli_args(parser)
    # 替换默认模型为你的本地 smol_llama-4x220M-MoE 路径
    parser.set_defaults(model="/mnt/nvme0/home/chenyunling/models/Isotonic/smol_llama-4x220M-MoE")
    # 添加采样参数组
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int, default=512, help="生成的最大token数")
    sampling_group.add_argument("--temperature", type=float, default=0.7, help="随机性（0=确定性，1=高随机）")
    sampling_group.add_argument("--top-p", type=float, default=0.95, help="核采样阈值")
    sampling_group.add_argument("--top-k", type=int, default=50, help="Top-K采样阈值")
    # 添加聊天模板路径参数
    parser.add_argument("--chat-template-path", type=str, default=None, help="自定义聊天模板文件路径")
    # CPU 相关配置（可选）
    parser.add_argument("--cpu-threads", type=int, default=16, help="CPU推理的线程数，根据你的CPU核心数调整")

    return parser


def main(args: dict):
    # 提取并移除 LLM 初始化不需要的参数
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")
    chat_template_path = args.pop("chat_template_path")
    cpu_threads = args.pop("cpu_threads")

    # 强制使用 CPU 运行的关键配置
    # 1. device="cpu"：指定CPU设备
    # 2. tensor_parallel_size=1：CPU不支持张量并行，必须设为1
    # 3. cpu_offload_gb=0：禁用GPU显存占用
    # 4. num_cpu_workers：设置CPU工作线程数，优化推理速度
    llm = LLM(
        **args,
        device="cpu",          # 核心：指定CPU运行
        tensor_parallel_size=1, # CPU必须设为1（禁用张量并行）
        cpu_offload_gb=0,       # 禁用GPU显存
        num_cpu_workers=cpu_threads,  # CPU线程数，根据你的CPU核心数调整
        gpu_memory_utilization=0.0, # 彻底禁用GPU显存占用
    )

    # 配置采样参数（覆盖默认值）
    sampling_params = llm.get_default_sampling_params()
    if max_tokens is not None:
        sampling_params.max_tokens = max_tokens
    if temperature is not None:
        sampling_params.temperature = temperature
    if top_p is not None:
        sampling_params.top_p = top_p
    if top_k is not None:
        sampling_params.top_k = top_k

    # 定义输出打印函数
    def print_outputs(outputs):
        print("\nGenerated Outputs:\n" + "-" * 80)
        for idx, output in enumerate(outputs):
            prompt = output.prompt  # 模型实际接收的拼接后的prompt
            generated_text = output.outputs[0].text  # 生成的回复
            print(f"【对话 {idx+1}】")
            print(f"原始Prompt: {prompt[:200]}..." if len(prompt) > 200 else f"原始Prompt: {prompt}")
            print(f"\n生成回复: {generated_text}")
            print("-" * 80)

    print("=" * 80)
    print("开始CPU环境下的单轮多角色对话测试...")
    print(f"CPU线程数配置：{cpu_threads}")
    print("=" * 80)

    # 构造多轮对话示例（适配 smol_llama 的对话格式）
    conversation = [
        {"role": "system", "content": "You are a concise and helpful assistant, answer in Chinese."},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么我能帮助你的吗？"},
        {"role": "user", "content": "解释一下什么是MoE模型，用简单的话说明"},
    ]

    # 单轮对话推理（关闭进度条）
    outputs = llm.chat(conversation, sampling_params, use_tqdm=False)
    print_outputs(outputs)

    # 批量推理测试（减少批量数，CPU推理速度慢，避免等待过久）
    print("\n" + "=" * 80)
    print("开始CPU环境下的批量推理测试（2个相同对话）...")
    print("=" * 80)
    conversations = [conversation for _ in range(2)]  # CPU下减少批量数，从10改为2
    outputs = llm.chat(conversations, sampling_params, use_tqdm=True)
    print_outputs(outputs[:2])

    # 自定义聊天模板测试（如果指定了模板文件）
    if chat_template_path is not None:
        print("\n" + "=" * 80)
        print("使用自定义聊天模板推理...")
        print("=" * 80)
        with open(chat_template_path, "r", encoding="utf-8") as f:
            chat_template = f.read()
        outputs = llm.chat(
            conversations,
            sampling_params,
            use_tqdm=False,
            chat_template=chat_template,
        )
        print_outputs(outputs[:1])


if __name__ == "__main__":
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    main(args)