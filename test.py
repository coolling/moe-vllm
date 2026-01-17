from safetensors.torch import safe_open, save_file,load_file
import os
import re

def save_expert_weights_single(
    hf_weights_files: list[str]
):
    # 清理文件名：仅替换非法字符，保留层级
    def sanitize_filename(name: str) -> str:
        return re.sub(r'[\\/:*?"<>|]', '_', name)

    # MoE专家配置（根据你的模型调整）
    EXPERT_DIM = 256  # 单个专家的输出维度
    EXPERT_KEY_PATTERN = r'experts.(\d+)'  # 匹配experts.0/1/2...

    for st_file in hf_weights_files:
        try:
            # 动态拼接experts目录
            st_file_dir = os.path.dirname(st_file)
            save_expert_dir = os.path.join(st_file_dir, "experts")
            os.makedirs(save_expert_dir, exist_ok=True)
            base_filename = os.path.basename(st_file).replace(".safetensors", "")

            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():
                    if "experts" in name.lower():
                        # ========== 核心修改：拆分拼接张量 ==========
                        # 1. 读取原始拼接张量
                        full_tensor = f.get_tensor(name)
                        print(f"原始拼接张量形状：{name} → {full_tensor.shape}")
                        
                        

                        # ========== 保存拆分后的张量 ==========
                        # 生成唯一文件名
                        safe_name = sanitize_filename(name)
                        save_path = os.path.join(
                            save_expert_dir,
                            f"{safe_name}.safetensors"
                        )

                        # 防覆盖：文件已存在则删除后重新保存
                        if os.path.exists(save_path):
                            os.remove(save_path)
                            print(f"删除旧文件：{save_path}")

                        # 保存拆分后的单个专家张量
                        save_file({name: full_tensor}, save_path)
                        print(f"✅ 保存成功：{save_path} → 形状{full_tensor.shape}")

                        # ========== 验证保存的文件 ==========
                        verify_dict = load_file(save_path, device="cpu")
                        verify_tensor = verify_dict[name]
                        print(f"✅ 验证保存文件：形状{verify_tensor.shape}")

        except Exception as e:
            print(f"处理文件 {st_file} 失败：{e}")
            continue

# ==================== 执行保存 ====================
if __name__ == "__main__":
    weights_files = ["/mnt/nvme0/home/chenyunling/models/mistralai/Mixtral-8x7B-Instruct-v0.1/model-00001-of-00019.safetensors",""]
    # 清空旧的experts文件（避免污染）
    experts_dir = "/mnt/nvme0/home/chenyunling/models/mistralai/Mixtral-8x7B-Instruct-v0.1/experts"
    if os.path.exists(experts_dir):
        for file in os.listdir(experts_dir):
            os.remove(os.path.join(experts_dir, file))
        print("已清空旧的experts文件")
    # 执行保存（先拆分再保存）
    save_expert_weights_single(weights_files)