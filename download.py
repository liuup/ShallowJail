
from huggingface_hub import snapshot_download

model_id = "Qwen/Qwen3-4B-Instruct-2507"  # 替换为你想要下载的模型ID
local_dir = "./models/Qwen3-4B-Instruct-2507" # 本地保存路径

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False, # 重要：设置为 False 以便下载真实文件，而不是缓存的软链接
    resume_download=True,         # 支持断点续传
    # token="hf_xxxx",      # 如果是私有模型 (如 Llama 3)，需要填入 Token
)

print(f"Downloaded to: {local_dir}")