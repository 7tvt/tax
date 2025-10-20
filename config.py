import os

# 税小助配置
CONFIG = {
    # 文档处理配置 - 调整为当前文件夹结构
    "pdf_path": "tax_laws.pdf",  # 直接在tax文件夹下
    "save_dir": "tax_db",
    "chunk_size": 512,
    "chunk_overlap": 64,

    # 模型配置（适合CPU运行）
    "embedding_model": "all-MiniLM-L6-v2",
    "llm_name": "distilgpt2",  # 统一为distilgpt2模型
    "max_new_tokens": 512,
    "temperature": 0.3,

    # 路径配置
    "data_dir": ".",  # 当前目录
    "checkpoint_dir": "checkpoints"
}

# 创建必要的目录
os.makedirs(CONFIG["data_dir"], exist_ok=True)
os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
os.makedirs(CONFIG["save_dir"], exist_ok=True)