import json
from pathlib import Path

KEY_FILE = "key_api.txt"

def load_api_keys(file_path=KEY_FILE):
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def pop_api_key(line_num: int, file_path=KEY_FILE):
    keys = load_api_keys(file_path)
    if line_num < 1 or line_num > len(keys):
        raise ValueError(f"⚠️ Không có API key ở dòng {line_num} trong {file_path}")

    # Trả về key theo dòng chỉ định nhưng KHÔNG xóa khỏi file
    return keys[line_num - 1]

def load_config_with_keys(config_file: str):
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file config: {config_file}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    for item in config:
        api_key = item.get("api_key")
        if isinstance(api_key, str) and api_key.startswith("KEY_LINE_"):
            line_num = int(api_key.replace("KEY_LINE_", ""))
            item["api_key"] = pop_api_key(line_num)  # Lấy & xóa key

    return config
