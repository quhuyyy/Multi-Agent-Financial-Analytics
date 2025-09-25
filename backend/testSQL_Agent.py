import json
from text_to_sql_agent import TextToSQLAgent
with open(r'./OAI_CONFIG_LIST_GPT') as config_file:
    config = json.load(config_file)
config_item = config[0]
sqlagent = TextToSQLAgent(
    name="TextToSQLAgent",
    config_item = config_item,
    termination_msg="Nếu tất cả đều đạt chính xác, vui lòng phản hồi 'Đã hoàn tất'.",
    code_execution_config={"work_dir": "coding", "use_docker": False}
)
# sqlagent.initiate_conversation("Cổ phiếu BID đã giảm bao nhiêu phần trăm từ ngày 1 đến ngày 29 tháng 3 năm 2024?")

# sqlagent.initiate_conversation("Lấy lợi nhuận sau thuế của công ty có mã VNM trong năm 2023.")
sqlagent.initiate_conversation("Cổ phiếu có mã ACB là của công ty nào?")