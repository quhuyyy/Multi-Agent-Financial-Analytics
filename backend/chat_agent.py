# manager_agent.py
import os
from autogen import ConversableAgent, config_list_from_json, AssistantAgent, UserProxyAgent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.rate_limiters import InMemoryRateLimiter
from typing import Optional, Dict, Literal
from litellm import completion

import logging


logging.basicConfig(level=logging.INFO)
logging.getLogger("langchain").setLevel(logging.DEBUG)

SYSTEM_PROMPT="""
    Bạn là một trợ lý tài chính chuyên sâu, có kiến thức rộng về các lĩnh vực tài chính, kinh tế, đầu tư và thị trường. Nhiệm vụ của bạn là trả lời mọi câu hỏi về tài chính một cách chính xác, chi tiết và đáng tin cậy, không được phép trả lời sai.

    Các hướng dẫn quan trọng:
    1. Hiểu rõ và phân tích câu hỏi của người dùng, xác định mục tiêu cần tìm kiếm thông tin.
    2. Trả lời dựa trên các dữ liệu, nguyên tắc tài chính đã được kiểm chứng và thông tin cập nhật từ các nguồn uy tín.
    3. Kiểm tra và xác thực mọi thông tin trước khi trả lời; nếu không chắc chắn, hãy yêu cầu làm rõ câu hỏi hoặc thông báo giới hạn kiến thức.
    4. Cung cấp thông tin minh bạch, có trích dẫn nguồn (nếu cần) để chứng minh tính chính xác của dữ liệu.
    5. Sử dụng ngôn từ chuyên nghiệp, rõ ràng và dễ hiểu, tránh gây hiểu lầm cho người dùng.
    6. Không đưa ra các suy đoán không có căn cứ hoặc thông tin lỗi thời.
    7. Luôn cập nhật kiến thức và nhấn mạnh việc sử dụng các quy chuẩn, phương pháp kiểm định thông tin để đảm bảo tính chính xác cao nhất.

    Lịch sử trò chuyện: {history}

    Câu hỏi: {question}
"""

class ChatAgent(ConversableAgent):
    def __init__(self, config_item, termination_msg):
        self.config_item = config_item
        self._termination_msg = termination_msg

        self._assistant_chat_agent= AssistantAgent(
            name="AsisstantChatAgent",
            system_message=SYSTEM_PROMPT + self._termination_msg,
            llm_config={"config_list": config_item, "timeout": 60, "temperature": 0},
        )

        self._chat_agent = UserProxyAgent(
            name="ChatAgent",
            # termination_msg=termination_msg,
            human_input_mode="NEVER",
            is_termination_msg=self.is_termination_msg,
            code_execution_config={
                "work_dir": "coding", 
                "use_docker": False
            }
        )

    def initiate_conversation(self, question, history):
        response = self._chat_agent.initiate_chat(self._assistant_chat_agent, message=SYSTEM_PROMPT.format(question=question, history=history, termination_msg = self._termination_msg), clear_history=True)
        result = response.chat_history[-1]['content'].replace("Đã hoàn tất.", "").strip()
        return result
    
    @staticmethod
    def is_termination_msg(content):
        have_content = content.get("content", None) is not None
        if have_content and "Đã hoàn tất" in content["content"]:
            return True
        return False