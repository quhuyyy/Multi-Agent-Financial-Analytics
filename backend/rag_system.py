from vi_rag_agent import VietnameseRAGAgent
from autogen import UserProxyAgent, config_list_from_json, AssistantAgent, ConversableAgent
from typing import Optional, Dict
from prompts import assistan_rag_prompt

SYSTEM_PROMPT = """Bạn là một trợ lý chuyên trả lời câu hỏi. Hãy trả lời câu hỏi chỉ dựa trên bối cảnh được cung cấp.
    Nếu câu hỏi không thể được trả lời bằng bối cảnh đó, chỉ cần nói “Tôi không biết”. Đừng bịa chuyện.

    Lịch sử trò chuyện là: {history}

    Câu hỏi của người dùng là: {question}

    Bối cảnh là: {context}

    {termination_msg}

    """

user_prompt = """
    #     Question: {question}

    #     Answer:"""


class Rag_System(ConversableAgent):
    def __init__(self, config_item,termination_msg, param_rag: Optional[Dict] = None):
        self.config_item = config_item
        self._param_rag = {} if param_rag is None else param_rag
        self._termination_msg = termination_msg
        
        self._assistant_rag= AssistantAgent(
            name="assistant_rag",
            system_message=assistan_rag_prompt + self._termination_msg,
            llm_config={"config_list": config_item, "timeout": 60, "temperature": 0},
        )

        print(self._param_rag.get("input_path", None))

        self._Vi_rag = VietnameseRAGAgent(
            name="vietnamese_rag",
            termination_msg=termination_msg,
            human_input_mode="NEVER",
            param_rag={
                "input_path": self._param_rag.get("input_path", None),  # Đảm bảo bạn có giá trị cho input_path
                "collection_name": self._param_rag.get("collection_name", "collection_1"),
                "model_name": self._param_rag.get("model_name", "BAAI/bge-small-en-v1.5"),
                "create_collection": self._param_rag.get("create_collection", True),
                "va_api_key": self._param_rag.get("va_api_key", None),
                "url_va": self._param_rag.get("url_va", "https://api.va.landing.ai/v1/tools/agentic-document-analysis"),
                "untructured_api_key": self._param_rag.get("untructured_api_key", None),
                "untructured_api_url": self._param_rag.get("untructured_api_url", "https://api.unstructured.io"),
                "openai_api_key": self._param_rag.get("openai_api_key", None),
                "requests_per_second": self._param_rag.get("requests_per_second", 1/20),
                "check_every_n_seconds": self._param_rag.get("check_every_n_seconds", 0.1),
                "max_bucket_size": self._param_rag.get("max_bucket_size", 10),
                "max_concurrency": self._param_rag.get("max_concurrency", 5),
                "context_max_tokens": self._param_rag.get("context_max_tokens", 1500),
                "top_k": self._param_rag.get("top_k", 5),
                "customized_prompt": self._param_rag.get("customized_prompt", None),
            },
            is_termination_msg=self.is_termination_msg,
            code_execution_config={
                "work_dir": "coding", 
                "use_docker": False
            }
        )

    @staticmethod
    def is_termination_msg(content):
        have_content = content.get("content", None) is not None
        if have_content and "Đã hoàn tất" in content["content"]:
            return True
        return False

    def initiate_conversation(self, question, history):
        results = self._Vi_rag.search(question)
        references = [obj.payload["content"] for obj in results]
        context_text = "\n\n".join(references)
        response = self._Vi_rag.initiate_chat(self._assistant_rag, message=SYSTEM_PROMPT.format(question=question, history=history, context=context_text, termination_msg = self._termination_msg), clear_history=True)
        return response.chat_history[-1]['content'].replace("Đã hoàn tất.", "")

