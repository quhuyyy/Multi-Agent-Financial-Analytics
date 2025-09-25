from vi_rag_agent import VietnameseRAGAgent
from autogen import UserProxyAgent, config_list_from_json, AssistantAgent, ConversableAgent, GroupChat, GroupChatManager
from typing import Optional, Dict
from prompts import system_prompt, data_engineer_prompt, admin_prompt, query_maker_gpt_system_prompt, gpt_turbo_config, gpt_turbo_config_execute, visualize_agent_promt, generate_code_python_promt
from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain.schema.runnable import RunnableSequence
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import connectDB
import time
from concurrent.futures import ThreadPoolExecutor
import re
import os
GLOBAL_QUERY_CACHE = {} 

class VisualizeSQLDataSystem(ConversableAgent):
    def __init__(self, config_item, termination_msg, config_agent: Optional[Dict] = None):
        self._config_item = config_item
        self._termination_msg = termination_msg
        self._config_agent = {} if config_agent is None else config_agent
        self._schema = self._config_agent.get("schema", None)
        self._termination_msg=termination_msg
        self._connection = connectDB.get_Connection()

        self._rate_limiter = InMemoryRateLimiter(
            requests_per_second=1/20,
            check_every_n_seconds=0.1,
            max_bucket_size=10,
        )

        self._openaiLLM = ChatOpenAI(
            model=self._config_item["model"],
            temperature=self._config_item["temperature"],
            openai_api_key=self._config_item["api_key"],
            base_url = self._config_item["base_url"],
            rate_limiter=self._rate_limiter,
            cache=False
        )

        self._user_proxy = UserProxyAgent(
            name="Admin",
            human_input_mode="NEVER",
            system_message=admin_prompt + termination_msg,
            is_termination_msg=self.is_termination_msg
        )
        self._sql_agent = AssistantAgent(
            name="SQLAgent",
            llm_config=gpt_turbo_config,
            system_message=data_engineer_prompt + termination_msg,
            function_map={
                "query_maker": self.query_maker,
                "run_sql_query": self.run_sql_query,
            }
        )

        self._visualize_agent = AssistantAgent(
            name="VisualizeAgent",
            llm_config=gpt_turbo_config,
            system_message=visualize_agent_promt + termination_msg,
            function_map={
                "generate_plot_code": self.generate_plot_code,
                "run_python_code": self.run_python_code,
            }
        )
        self._user_proxy.register_function(function_map={
            "query_maker": self.query_maker,
            "run_sql_query": self.run_sql_query,
            "generate_plot_code": self.generate_plot_code,
            "run_python_code": self.run_python_code
        })

    @staticmethod
    def is_termination_msg(content):
        have_content = content.get("content", None) is not None
        if have_content and "Đã hoàn tất" in content["content"]:
            return True
        return False
    
    def query_maker(self, user_input):
        prompt_template = PromptTemplate.from_template(
            "{system_prompt} + '\n' +  {user_input}."
        )
        chain = RunnableSequence(prompt_template | self._openaiLLM)
        query = chain.invoke({"system_prompt": query_maker_gpt_system_prompt, "user_input": user_input})
        return query

    def run_sql_query(self, sql_query):
        cursor = self._connection.cursor()
        try:
            cursor.execute(sql_query)
            result = cursor.fetchall()
        except Exception as e:
            return e
        cursor.close()
        return result
    
    def generate_plot_code(self, plot_instructions: str, data_info: str, output_image_path: str):
        prompt_template = PromptTemplate.from_template(
            "{system_prompt}\nData info: {data_info}\nPlot instructions: {plot_instructions}\nOutput file: {output_image_path}"
        )
        chain = RunnableSequence(prompt_template | self._openaiLLM)
        code = chain.invoke({
            "system_prompt": generate_code_python_promt,
            "data_info": data_info,
            "plot_instructions": plot_instructions,
            "output_image_path": output_image_path
        })
        return code


    def run_python_code(self, python_code: str):
        try:
            local_namespace = {}
            exec(python_code, {}, local_namespace)
            return "Code executed successfully."
        except Exception as e:
            return f"Error during code execution: {str(e)}"

    def initiate_conversation(self, question, history):
        print(question)
        response = self._user_proxy.initiate_chat(
            self._sql_agent,
            message=f"Lịch sử trò chuyện: {history} \n Câu hỏi: {question}",
            clear_history=True
        )
        print(response.chat_history[-1]['content'].replace("Đã hoàn tất.", ""))
        pattern = r"(/app/images/[^\s)]+\.png)"
        match = re.search(pattern, response.chat_history[-1]['content'].replace("Đã hoàn tất.", ""))
        if match:
            link = match.group(1)
            print("Link:", link)
        else:
            print("Không tìm thấy link.")
        return link