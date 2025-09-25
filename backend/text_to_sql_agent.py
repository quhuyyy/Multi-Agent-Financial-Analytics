import connectDB
import autogen
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from autogen import ConversableAgent, GroupChat, GroupChatManager
from prompts2 import gpt_turbo_config, query_maker_gpt_system_prompt, agent_excute, data_engineer_prompt
from typing import Optional, Dict
from langchain_core.runnables import RunnableSequence
from sqlalchemy import create_engine
import pandas as pd
import os
import time
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import text
import asyncio
GLOBAL_QUERY_CACHE = {} 

class TextToSQLAgent(ConversableAgent):
    def __init__(self, name, termination_msg, config_item, config_openai,code_execution_config, param_text2sql: Optional[Dict] = None):
        super().__init__(
            name=name,
            human_input_mode="NEVER",
            llm_config=gpt_turbo_config,
            system_message=data_engineer_prompt + termination_msg,
            is_termination_msg=self.is_termination_msg,
            code_execution_config=code_execution_config,
        )
        self._param_text2sql = {} if param_text2sql is None else param_text2sql
        self._create_database = self._param_text2sql.get("create_database", False)
        self.termination_msg=termination_msg
        self.code_execution_config = code_execution_config
        self.config_item = config_item
        self.register_function(function_map={
            "query_maker": self.query_maker,
            "run_sql_query": self.run_sql_query
        })
        
        self.executor_agent = autogen.ConversableAgent(
            name="Executor_Agent",
            llm_config=gpt_turbo_config,
            code_execution_config=code_execution_config,
            system_message=agent_excute + termination_msg,
            is_termination_msg=self.is_termination_msg,
            function_map={
                "query_maker": self.query_maker,
                "run_sql_query": self.run_sql_query,
            }
        )
        self.config_openai = config_openai
        openaiLLM = ChatOpenAI(
            model=self.config_openai["model"],
            temperature=self.config_openai["temperature"],
            openai_api_key=self.config_openai["api_key"],
            base_url = self.config_openai["base_url"],
            cache=False
        )
        prompt_template = PromptTemplate.from_template(
            "{system_prompt} + '\n' +  {user_input}."
        )
        self.chain = RunnableSequence(prompt_template | openaiLLM)
        
        self.executor = ThreadPoolExecutor(max_workers=2)
        if self._create_database == True:
            for i in os.listdir("./data_main"):
                if i[-4:] == ".csv":
                    self.import_data(i[:-4])
            self.add_contrains()
        self.connection = connectDB.get_Connection()

    def import_data(self, data):
        username = "root"  
        password = "root"  
        host = "mysql_container"  
        port = 3306
        database = "Financial"

        engine = create_engine(f"mysql+pymysql://{username}:{password}@{host}:{port}")

        sql = f"CREATE DATABASE IF NOT EXISTS {database};"

        try:
            with engine.connect() as connection:
                try:
                    result = connection.execute(text(sql))
                    print("Tạo database thành công")
                except Exception as e:
                    print(f"Lỗi khi chạy: {sql}\nError: {e}")
        except Exception as e:
            print(f"Lỗi kết nối MySQL: {e}")

        engine = create_engine(f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}")

        df = pd.read_csv(f"./data_main/{data}.csv")

        df.to_sql(f"{data}", con=engine, if_exists="append", index=False)
        print("Dữ liệu đã được nhập thành công vào MySQL!")

    def add_contrains(self):
        username = "root"  
        password = "root"  
        host = "mysql_container"  
        port = 3306
        database = "Financial"

        engine = create_engine(f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}")

        sql_statements = [
            "USE Financial;",
            "ALTER TABLE company_overview MODIFY Tickers VARCHAR(3) NOT NULL;",
            "ALTER TABLE company_overview ADD PRIMARY KEY (Tickers);",
            "ALTER TABLE stock_codes_by_exchange MODIFY symbol VARCHAR(3) NOT NULL;",
            "ALTER TABLE stock_codes_by_exchange ADD PRIMARY KEY (symbol);",
            "ALTER TABLE company_overview ADD CONSTRAINT fk_company_stock FOREIGN KEY (Tickers) REFERENCES stock_codes_by_exchange (symbol);",
            "ALTER TABLE all_tickers_history MODIFY Ticker VARCHAR(3) NOT NULL;",
            "ALTER TABLE all_tickers_history ADD CONSTRAINT fk_all_tickers FOREIGN KEY (Ticker) REFERENCES company_overview (Tickers);",
            "ALTER TABLE company_dividends MODIFY Symbol VARCHAR(3) NOT NULL;",
            "ALTER TABLE company_dividends ADD CONSTRAINT fk_company_dividends FOREIGN KEY (Symbol) REFERENCES company_overview (Tickers);",
            "ALTER TABLE company_news MODIFY Symbol VARCHAR(3) NOT NULL;",
            "ALTER TABLE company_news ADD CONSTRAINT fk_company_news FOREIGN KEY (Symbol) REFERENCES company_overview (Tickers);",
            "ALTER TABLE events MODIFY Symbol VARCHAR(3) NOT NULL;",
            "ALTER TABLE events ADD CONSTRAINT fk_events FOREIGN KEY (Symbol) REFERENCES company_overview (Tickers);",
            "ALTER TABLE insider_trading MODIFY Ticker VARCHAR(3) NOT NULL;",
            "ALTER TABLE insider_trading ADD CONSTRAINT fk_insider_trading FOREIGN KEY (Ticker) REFERENCES company_overview (Tickers);",
            "ALTER TABLE profit_loss MODIFY CP VARCHAR(3) NOT NULL;",
            "ALTER TABLE profit_loss ADD CONSTRAINT fk_profit_loss FOREIGN KEY (CP) REFERENCES company_overview (Tickers);",
            "ALTER TABLE profit_loss CHANGE COLUMN `CP` `stock_code` VARCHAR(10);",
            "ALTER TABLE profit_loss CHANGE COLUMN `Năm` `year` BIGINT;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Tăng trưởng doanh thu (%)` `revenue_growth` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Doanh thu (Tỷ đồng)` `revenue` BIGINT;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Lợi nhuận sau thuế của Cổ đông công ty mẹ (Tỷ đồng)` `net_profit_after_tax_for_parent_company` BIGINT;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Tăng trưởng lợi nhuận (%)` `profit_growth` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Thu nhập tài chính` `financial_income` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Doanh thu bán hàng và cung cấp dịch vụ` `sales_revenue` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Các khoản giảm trừ doanh thu` `revenue_deductions` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Doanh thu thuần` `net_revenue` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Giá vốn hàng bán` `cost_of_goods_sold` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Lãi gộp` `gross_profit` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Chi phí tài chính` `financial_expenses` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Chi phí bán hàng` `sales_expenses` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Chi phí quản lý DN` `management_expenses` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Lãi/Lỗ từ hoạt động kinh doanh` `operating_profit_loss` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Thu nhập khác` `other_income` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Thu nhập/Chi phí khác` `other_income_expenses` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Lợi nhuận khác` `other_profit` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `LN trước thuế` `profit_before_tax` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Chi phí thuế TNDN hiện hành` `current_tax_expenses` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Chi phí thuế TNDN hoãn lại` `deferred_tax_expenses` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Lợi nhuận thuần` `net_profit_final` BIGINT;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Cổ đông của Công ty mẹ` `parent_company_shareholders` BIGINT;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Ticker` `ticker_symbol` TEXT;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Chi phí tiền lãi vay` `interest_expenses` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Lãi/lỗ từ công ty liên doanh` `profit_loss_from_joint_ventures` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Lãi lỗ trong công ty liên doanh, liên kết` `profit_loss_from_associates` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Cổ đông thiểu số` `minority_shareholders` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Thu nhập lãi và các khoản tương tự` `interest_and_related_income` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Chi phí lãi và các khoản tương tự` `interest_and_related_expenses` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Thu nhập lãi thuần` `net_interest_income` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Thu nhập từ hoạt động dịch vụ` `service_revenue` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Chi phí hoạt động dịch vụ` `service_expenses` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Lãi thuần từ hoạt động dịch vụ` `net_service_profit` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Kinh doanh ngoại hối và vàng` `forex_and_gold_trading` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Chứng khoán kinh doanh` `trading_securities` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Chứng khoán đầu tư` `investment_securities` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Hoạt động khác` `other_operations` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Chi phí hoạt động khác` `other_operating_expenses` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Lãi/lỗ thuần từ hoạt động khác` `net_profit_loss_from_other_operations` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Cố tức đã nhận` `dividends_received` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Tổng thu nhập hoạt động` `total_operating_income` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `LN từ HĐKD trước CF dự phòng` `operating_profit_before_provision` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Chi phí dự phòng rủi ro tín dụng` `credit_risk_provision_expenses` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Thuế TNDN` `corporate_income_tax` DOUBLE;",
            "ALTER TABLE profit_loss CHANGE COLUMN `Lãi cơ bản trên cổ phiếu` `basic_eps` DOUBLE;",
            "SHOW TABLES;"
        ]

        try:
            with engine.connect() as connection:
                for sql in sql_statements:
                    try:
                        result = connection.execute(text(sql))
                        print(f"Đã chạy {sql}")
                        print(result)
                    except Exception as e:
                        print(f"Lỗi khi chạy: {sql}\nError: {e}")
        except Exception as e:
            print(f"Lỗi kết nối MySQL: {e}")


    @staticmethod
    def is_termination_msg(content):
        have_content = content.get("content", None) is not None
        if have_content and "Đã hoàn tất" in content["content"]:
            return True
        return False
    
    @staticmethod
    def is_sql_error(result):
        """Kiểm tra kết quả trả về có phải là lỗi SQL hoặc kết quả không hợp lệ."""
        if isinstance(result, str) and any(err in result.lower() for err in ["không thể tạo truy vấn sql", "error", "invalid", "unknown", "division by zero","does not exist"]):
            return result
        if isinstance(result, list) and (len(result) == 0 or result[0][0] is None):
            return result
        return False
    
    async def auto_fix_query(self, user_input, max_retries=5):
        """
        Thực hiện tự động tạo lại truy vấn dựa trên phản hồi lỗi.
        - user_input: yêu cầu ban đầu của người dùng.
        - max_retries: số lần thử tối đa.
        """
        attempt = 0
        while attempt < max_retries:
            # Tạo truy vấn bằng cách gọi hàm bất đồng bộ
            query = await self.query_maker_async(user_input)
            print(f"[Attempt {attempt+1}] Generated query:\n{query}")
            # Thực thi truy vấn SQL
            result = self.run_sql_query(query)
            if not self.is_sql_error(result):
                # Nếu không có lỗi, trả về kết quả
                return result
            else:
                # Nếu có lỗi, ghi nhận thông báo lỗi để cập nhật lại đầu vào và thử lại
                print(f"Lỗi phát sinh: {result}\nĐang thử tạo lại truy vấn...")
                # Cập nhật đầu vào để agent có thêm ngữ cảnh về lỗi (ví dụ: thêm thông báo lỗi vào yêu cầu)
                self.user_input += f" Lỗi: {result}"[:500]
                attempt += 1
        return result

    def query_maker_with_auto_fix(self, user_input):
        """
        Hàm đồng bộ bọc lớp auto_fix_query để sử dụng trong môi trường không hỗ trợ async trực tiếp.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self.auto_fix_query(user_input), loop)
            return future.result()
        else:
            return asyncio.run(self.auto_fix_query(user_input))

    # async def query_maker_async(self, user_input):
    #     try:
    #         result = await self.chain.arun({
    #             "system_prompt": query_maker_gpt_system_prompt,
    #             "user_input": user_input
    #         })
            
    #         return result
    #     except Exception as e:
    #         print(f"Lỗi khi gọi OpenAI: {e}")
    #         # Không cache kết quả lỗi, trả về thông báo lỗi để cho phép thử lại sau
    #         return f"Lỗi: {e}"
    async def query_maker_async(self, user_input):
        try:
            # # Gọi model Gemini
            # response = self.chain.generate_content(
            #     f"{query_maker_gpt_system_prompt}\n{user_input}"
            # )
            
            # result = response.text  # Lấy nội dung kết quả
            result = await self.chain.ainvoke({
                "system_prompt": query_maker_gpt_system_prompt,
                "user_input": user_input
            })
            
            return result.content
        except Exception as e:
            print(f"Lỗi khi gọi OpenAI: {e}")
            return f"Lỗi: {e}"
    

    def query_maker(self, user_input):
        # Nếu đã có event loop, ta nên dùng asyncio.create_task thay vì asyncio.run
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Nếu đang trong event loop, tạo task và đợi kết quả
            return asyncio.run_coroutine_threadsafe(self.query_maker_async(user_input), loop).result()
        else:
            return asyncio.run(self.query_maker_async(user_input))

    def run_sql_query(self, sql_query):
        """
        Thực thi truy vấn SQL. Nếu kết quả đã được cache thì trả về cache,
        nếu không, chạy truy vấn bất đồng bộ và cache kết quả lại.
        """
        
        def execute_query():
            start_time = time.time()
            cursor = self.connection.cursor()
            try:
                cursor.execute(sql_query)
                result = cursor.fetchall()
            except Exception as e:
                result = f"Không thể tạo truy vấn SQL mới do lỗi: {e}"
            finally:
                cursor.close()
            elapsed = time.time() - start_time
            print(f"⏱️ Thời gian thực thi truy vấn: {elapsed:.3f} giây")
            return result
        
        future = self.executor.submit(execute_query)
        result = future.result()
        
        return result
    
    def initiate_conversation(self, message, history):
        print(f"💬 Bắt đầu cuộc hội thoại với nội dung: {message}")
        
        response = self.initiate_chat(self,message=f"Lịch sử trò chuyện: {history} \n Câu hỏi: {message}", clear_history=True)
        return response.chat_history[-2]['content'].replace("Đã hoàn tất.", "")