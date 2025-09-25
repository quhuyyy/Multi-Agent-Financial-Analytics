# manager_agent.py
import os
from autogen import ConversableAgent, config_list_from_json, AssistantAgent, UserProxyAgent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.rate_limiters import InMemoryRateLimiter
from typing import Optional, Dict, Literal

import logging

prompt_template = """
Bạn là trợ lý phân loại câu hỏi. Dựa trên nội dung câu hỏi dưới đây, hãy xác định loại xử lý phù hợp theo các quy tắc sau:

    - flow_1: Nếu câu hỏi có thể được trả lời trực tiếp bởi LLM mà không cần tra cứu dữ liệu bên ngoài. Ví dụ: "Thời tiết hôm nay như thế nào?" hay "Hãy kể cho tôi nghe một câu chuyện cười."

    - flow_2: Nếu câu hỏi yêu cầu truy vấn cơ sở dữ liệu có metadata như sau: {metadata}, và câu hỏi không yêu cầu trình bày dữ liệu dưới dạng biểu đồ. Ví dụ: "Doanh thu quý 1 của công ty ABC là bao nhiêu?"

    - flow_3: Nếu câu hỏi yêu cầu truy vấn cơ sở dữ liệu kết hợp với việc vẽ biểu đồ để hiển thị dữ liệu trực quan. Ví dụ: "Hãy vẽ biểu đồ so sánh doanh thu các quý của công ty ABC" hay "Biểu đồ tăng trưởng lợi nhuận qua các năm của công ty XYZ."

    - flow_4: Nếu câu hỏi liên quan đến báo cáo tài chính hoặc các chỉ số tài chính và các thông tin trong báo cáo tài chính như sau: {metadata_bctc}, yêu cầu trích xuất thông tin từ báo cáo tài chính. Ví dụ: "Lợi nhuận sau thuế quý 2 của ACB là bao nhiêu?" hay "Tỉ lệ tăng trưởng doanh thu năm 2023 của công ty nào?"
    
    - flow_5: Nếu câu hỏi liên quan đến dự báo nên mua bán hay giữ một cổ phiếu nào đó trong 1 ngày mà không có bất kì thay đổi feature nào.
    
    - flow_6: Nếu câu hỏi liên quan đến việc giả sử 1 giá trị một khoảng và muốn dự đoán đầu ra của mô hình (scenario analysis). Ví dụ: tăng P/E_Previous_Quarter lên 10% thì giá cổ phiếu sẽ tăng bao nhiêu? (Thường câu hỏi này sẽ xuất hiện sau khi người dùng dự đoán 1 ngày và họ sẽ hỏi tiếp)

Lịch sử trò chuyện trước đó: {history}
Câu hỏi: {question}
Chỉ trả lời bằng một trong các từ khóa: flow_1, flow_2, flow_3, flow_4, flow_5, flow_6
Trả lời trên cùng 1 hàng.
"""

metadata = """
1.	Bảng dữ liệu giá cổ phiếu (all_tickers_history): Ghi nhận thông tin lịch sử giá cổ phiếu theo từng ngày giao dịch.
•	time: Ngày giao dịch
•	open: Giá mở cửa
•	high: Giá cao nhất trong ngày
•	low: Giá thấp nhất trong ngày
•	close: Giá đóng cửa
•	volume: Khối lượng giao dịch
•	Ticker: Mã cổ phiếu
2.	Bảng cổ tức (company_dividen): Thông tin về việc chi trả cổ tức của các công ty.
•	exercise_date: Ngày thực hiện chi trả cổ tức
•	cash_year: Năm tài chính trả cổ tức
•	cash_dividend_percentage: Tỷ lệ cổ tức bằng tiền mặt
•	issue_method: Phương thức chi trả (tiền mặt, cổ phiếu,...)
•	Symbol: Mã cổ phiếu
3.	Bảng tin tức chứng khoán (company_news): Thông tin tin tức liên quan đến các cổ phiếu.
•	rsi: Chỉ số sức mạnh tương đối (Relative Strength Index)
•	rs: Chỉ số sức mạnh tương đối theo khối lượng
•	price: Giá cổ phiếu
•	price_change: Thay đổi giá
•	price_change_ratio: Tỷ lệ thay đổi giá
•	price_change_ratio_1m: Tỷ lệ thay đổi giá trong 1 tháng
•	id: Mã tin tức
•	title: Tiêu đề tin tức
•	source: Nguồn tin tức
•	publish_date: Ngày xuất bản tin tức
•	Symbol: Mã cổ phiếu
4.	Bảng thông tin tổng quan công ty (company_overview): Thông tin cơ bản và tình hình hoạt động của các công ty.
•	exchange: Sàn giao dịch
•	industry: Ngành nghề hoạt động
•	company_type: Loại hình doanh nghiệp
•	no_shareholders: Số lượng cổ đông
•	foreign_percent: Tỷ lệ sở hữu nước ngoài
•	outstanding_share: Số lượng cổ phiếu đang lưu hành
•	issue_share: Số cổ phiếu phát hành
•	established_year: Năm thành lập
•	no_employees: Số lượng nhân viên
•	stock_rating: Đánh giá cổ phiếu
•	delta_in_week: Biến động giá trong tuần
•	delta_in_month: Biến động giá trong tháng
•	delta_in_year: Biến động giá trong năm
•	short_name: Tên viết tắt công ty
•	website: Trang web công ty
•	industry_id: Mã ngành
•	industry_id_v2: Mã ngành (phiên bản 2)
•	Tickers: Mã cổ phiếu
5.	Bảng sự kiện chứng khoán (events): Thông tin về các sự kiện quan trọng liên quan đến cổ phiếu.
•	rsi: Chỉ số sức mạnh tương đối
•	rs: Chỉ số sức mạnh tương đối theo khối lượng
•	id: Mã sự kiện
•	price: Giá cổ phiếu
•	price_change: Thay đổi giá
•	price_change_ratio: Tỷ lệ thay đổi giá
•	price_change_ratio_1m: Tỷ lệ thay đổi giá trong 1 tháng
•	event_name: Tên sự kiện
•	event_code: Mã sự kiện
•	notify_date: Ngày thông báo sự kiện
•	exer_date: Ngày thực hiện sự kiện
•	reg_final_date: Ngày đăng ký cuối cùng
•	exer_right_date: Ngày giao dịch không hưởng quyền
•	event_desc: Mô tả chi tiết sự kiện
•	Symbol: Mã cổ phiếu
6.	Bảng giao dịch cổ đông lớn/nội bộ (Insider_trading): Thông tin giao dịch cổ phiếu của cổ đông lớn hoặc nội bộ công ty.
•	deal_announce_date: Ngày thông báo giao dịch
•	deal_method: Phương thức giao dịch
•	deal_action: Hành động (mua/bán)
•	deal_quantity: Số lượng cổ phiếu giao dịch
•	deal_price: Giá giao dịch
•	deal_ratio: Tỷ lệ giao dịch (nếu có)
•	Ticker: Mã cổ phiếu
7.	Bảng chỉ số thị trường VNINDEX (market_index): Lịch sử chỉ số thị trường VNINDEX theo từng ngày giao dịch.
•	time: Ngày giao dịch
•	open: Giá mở cửa
•	high: Giá cao nhất trong ngày
•	low: Giá thấp nhất trong ngày
•	close: Giá đóng cửa
•	volume: Khối lượng giao dịch
•	Index: Tên chỉ số thị trường
8.	Bảng báo cáo tài chính tóm tắt (profitloss): Tóm tắt dữ liệu tài chính quan trọng của doanh nghiệp.
•	CP: Mã cổ phiếu
•	Năm: Năm tài chính
•	Tăng trưởng doanh thu (%): Tỷ lệ tăng trưởng doanh thu so với năm trước
•	Doanh thu (Tỷ đồng): Tổng doanh thu
•	Lợi nhuận sau thuế của Cổ đông công ty mẹ (Tỷ đồng): Lợi nhuận thuần thuộc về cổ đông công ty mẹ
•	Tăng trưởng lợi nhuận (%): Tỷ lệ tăng trưởng lợi nhuận so với năm trước
•	Thu nhập tài chính: Thu nhập từ hoạt động tài chính
•	Doanh thu bán hàng và cung cấp dịch vụ: Tổng doanh thu bán hàng và cung cấp dịch vụ
•	Các khoản giảm trừ doanh thu: Các khoản trừ khỏi doanh thu
•	Doanh thu thuần: Doanh thu sau khi trừ các khoản giảm trừ
•	Giá vốn hàng bán: Chi phí trực tiếp của sản phẩm, dịch vụ
•	Lãi gộp: Lợi nhuận gộp
•	Chi phí tài chính: Chi phí liên quan hoạt động tài chính
•	Chi phí bán hàng: Chi phí bán hàng và marketing
•	Chi phí quản lý DN: Chi phí quản lý doanh nghiệp
•	Lãi/Lỗ từ hoạt động kinh doanh: Lợi nhuận hoặc lỗ từ hoạt động kinh doanh
•	Thu nhập khác: Các khoản thu nhập không thường xuyên
•	Thu nhập/Chi phí khác: Chênh lệch thu nhập và chi phí khác
•	Lợi nhuận khác: Lợi nhuận từ hoạt động khác
•	LN trước thuế: Lợi nhuận trước thuế
•	Chi phí thuế TNDN hiện hành: Thuế thu nhập doanh nghiệp hiện hành
•	Chi phí thuế TNDN hoãn lại: Thuế thu nhập doanh nghiệp hoãn lại
•	Lợi nhuận thuần: Lợi nhuận sau thuế
•	Cổ đông của Công ty mẹ: Lợi nhuận thuộc về cổ đông công ty mẹ
•	Ticker: Mã cổ phiếu
9.	Bảng thông tin tổ chức niêm yết (stock_codes_by_exchange): Thông tin cơ bản của các tổ chức niêm yết trên các sàn giao dịch.
•	symbol: Mã cổ phiếu
•	id: ID tổ chức niêm yết
•	type: Loại chứng khoán
•	exchange: Sàn giao dịch
•	en_organ_name: Tên tổ chức bằng tiếng Anh
•	en_organ_short_name: Tên viết tắt bằng tiếng Anh
•	organ_short_name: Tên viết tắt bằng tiếng Việt
•	organ_name: Tên đầy đủ của tổ chức bằng tiếng Việt

Quan hệ các bảng:
•	One-to-Many:
•	company_overview → all_tickers_history, company_dividen, company_news, events, Insider_trading, profitloss, stock_codes_by_exchange
•	One-to-One:
•	stock_codes_by_exchange → company_overview



Công thức 
1. Công thức về Doanh nghiệp (Tài chính, Hiệu suất, Tình trạng Cổ phiếu)
1.1. Chỉ số tài chính từ bảng profitloss
•	Biên lợi nhuận gộp = (Lãi gộp / Doanh thu thuần) * 100%
•	Biên lợi nhuận ròng = (Lợi nhuận thuần / Doanh thu thuần) * 100%
•	ROE (Tỷ suất lợi nhuận trên vốn chủ sở hữu) = (Lợi nhuận thuần / Vốn chủ sở hữu) * 100%
•	ROA (Tỷ suất lợi nhuận trên tổng tài sản) = (Lợi nhuận thuần / Tổng tài sản) * 100%
•	EPS (Lợi nhuận trên mỗi cổ phiếu) = Lợi nhuận thuần / Số cổ phiếu đang lưu hành
•	P/E Ratio (Hệ số giá trên thu nhập) = Giá cổ phiếu / EPS
•	P/B Ratio (Hệ số giá trên giá trị sổ sách) = Giá cổ phiếu / (Giá trị sổ sách trên mỗi cổ phiếu)
•	Tỷ lệ nợ/Tổng tài sản = (Tổng nợ / Tổng tài sản) * 100%
•	Tỷ lệ nợ/Vốn chủ sở hữu = (Tổng nợ / Vốn chủ sở hữu) * 100%
•	Tỷ lệ cổ tức/ EPS = (Cổ tức trên mỗi cổ phiếu / EPS) * 100%
•	Tỷ lệ chi trả cổ tức = (Tổng cổ tức / Lợi nhuận thuần) * 100%
•	Tốc độ tăng trưởng lợi nhuận = (Lợi nhuận năm nay - Lợi nhuận năm trước) / Lợi nhuận năm trước * 100%
•	Tốc độ tăng trưởng doanh thu = (Doanh thu năm nay - Doanh thu năm trước) / Doanh thu năm trước * 100%
1.2. Hiệu suất giao dịch cổ phiếu từ bảng all_tickers_history
•	Biến động giá trong ngày = ((high - low) / open) * 100%
•	Tỷ lệ thay đổi giá đóng cửa so với ngày trước = ((close_hôm_nay - close_hôm_trước) / close_hôm_trước) * 100%
•	Khối lượng giao dịch trung bình 10 ngày = SUM(volume 10 ngày gần nhất) / 10
•	Khối lượng giao dịch trung bình 30 ngày = SUM(volume 30 ngày gần nhất) / 30
•	Mức độ thanh khoản = (Tổng khối lượng giao dịch / Tổng số cổ phiếu đang lưu hành) * 100%

2. Công thức về Thị trường (VNINDEX, Xu hướng, Chỉ số sức mạnh)
2.1. Chỉ số VNINDEX từ bảng market_index
•	Tỷ lệ thay đổi chỉ số VNINDEX theo ngày = ((close_hôm_nay - close_hôm_trước) / close_hôm_trước) * 100%
•	Tỷ lệ thay đổi chỉ số VNINDEX theo tuần/tháng = ((close_hôm_nay - close_tuần_trước) / close_tuần_trước) * 100%
•	Chỉ số biến động trung bình (ATR - Average True Range) = SUM(High - Low 14 ngày gần nhất) / 14
2.2. Chỉ báo kỹ thuật (Technical Indicators)
•	RSI (Chỉ số sức mạnh tương đối, từ company_news và events)
•	RS = (Trung bình tăng giá / Trung bình giảm giá)
•	RSI = 100 - (100 / (1 + RS))
•	MACD (Moving Average Convergence Divergence)
•	MACD Line = EMA(12) - EMA(26)
•	Signal Line = EMA(9) của MACD Line
•	Histogram = MACD Line - Signal Line
•	SMA (Simple Moving Average) của giá cổ phiếu
•	SMA(n) = SUM(close giá trong n ngày) / n
•	EMA (Exponential Moving Average) của giá cổ phiếu
•	EMA(n) = (Giá đóng cửa hôm nay * α) + (EMA hôm qua * (1 - α)) với α = 2/(n+1)

3. Công thức về Sự kiện và Giao dịch Nội bộ
3.1. Phân tích tác động của sự kiện từ bảng events
•	Tỷ lệ thay đổi giá trước và sau sự kiện = ((close_sau_sự_kiện - close_trước_sự_kiện) / close_trước_sự_kiện) * 100%
•	Mức độ ảnh hưởng của sự kiện = Tỷ lệ thay đổi giá / Biến động giá trung bình trong 1 tháng trước sự kiện
3.2. Ảnh hưởng của giao dịch nội bộ từ Insider_trading
•	Khối lượng giao dịch nội bộ so với tổng khối lượng = (Tổng số lượng giao dịch nội bộ / Tổng khối lượng giao dịch thị trường) * 100%
•	Tỷ lệ thay đổi giá sau giao dịch nội bộ = ((close_sau_giao_dịch - close_trước_giao_dịch) / close_trước_giao_dịch) * 100%

4. Công thức Đánh giá Đầu tư & Định giá Doanh nghiệp
•	Intrinsic Value (Giá trị nội tại theo phương pháp chiết khấu dòng tiền - DCF)
•	Giá trị nội tại = (Dòng tiền tự do 1 năm sau) / (1 + WACC)^1 + ... + (Dòng tiền tự do n năm sau) / (1 + WACC)^n
•	Trong đó, WACC là Chi phí vốn bình quân gia quyền
•	PEG Ratio (Chỉ số định giá theo tăng trưởng)
•	PEG = P/E Ratio / Tốc độ tăng trưởng EPS
•	EV/EBITDA Ratio
•	EV/EBITDA = (Vốn hóa thị trường + Tổng nợ - Tiền mặt) / EBITDA
•	Tỷ lệ giá trị doanh nghiệp trên doanh thu (EV/Sales)
•	EV/Sales = (Vốn hóa thị trường + Tổng nợ - Tiền mặt) / Doanh thu
•	Tỷ lệ tăng trưởng EPS trung bình hàng năm
•	CAGR(EPS) = ((EPS_năm_n / EPS_năm_0)^(1/n) - 1) * 100%
"""

metadata_bctc = """
a
"""


logging.basicConfig(level=logging.INFO)
logging.getLogger("langchain").setLevel(logging.DEBUG)

class ManagerAgent(AssistantAgent):
    """
    ManagerAgent nhận câu hỏi của người dùng, sử dụng ChatOpenAI để phân loại câu hỏi thành 7 loại:
      - flow_1: Câu hỏi có thể được trả lời trực tiếp bởi LLM (LLMs Agent).
      - flow_2: Câu hỏi yêu cầu trả lời dựa trên metadata (Metadata Agent).
      - flow_3: Câu hỏi yêu cầu truy vấn cơ sở dữ liệu (Text2SQL Agent).
      - flow_4: Câu hỏi yêu cầu truy vấn kết hợp vẽ biểu đồ (Drawing Agent).
      - flow_5: Câu hỏi yêu cầu sử dụng RAG với báo cáo tài chính (RAG Agent).
      - flow_6: Câu hỏi yêu cầu việc giả sử 1 giá trị một khoảng và muốn dự đoán đầu ra của mô hình (Scenario Analysis Agent).
    Các agent con được truyền vào qua config_agent.
    """

    def __init__(self, name, code_execution_config, human_input_mode, config_item, termination_msg, config_agent: Optional[Dict] = None):
        super().__init__(
            name=name,
            code_execution_config=code_execution_config,
            human_input_mode=human_input_mode,
            
        )
        self.config_item = config_item
        self._termination_msg = termination_msg
        self._config_agent = {} if config_agent is None else config_agent
        self._openai_api_key = self._config_agent.get("openai_api_key", None)
        self._llms_agent = self._config_agent.get("llms_agent", None)
        self._text2sql_agent = self._config_agent.get("text2sql_agent", None)
        self._visualize_data = self._config_agent.get("visualize_data", None)
        self._rag_agent = self._config_agent.get("rag_agent", None)
        self._ml_agent = self._config_agent.get("ml_agent", None)
        self._schema = self._config_agent.get("schema", None)
        self._scenario_analysis_agent = self._config_agent.get("scenario_analysis_agent", None)

        self._assistant_manager= AssistantAgent(
            name="assistant_rag",
            system_message=prompt_template + self._termination_msg,
            llm_config={"config_list": config_item, "timeout": 60, "temperature": 0},
        )

        self._manager_agent = UserProxyAgent(
            name="ManagerAgent",
            # termination_msg=termination_msg,
            human_input_mode="NEVER",
            is_termination_msg=self.is_termination_msg,
            code_execution_config={
                "work_dir": "coding", 
                "use_docker": False
            }
        )

    def determine_domain(self, question: str, history: str) -> str:
        response = self._manager_agent.initiate_chat(self._assistant_manager, message=prompt_template.format(question=question, history=history, metadata=metadata, metadata_bctc=metadata_bctc,  termination_msg = self._termination_msg), clear_history=True)
        domain = response.chat_history[-1]['content'].replace("Đã hoàn tất.", "").strip()
        print(domain)
        if domain not in {"flow_1", "flow_2", "flow_3", "flow_4", "flow_5", "flow_6"}:
            domain = "flow_1"
        return domain

    def process_question(self, question: str, history: str) -> str:
        domain = self.determine_domain(question, history)
        print(f"Phân loại câu hỏi sang: {domain}")

        if domain == "flow_1":
            response = self._llms_agent.initiate_conversation(question, history)
            return response
        elif domain == "flow_2":
            response = self._text2sql_agent.initiate_conversation(question, history)
            return response
        elif domain == "flow_3":
            return self._visualize_data.initiate_conversation(question, history)
        elif domain == "flow_4":
            response = self._rag_agent.initiate_conversation(question, history)
            return response
        elif domain == "flow_5":
            response = self._ml_agent.initiate_conversation(question, history)
            return response
        elif domain == "flow_6":
            response = self._scenario_analysis_agent.initiate_conversation(question, history)
            return response
        else:
            return "Không nhận diện được loại câu hỏi. Vui lòng thử lại."
        
    @staticmethod
    def is_termination_msg(content):
        have_content = content.get("content", None) is not None
        if have_content and "Đã hoàn tất" in content["content"]:
            return True
        return False