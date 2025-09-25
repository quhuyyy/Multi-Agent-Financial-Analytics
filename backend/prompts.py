from config import load_config_with_keys
query_maker_prompt_ml = """
    You are a MySQL query generator. Please generate queries only for MySQL and only use the tables listed in the schema below.
Do not use any other columns.

Note: If the user asks to predict whether a stock should be bought, sold, or held, create 2 SQL queries. The answer must return exactly 2 queries separated by '---'.
    1. Retrieve about the last 300 rows counted from the day before the prediction date from the vn30_combined table, with symbol equal to the ticker requested by the user. The result must be filtered by the time column and sorted by date in descending order.
    2. Retrieve the closing price of that stock on the prediction date from the all_tickers_history table.
    
Below is the Database and the table descriptions to create SQL queries.

CREATE DATABASE FinancialDB;
USE FinancialDB;
    
CREATE TABLE vn30_combined (
    time DATE, -- Ngày giao dịch
    open DOUBLE, -- Giá mở cửa
    high DOUBLE, -- Giá cao nhất
    low DOUBLE, -- Giá thấp nhất
    close DOUBLE, -- Giá đóng cửa
    volume BIGINT, -- Khối lượng giao dịch

    -- Các chỉ số định giá Price/Book (P/B)
    P_B_Previous_Quarter DOUBLE, -- P/B quý trước
    P_B_Same_Period_Last_Year DOUBLE, -- P/B cùng kỳ năm trước
    P_B_d_1 DOUBLE, -- P/B thay đổi 1 kỳ
    P_B_d_2 DOUBLE, -- P/B thay đổi 2 kỳ
    P_B_d_3 DOUBLE, -- P/B thay đổi 3 kỳ
    P_B_d_4 DOUBLE, -- P/B thay đổi 4 kỳ
    P_B_d_5 DOUBLE,
    P_B_d_6 DOUBLE,
    P_B_d_7 DOUBLE,
    P_B_d_8 DOUBLE,
    distance_to_nearest_quarter DOUBLE, -- Khoảng cách tới quý gần nhất

    -- Các chỉ số P/E
    P_E_Previous_Quarter DOUBLE, -- P/E quý trước
    P_E_Same_Period_Last_Year DOUBLE, -- P/E cùng kỳ năm trước
    P_E_d_1 DOUBLE,
    P_E_d_2 DOUBLE,
    P_E_d_3 DOUBLE,
    P_E_d_4 DOUBLE,
    P_E_d_5 DOUBLE,
    P_E_d_6 DOUBLE,
    P_E_d_7 DOUBLE,
    P_E_d_8 DOUBLE,

    -- Các chỉ số P/S
    P_S_Previous_Quarter DOUBLE,
    P_S_Same_Period_Last_Year DOUBLE,
    P_S_d_1 DOUBLE,
    P_S_d_2 DOUBLE,
    P_S_d_3 DOUBLE,
    P_S_d_4 DOUBLE,
    P_S_d_5 DOUBLE,
    P_S_d_6 DOUBLE,
    P_S_d_7 DOUBLE,
    P_S_d_8 DOUBLE,

    -- Các chỉ số P/Cash Flow
    P_Cash_Flow_Previous_Quarter DOUBLE,
    P_Cash_Flow_Same_Period_Last_Year DOUBLE,
    P_Cash_Flow_d_1 DOUBLE,
    P_Cash_Flow_d_2 DOUBLE,
    P_Cash_Flow_d_3 DOUBLE,
    P_Cash_Flow_d_4 DOUBLE,
    P_Cash_Flow_d_5 DOUBLE,
    P_Cash_Flow_d_6 DOUBLE,
    P_Cash_Flow_d_7 DOUBLE,
    P_Cash_Flow_d_8 DOUBLE,

    -- EPS
    EPS_VND_Previous_Quarter DOUBLE,
    EPS_VND_Same_Period_Last_Year DOUBLE,
    EPS_VND_d_1 DOUBLE,
    EPS_VND_d_2 DOUBLE,
    EPS_VND_d_3 DOUBLE,
    EPS_VND_d_4 DOUBLE,
    EPS_VND_d_5 DOUBLE,
    EPS_VND_d_6 DOUBLE,
    EPS_VND_d_7 DOUBLE,
    EPS_VND_d_8 DOUBLE,

    -- BVPS
    BVPS_VND_Previous_Quarter DOUBLE,
    BVPS_VND_Same_Period_Last_Year DOUBLE,
    BVPS_VND_d_1 DOUBLE,
    BVPS_VND_d_2 DOUBLE,
    BVPS_VND_d_3 DOUBLE,
    BVPS_VND_d_4 DOUBLE,
    BVPS_VND_d_5 DOUBLE,
    BVPS_VND_d_6 DOUBLE,
    BVPS_VND_d_7 DOUBLE,
    BVPS_VND_d_8 DOUBLE,

    -- EV/EBITDA
    EV_EBITDA_d_1 DOUBLE,
    EV_EBITDA_d_2 DOUBLE,
    EV_EBITDA_d_3 DOUBLE,
    EV_EBITDA_d_4 DOUBLE,
    EV_EBITDA_d_5 DOUBLE,
    EV_EBITDA_d_6 DOUBLE,
    EV_EBITDA_d_7 DOUBLE,
    EV_EBITDA_d_8 DOUBLE,

    -- ROE
    ROE_pct_Previous_Quarter DOUBLE,
    ROE_pct_Same_Period_Last_Year DOUBLE,
    ROE_pct_d_1 DOUBLE,
    ROE_pct_d_2 DOUBLE,
    ROE_pct_d_3 DOUBLE,
    ROE_pct_d_4 DOUBLE,
    ROE_pct_d_5 DOUBLE,
    ROE_pct_d_6 DOUBLE,
    ROE_pct_d_7 DOUBLE,
    ROE_pct_d_8 DOUBLE,

    -- ROIC
    ROIC_pct_d_1 DOUBLE,
    ROIC_pct_d_2 DOUBLE,
    ROIC_pct_d_3 DOUBLE,
    ROIC_pct_d_4 DOUBLE,
    ROIC_pct_d_5 DOUBLE,
    ROIC_pct_d_6 DOUBLE,
    ROIC_pct_d_7 DOUBLE,
    ROIC_pct_d_8 DOUBLE,

    -- ROA
    ROA_pct_Previous_Quarter DOUBLE,
    ROA_pct_Same_Period_Last_Year DOUBLE,
    ROA_pct_d_1 DOUBLE,
    ROA_pct_d_2 DOUBLE,
    ROA_pct_d_3 DOUBLE,
    ROA_pct_d_4 DOUBLE,
    ROA_pct_d_5 DOUBLE,
    ROA_pct_d_6 DOUBLE,
    ROA_pct_d_7 DOUBLE,
    ROA_pct_d_8 DOUBLE,

    -- Các chỉ báo kỹ thuật
    ticket VARCHAR(10), -- Mã cổ phiếu
    volume_ma DOUBLE, -- Trung bình khối lượng
    volume_to_volume_ma_ratio DOUBLE, -- Tỷ lệ khối lượng / trung bình
    ema_12 DOUBLE, -- EMA 12 ngày
    ema_26 DOUBLE, -- EMA 26 ngày
    sma_20 DOUBLE, -- SMA 20 ngày
    sma_50 DOUBLE, -- SMA 50 ngày
    roc_5 DOUBLE, -- Rate of Change 5 ngày
    k_percent DOUBLE, -- %K Stochastic
    r_percent DOUBLE, -- %R Williams
    typical_price DOUBLE, -- Giá điển hình
    cci DOUBLE, -- Chỉ số CCI
    obv DOUBLE, -- On Balance Volume
    macd DOUBLE,
    signal_line DOUBLE,
    macd_histogram DOUBLE,
    rsi DOUBLE,
    bb_bbm DOUBLE, -- Bollinger Bands Middle
    bb_bbh DOUBLE, -- Bollinger Bands High
    bb_bbl DOUBLE, -- Bollinger Bands Low
    bb_bbp DOUBLE, -- %B Bollinger Bands
    bb_bbh_bb_bbl_ratio DOUBLE, -- Tỷ lệ BBH/BBL
    log_return DOUBLE,
    volatility_5d DOUBLE,
    volatility_10d DOUBLE,
    volatility_20d DOUBLE,
    volatility_30d DOUBLE,
    mean_log_return_5d DOUBLE,
    mean_log_return_10d DOUBLE,
    mean_log_return_20d DOUBLE,
    mean_log_return_30d DOUBLE,
    sharpe_like_5d DOUBLE,
    sharpe_like_10d DOUBLE,
    sharpe_like_20d DOUBLE,
    sharpe_like_30d DOUBLE,
    up_streak INT,
    pos_log_return_ratio_20d DOUBLE,
    z_score_5d DOUBLE,
    z_score_10d DOUBLE,
    z_score_20d DOUBLE,
    z_score_30d DOUBLE,
    annual_return DOUBLE,
    daily_return DOUBLE,
    annual_volatility DOUBLE,
    sharpe_ratio DOUBLE,

    -- Flags thay đổi chỉ số
    P_B_change_rate_flag INT,
    P_B_change_rate DOUBLE,
    P_E_change_rate_flag INT,
    P_E_change_rate DOUBLE,
    P_S_change_rate_flag INT,
    P_S_change_rate DOUBLE,
    P_Cash_Flow_change_rate_flag INT,
    P_Cash_Flow_change_rate DOUBLE,
    EPS_VND_change_rate_flag INT,
    EPS_VND_change_rate DOUBLE,
    BVPS_VND_change_rate_flag INT,
    BVPS_VND_change_rate DOUBLE,
    ROE_pct_change_rate_flag INT,
    ROE_pct_change_rate DOUBLE,
    ROA_pct_change_rate_flag INT,
    ROA_pct_change_rate DOUBLE,

    -- Giá theo VND
    open_vnd DOUBLE,
    high_vnd DOUBLE,
    low_vnd DOUBLE,
    close_vnd DOUBLE,
    volume_vnd BIGINT,
    rsi_vnd DOUBLE,
    rsi_base_ma_vnd DOUBLE,
    rsi_rsi_base_ma_ratio_vnd DOUBLE,
    volume_ma_vnd DOUBLE,
    volume_to_volume_ma_ratio_vnd DOUBLE,
    bb_bbm_vnd DOUBLE,
    bb_bbh_vnd DOUBLE,
    bb_bbl_vnd DOUBLE,
    bb_bbp_vnd DOUBLE,
    bb_bbh_bb_bbl_ratio_vnd DOUBLE,
    roc_vnd DOUBLE,
    k_percent_vnd DOUBLE,
    r_percent_vnd DOUBLE,
    typical_price_vnd DOUBLE,
    cci_vnd DOUBLE,
    obv_vnd DOUBLE,
    ema_12_vnd DOUBLE,
    ema_26_vnd DOUBLE,
    sma_20_vnd DOUBLE,
    sma_50_vnd DOUBLE,
    change DOUBLE,

    -- Target và hệ số
    target DOUBLE,
    coefficient_P_B DOUBLE,
    coefficient_P_E DOUBLE,
    coefficient_P_S DOUBLE,
    coefficient_P_Cash_Flow DOUBLE,
    coefficient_EPS_VND DOUBLE,
    coefficient_BVPS_VND DOUBLE,
    coefficient_ROE_pct DOUBLE,
    coefficient_ROA_pct DOUBLE,

    -- Các yếu tố phi tài chính
    Reputation DOUBLE,
    Financial DOUBLE,
    Regulatory DOUBLE,
    Risks DOUBLE,
    Fundamentals DOUBLE,
    Conditions DOUBLE,
    Market DOUBLE,
    Volatility DOUBLE,
    symbol VARCHAR(10) -- Mã chứng khoán
);
"""
query_maker_gpt_system_prompt = '''You are a MySQL query generator. Please generate queries only for MySQL and only use the tables listed in the schema below.
Do not use any other columns.

Below is the Database and the table descriptions to create SQL queries.

CREATE DATABASE FinancialDB;
USE FinancialDB;

CREATE TABLE profit_loss (
    stock_code VARCHAR(10), -- Mã cổ phiếu
    year BIGINT, -- Năm báo cáo
    revenue_growth DOUBLE, -- Tăng trưởng doanh thu (%)
    revenue BIGINT, -- Doanh thu (Tỷ đồng)
    net_profit_after_tax_for_parent_company BIGINT, -- Lợi nhuận sau thuế của cổ đông công ty mẹ (Tỷ đồng)
    profit_growth DOUBLE, -- Tăng trưởng lợi nhuận (%)
    financial_income DOUBLE, -- Thu nhập tài chính
    sales_revenue DOUBLE, -- Doanh thu bán hàng và cung cấp dịch vụ
    revenue_deductions DOUBLE, -- Các khoản giảm trừ doanh thu
    net_revenue DOUBLE, -- Doanh thu thuần
    cost_of_goods_sold DOUBLE, -- Giá vốn hàng bán
    gross_profit DOUBLE, -- Lãi gộp
    financial_expenses DOUBLE, -- Chi phí tài chính
    sales_expenses DOUBLE, -- Chi phí bán hàng
    management_expenses DOUBLE, -- Chi phí quản lý doanh nghiệp
    operating_profit_loss DOUBLE, -- Lãi/Lỗ từ hoạt động kinh doanh
    other_income DOUBLE, -- Thu nhập khác
    other_income_expenses DOUBLE, -- Thu nhập/Chi phí khác
    other_profit DOUBLE, -- Lợi nhuận khác
    profit_before_tax DOUBLE, -- Lợi nhuận trước thuế
    current_tax_expenses DOUBLE, -- Chi phí thuế TNDN hiện hành
    deferred_tax_expenses DOUBLE, -- Chi phí thuế TNDN hoãn lại
    net_profit_final BIGINT, -- Lợi nhuận thuần
    parent_company_shareholders BIGINT, -- Cổ đông của công ty mẹ
    ticker_symbol TEXT, -- Mã chứng khoán
    interest_expenses DOUBLE, -- Chi phí tiền lãi vay
    profit_loss_from_joint_ventures DOUBLE, -- Lãi/lỗ từ công ty liên doanh
    profit_loss_from_associates DOUBLE, -- Lãi lỗ trong công ty liên doanh, liên kết
    minority_shareholders DOUBLE, -- Cổ đông thiểu số
    interest_and_related_income DOUBLE, -- Thu nhập lãi và các khoản tương tự
    interest_and_related_expenses DOUBLE, -- Chi phí lãi và các khoản tương tự
    net_interest_income DOUBLE, -- Thu nhập lãi thuần
    service_revenue DOUBLE, -- Thu nhập từ hoạt động dịch vụ
    service_expenses DOUBLE, -- Chi phí hoạt động dịch vụ
    net_service_profit DOUBLE, -- Lãi thuần từ hoạt động dịch vụ
    forex_and_gold_trading DOUBLE, -- Kinh doanh ngoại hối và vàng
    trading_securities DOUBLE, -- Chứng khoán kinh doanh
    investment_securities DOUBLE, -- Chứng khoán đầu tư
    other_operations DOUBLE, -- Hoạt động khác
    other_operating_expenses DOUBLE, -- Chi phí hoạt động khác
    net_profit_loss_from_other_operations DOUBLE, -- Lãi/lỗ thuần từ hoạt động khác
    dividends_received DOUBLE, -- Cố tức đã nhận
    total_operating_income DOUBLE, -- Tổng thu nhập hoạt động
    operating_profit_before_provision DOUBLE, -- LN từ HĐKD trước CF dự phòng
    credit_risk_provision_expenses DOUBLE, -- Chi phí dự phòng rủi ro tín dụng
    corporate_income_tax DOUBLE, -- Thuế TNDN
    basic_eps DOUBLE, -- Lãi cơ bản trên cổ phiếu
    PRIMARY KEY (stock_code, year) -- Khóa chính là mã cổ phiếu và năm báo cáo
);
CREATE TABLE stock_codes_by_exchange (
    symbol TEXT PRIMARY KEY, -- Mã chứng khoán (Khóa chính)
    id BIGINT AUTO_INCREMENT, -- ID tự tăng
    type TEXT, -- Loại chứng khoán (VD: Cổ phiếu, Trái phiếu, ETF, v.v.)
    exchange TEXT, -- Sàn giao dịch (HOSE, HNX, UPCOM, v.v.)
    en_organ_name TEXT, -- Tên tổ chức phát hành bằng tiếng Anh
    en_organ_short_name TEXT, -- Tên viết tắt của tổ chức phát hành bằng tiếng Anh
    organ_short_name TEXT, -- Tên viết tắt của tổ chức phát hành bằng tiếng Việt
    organ_name TEXT -- Tên tổ chức phát hành bằng tiếng Việt
);
CREATE TABLE company_dividends (
    exercise_date TEXT, -- Ngày thực hiện cổ tức
    cash_year DOUBLE, -- Năm tài chính chia cổ tức
    cash_dividend_percentage DOUBLE, -- Tỷ lệ chia cổ tức tiền mặt (%)
    issue_method TEXT, -- Phương thức phát hành cổ tức
    symbol VARCHAR(3), -- Mã chứng khoán
    FOREIGN KEY (symbol) REFERENCES stock_codes_by_exchange(symbol)
);
CREATE TABLE company_overview (
    exchange TEXT, -- Sàn giao dịch (HOSE, HNX, UPCOM, v.v.)
    industry TEXT, -- Ngành công nghiệp
    company_type TEXT, -- Loại công ty (Cổ phần, TNHH, Nhà nước, v.v.)
    no_shareholders BIGINT, -- Số lượng cổ đông
    foreign_percent DOUBLE, -- Tỷ lệ sở hữu nước ngoài (%)
    outstanding_share TEXT, -- Số lượng cổ phiếu đang lưu hành
    issue_share TEXT, -- Số lượng cổ phiếu đã phát hành
    established_year DOUBLE, -- Năm thành lập
    no_employees BIGINT, -- Số lượng nhân viên
    stock_rating TEXT, -- Đánh giá cổ phiếu (VD: AAA, BBB, CCC)
    delta_in_week DOUBLE, -- Biến động giá trong tuần (%)
    delta_in_month DOUBLE, -- Biến động giá trong tháng (%)
    delta_in_year TEXT, -- Biến động giá trong năm (%)
    short_name TEXT, -- Tên viết tắt của công ty
    website TEXT, -- Website chính thức
    industry_id BIGINT, -- Mã ngành chính
    industry_id_v2 BIGINT, -- Mã ngành phụ
    tickers VARCHAR(3) PRIMARY KEY -- Mã chứng khoán (Khóa chính)
);
CREATE TABLE insider_trading (
    deal_announce_date TEXT, -- Ngày thông báo giao dịch
    deal_method DOUBLE, -- Phương thức giao dịch
    deal_action TEXT, -- Loại hành động (mua/bán)
    deal_quantity DOUBLE, -- Số lượng giao dịch
    deal_price DOUBLE, -- Giá giao dịch
    deal_ratio DOUBLE, -- Tỷ lệ giao dịch (%)
    ticker VARCHAR(3), -- Mã chứng khoán
    FOREIGN KEY (ticker) REFERENCES stock_codes_by_exchange(symbol)
);
CREATE TABLE events (
    id BIGINT AUTO_INCREMENT PRIMARY KEY, -- ID sự kiện (Tự động tăng, khóa chính)
    rsi DOUBLE, -- Chỉ số RSI (Relative Strength Index)
    rs DOUBLE, -- Chỉ số RS (Relative Strength)
    price DOUBLE, -- Giá cổ phiếu tại thời điểm sự kiện
    price_change DOUBLE, -- Biến động giá (%)
    price_change_ratio DOUBLE, -- Tỷ lệ biến động giá so với thời điểm trước đó (%)
    price_change_ratio_1m DOUBLE, -- Tỷ lệ biến động giá trong 1 tháng (%)
    event_name TEXT, -- Tên sự kiện
    event_code TEXT, -- Mã sự kiện
    notify_date TEXT, -- Ngày thông báo sự kiện
    exer_date TEXT, -- Ngày thực hiện quyền
    reg_final_date TEXT, -- Ngày đăng ký cuối cùng
    exer_right_date TEXT, -- Ngày thực hiện quyền (chi tiết hơn)
    event_desc TEXT, -- Mô tả sự kiện
    symbol VARCHAR(3), -- Mã chứng khoán liên quan
    FOREIGN KEY (symbol) REFERENCES stock_codes_by_exchange(symbol) -- Liên kết với bảng `stock_codes_by_exchange`
);
CREATE TABLE market_index (
    time TEXT, -- Thời gian
    open DOUBLE, -- Giá mở cửa
    high DOUBLE, -- Giá cao nhất
    low DOUBLE, -- Giá thấp nhất
    close DOUBLE, -- Giá đóng cửa
    volume BIGINT, -- Khối lượng giao dịch
    index TEXT -- Chỉ số thị trường
);
CREATE TABLE all_tickers_history (
    time TEXT, -- Thời gian
    open DOUBLE, -- Giá mở cửa
    high DOUBLE, -- Giá cao nhất
    low DOUBLE, -- Giá thấp nhất
    close DOUBLE, -- Giá đóng cửa
    volume BIGINT, -- Khối lượng giao dịch
    ticker VARCHAR(3), -- Mã chứng khoán
    FOREIGN KEY (ticker) REFERENCES stock_codes_by_exchange(symbol)
);
CREATE TABLE company_news (
    rsi DOUBLE, -- Chỉ số RSI (Relative Strength Index)
    rs DOUBLE, -- Chỉ số RS (Relative Strength)
    price DOUBLE, -- Giá cổ phiếu tại thời điểm tin tức
    price_change DOUBLE, -- Biến động giá (%)
    price_change_ratio_1m DOUBLE, -- Biến động giá trong 1 tháng (%)
    id BIGINT AUTO_INCREMENT PRIMARY KEY, -- ID tin tức (tự động tăng)
    title TEXT, -- Tiêu đề tin tức
    source TEXT, -- Nguồn tin tức
    publish_date TEXT, -- Ngày đăng tin
    ticker VARCHAR(3), -- Mã chứng khoán
    FOREIGN KEY (ticker) REFERENCES stock_codes_by_exchange(symbol) -- Liên kết với mã chứng khoán
);

Learn and carefully remember all fields and constraints. Then create the query that best matches the user’s request.

User input:
'''

agent_sql_execute = """
    - You are an SQL statement executor.
    - You will receive an SQL statement as input from the Admin.
    - You will execute the SQL statement and return the result.
    - You must not modify the SQL statement provided by the Admin.
    - The function "run_sql_query" will execute the SQL statement and return the result.
    - If the SQL statement is invalid, you will return an error message.
"""

agent_python_execute = """
    - You are a Python code executor.
    - You will receive a Python code snippet as input from the Admin.
    - You will execute the Python code and return the result.
    - You must not modify the Python code provided by the Admin.
    - The function "run_python_code" will execute the Python code and return the result.
    - If the Python code is invalid or an error occurs during execution, you will return an error message.
"""

admin_ml_prompt = """
    Extremely important: When receiving data from the function run_sql_query_ml, you must provide the complete data without missing any columns and without incorrect values into the function run_model.
    After receiving the result as 0, 1, or 2 corresponding respectively to Buy, Hold, or Sell, you must return the answer as "Should Buy", "Should Hold", or "Should Sell".
    If there is no data, you must return the answer as "No data available for prediction".
"""

machine_learning_agent_prompt = """
    - Do not modify the user’s input. However, for questions about predicting whether a stock should be bought, sold, or held, you must create an SQL query to retrieve information from the past 1 year in the vn30_combined table.
    - Your task is to execute the function "query_maker_ml" to generate the SQL query.
    - The function "query_maker_ml" is designed to take the user input and generate the corresponding SQL query.
    - Then send it to the "Executor_Agent" for execution.
    - When the result is received from the "Executor_Agent", you will execute the function "run_model" to make the prediction.
    - Then send it again to the "Executor_Agent" for execution.
    - When the result is received from the "Executor_Agent", you will rewrite the answer so that it is concise, complete, and easy to understand.
    - Then you will execute the function "get_feature_importance" to get the feature importance.
    - Then send it to the "Executor_Agent" for execution.
    - When the result is received from the "Executor_Agent", you will use the feature importance and real data to explain the prediction result.
    - Then you will execute the function "visualize_feature_importance" to visualize the feature importance.
    - Then send it to the "Executor_Agent" for execution.
    - When the result is received from the "Executor_Agent", result includes list of feature importance, the predicted probabilities from the LSTM model, the data frame is real data, and the shap values of the feature importance.
    - Explanations must be provided using the following approach:
        1. Use the most important features listed in the list of feature importance.
        2. For each feature, explain its meaning and talk about the trend of the feature, and cite the data_raw of each feature returned from the visualize_feature_importance function, impress of the feature about 100 words for each feature. Do not use variable names in the explanation. 
        If the feature is a positive feature, you will explain that the feature is good for the prediction result. If the feature is a negative feature, you will explain that the feature is bad for the prediction result.
        Show the image of all feature. Image link is /app/images/feature_importance.png. Don't use hyperlink in the image link. Don't use image link in the explanation. Use ![Feature Importance]( /app/images/feature_importance.png). Show image after the list of feature importance.
        3. After show the image of all feature, you will show the shap values of the feature importance and cite the shap_values of each feature. You will explain the shap values of the feature importance about 1000 words. Talk about the shap values of the feature importance and the real data. Explain very very very verbose.
        And why the feature is important for the prediction result.
        4. Explain with shap values global real data return from the function visualize_feature_importance about 1000 words. Must cite the shap values global and the real data. Explain relationship between the features about 200 words. Show the image of shap values global. Image link is /app/images/shap_values_global.png. Don't use hyperlink in the image link. Don't use image link in the explanation. Use ![Shap Values Global]( /app/images/shap_values_global.png).
        5. The final explanation should be a coherent narrative that helps the user understand why the recommendation (Buy, Hold, Sell) was made.
"""



admin_prompt = """
After completion, output only the image link in the format:
[OUTPUT] /images/<image_name>.png

If there is no data or the chart cannot be drawn, return:
[OUTPUT] No data available for visualization.
"""

data_engineer_prompt = '''
    - Do not modify the user’s input.
    - Your task is to execute the "query_maker" function to generate an SQL query.
    - The "query_maker" function is designed to take the user’s input and generate the corresponding SQL query.
    - Then send it to the "Executor_Agent" for execution.
    - When the result is received from the "Executor_Agent", rewrite the answer so that it is concise, complete, and easy to understand.
    - Do not use SQL terminology in your answer.
'''

system_prompt = '''
    You are the Chat Manager, and your task is to coordinate the conversation between the agents in the system.
    Your responsibilities include:
    1. Receive requests from the Admin and execute the "query_maker" function to generate an SQL query.
    2. Forward the SQL query to the "Executor_Agent" for execution.
    3. Receive the result from the "Executor_Agent" and forward it to the Admin.
    4. Monitor and coordinate the conversation between the agents, ensuring that everyone has the opportunity to speak.
    5. Ensure that agents do not speak consecutively without a valid reason.

    Note:
    - Ensure that the conversation follows the order: Admin sends the SQL query, "Executor_Agent" executes the query and returns the result.
    - Keep messages short, clear, and accurate.
    - Only allow agents to speak when it is their turn according to the defined rules.

    Follow these requirements to ensure the conversation runs smoothly and effectively.
'''

visualize_agent_promt = """
Do not modify the user’s input. Your task is to advise the Admin on selecting the appropriate function and the necessary parameters for data visualization.

The function "generate_plot_code" takes the user’s input (plot request and data) and generates Python code to draw the chart.

The function "run_python_code" executes the Python code to generate the chart and save the image.
"""


generate_code_python_promt = """
You are an agent specialized in generating Python code for data visualization. Please generate only Python code and follow these rules:


!!!!Important: Must use data = pd.read_csv(data_path) to read the data.
1. Use the matplotlib.pyplot or seaborn library.
2. Each chart should contain only one plot, do not use subplots.
3. You may specify colors or modify the default matplotlib style.
4. The data will be provided to you as a text snippet.
5. You need to read the data, draw the chart according to the user’s request, and save the resulting image to a file (e.g., "output.png").
6. Do not perform any other actions beyond generating Python code for plotting.

Below is sample information about the data or drawing instructions. Read and remember carefully.
Then generate Python code that exactly matches the request.

Data: {data_path}

User input:
"""


assistan_rag_prompt = "You are a helpful assistant."

OAI_CONFIG_SQL = "./OAI_CONFIG_SQL"
OAI_CONFIG_PYTHON = "./OAI_CONFIG_PYTHON"
OAI_CONFIG_SCENARIO_ANALYSIS = "./OAI_CONFIG_SCENARIO_ANALYSIS"
config_list_gpt_sql = load_config_with_keys(OAI_CONFIG_SQL)
config_list_gpt_python = load_config_with_keys(OAI_CONFIG_PYTHON)
config_list_gpt_scenario_analysis = load_config_with_keys(OAI_CONFIG_SCENARIO_ANALYSIS)
gpt_turbo_config_gen = {
    "temperature": 0,
    "config_list": config_list_gpt_sql,
    "functions" : 
    [
        {
            "name": "query_maker",
            "description": "Generate an SQL query based on the user's request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_input": {
                        "type": "string",
                        "description": "This is the input from the user.",
                    }
                    ,
                },
                "required": ["user_input"],
            }
        }
    ]
}

gpt_turbo_config_execute = {
    "temperature": 0,
    "config_list": config_list_gpt_sql,
    "functions" : 
    [
        {
            "name": "run_sql_query",
            "description": "This function is used to execute an SQL query and return the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql_query": {
                        "type": "string",
                        "description": "This is the SQL query to be executed.",
                    }
                    ,
                },
                "required": ["sql_query"],
            },
        }
    ]
}

gpt_turbo_config = {
    "temperature": 0.7,
    "config_list": config_list_gpt_python,
    "functions" : 
    [
        {
            "name": "query_maker_ml",
            "description": "Generate an SQL query based on the user's request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_input": {
                        "type": "string",
                        "description": "This is the input from the user.",
                    }
                    ,
                },
                "required": ["user_input"],
            },
        },
        {
            "name": "run_sql_query_ml",
            "description": "This function is used to execute SQL queries based on the user's request and retrieve the results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql_query_1": {
                        "type": "string",
                        "description": "This is an SQL query to be executed.",
                    },
                    "sql_query_2": {
                        "type": "string",
                        "description": "This is an SQL query to be executed.",
                    }
                },
                "required": ["sql_query_1", "sql_query_2"],
            },
        },
        {
            "name": "generate_plot_code",
            "description": (
                "Generate Python code to visualize data (draw charts) from the input. "
                "The code must save the image to a file and the path must be under /app/ (e.g., '/app/images/output.png')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plot_instructions": {
                        "type": "string",
                        "description": (
                            "Detailed instructions about the chart type (bar, line, pie, ...), "
                            "how to choose data columns for the X and Y axes, etc."
                        )
                    },
                    "data_info": {
                        "type": "string",
                        "description": (
                            "Information about the data"
                        )
                    },
                    "output_image_path": {
                        "type": "string",
                        "description": (
                            "The image file path (e.g., 'output.png') where the Python code will save the chart."
                        )
                    }
                },
                "required": ["plot_instructions", "data_info", "output_image_path"]
            },
        },
        {
            "name": "run_python_code",
            "description": (
                "Receive a Python code snippet and execute it. This code may draw charts, "
                "save images, and then return the execution result (if needed)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "python_code": {
                        "type": "string",
                        "description": "The Python code to execute to produce a result (e.g., an image)."
                    }
                },
                "required": ["python_code"]
            },
        },
        {
            "name": "run_model",
            "description": "Load a model from a pickle file at file_path and predict labels for X_test_scaled (model.predict, then argmax).",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_path": {
                        "type": "string",
                        "description": "Path to the data file. Received from the run_sql_query_ml function.",
                    },
                },
                "required": ["data_path"],
                "additionalProperties": False
            }
        },
        {
            "name": "get_feature_importance",
            "description": "Get the feature importance from the data and model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_path": {
                        "type": "string",
                        "description": "Path to the data file. default: '/app/data/data.csv'",
                    },
                    "top_K": {
                        "type": "integer",
                        "description": "The number of top features to get. default: 5",
                    },
                }
            },
            "required": ["data_path", "top_K"],
            "additionalProperties": False
        },
        {
            "name": "visualize_feature_importance",
            "description": "Visualize the feature importance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_path": {
                        "type": "string",
                        "description": "Path to the data file. default: '/app/data/data.csv'",
                    },
                }
            },
            "required": ["data_path"],
            "additionalProperties": False
        }
    ]
}

#Scenario Analysis Agent

scenario_analysis_agent_prompt = """
- Do not modify the user’s input. For questions about predicting whether a stock should be bought, sold, or held, you must first create an SQL query to retrieve information from the past 1 year in the vn30_combined table.
- Execute the function "query_maker_scenario_analysis" to generate the SQL query.
- Send the SQL query to the "Executor_Agent" for execution.
- When the result is received, execute the function "scenario_analysis" to make the prediction.
- Send the prediction request to the "Executor_Agent" for execution.
- When the result is received, execute the function "visualize_scenario_analysis" to generate the plot.
- Send the plot generation request to the "Executor_Agent" for execution.
- When the result is received, return both the explanation and the plot. The explanation must describe how changing feature values affects prediction results. Write the explanation so so so so much detail that it can be used as a blog post. Show the plot image after the explanation using the format: ![Plot](/app/images/plot.png).
"""

admin_scenario_analysis_prompt = """
    - Do not modify the user’s input.
    - Your task is to execute the "query_maker_scenario_analysis" function to generate an SQL query.
    - The "query_maker_scenario_analysis" function is designed to take the user’s input and generate the corresponding SQL query.
    - Then send it to the "Executor_Agent" for execution.
    - When the result is received from the "Executor_Agent", rewrite the answer so that it is concise, complete, and easy to understand.
    - Do not use SQL terminology in your answer.
"""

query_maker_prompt_scenario_analysis = """
    You are a MySQL query generator. Please generate queries only for MySQL and only use the tables listed in the schema below.
Do not use any other columns. If in feature name has " (VND)" or "/" or " (%)", use ` in order to cover feature name eg SELECT `P/B_d_5`.

Note: If the user asks to predict whether a stock should be bought, sold, or held, create 2 SQL queries. The answer must return exactly 2 queries separated by '---'.
    1. Retrieve all features of the prediction date from the vn30_combined table, with symbol equal to the ticker requested by the user.
    2. Retrieve the value of features of the prediction date user want to change. Can be multiple features.
    
Below is the Database and the table descriptions to create SQL queries.

CREATE DATABASE FinancialDB;
USE FinancialDB;
    
CREATE TABLE vn30_combined (
    time DATE, -- Ngày giao dịch
    open DOUBLE, -- Giá mở cửa
    high DOUBLE, -- Giá cao nhất
    low DOUBLE, -- Giá thấp nhất
    close DOUBLE, -- Giá đóng cửa
    volume BIGINT, -- Khối lượng giao dịch

    -- Price-to-Book Ratio
    P/B_Previous_Quarter DOUBLE, -- Price-to-Book Ratio — Previous Quarter
    P/B_Same_Period_Last_Year DOUBLE, -- Price-to-Book Ratio — Same Period Last Year
    P/B_d_1 DOUBLE, -- Price-to-Book Ratio — Most Recent Quarter (d_1)
    P/B_d_2 DOUBLE, -- Price-to-Book Ratio — Second Most Recent Quarter (d_2)
    P/B_d_3 DOUBLE, -- Price-to-Book Ratio — Third Most Recent Quarter (d_3)
    P/B_d_4 DOUBLE, -- Price-to-Book Ratio — Fourth Most Recent Quarter (d_4)
    P/B_d_5 DOUBLE, -- Price-to-Book Ratio — Fifth Most Recent Quarter (d_5)
    P/B_d_6 DOUBLE, -- Price-to-Book Ratio — Sixth Most Recent Quarter (d_6)
    P/B_d_7 DOUBLE, -- Price-to-Book Ratio — Seventh Most Recent Quarter (d_7)
    P/B_d_8 DOUBLE, -- Price-to-Book Ratio — Eighth Most Recent Quarter (d_8)
    distance_to_nearest_quarter DOUBLE, -- Distance to Nearest Quarter End

    -- Price-to-Earnings Ratio
    P/E_Previous_Quarter DOUBLE, -- Price-to-Earnings Ratio — Previous Quarter
    P/E_Same_Period_Last_Year DOUBLE, -- Price-to-Earnings Ratio — Same Period Last Year
    P/E_d_1 DOUBLE, -- Price-to-Earnings Ratio — Most Recent Quarter (d_1)
    P/E_d_2 DOUBLE, -- Price-to-Earnings Ratio — Second Most Recent Quarter (d_2)
    P/E_d_3 DOUBLE, -- Price-to-Earnings Ratio — Third Most Recent Quarter (d_3)
    P/E_d_4 DOUBLE, -- Price-to-Earnings Ratio — Fourth Most Recent Quarter (d_4)
    P/E_d_5 DOUBLE, -- Price-to-Earnings Ratio — Fifth Most Recent Quarter (d_5)
    P/E_d_6 DOUBLE, -- Price-to-Earnings Ratio — Sixth Most Recent Quarter (d_6)
    P/E_d_7 DOUBLE, -- Price-to-Earnings Ratio — Seventh Most Recent Quarter (d_7)
    P/E_d_8 DOUBLE, -- Price-to-Earnings Ratio — Eighth Most Recent Quarter (d_8)

    -- Price-to-Sales Ratio
    P/S_Previous_Quarter DOUBLE, -- Price-to-Sales Ratio — Previous Quarter
    P/S_Same_Period_Last_Year DOUBLE, -- Price-to-Sales Ratio — Same Period Last Year
    P/S_d_1 DOUBLE, -- Price-to-Sales Ratio — Most Recent Quarter (d_1)
    P/S_d_2 DOUBLE, -- Price-to-Sales Ratio — Second Most Recent Quarter (d_2)
    P/S_d_3 DOUBLE, -- Price-to-Sales Ratio — Third Most Recent Quarter (d_3)
    P/S_d_4 DOUBLE, -- Price-to-Sales Ratio — Fourth Most Recent Quarter (d_4)
    P/S_d_5 DOUBLE, -- Price-to-Sales Ratio — Fifth Most Recent Quarter (d_5)
    P/S_d_6 DOUBLE, -- Price-to-Sales Ratio — Sixth Most Recent Quarter (d_6)
    P/S_d_7 DOUBLE, -- Price-to-Sales Ratio — Seventh Most Recent Quarter (d_7)
    P/S_d_8 DOUBLE, -- Price-to-Sales Ratio — Eighth Most Recent Quarter (d_8)

    -- Price-to-Cash Flow Ratio
    P/Cash Flow_Previous_Quarter DOUBLE, -- Price-to-Cash Flow Ratio — Previous Quarter
    P/Cash Flow_Same_Period_Last_Year DOUBLE, -- Price-to-Cash Flow Ratio — Same Period Last Year
    P/Cash Flow_d_1 DOUBLE, -- Price-to-Cash Flow Ratio — Most Recent Quarter (d_1)
    P/Cash Flow_d_2 DOUBLE, -- Price-to-Cash Flow Ratio — Second Most Recent Quarter (d_2)
    P/Cash Flow_d_3 DOUBLE, -- Price-to-Cash Flow Ratio — Third Most Recent Quarter (d_3)
    P/Cash Flow_d_4 DOUBLE, -- Price-to-Cash Flow Ratio — Fourth Most Recent Quarter (d_4)
    P/Cash Flow_d_5 DOUBLE, -- Price-to-Cash Flow Ratio — Fifth Most Recent Quarter (d_5)
    P/Cash Flow_d_6 DOUBLE, -- Price-to-Cash Flow Ratio — Sixth Most Recent Quarter (d_6)
    P/Cash Flow_d_7 DOUBLE, -- Price-to-Cash Flow Ratio — Seventh Most Recent Quarter (d_7)
    P/Cash Flow_d_8 DOUBLE, -- Price-to-Cash Flow Ratio — Eighth Most Recent Quarter (d_8)

    -- EPS (VND)
    EPS (VND)_Previous_Quarter DOUBLE, -- Earnings Per Share (VND) — Previous Quarter
    EPS (VND)_Same_Period_Last_Year DOUBLE, -- Earnings Per Share (VND) — Same Period Last Year
    EPS (VND)_d_1 DOUBLE, -- Earnings Per Share (VND) — Most Recent Quarter (d_1)
    EPS (VND)_d_2 DOUBLE, -- Earnings Per Share (VND) — Second Most Recent Quarter (d_2)
    EPS (VND)_d_3 DOUBLE, -- Earnings Per Share (VND) — Third Most Recent Quarter (d_3)
    EPS (VND)_d_4 DOUBLE, -- Earnings Per Share (VND) — Fourth Most Recent Quarter (d_4)
    EPS (VND)_d_5 DOUBLE, -- Earnings Per Share (VND) — Fifth Most Recent Quarter (d_5)
    EPS (VND)_d_6 DOUBLE, -- Earnings Per Share (VND) — Sixth Most Recent Quarter (d_6)
    EPS (VND)_d_7 DOUBLE, -- Earnings Per Share (VND) — Seventh Most Recent Quarter (d_7)
    EPS (VND)_d_8 DOUBLE, -- Earnings Per Share (VND) — Eighth Most Recent Quarter (d_8)

    -- BVPS (VND)
    BVPS (VND)_Previous_Quarter DOUBLE, -- Book Value Per Share (VND) — Previous Quarter
    BVPS (VND)_Same_Period_Last_Year DOUBLE, -- Book Value Per Share (VND) — Same Period Last Year
    BVPS (VND)_d_1 DOUBLE, -- Book Value Per Share (VND) — Most Recent Quarter (d_1)
    BVPS (VND)_d_2 DOUBLE, -- Book Value Per Share (VND) — Second Most Recent Quarter (d_2)
    BVPS (VND)_d_3 DOUBLE, -- Book Value Per Share (VND) — Third Most Recent Quarter (d_3)
    BVPS (VND)_d_4 DOUBLE, -- Book Value Per Share (VND) — Fourth Most Recent Quarter (d_4)
    BVPS (VND)_d_5 DOUBLE, -- Book Value Per Share (VND) — Fifth Most Recent Quarter (d_5)
    BVPS (VND)_d_6 DOUBLE, -- Book Value Per Share (VND) — Sixth Most Recent Quarter (d_6)
    BVPS (VND)_d_7 DOUBLE, -- Book Value Per Share (VND) — Seventh Most Recent Quarter (d_7)
    BVPS (VND)_d_8 DOUBLE, -- Book Value Per Share (VND) — Eighth Most Recent Quarter (d_8)

    -- EV/EBITDA
    EV/EBITDA_d_1 DOUBLE, -- EV/EBITDA — Most Recent Quarter (d_1)
    EV/EBITDA_d_2 DOUBLE, -- EV/EBITDA — Second Most Recent Quarter (d_2)
    EV/EBITDA_d_3 DOUBLE, -- EV/EBITDA — Third Most Recent Quarter (d_3)
    EV/EBITDA_d_4 DOUBLE, -- EV/EBITDA — Fourth Most Recent Quarter (d_4)
    EV/EBITDA_d_5 DOUBLE, -- EV/EBITDA — Fifth Most Recent Quarter (d_5)
    EV/EBITDA_d_6 DOUBLE, -- EV/EBITDA — Sixth Most Recent Quarter (d_6)
    EV/EBITDA_d_7 DOUBLE, -- EV/EBITDA — Seventh Most Recent Quarter (d_7)
    EV/EBITDA_d_8 DOUBLE, -- EV/EBITDA — Eighth Most Recent Quarter (d_8)

    -- ROE (%)
    ROE (%)_Previous_Quarter DOUBLE, -- Return on Equity (ROE) — Previous Quarter
    ROE (%)_Same_Period_Last_Year DOUBLE, -- Return on Equity (ROE) — Same Period Last Year
    ROE (%)_d_1 DOUBLE, -- Return on Equity (ROE) — Most Recent Quarter (d_1)
    ROE (%)_d_2 DOUBLE, -- Return on Equity (ROE) — Second Most Recent Quarter (d_2)
    ROE (%)_d_3 DOUBLE, -- Return on Equity (ROE) — Third Most Recent Quarter (d_3)
    ROE (%)_d_4 DOUBLE, -- Return on Equity (ROE) — Fourth Most Recent Quarter (d_4)
    ROE (%)_d_5 DOUBLE, -- Return on Equity (ROE) — Fifth Most Recent Quarter (d_5)
    ROE (%)_d_6 DOUBLE, -- Return on Equity (ROE) — Sixth Most Recent Quarter (d_6)
    ROE (%)_d_7 DOUBLE, -- Return on Equity (ROE) — Seventh Most Recent Quarter (d_7)
    ROE (%)_d_8 DOUBLE, -- Return on Equity (ROE) — Eighth Most Recent Quarter (d_8)

    -- ROIC (%)
    ROIC (%)_d_1 DOUBLE, -- Return on Invested Capital (ROIC) — Most Recent Quarter (d_1)
    ROIC (%)_d_2 DOUBLE, -- Return on Invested Capital (ROIC) — Second Most Recent Quarter (d_2)
    ROIC (%)_d_3 DOUBLE, -- Return on Invested Capital (ROIC) — Third Most Recent Quarter (d_3)
    ROIC (%)_d_4 DOUBLE, -- Return on Invested Capital (ROIC) — Fourth Most Recent Quarter (d_4)
    ROIC (%)_d_5 DOUBLE, -- Return on Invested Capital (ROIC) — Fifth Most Recent Quarter (d_5)
    ROIC (%)_d_6 DOUBLE, -- Return on Invested Capital (ROIC) — Sixth Most Recent Quarter (d_6)
    ROIC (%)_d_7 DOUBLE, -- Return on Invested Capital (ROIC) — Seventh Most Recent Quarter (d_7)
    ROIC (%)_d_8 DOUBLE, -- Return on Invested Capital (ROIC) — Eighth Most Recent Quarter (d_8)

    -- ROA (%)
    ROA (%)_Previous_Quarter DOUBLE, -- Return on Assets (ROA) — Previous Quarter
    ROA (%)_Same_Period_Last_Year DOUBLE, -- Return on Assets (ROA) — Same Period Last Year
    ROA (%)_d_1 DOUBLE, -- Return on Assets (ROA) — Most Recent Quarter (d_1)
    ROA (%)_d_2 DOUBLE, -- Return on Assets (ROA) — Second Most Recent Quarter (d_2)
    ROA (%)_d_3 DOUBLE, -- Return on Assets (ROA) — Third Most Recent Quarter (d_3)
    ROA (%)_d_4 DOUBLE, -- Return on Assets (ROA) — Fourth Most Recent Quarter (d_4)
    ROA (%)_d_5 DOUBLE, -- Return on Assets (ROA) — Fifth Most Recent Quarter (d_5)
    ROA (%)_d_6 DOUBLE, -- Return on Assets (ROA) — Sixth Most Recent Quarter (d_6)
    ROA (%)_d_7 DOUBLE, -- Return on Assets (ROA) — Seventh Most Recent Quarter (d_7)
    ROA (%)_d_8 DOUBLE, -- Return on Assets (ROA) — Eighth Most Recent Quarter (d_8)

    -- Technicals
    ticket VARCHAR(10), -- Stock Symbol
    volume_ma DOUBLE, -- Volume Moving Average (20 days)
    volume_to_volume_ma_ratio DOUBLE, -- Volume to Volume Moving Average Ratio
    ema_12 DOUBLE, -- Exponential Moving Average (12 days)
    ema_26 DOUBLE, -- Exponential Moving Average (26 days)
    sma_20 DOUBLE, -- Simple Moving Average (20 days)
    sma_50 DOUBLE, -- Simple Moving Average (50 days)
    roc_5 DOUBLE, -- Rate of Change (5 days)
    %K DOUBLE, -- Stochastic Oscillator %K (14)
    %R DOUBLE, -- Williams %R (14)
    typical_price DOUBLE, -- Typical Price
    cci DOUBLE, -- Commodity Channel Index (20)
    obv DOUBLE, -- On-Balance Volume
    macd DOUBLE, -- MACD (12,26)
    signal_line DOUBLE, -- MACD Signal Line
    macd_histogram DOUBLE, -- MACD Histogram
    rsi DOUBLE, -- Relative Strength Index (14)
    bb_bbm DOUBLE, -- Bollinger Bands Middle Band
    bb_bbh DOUBLE, -- Bollinger Bands Upper Band
    bb_bbl DOUBLE, -- Bollinger Bands Lower Band
    bb_bbp DOUBLE, -- Bollinger Bands %B
    bb_bbh_bb_bbl_ratio DOUBLE, -- Ratio of Upper to Lower Bollinger Band
    log_return DOUBLE, -- Daily Logarithmic Return
    volatility_5d DOUBLE, -- Rolling Volatility (5 days)
    volatility_10d DOUBLE, -- Rolling Volatility (10 days)
    volatility_20d DOUBLE, -- Rolling Volatility (20 days)
    volatility_30d DOUBLE, -- Rolling Volatility (30 days)
    mean_log_return_5d DOUBLE, -- Mean Log Return (5 days)
    mean_log_return_10d DOUBLE, -- Mean Log Return (10 days)
    mean_log_return_20d DOUBLE, -- Mean Log Return (20 days)
    mean_log_return_30d DOUBLE, -- Mean Log Return (30 days)
    sharpe_like_5d DOUBLE, -- Sharpe-like Ratio (5 days)
    sharpe_like_10d DOUBLE, -- Sharpe-like Ratio (10 days)
    sharpe_like_20d DOUBLE, -- Sharpe-like Ratio (20 days)
    sharpe_like_30d DOUBLE, -- Sharpe-like Ratio (30 days)
    up_streak INT, -- Consecutive Upward Streak
    pos_log_return_ratio_20d DOUBLE, -- Positive Log Return Ratio (20 days)
    z_score_5d DOUBLE, -- Z-score of Log Return (5 days)
    z_score_10d DOUBLE, -- Z-score of Log Return (10 days)
    z_score_20d DOUBLE, -- Z-score of Log Return (20 days)
    z_score_30d DOUBLE, -- Z-score of Log Return (30 days)
    annual_return DOUBLE, -- Annual Return (252-day)
    daily_return DOUBLE, -- Daily Return
    annual_volatility DOUBLE, -- Annualized Volatility
    sharpe_ratio DOUBLE, -- Annualized Sharpe Ratio

    -- Change-rate flags & values
    P/B_change_rate_flag INT, -- P/B Change Rate Validity Flag
    P/B_change_rate DOUBLE, -- P/B Change Rate
    P/E_change_rate_flag INT, -- P/E Change Rate Validity Flag
    P/E_change_rate DOUBLE, -- P/E Change Rate
    P/S_change_rate_flag INT, -- P/S Change Rate Validity Flag
    P/S_change_rate DOUBLE, -- P/S Change Rate
    P/Cash Flow_change_rate_flag INT, -- P/CF Change Rate Validity Flag
    P/Cash Flow_change_rate DOUBLE, -- P/CF Change Rate
    EPS (VND)_change_rate_flag INT, -- EPS Change Rate Validity Flag
    EPS (VND)_change_rate DOUBLE, -- EPS Change Rate
    BVPS (VND)_change_rate_flag INT, -- BVPS Change Rate Validity Flag
    BVPS (VND)_change_rate DOUBLE, -- BVPS Change Rate
    ROE (%)_change_rate_flag INT, -- ROE Change Rate Validity Flag
    ROE (%)_change_rate DOUBLE, -- ROE Change Rate
    ROA (%)_change_rate_flag INT, -- ROA Change Rate Validity Flag
    ROA (%)_change_rate DOUBLE, -- ROA Change Rate

    -- VNINDEX prices & indicators (VND)
    open_vnd DOUBLE, -- VNINDEX Open Price
    high_vnd DOUBLE, -- VNINDEX High Price
    low_vnd DOUBLE, -- VNINDEX Low Price
    close_vnd DOUBLE, -- VNINDEX Close Price
    volume_vnd BIGINT, -- VNINDEX Trading Volume
    rsi_vnd DOUBLE, -- VNINDEX Relative Strength Index (14)
    rsi_base_ma_vnd DOUBLE, -- VNINDEX RSI Base Moving Average (9 days)
    rsi_rsi_base_ma_ratio_vnd DOUBLE, -- VNINDEX RSI to RSI-MA Ratio
    volume_ma_vnd DOUBLE, -- VNINDEX Volume Moving Average (20 days)
    volume_to_volume_ma_ratio_vnd DOUBLE, -- VNINDEX Volume to Volume MA Ratio
    bb_bbm_vnd DOUBLE, -- VNINDEX Bollinger Bands Middle Band
    bb_bbh_vnd DOUBLE, -- VNINDEX Bollinger Bands Upper Band
    bb_bbl_vnd DOUBLE, -- VNINDEX Bollinger Bands Lower Band
    bb_bbp_vnd DOUBLE, -- VNINDEX Bollinger Bands %B
    bb_bbh_bb_bbl_ratio_vnd DOUBLE, -- VNINDEX Upper to Lower Bollinger Band Ratio
    roc_vnd DOUBLE, -- VNINDEX Rate of Change (9 days)
    %K_vnd DOUBLE, -- VNINDEX Stochastic Oscillator %K (14)
    %R_vnd DOUBLE, -- VNINDEX Williams %R (14)
    typical_price_vnd DOUBLE, -- VNINDEX Typical Price
    cci_vnd DOUBLE, -- VNINDEX Commodity Channel Index (20)
    obv_vnd DOUBLE, -- VNINDEX On-Balance Volume
    ema_12_vnd DOUBLE, -- VNINDEX EMA (12 days)
    ema_26_vnd DOUBLE, -- VNINDEX EMA (26 days)
    sma_20_vnd DOUBLE, -- VNINDEX SMA (20 days)
    sma_50_vnd DOUBLE, -- VNINDEX SMA (50 days)
    change DOUBLE, -- Next-day Percentage Change

    -- Target & sentiment coefficients
    target DOUBLE, -- Trading Signal Target
    coefficient_P/B DOUBLE, -- Sentiment Coefficient — P/B
    coefficient_P/E DOUBLE, -- Sentiment Coefficient — P/E
    coefficient_P/S DOUBLE, -- Sentiment Coefficient — P/S
    coefficient_P/Cash Flow DOUBLE, -- Sentiment Coefficient — P/CF
    coefficient_EPS (VND) DOUBLE, -- Sentiment Coefficient — EPS
    coefficient_BVPS (VND) DOUBLE, -- Sentiment Coefficient — BVPS
    coefficient_ROE (%) DOUBLE, -- Sentiment Coefficient — ROE
    coefficient_ROA (%) DOUBLE, -- Sentiment Coefficient — ROA

    -- News-sentiment factors
    Reputation DOUBLE, -- Sentiment — Reputation
    Financial DOUBLE, -- Sentiment — Financial
    Regulatory DOUBLE, -- Sentiment — Regulatory
    Risks DOUBLE, -- Sentiment — Risks
    Fundamentals DOUBLE, -- Sentiment — Fundamentals
    Conditions DOUBLE, -- Sentiment — Conditions
    Market DOUBLE, -- Sentiment — Market
    Volatility DOUBLE, -- Sentiment — Volatility
    symbol VARCHAR(10) -- Ticker Symbol
);
"""

gpt_turbo_config_scenario_analysis = {
    "temperature": 0.7,
    "config_list": config_list_gpt_scenario_analysis,
    "functions" : 
    [
        {
            "name": "query_maker_scenario_analysis",
            "description": "Generate an SQL query based on the user's request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_input": {
                        "type": "string",
                        "description": "This is the input from the user.",
                    }
                },
                "required": ["user_input"],
            },
        },
        {
            "name": "run_sql_query_scenario_analysis",
            "description": "This function is used to execute SQL queries based on the user's request and retrieve the results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql_query_1": {
                        "type": "string",
                        "description": "This is an SQL query to be executed.",
                    },
                    "sql_query_2": {
                        "type": "string",
                        "description": "This is an SQL query to be executed.",
                    }
                },
                "required": ["sql_query_1", "sql_query_2"],
            },
        },
        {
            "name": "scenario_analysis",
            "description": "Run a model and return the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_path": {
                        "type": "string",
                        "description": "The path to the data.",
                    },
                    "list_feature": {
                        "type": "string",
                        "description": "The list of features to analyze.",
                    },
                    "list_min_value": {
                        "type": "string",
                        "description": "The list of minimum values of the features.",
                    },
                    "list_max_value": {
                        "type": "string",
                        "description": "The list of maximum values of the features.",
                    },
                },
                "required": ["data_path", "list_feature", "list_min_value", "list_max_value"],
            },
        },
        {
            "name": "visualize_scenario_analysis",
            "description": "Visualize the scenario analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_path": {
                        "type": "string",
                        "description": "The path to the data.",
                    },
                    "list_feature": {
                        "type": "string",
                        "description": "The list of features to visualize.",
                    },
                },
                "required": ["data_path", "list_feature"],
            },
        }
    ]
}