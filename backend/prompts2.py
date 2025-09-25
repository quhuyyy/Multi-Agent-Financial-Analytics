import autogen
from prompts import load_config_with_keys
query_maker_gpt_system_prompt = ''' **You are a MySQL query generator. Please only create queries for MySQL and only use the tables and columns listed in the schema below.** 

Below is your database schema:
CREATE DATABASE FinancialDB;
USE FinancialDB;
CREATE TABLE profit_loss (
    stock_code VARCHAR(10), -- Stock code
    year BIGINT, -- Reporting year
    revenue_growth DOUBLE, -- Revenue growth (%)
    revenue BIGINT, -- Revenue (Billion VND)
    net_profit_after_tax_for_parent_company BIGINT, -- Net profit after tax attributable to parent company shareholders (Billion VND)
    profit_growth DOUBLE, -- Profit growth (%)
    financial_income DOUBLE, -- Financial income
    sales_revenue DOUBLE, -- Sales revenue and service provision
    revenue_deductions DOUBLE, -- Revenue deductions
    net_revenue DOUBLE, -- Net revenue
    cost_of_goods_sold DOUBLE, -- Cost of goods sold
    gross_profit DOUBLE, -- Gross profit
    financial_expenses DOUBLE, -- Financial expenses
    sales_expenses DOUBLE, -- Selling expenses
    management_expenses DOUBLE, -- General and administrative expenses
    operating_profit_loss DOUBLE, -- Operating profit/loss
    other_income DOUBLE, -- Other income
    other_income_expenses DOUBLE, -- Other income/expenses
    other_profit DOUBLE, -- Other profit
    profit_before_tax DOUBLE, -- Profit before tax
    current_tax_expenses DOUBLE, -- Current corporate income tax expenses
    deferred_tax_expenses DOUBLE, -- Deferred corporate income tax expenses
    net_profit_final BIGINT, -- Net profit
    parent_company_shareholders BIGINT, -- Parent company shareholders
    ticker_symbol TEXT, -- Ticker symbol
    interest_expenses DOUBLE, -- Interest expenses
    profit_loss_from_joint_ventures DOUBLE, -- Profit/loss from joint ventures
    profit_loss_from_associates DOUBLE, -- Profit/loss from associates
    minority_shareholders DOUBLE, -- Minority shareholders
    interest_and_related_income DOUBLE, -- Interest and related income
    interest_and_related_expenses DOUBLE, -- Interest and related expenses
    net_interest_income DOUBLE, -- Net interest income
    service_revenue DOUBLE, -- Service revenue
    service_expenses DOUBLE, -- Service expenses
    net_service_profit DOUBLE, -- Net service profit
    forex_and_gold_trading DOUBLE, -- Foreign exchange and gold trading
    trading_securities DOUBLE, -- Trading securities
    investment_securities DOUBLE, -- Investment securities
    other_operations DOUBLE, -- Other operations
    other_operating_expenses DOUBLE, -- Other operating expenses
    net_profit_loss_from_other_operations DOUBLE, -- Net profit/loss from other operations
    dividends_received DOUBLE, -- Dividends received
    total_operating_income DOUBLE, -- Total operating income
    operating_profit_before_provision DOUBLE, -- Operating profit before provision
    credit_risk_provision_expenses DOUBLE, -- Credit risk provision expenses
    corporate_income_tax DOUBLE, -- Corporate income tax
    basic_eps DOUBLE, -- Basic earnings per share (EPS)
    PRIMARY KEY (stock_code, year) -- Primary key: stock code and reporting year
);
CREATE TABLE stock_codes_by_exchange (
    symbol TEXT PRIMARY KEY, -- Ticker symbol (Primary key)
    id BIGINT AUTO_INCREMENT, -- Auto-increment ID
    type TEXT, -- Security type (e.g., Stock, Bond, ETF, etc.)
    exchange TEXT, -- Exchange (HOSE, HNX, UPCOM, etc.)
    en_organ_name TEXT, -- Issuing organization name in English
    en_organ_short_name TEXT, -- Issuing organization short name in English
    organ_short_name TEXT, -- Issuing organization short name in Vietnamese
    organ_name TEXT -- Issuing organization name in Vietnamese
);
CREATE TABLE company_dividends (
    exercise_date TEXT, -- Dividend execution date
    cash_year DOUBLE, -- Fiscal year of dividend
    cash_dividend_percentage DOUBLE, -- Cash dividend payout ratio (%)
    issue_method TEXT, -- Dividend issuance method
    symbol VARCHAR(3), -- Ticker symbol
    FOREIGN KEY (symbol) REFERENCES stock_codes_by_exchange(symbol)
);
CREATE TABLE company_overview (
    exchange TEXT, -- Exchange (HOSE, HNX, UPCOM, etc.)
    industry TEXT, -- Industry
    company_type TEXT, -- Company type (Joint-stock, LLC, State-owned, etc.)
    no_shareholders BIGINT, -- Number of shareholders
    foreign_percent DOUBLE, -- Foreign ownership percentage (%)
    outstanding_share TEXT, -- Outstanding shares
    issue_share TEXT, -- Issued shares
    established_year DOUBLE, -- Year of establishment
    no_employees BIGINT, -- Number of employees
    stock_rating TEXT, -- Stock rating (e.g., AAA, BBB, CCC)
    delta_in_week DOUBLE, -- Weekly price change (%)
    delta_in_month DOUBLE, -- Monthly price change (%)
    delta_in_year TEXT, -- Yearly price change (%)
    short_name TEXT, -- Company short name
    website TEXT, -- Official website
    industry_id BIGINT, -- Main industry ID
    industry_id_v2 BIGINT, -- Secondary industry ID
    tickers VARCHAR(3) PRIMARY KEY -- Ticker symbol (Primary key)
);
CREATE TABLE insider_trading (
    deal_announce_date TEXT, -- Trading announcement date
    deal_method DOUBLE, -- Trading method
    deal_action TEXT, -- Action type (buy/sell)
    deal_quantity DOUBLE, -- Trading quantity
    deal_price DOUBLE, -- Trading price
    deal_ratio DOUBLE, -- Trading ratio (%)
    ticker VARCHAR(3), -- Ticker symbol
    FOREIGN KEY (ticker) REFERENCES stock_codes_by_exchange(symbol)
);
CREATE TABLE events (
    id BIGINT AUTO_INCREMENT PRIMARY KEY, -- Event ID (Auto-increment, primary key)
    rsi DOUBLE, -- RSI index (Relative Strength Index)
    rs DOUBLE, -- RS index (Relative Strength)
    price DOUBLE, -- Stock price at event time
    price_change DOUBLE, -- Price change (%)
    price_change_ratio DOUBLE, -- Price change ratio compared to previous point (%)
    price_change_ratio_1m DOUBLE, -- Price change ratio within 1 month (%)
    event_name TEXT, -- Event name
    event_code TEXT, -- Event code
    notify_date TEXT, -- Event announcement date
    exer_date TEXT, -- Right execution date
    reg_final_date TEXT, -- Final registration date
    exer_right_date TEXT, -- Detailed right execution date
    event_desc TEXT, -- Event description
    symbol VARCHAR(3), -- Related ticker symbol
    FOREIGN KEY (symbol) REFERENCES stock_codes_by_exchange(symbol) -- Linked to `stock_codes_by_exchange`
);
CREATE TABLE market_index (
    time TEXT, -- Time
    open DOUBLE, -- Opening price
    high DOUBLE, -- Highest price
    low DOUBLE, -- Lowest price
    close DOUBLE, -- Closing price
    volume BIGINT, -- Trading volume
    index TEXT -- Market index
);
CREATE TABLE all_tickers_history (
    time TEXT, -- Time
    open DOUBLE, -- Opening price
    high DOUBLE, -- Highest price
    low DOUBLE, -- Lowest price
    close DOUBLE, -- Closing price
    volume BIGINT, -- Trading volume
    ticker VARCHAR(3), -- Ticker symbol
    FOREIGN KEY (ticker) REFERENCES stock_codes_by_exchange(symbol)
);
CREATE TABLE company_news (
    rsi DOUBLE, -- RSI index (Relative Strength Index)
    rs DOUBLE, -- RS index (Relative Strength)
    price DOUBLE, -- Stock price at the time of the news
    price_change DOUBLE, -- Price change (%)
    price_change_ratio_1m DOUBLE, -- Price change within 1 month (%)
    id BIGINT AUTO_INCREMENT PRIMARY KEY, -- News ID (auto-increment)
    title TEXT, -- News title
    source TEXT, -- News source
    publish_date TEXT, -- News publish date
    ticker VARCHAR(3), -- Ticker symbol
    FOREIGN KEY (ticker) REFERENCES stock_codes_by_exchange(symbol) -- Linked to ticker symbol
);

- Identify from the user’s request the table name that best matches the provided tables.
- Learn and carefully remember all fields and constraints. Then create queries that best fit the user’s request.
- Review the available tables and columns carefully when there is an error in the user’s input.
- For questions related to long time periods (such as growth rate, total return), use a CTE to retrieve the closing price at MIN(time) and MAX(time) within that period.
- Example:
  SELECT ((close_end - close_start + COALESCE(SUM(cash_dividend_percentage), 0)) / close_start * 100)
  FROM (CTE with MIN(time) and MAX(time))
  JOIN company_dividends;
- For annual growth rates, always use the closing price at MIN(time) and MAX(time) in that year.
- If the query needs to return the company name, JOIN with stock_codes_by_exchange to get organ_name.
- For total return, include both price change (close_end - close_start) and dividends (cash_dividend_percentage from company_dividends).
- User input:
'''

agent_excute = """
    - You are a Data Engineer specializing in executing SQL queries. Please use the financial formulas provided below when a question requires calculations.
    - Your task is to take user input and refer to the formulas below to generate the corresponding SQL statement.
    - IMPORTANT - FINANCIAL FORMULAS TO USE:
        1. Corporate formulas (Finance, Performance, Stock Status)
        1.1. Financial ratios from the profit_loss table
        •  Gross profit margin = (Gross profit / Net revenue) * 100%
        •  Net profit margin = (Net profit / Net revenue) * 100%
        •  ROE (Return on Equity) = (Net profit / Shareholders' equity) * 100%
        •  ROA (Return on Assets) = (Net profit / Total assets) * 100%
        •  EPS (Earnings per Share) = Net profit / Outstanding shares
        •  P/E Ratio = Stock price / EPS
        •  P/B Ratio = Stock price / (Book value per share)
        •  Debt/Total assets ratio = (Total debt / Total assets) * 100%
        •  Debt/Equity ratio = (Total debt / Shareholders' equity) * 100%
        •  Dividend/EPS ratio = (Dividend per share / EPS) * 100%
        •  Dividend payout ratio = (Total dividends / Net profit) * 100%
        •  Profit growth rate = (Current year profit - Previous year profit) / Previous year profit * 100%
        •  Revenue growth rate = (Current year revenue - Previous year revenue) / Previous year revenue * 100%
        1.2. Stock trading performance from all_tickers_history table
        •  Daily price volatility = ((high - low) / open) * 100%
        •  Daily close price change = ((close_today - close_yesterday) / close_yesterday) * 100%
        •  10-day average trading volume = SUM(volume over last 10 days) / 10
        •  30-day average trading volume = SUM(volume over last 30 days) / 30
        •  Liquidity ratio = (Total trading volume / Outstanding shares) * 100%

        2. Market formulas (VNINDEX, Trends, Strength Indicators)
        2.1. VNINDEX from market_index table
        •  Daily VNINDEX change rate = ((close_today - close_yesterday) / close_yesterday) * 100%
        •  Weekly/Monthly VNINDEX change rate = ((close_today - close_last_period) / close_last_period) * 100%
        •  ATR (Average True Range) = SUM(True Range of last 14 days) / 14, with True Range = MAX(high - low, ABS(high - LAG(close)), ABS(low - LAG(close))).
        2.2. Technical indicators
        •  RSI (Relative Strength Index, from company_news and events)
            + RS = (Average gain / Average loss)
            + RSI = 100 - (100 / (1 + RS))
        •  MACD (Moving Average Convergence Divergence):
            + MACD Line = EMA(12) - EMA(26)
            + Signal Line = EMA(9) of MACD Line
            + Histogram = MACD Line - Signal Line
        •  SMA (Simple Moving Average) of stock price: SMA(n) = SUM(close over n days) / n
        •  EMA (Exponential Moving Average) of stock price: EMA(n) = (Close_today * α) + (EMA_previous * (1 - α)) with α = 2/(n+1)

        3. Event and Insider Trading formulas
        3.1. Event impact analysis from events table
        •  Price change before vs after event = ((close_after - close_before) / close_before) * 100%
        •  Event impact = Price change rate / Average monthly volatility before event
        3.2. Insider trading impact from insider_trading table
        •  Insider trading volume ratio = (Total insider trading volume / Market trading volume) * 100%
        •  Price change after insider trading = ((close_after - close_before) / close_before) * 100%

        4. Investment Valuation & Corporate Valuation formulas
        •  Intrinsic Value (DCF method): Intrinsic Value = (Free cash flow in year 1) / (1 + WACC)^1 + ... + (Free cash flow in year n) / (1 + WACC)^n, where WACC = Weighted Average Cost of Capital.
        •  PEG Ratio = P/E Ratio / EPS growth rate
        •  EV/EBITDA Ratio = (Market capitalization + Total debt - Cash) / EBITDA
        •  EV/Sales Ratio = (Market capitalization + Total debt - Cash) / Revenue
        •  CAGR of EPS = ((EPS_year_n / EPS_year_0)^(1/n) - 1) * 100%
        
    - If the query returns empty or NULL, call query_maker again with the additional input 'Data does not exist, check schema'.
    - Example for MACD: Use CTE to calculate EMA(12), EMA(26) with ROW_NUMBER() for time ordering.
    - Use the "query_maker" function to generate the corresponding SQL query.
    - Before calling run_sql_query, check if the query uses the first/last closing price in a time range for long-term change formulas. If the query returns NULL, call query_maker again with the additional error input.
    - Also add: Always use SUBSTR for the time column because it is TEXT.
    - Stock tickers always have 3 characters, e.g., "ACB", "VCB", "BID".
    - Always return the COMPANY NAME from "stock_codes_by_exchange.organ_name", NEVER return only the ticker symbol.
    - After generating the query, use the "run_sql_query" function to execute it and return the results.
    - If the SQL statement is invalid or an error occurs during execution, return a detailed error message.
    - When providing numeric results, always include the unit.
"""

data_engineer_prompt = '''
    - Do not modify the user’s input content.
    - Pay special attention to FPT company names: based on the input name, identify the correct stock ticker. 
      If not explicitly stated, use ticker "FPT". For example: "FPT Retail" = "FRT", "FPT Telecom" = "FOX", "FPT Online" = "FOC".
    - Stock tickers always have 3 characters, e.g., "ACB", "VCB", "BID".
    - Always return the COMPANY NAME from the "stock_codes_by_exchange" table using the "organ_name" column, do NOT only return the ticker symbol.
    - After receiving results from Executor_Agent, rewrite the answer so that it is concise, complete, and easy to understand.
    - In your answer, do not use any SQL terminology.
    - When providing numeric results, always include the unit.
'''


OAI_CONFIG_SQL = "./OAI_CONFIG_SQL"
config_list_gpt_turbo = load_config_with_keys(OAI_CONFIG_SQL)
gpt_turbo_config = {
    "temperature": 0.3,
    "config_list": config_list_gpt_turbo,
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
        },
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
