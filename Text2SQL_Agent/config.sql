use financial;

ALTER TABLE company_overview MODIFY Tickers VARCHAR(3) NOT NULL;

ALTER TABLE company_overview ADD PRIMARY KEY (Tickers);
ALTER TABLE stock_codes_by_exchange MODIFY symbol VARCHAR(3) NOT NULL;

ALTER TABLE stock_codes_by_exchange ADD PRIMARY KEY (symbol);

desc all_tickers_history;
ALTER TABLE all_tickers_history MODIFY Ticker VARCHAR(3) NOT NULL;
ALTER TABLE all_tickers_history ADD CONSTRAINT fk_all_tickers FOREIGN KEY (Ticker) REFERENCES company_overview (Tickers);

desc company_dividends;
ALTER TABLE company_dividends MODIFY Symbol VARCHAR(3) NOT NULL;
ALTER TABLE company_dividends ADD CONSTRAINT fk_company_dividends FOREIGN KEY (Symbol) REFERENCES company_overview (Tickers);

desc company_news;
ALTER TABLE company_news MODIFY Symbol VARCHAR(3) NOT NULL;
ALTER TABLE company_news ADD CONSTRAINT fk_company_news FOREIGN KEY (Symbol) REFERENCES company_overview (Tickers);

desc events;
ALTER TABLE events MODIFY Symbol VARCHAR(3) NOT NULL;
ALTER TABLE events ADD CONSTRAINT fk_events FOREIGN KEY (Symbol) REFERENCES company_overview (Tickers);

desc insider_trading;
ALTER TABLE insider_trading MODIFY Ticker VARCHAR(3) NOT NULL;
ALTER TABLE insider_trading ADD CONSTRAINT fk_insider_trading FOREIGN KEY (Ticker) REFERENCES company_overview (Tickers);

desc profit_loss;
ALTER TABLE profit_loss MODIFY CP VARCHAR(3) NOT NULL;
ALTER TABLE profit_loss ADD CONSTRAINT fk_profit_loss FOREIGN KEY (CP) REFERENCES company_overview (Tickers);

desc company_overview;
ALTER TABLE company_overview ADD CONSTRAINT fk_company_stock FOREIGN KEY (Tickers) REFERENCES stock_codes_by_exchange (symbol);
