import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from vnstock import *
from typing import Optional, Dict
from sklearn.preprocessing import StandardScaler
from langchain.schema.runnable import RunnableSequence
from langchain.prompts import PromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI
import connectDB
import matplotlib.pyplot as plt
from prompts import gpt_turbo_config, admin_ml_prompt, query_maker_prompt_ml
from autogen import UserProxyAgent, AssistantAgent
from prompts import machine_learning_agent_prompt
import logging
import shap
import re
import json

logging.basicConfig(level=logging.INFO)
logging.getLogger("langchain").setLevel(logging.DEBUG)



class ML_System(UserProxyAgent):
    def __init__(self, config_item, name, human_input_mode, termination_msg, code_execution_config, param_ml: Optional[Dict] = None):
        super().__init__(
            name=name,
            code_execution_config=code_execution_config,
            human_input_mode=human_input_mode,
        )
        self._config_item = config_item
        self._param_ml = {} if param_ml is None else param_ml
        self._model_path = r"/app/model/random_forest_model.pkl"
        self._connection = connectDB.get_Connection()
        self._df_list = []
        self._y_pred_probs_lstm = None
        self._scaler_path = r"/app/model/scaler.pkl"
        self._feature_importance_path = r"/app/data/feature_importance.csv"
        self._shap_values_global_path = r"/app/model/shap_feature_importance.csv"
        
        self._rate_limiter = InMemoryRateLimiter(
            requests_per_second=1/20,
            check_every_n_seconds=0.1,
            max_bucket_size=10,
        )
        
        self._user_proxy = UserProxyAgent(
            name="Admin",
            human_input_mode="NEVER",
            system_message= admin_ml_prompt + termination_msg,
            is_termination_msg=self.is_termination_msg
        )
        
        self._openaiLLM = ChatOpenAI(
            model=self._config_item["model"],
            temperature=self._config_item["temperature"],
            openai_api_key=self._config_item["api_key"],
            base_url = self._config_item["base_url"],
            rate_limiter=self._rate_limiter,
            cache=False
        )
        
        self._ml_agent = AssistantAgent(
            name="MLAgent",
            llm_config=gpt_turbo_config,
            system_message=machine_learning_agent_prompt + termination_msg,
            function_map={
                "query_maker_ml": self.query_maker_ml,
                "run_sql_query_ml": self.run_sql_query_ml,
                "run_model": self.run_model,
                "get_feature_importance": self.get_feature_importance,
                "visualize_feature_importance": self.visualize_feature_importance,
            }
        )
        
        self._user_proxy.register_function(function_map={
            "run_model": self.run_model,
            "query_maker_ml": self.query_maker_ml,
            "run_sql_query_ml": self.run_sql_query_ml,
            "get_feature_importance": self.get_feature_importance,
            "visualize_feature_importance": self.visualize_feature_importance,
        })
    
    @staticmethod
    def is_termination_msg(content):
        have_content = content.get("content", None) is not None
        if have_content and "Đã hoàn tất." in content["content"]:
            return True
        return False
    
    def query_maker_ml(self, user_input):
        prompt_template = PromptTemplate.from_template(
            "{system_prompt} + '\n' +  {user_input}."
        )
        chain = RunnableSequence(prompt_template | self._openaiLLM)
        query = chain.invoke({"system_prompt": query_maker_prompt_ml, "user_input": user_input}).content.split("---")
        print(query)
        return query[0], query[1]

    def run_sql_query_ml(self, sql_query_1, sql_query_2):
        cursor = self._connection.cursor()
        try:
            # Query 1
            cursor.execute(sql_query_1)
            rows_1 = cursor.fetchall()
            cols_1 = [desc[0] for desc in cursor.description]  # tên cột
            df_1 = pd.DataFrame(rows_1, columns=cols_1)
            print(df_1.head())  # in 5 dòng đầu
            print(type(df_1))

            # Query 2
            cursor.execute(sql_query_2)
            rows_2 = cursor.fetchall()
            cols_2 = [desc[0] for desc in cursor.description]
            df_2 = pd.DataFrame(rows_2, columns=cols_2)
            print(df_2.head())
            print(type(df_2))

        except Exception as e:
            print("Error:", e)
        cursor.close()
        data_path = "/app/data/result.csv"
        df_1.to_csv(data_path, index=False)
        return data_path
    
    def run_model(self, data_path):
        model = joblib.load(self._model_path)
            
        df = pd.read_csv(data_path)
        ta_feature = ['volume_ma','volume_to_volume_ma_ratio','ema_12','ema_26','sma_20','sma_50','roc_5','%K','%R','cci','obv','macd','signal_line','macd_histogram','rsi','bb_bbm','bb_bbh','bb_bbl','bb_bbp','bb_bbh_bb_bbl_ratio','rsi_vnd','rsi_base_ma_vnd','rsi_rsi_base_ma_ratio_vnd','volume_ma_vnd','volume_to_volume_ma_ratio_vnd','bb_bbm_vnd','bb_bbh_vnd','bb_bbl_vnd','bb_bbp_vnd','bb_bbh_bb_bbl_ratio_vnd','roc_vnd','%K_vnd','%R_vnd','cci_vnd','obv_vnd','ema_12_vnd','ema_26_vnd','sma_20_vnd','sma_50_vnd']
        fa_feature = ['P/B_Previous_Quarter', 'P/B_change_rate','P/B_change_rate_flag','P/E_Previous_Quarter','P/E_change_rate','P/E_change_rate_flag','P/S_Previous_Quarter','P/S_change_rate','P/S_change_rate_flag','P/Cash Flow_Previous_Quarter','P/Cash Flow_change_rate','P/Cash Flow_change_rate_flag','EPS (VND)_Previous_Quarter','EPS (VND)_change_rate', 'EPS (VND)_change_rate_flag','BVPS (VND)_Previous_Quarter','BVPS (VND)_change_rate', 'BVPS (VND)_change_rate_flag','ROE (%)_Previous_Quarter','ROE (%)_change_rate','ROE (%)_change_rate_flag','ROA (%)_Previous_Quarter','ROA (%)_change_rate','ROA (%)_change_rate_flag','log_return','volatility_5d','volatility_10d','volatility_20d','volatility_30d','mean_log_return_5d','mean_log_return_10d','mean_log_return_20d','mean_log_return_30d','sharpe_like_5d','sharpe_like_10d','sharpe_like_20d','sharpe_like_30d','up_streak','pos_log_return_ratio_20d','z_score_5d','z_score_10d','z_score_20d','z_score_30d','annual_return','daily_return','sharpe_ratio','coefficient_P/B','coefficient_P/E','coefficient_P/S','coefficient_P/Cash Flow','coefficient_EPS (VND)','coefficient_BVPS (VND)','coefficient_ROE (%)','coefficient_ROA (%)','distance_to_nearest_quarter']
        sa_feature = ['Reputation', 'Financial', 'Regulatory', 'Risks','Fundamentals', 'Conditions', 'Market', 'Volatility']
        features = ta_feature + fa_feature + sa_feature
        target = 'target'
        scaler = joblib.load(self._scaler_path)
        X_test = df[features]
        print(df['target'].value_counts())
        y_test = df[target]
        print(X_test.columns)
        X_test_all = scaler.transform(X_test)
        y_pred_probs_lstm = model.predict_proba(X_test_all[0].reshape(1, -1))
        self._y_pred_probs_lstm = y_pred_probs_lstm
        return y_pred_probs_lstm
    
    ##SHAP
    def clean_array(self, a):
        return np.nan_to_num(a, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # ---------- Tính (hoặc tái sử dụng) SHAP ----------
    def to_3d(self, sv):
        """Chuẩn hóa về (n_samples, n_features, n_classes). Nếu nhị phân 2D -> C=1."""
        if isinstance(sv, list):
            arrs = [s.values if hasattr(s, "values") else s for s in sv]
            return np.stack(arrs, axis=-1)  # (n, f, C)
        else:
            arr = sv.values if hasattr(sv, "values") else sv
            if arr.ndim == 3:     # (n, f, C)
                return arr
            elif arr.ndim == 2:   # (n, f)
                return arr[:, :, None]
            else:
                raise ValueError(f"SHAP array ndim không hỗ trợ: {arr.ndim}")

    
    def get_feature_importance(self, data_path, top_K = 5):
        ta_feature = ['volume_ma','volume_to_volume_ma_ratio','ema_12','ema_26','sma_20','sma_50','roc_5','%K','%R','cci','obv','macd','signal_line','macd_histogram','rsi','bb_bbm','bb_bbh','bb_bbl','bb_bbp','bb_bbh_bb_bbl_ratio','rsi_vnd','rsi_base_ma_vnd','rsi_rsi_base_ma_ratio_vnd','volume_ma_vnd','volume_to_volume_ma_ratio_vnd','bb_bbm_vnd','bb_bbh_vnd','bb_bbl_vnd','bb_bbp_vnd','bb_bbh_bb_bbl_ratio_vnd','roc_vnd','%K_vnd','%R_vnd','cci_vnd','obv_vnd','ema_12_vnd','ema_26_vnd','sma_20_vnd','sma_50_vnd']
        fa_feature = ['P/B_Previous_Quarter', 'P/B_change_rate','P/B_change_rate_flag','P/E_Previous_Quarter','P/E_change_rate','P/E_change_rate_flag','P/S_Previous_Quarter','P/S_change_rate','P/S_change_rate_flag','P/Cash Flow_Previous_Quarter','P/Cash Flow_change_rate','P/Cash Flow_change_rate_flag','EPS (VND)_Previous_Quarter','EPS (VND)_change_rate', 'EPS (VND)_change_rate_flag','BVPS (VND)_Previous_Quarter','BVPS (VND)_change_rate', 'BVPS (VND)_change_rate_flag','ROE (%)_Previous_Quarter','ROE (%)_change_rate','ROE (%)_change_rate_flag','ROA (%)_Previous_Quarter','ROA (%)_change_rate','ROA (%)_change_rate_flag','log_return','volatility_5d','volatility_10d','volatility_20d','volatility_30d','mean_log_return_5d','mean_log_return_10d','mean_log_return_20d','mean_log_return_30d','sharpe_like_5d','sharpe_like_10d','sharpe_like_20d','sharpe_like_30d','up_streak','pos_log_return_ratio_20d','z_score_5d','z_score_10d','z_score_20d','z_score_30d','annual_return','daily_return','sharpe_ratio','coefficient_P/B','coefficient_P/E','coefficient_P/S','coefficient_P/Cash Flow','coefficient_EPS (VND)','coefficient_BVPS (VND)','coefficient_ROE (%)','coefficient_ROA (%)','distance_to_nearest_quarter']
        sa_feature = ['Reputation', 'Financial', 'Regulatory', 'Risks','Fundamentals', 'Conditions', 'Market', 'Volatility']
        features = ta_feature + fa_feature + sa_feature
        data = pd.read_csv(data_path)
        X_data = data.drop(columns=['target'])
        X_data = X_data[features]
        
        X_test_all = X_data.iloc[[0]]
        print(X_test_all)
        scaler = joblib.load(self._scaler_path)
        X_test_all = scaler.transform(X_test_all)
        
        X_te = self.clean_array(X_test_all.values  if hasattr(X_test_all,  "values")  else X_test_all)
        features = list(features)
        assert X_te.shape[1] == len(features), "Số cột X_test_all không khớp 'features'."

        xgb_model = joblib.load(self._model_path)
        
        explainer = shap.TreeExplainer(xgb_model)

        sv_raw = explainer.shap_values(X_te, check_additivity=False)
            
        sv3 = self.to_3d(sv_raw)  # (n, f, C)
        
        # ---------- Importance: mean(|SHAP|) gộp các lớp ----------
        imp = np.mean(np.abs(sv3), axis=(0, 2))  # (n_features,)

        # ---------- Chọn Top-M ----------
        order = np.argsort(imp)[-top_K:][::-1]
        feat_names = [features[i] for i in order]
        vals = imp[order]
        
        df_feature_importance = pd.DataFrame([vals], columns=feat_names)
        
        df_feature_importance.to_csv(self._feature_importance_path, index=False)
        
        return data_path
    
    def visualize_feature_importance(self, data_path):
        feature_importance = pd.read_csv(self._feature_importance_path)
        print(feature_importance)
        feature_names = feature_importance.columns
        data = pd.read_csv(data_path)
        data = data.iloc[:65]
        data = data.iloc[::-1].reset_index(drop=True)
        data_feature = data[feature_names]
        data_time = data['time']
        print(data)
        
        feature_list = [f for f in feature_names if f not in ["Buy_Prob", "Hold_Prob", "Sell_Prob"]]
        n_features = len(feature_list)

        fig, axes = plt.subplots(n_features, 1, figsize=(14, 4*n_features))
        if n_features == 1:
            axes = [axes]

        with open(r"/app/model/feature_definitions.json", "r", encoding="utf-8") as f:
            info_feature = json.load(f)

        print(info_feature)
        
        bar_width = 0.5  # độ rộng cột xác suất

        for i, feature in enumerate(feature_list):
            ax = axes[i]
            
            f = next((f for f in info_feature if f["feature"] == feature), None)
            
            # Vẽ feature gốc trên trục y trái
            line_feature, = ax.plot(data_time, data_feature[feature], label=feature, color='black', alpha=1, linewidth=1.5)
            ax.set_ylabel(f"{f['unit']}", color='blue')
            ax.tick_params(axis='y', labelcolor='blue')

            # Trục y thứ 2 cho xác suất
            ax2 = ax.twinx()
            buy = data["Buy_Prob"].values
            hold = data["Hold_Prob"].values
            sell = data["Sell_Prob"].values

            bottom_hold = buy
            bottom_sell = buy + hold

            # Stacked bar
            bar_buy = ax2.bar(data_time, buy, width=bar_width, color="green", alpha=0.6, label="Buy")
            bar_hold = ax2.bar(data_time, hold, width=bar_width, bottom=bottom_hold, color="orange", alpha=0.6, label="Hold")
            bar_sell = ax2.bar(data_time, sell, width=bar_width, bottom=bottom_sell, color="red", alpha=0.6, label="Sell")

            ax2.set_ylim(0, 1)
            ax2.set_ylabel("Probability", color='black')
            ax2.tick_params(axis='y', labelcolor='black')

            # Gộp legend feature và xác suất
            lines = [line_feature] + [bar_buy, bar_hold, bar_sell]
            labels = [line_feature.get_label(), "Buy", "Hold", "Sell"]
            ax2.legend(lines, labels, loc='upper left', fontsize=8)

            ax.set_title(f"{f['full_name']} vs Predicted Probabilities", fontsize=14)

            # Thêm ngày cụ thể cho từng subplot
            step = max(1, len(data_time) // 15)  # hiển thị 10 ngày
            tick_positions = np.arange(0, len(data_time), step)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([data_time[i] for i in tick_positions], rotation=45)

        # Chỉ cần xlabel ở subplot cuối cùng
        axes[-1].set_xlabel("Time (Date)")

        plt.tight_layout()
        plt.savefig("/app/images/feature_importance.png", dpi=200, bbox_inches='tight')
        plt.show()
        
        
        
        list_feature_importance = []
        for i, feature in enumerate(feature_list):
            f = next((f for f in info_feature if f["feature"] == feature), None)
            list_feature_importance.append({
                "feature": feature,
                "full_name": f["full_name"],
                "meaning": f["meaning"],
            })
            
        shap_values_global = pd.read_csv(self._shap_values_global_path)

        list_feature_importance_global = []
        for i, feature in enumerate(shap_values_global.columns[:5]):
            f = next((f for f in info_feature if f["feature"] == feature), None)
            list_feature_importance_global.append({
                "feature": feature,
                "full_name": f["full_name"],
                "meaning": f["meaning"],
            })
        return {
            "list_feature_importance": list_feature_importance,
            "Buy_Prob": self._y_pred_probs_lstm[0][0],
            "Hold_Prob": self._y_pred_probs_lstm[0][1],
            "Sell_Prob": self._y_pred_probs_lstm[0][2],
            "data_raw": data.iloc[-1][feature_names],
            "shap_values": feature_importance,
            "shap_values_global": shap_values_global.iloc[0, :5],
            "list_feature_importance_global": list_feature_importance_global
        }
    
    def initiate_conversation(self, question, history):
        print(question)
        response = self._user_proxy.initiate_chat(
            self._ml_agent,
            message=f"Lịch sử trò chuyện: {history} \n Câu hỏi: {question}",
            clear_history=True
        )
 
        return response.chat_history[-1]['content'].replace("Đã hoàn tất.", "")