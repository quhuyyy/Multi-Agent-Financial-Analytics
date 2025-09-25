from autogen import ConversableAgent, UserProxyAgent, AssistantAgent
from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from prompts import scenario_analysis_agent_prompt, admin_scenario_analysis_prompt, gpt_turbo_config, gpt_turbo_config_scenario_analysis, query_maker_prompt_scenario_analysis, generate_code_python_promt
from typing import Optional, Dict, Union, Sequence
from itertools import product
import connectDB
from typing import List
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
import matplotlib as mpl
from matplotlib.patches import Patch


class ScenarioAnalysisAgent(ConversableAgent):
    def __init__(self, config_item, name, human_input_mode, termination_msg, code_execution_config, param_scenario_analysis: Optional[Dict] = None):
        super().__init__(
            name=name,
            code_execution_config=code_execution_config,
            human_input_mode=human_input_mode,
        )
        self._config_item = config_item
        self._param_scenario_analysis = {} if param_scenario_analysis is None else param_scenario_analysis
        self._model_path = r"/app/model/random_forest_model.pkl"
        self._connection = connectDB.get_Connection()
        self._df_list = []
        self._y_pred_probs_lstm = None
        self._scaler_path = r"/app/model/scaler.pkl"
        self._feature_importance_path = r"/app/data/feature_importance.csv"
        ta_feature = ['volume_ma','volume_to_volume_ma_ratio','ema_12','ema_26','sma_20','sma_50','roc_5','%K','%R','cci','obv','macd','signal_line','macd_histogram','rsi','bb_bbm','bb_bbh','bb_bbl','bb_bbp','bb_bbh_bb_bbl_ratio','rsi_vnd','rsi_base_ma_vnd','rsi_rsi_base_ma_ratio_vnd','volume_ma_vnd','volume_to_volume_ma_ratio_vnd','bb_bbm_vnd','bb_bbh_vnd','bb_bbl_vnd','bb_bbp_vnd','bb_bbh_bb_bbl_ratio_vnd','roc_vnd','%K_vnd','%R_vnd','cci_vnd','obv_vnd','ema_12_vnd','ema_26_vnd','sma_20_vnd','sma_50_vnd']
        fa_feature = ['P/B_Previous_Quarter', 'P/B_change_rate','P/B_change_rate_flag','P/E_Previous_Quarter','P/E_change_rate','P/E_change_rate_flag','P/S_Previous_Quarter','P/S_change_rate','P/S_change_rate_flag','P/Cash Flow_Previous_Quarter','P/Cash Flow_change_rate','P/Cash Flow_change_rate_flag','EPS (VND)_Previous_Quarter','EPS (VND)_change_rate', 'EPS (VND)_change_rate_flag','BVPS (VND)_Previous_Quarter','BVPS (VND)_change_rate', 'BVPS (VND)_change_rate_flag','ROE (%)_Previous_Quarter','ROE (%)_change_rate','ROE (%)_change_rate_flag','ROA (%)_Previous_Quarter','ROA (%)_change_rate','ROA (%)_change_rate_flag','log_return','volatility_5d','volatility_10d','volatility_20d','volatility_30d','mean_log_return_5d','mean_log_return_10d','mean_log_return_20d','mean_log_return_30d','sharpe_like_5d','sharpe_like_10d','sharpe_like_20d','sharpe_like_30d','up_streak','pos_log_return_ratio_20d','z_score_5d','z_score_10d','z_score_20d','z_score_30d','annual_return','daily_return','sharpe_ratio','coefficient_P/B','coefficient_P/E','coefficient_P/S','coefficient_P/Cash Flow','coefficient_EPS (VND)','coefficient_BVPS (VND)','coefficient_ROE (%)','coefficient_ROA (%)','distance_to_nearest_quarter']
        sa_feature = ['Reputation', 'Financial', 'Regulatory', 'Risks','Fundamentals', 'Conditions', 'Market', 'Volatility']
        self._features = ta_feature + fa_feature + sa_feature
        
        self._rate_limiter = InMemoryRateLimiter(
            requests_per_second=1/20,
            check_every_n_seconds=0.1,
            max_bucket_size=10,
        )
        
        self._user_proxy = UserProxyAgent(
            name="Admin",
            human_input_mode="NEVER",
            system_message= admin_scenario_analysis_prompt + termination_msg,
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
        
        self._scenario_analysis_agent = AssistantAgent(
            name="ScenarioAnalysisAgent",
            llm_config=gpt_turbo_config_scenario_analysis,
            system_message=scenario_analysis_agent_prompt + termination_msg,
            function_map={
                "query_maker_scenario_analysis": self.query_maker_scenario_analysis,
                "run_sql_query_scenario_analysis": self.run_sql_query_scenario_analysis,
                "scenario_analysis": self.scenario_analysis,
                "visualize_scenario_analysis": self.visualize_scenario_analysis,
            }
        )
        
        self._user_proxy.register_function(function_map={
            "query_maker_scenario_analysis": self.query_maker_scenario_analysis,
            "run_sql_query_scenario_analysis": self.run_sql_query_scenario_analysis,
            "scenario_analysis": self.scenario_analysis,
            "visualize_scenario_analysis": self.visualize_scenario_analysis,
        })
        
    @staticmethod
    def is_termination_msg(content):
        have_content = content.get("content", None) is not None
        if have_content and "Đã hoàn tất." in content["content"]:
            return True
        return False
    
    def query_maker_scenario_analysis(self, user_input):
        prompt_template = PromptTemplate.from_template(
            "{system_prompt} + '\n' +  {user_input}."
        )
        chain = RunnableSequence(prompt_template | self._openaiLLM)
        query = chain.invoke({"system_prompt": query_maker_prompt_scenario_analysis, "user_input": user_input}).content.split("---")
        print(query)
        return query[0], query[1]

    def run_sql_query_scenario_analysis(self, sql_query_1, sql_query_2):
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
        feature = df_2.columns
        df_1.to_csv(data_path, index=False)
        
        return {
            "data_path": data_path,
            "feature": feature,
            "current_value": df_2
        }

    def scenario_analysis(self,
                      data_path: str,
                      list_feature: Union[str, Sequence[str]],
                      list_min_value: Union[float, Sequence[float], str],
                      list_max_value: Union[float, Sequence[float], str],
                      bins: Union[int, Sequence[int], str] = 10,
                      out_path: str = None):

        def _split(s):
            return [x.strip() for x in s.split(",") if x.strip()]

        if isinstance(list_feature, str):
            features: List[str] = _split(list_feature)
        else:
            features = list(list_feature)

        if isinstance(list_min_value, str):
            mins = [float(x) for x in _split(list_min_value)]
        elif not isinstance(list_min_value, (list, tuple, np.ndarray)):
            mins = [float(list_min_value)] * len(features)
        else:
            mins = list(map(float, list_min_value))

        if isinstance(list_max_value, str):
            maxs = [float(x) for x in _split(list_max_value)]
        elif not isinstance(list_max_value, (list, tuple, np.ndarray)):
            maxs = [float(list_max_value)] * len(features)
        else:
            maxs = list(map(float, list_max_value))

        if isinstance(bins, str):
            bins_list = [int(x) for x in _split(bins)]
        elif isinstance(bins, int):
            bins_list = [bins] * len(features)
        else:
            bins_list = list(bins)

        assert len(features) == len(mins) == len(maxs), "features/mins/maxs must have same length."
        if len(bins_list) != len(features):
            assert len(bins_list) == 1, "Length of bins list must match features."
            bins_list = bins_list * len(features)

        df = pd.read_csv(data_path)
        base_row = df.iloc[0].copy()

        values_per_feature = [np.linspace(mn, mx, b) for mn, mx, b in zip(mins, maxs, bins_list)]
        combos = list(product(*values_per_feature))

        df_new = pd.DataFrame([base_row] * len(combos))
        for col_idx, feat in enumerate(features):
            df_new[feat] = [c[col_idx] for c in combos]

        model = joblib.load(self._model_path)
        scaler = joblib.load(self._scaler_path)

        X_test = df_new[self._features]
        X_test_all = scaler.transform(X_test)

        y_pred_probs = model.predict_proba(X_test_all)

        df_new["Buy_Prob"]  = y_pred_probs[:, 0]
        df_new["Hold_Prob"] = y_pred_probs[:, 1]
        df_new["Sell_Prob"] = y_pred_probs[:, 2]

        keep_cols = features + ["Buy_Prob", "Hold_Prob", "Sell_Prob"]
        df_new = df_new[keep_cols]

        if out_path is None:
            if data_path.lower().endswith(".csv"):
                out_path = data_path[:-4] + "_scenario.csv"
            else:
                out_path = data_path + "_scenario.csv"

        df_new.to_csv(out_path, index=False)

        return {
            "data_scenario": df_new,
            "data_path": out_path,
            "list_feature": features,
        }


        
    def visualize_scenario_analysis(self, data_path: str, list_feature: str, out_path: str = "/app/images/plot.png"):
        df = pd.read_csv(data_path)
        feats = [f.strip() for f in str(list_feature).split(",") if f.strip()]
        if len(feats) == 1:
            xcol = feats[0]
            cols = ["Buy_Prob", "Hold_Prob", "Sell_Prob"]
            dfp = df.sort_values(by=xcol)
            plt.figure(figsize=(10, 6))
            plt.plot(dfp[xcol], dfp["Buy_Prob"], label="Buy Probability")
            plt.plot(dfp[xcol], dfp["Hold_Prob"], label="Hold Probability")
            plt.plot(dfp[xcol], dfp["Sell_Prob"], label="Sell Probability")
            plt.xlabel(xcol)
            plt.ylabel("Predicted Probability")
            plt.title("Predicted Probabilities by Feature")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
            return {"image_path": out_path, "error": None}

        xcol, ycol = feats[:2]
        X = df[[xcol, ycol]].to_numpy()

        if {"Buy_Prob", "Hold_Prob", "Sell_Prob"}.issubset(df.columns):
            Y = df[["Buy_Prob", "Hold_Prob", "Sell_Prob"]].to_numpy()
        else:
            if "pred_class" not in df.columns:
                return {"image_path": None, "error": "Need either probability columns or pred_class"}
            lab2idx = {"Buy": 0, "Hold": 1, "Sell": 2}
            y_raw = df["pred_class"].astype(str).map(lab2idx).fillna(1).astype(int).to_numpy()
            Y = np.zeros((len(y_raw), 3), dtype=float)
            Y[np.arange(len(y_raw)), y_raw] = 1.0

        lr = LinearRegression()
        lr.fit(X, Y)  

        pad_x = (X[:, 0].max() - X[:, 0].min()) / 10 or 1.0
        pad_y = (X[:, 1].max() - X[:, 1].min()) / 10 or 1.0
        x_min, x_max = X[:, 0].min() - pad_x, X[:, 0].max() + pad_x
        y_min, y_max = X[:, 1].min() - pad_y, X[:, 1].max() + pad_y
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                            np.linspace(y_min, y_max, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = lr.predict(grid).reshape(xx.shape[0], xx.shape[1], 3)

        Z = np.clip(Z, 0.0, 1.0)
        Z_buy, Z_hold, Z_sell = Z[..., 0], Z[..., 1], Z[..., 2]
        
        def curve_surface(Z, power=0.7, scale=0.2):
            Z_new = Z.copy()
           
            mask = Z > 0.05  
            Z_new[mask] = Z[mask] + scale * np.power(Z[mask], power)
            
            return np.clip(Z_new, 0.0, 1.0)
        
        
        Z_buy_curved = curve_surface(Z_buy, power=0.7, scale=0.2)
        Z_hold_curved = curve_surface(Z_hold, power=0.7, scale=0.2)
        Z_sell_curved = curve_surface(Z_sell, power=0.7, scale=0.2)
        
        
        cls_map = np.argmax(Z, axis=2)  
        
        
        

        
        fig = plt.figure(figsize=(10, 6), dpi=120)  
        
        ax3d = fig.add_subplot(111, projection="3d")
        
        
        buy_col  = "#2c7b2c"   # xanh sáng
        hold_col = "#f8b147"   # vàng sáng
        sell_col = "#d73838"   # đỏ sáng
        

        s1 = ax3d.plot_surface(xx, yy, Z_buy_curved,  color=buy_col,  alpha=0.65,
                            linewidth=0, antialiased=True, shade=False, rstride=5, cstride=5)
        s2 = ax3d.plot_surface(xx, yy, Z_hold_curved, color=hold_col, alpha=0.65,
                            linewidth=0, antialiased=True, shade=False, rstride=5, cstride=5)
        s3 = ax3d.plot_surface(xx, yy, Z_sell_curved, color=sell_col, alpha=0.65,
                            linewidth=0, antialiased=True, shade=False, rstride=5, cstride=5)

        
        ax3d.xaxis._axinfo["grid"]['linewidth'] = 0.6
        ax3d.yaxis._axinfo["grid"]['linewidth'] = 0.6
        ax3d.zaxis._axinfo["grid"]['linewidth'] = 0.6

        ax3d.set_title("3D Probability Surfaces", fontsize=12, fontweight='bold')
        ax3d.set_xlabel(xcol, fontweight='bold', labelpad=10)
        ax3d.set_ylabel(ycol, fontweight='bold', labelpad=10)
        ax3d.set_zlabel("Probability", fontweight='bold', labelpad=10)
        ax3d.set_zlim(0.0, 1.0)  
        
        ax3d.view_init(elev=30, azim=-45)  
        ax3d.zaxis.set_major_locator(MaxNLocator(5))
        
        
        ax3d.grid(True, alpha=0.3, linestyle='--')
        
        ax3d.set_box_aspect((1, 1, 0.4))  
        
        xx_flat, yy_flat = np.meshgrid(
            np.linspace(x_min, x_max, 20), 
            np.linspace(y_min, y_max, 20)
        )

        if {"Buy_Prob", "Hold_Prob", "Sell_Prob"}.issubset(df.columns):
            y_pts = np.argmax(df[["Buy_Prob", "Hold_Prob", "Sell_Prob"]].to_numpy(), axis=1)
        else:
            y_pts = np.argmax(Y, axis=1)
        

        legend_elements = [
            Patch(facecolor=buy_col,  edgecolor='none', alpha=0.65, label='Buy'),
            Patch(facecolor=hold_col, edgecolor='none', alpha=0.65, label='Hold'),
            Patch(facecolor=sell_col, edgecolor='none', alpha=0.65, label='Sell'),
        ]
        ax3d.legend(handles=legend_elements, loc="lower right",
                    frameon=True, framealpha=0.95, fontsize=11,
                    facecolor='white', edgecolor='gray', shadow=True)

        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.show()
        plt.savefig("/app/images/plot_2_feature.png", dpi=150)
        plt.close()
        return {"image_path": "/app/images/plot_2_feature.png", "error": None}    
    
    def initiate_conversation(self, question, history):
        print(question)
        response = self._user_proxy.initiate_chat(
            self._scenario_analysis_agent,
            message=f"Lịch sử trò chuyện: {history} \n Câu hỏi: {question}",
            clear_history=True
        )
 
        return response.chat_history[-1]['content'].replace("Đã hoàn tất.", "")