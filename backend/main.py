# backend/main.py
from autogen import UserProxyAgent, config_list_from_json, AssistantAgent
from text_to_sql_agent import TextToSQLAgent
from manager_agent import ManagerAgent
from rag_system import Rag_System
from visualize_sql_data_system import VisualizeSQLDataSystem
from scenario_analysis_agent import ScenarioAnalysisAgent
import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
from chat_agent import ChatAgent
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import threading
import time
from ml_system import ML_System
from config import load_config_with_keys
import re



CONFIG_MANAGER = r"./OAI_CONFIG_MANAGER"
CONFIG_RAG = r"./OAI_CONFIG_RAG"
CONFIG_CHAT = r"./OAI_CONFIG_CHAT"
CONFIG_GPT = "./OAI_CONFIG_GPT"
CONFIG_SCENARIO_ANALYSIS = "./OAI_CONFIG_SCENARIO_ANALYSIS"
config_list_manager = load_config_with_keys(CONFIG_MANAGER)
config_list_rag = load_config_with_keys(CONFIG_RAG)
config_list_chat = load_config_with_keys(CONFIG_CHAT)
config_list_gpt = load_config_with_keys(CONFIG_GPT)
config_item = config_list_gpt[0]
config_item1 = config_list_gpt[1]
config_item_scenario_analysis = config_list_gpt[2]
COLLECTION_NAME = "financial-reports"


print("Done ml system!")

rag_system = Rag_System(
    param_rag={
        # "input_path": "./BaoCaoTaiChinh",
        "collection_name": COLLECTION_NAME,
        "create_collection": False,
        "va_api_key": "NGRiNWxtcWNsYXhnYTdjNnZsbWkwOkRudkJpeWR5cHF6S0hMejBqZlU5OE51eklRSjBzU3ZU",
        "untructured_api_key": "Sqs4EVz4ediU6vlgrRVBAnoatDdsav",
        "openai_api_key": config_list_rag[0]['api_key'],
    },
    config_item=config_list_rag,
    termination_msg="Sau khi trả lời, vui lòng phản hồi 'Đã hoàn tất'.",
)

sql_agent = TextToSQLAgent(
    name="TextToSQLAgent",
    config_item=config_item,
    config_openai=config_item1,
    param_text2sql={
        "create_database": False,
    },
    termination_msg="Sau khi trả lời, vui lòng phản hồi 'Đã hoàn tất'.",
    code_execution_config={"work_dir": "coding", "use_docker": False}
)

ml_system = ML_System(
    config_item=config_item,
    name="ml_system",
    human_input_mode="NEVER",
    termination_msg="Sau khi trả lời, vui lòng phản hồi 'Đã hoàn tất'.",
    code_execution_config={
        "work_dir": "coding", 
        "use_docker": False
    }
)

# ml_system.excute_code()

chat_agent = ChatAgent(
    config_item=config_list_chat,
    termination_msg="Sau khi trả lời, vui lòng phản hồi 'Đã hoàn tất'.",
)

visualize_data_agent = VisualizeSQLDataSystem(
    # name="VisualizeDataAgent",
    config_item=config_item,
    termination_msg="Sau khi trả lời, vui lòng phản hồi 'Đã hoàn tất'.",
    # config_openai=config_item1,
    # code_execution_config={"work_dir": "coding", "use_docker": False}
)

scenario_analysis_agent = ScenarioAnalysisAgent(
    name="ScenarioAnalysisAgent",
    human_input_mode="NEVER",
    config_item=config_item_scenario_analysis,
    termination_msg="Sau khi trả lời, vui lòng phản hồi 'Đã hoàn tất'.",
    code_execution_config={"work_dir": "coding", "use_docker": False}
)

manager = ManagerAgent(
    name="ManagerAgent",
    config_agent={
        "openai_api_key": config_list_manager[0]['api_key'],
        "llms_agent": chat_agent,          
        "text2sql_agent": sql_agent,               
        "visualize_data": visualize_data_agent,        
        "rag_agent": rag_system,
        "ml_agent": ml_system,
        "scenario_analysis_agent": scenario_analysis_agent,
    },
    human_input_mode="NEVER",
    config_item=config_list_manager,
    termination_msg="Quan trọng: sau khi trả lời, vui lòng phản hồi 'Đã hoàn tất.'",
    code_execution_config={
        "work_dir": "coding", 
        "use_docker": False
    }
)

user_proxy = UserProxyAgent(
    "user_proxy", 
    human_input_mode="ALWAYS",
    code_execution_config={"work_dir": "coding", "use_docker": False}
)
app = FastAPI()
app.mount("/images", StaticFiles(directory="/app/images"), name="images")

class AskRequest(BaseModel):
    question: str
    history: list

@app.get("/images/{filename}")
def get_image(filename: str):
    file_path = Path("/app/images") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path), media_type="image/png")

@app.post("/ask")
def ask_question(payload: AskRequest):
    question = payload.question
    history = payload.history

    formatted_history = [f"{msg['role']}: {msg['content']}" for msg in history[-3:]]
    formatted_history = ", ".join(formatted_history)

    if not question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    
    response = manager.process_question(question, formatted_history)

    # Tìm tất cả các link ảnh
    image_paths = re.findall(r"/app/images/[^ ]+\.png", response)
    public_links = []
    if image_paths:
        for img_path in image_paths:
            filename = img_path.split("/app/images/")[1]
            new_link = f"http://localhost:8000/images/{filename}"
            response = response.replace(img_path, new_link)
            public_links.append(new_link)

    return {
        "answer": response,
        "is_image": bool(public_links),
        "images": []
    }

@app.get("/")
def read_root():
    return {"message": "Hello from RAG-based FastAPI backend!"}