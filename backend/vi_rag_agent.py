from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from langchain_text_splitters import RecursiveCharacterTextSplitter
from autogen import UserProxyAgent, ConversableAgent, AssistantAgent
from pdf2image import convert_from_path
from pypdf import PdfReader, PdfWriter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from typing import Optional, Dict, Callable, Literal, Union
from litellm import completion
from unstructured.partition.pdf import partition_pdf
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared, operations
from unstructured_client.models.errors import SDKError
from unstructured.staging.base import dict_to_elements
from Element import Element
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from prompts import assistan_rag_prompt
# from google import genai
# from google.generativeai import types
# from tabulate import tabulate
from IPython.display import Markdown, display
# import google.generativeai as gg_genai
from tqdm import tqdm
import requests
import uuid
import torch
import os
import logging
import autogen

logging.basicConfig(level=logging.INFO)
logging.getLogger("langchain").setLevel(logging.DEBUG)


SYSTEM_PROMPT = """Bạn là một trợ lý chuyên trả lời câu hỏi. Hãy trả lời câu hỏi chỉ dựa trên bối cảnh được cung cấp.
Nếu câu hỏi không thể được trả lời bằng bối cảnh đó, chỉ cần nói “Tôi không biết”. Đừng bịa chuyện.

Câu hỏi của người dùng là: {question}

Bối cảnh là: {context}

{termination_msg}

"""
# Nguồn của bối cảnh là: {input_sources}

class VietnameseRAGAgent(UserProxyAgent):

    def __init__(self, name, human_input_mode, termination_msg, is_termination_msg, code_execution_config, param_rag: Optional[Dict] = None):
        super().__init__(
            name=name,
            code_execution_config=code_execution_config,
            human_input_mode=human_input_mode,
            is_termination_msg = is_termination_msg
        )
        self.termination_msg=termination_msg
        self._param_rag = {} if param_rag is None else param_rag
        self._input_path = self._param_rag.get("input_path", None)
        if not self._input_path:
            print("Chưa chỉ định input_path trong param_rag. Sử dụng lại collection")
        self._collection_name = self._param_rag.get("collection_name", "collection_1")
        self._model_name = self._param_rag.get("model_name", "BAAI/bge-small-en-v1.5")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embedding_model = SentenceTransformer(self._model_name, device=device) 
        self._model_embedding = embedding_model

        client = QdrantClient("http://qdrant:6333")
        self._client = client
        self._txt_files = []
        self._table_summaries = self._param_rag.get("table_summaries", [])
        self._table_content = []
        self._vectorstore = None
        self._untructured_api_key = self._param_rag.get("untructured_api_key", None)
        self._untructured_api_url = self._param_rag.get("untructured_api_url", "https://api.unstructured.io")
        self._va_api_key = self._param_rag.get("va_api_key", None)
        self._url_va = self._param_rag.get("url_va", "https://api.va.landing.ai/v1/tools/agentic-document-analysis")
        self._requests_per_second = self._param_rag.get("requests_per_second", 1/20)
        self._check_every_n_seconds = self._param_rag.get("check_every_n_seconds", 0.1)
        self._max_bucket_size = self._param_rag.get("max_bucket_size", 10)
        self._max_concurrency = self._param_rag.get("max_concurrency", 5)
        self._openai_api_key = self._param_rag.get("openai_api_key", None)
        self._create_collection = self._param_rag.get("create_collection", False)

        self._context_max_tokens = self._param_rag.get("context_max_tokens", 1500)
        self._top_k = self._param_rag.get("top_k", 5)

        self._customized_prompt = self._param_rag.get("customized_prompt", None)
        self._custom_token_count_function = lambda text, model: len(text.split())
        self._model = self._model_embedding
        self.initialize_index()
        
    def split_pdf(self, input_pdf, output_prefix, pages_per_split=1):
        # print("pdf_file =", input_pdf)
        # print("type(pdf_file) =", type(input_pdf))
        reader = PdfReader(input_pdf)
        total_pages = len(reader.pages)
        output_files = []

        for i in range(0, total_pages, pages_per_split):
            writer = PdfWriter()
            for j in range(i, min(i + pages_per_split, total_pages)):
                writer.add_page(reader.pages[j])

            output_filename = f"{output_prefix}_part_{i//pages_per_split + 1}.pdf"
            with open(output_filename, "wb") as output_pdf:
                writer.write(output_pdf)

            output_files.append(output_filename)
            # print(f"Đã tạo: {output_filename}")
        print("Đã tách file")

        return output_files
    
    def extract_infomation(self, pdf_file):
        # print("pdf_file =", pdf_file)
        # print("type(pdf_file) =", type(pdf_file))
        output_files = self.split_pdf(pdf_file, "split_pdf_output")

        entire_text = ""
        tables_content = []
        table_summaries = []
        text_content = []

        print(f"File {pdf_file} được chia thành {len(output_files)} files")
        for idx, file in enumerate(output_files):
            with open(file, "rb") as f:  # Mở file trong context manager
                files = {"pdf": f}
                headers = {"Authorization": "Basic {}".format(self._va_api_key)}
                response = requests.post(self._url_va, files=files, headers=headers)
            print(idx)
            print("Status code:", response.status_code)
            # print("Response text:", response.text)
            if response.status_code == 200:
                json_data = response.json()["data"]["chunks"]
                tables_content.extend([chunk["text"] for chunk in json_data if chunk["chunk_type"] == "table"])
                text_content.extend([chunk["text"] for chunk in json_data if chunk["chunk_type"] == "text"])
            else:
                # Trường hợp server trả về lỗi (4xx, 5xx, ...)
                print(f"Lỗi từ server: {response.status_code}, {response.text} tại file số {idx}")
        

        rate_limiter = InMemoryRateLimiter(
            requests_per_second=self._requests_per_second,
            check_every_n_seconds=self._check_every_n_seconds,
            max_bucket_size=self._max_bucket_size,
        )

        summary_chain = (
            {"doc": lambda x: x}
            | ChatPromptTemplate.from_template("""
                                            Tóm tắt bảng sau đây với đầy đủ các tiêu đề của bảng:\n\n{doc}
                                            Chỉ tóm tắt bảng chứ những thông tin về cái gì thôi không cần đưa ra số liệu và không cần đưa theo format của bảng, chỉ đưa format text
                                            """)
            | ChatOpenAI(openai_api_key= self._openai_api_key , max_retries=3, model="gpt-4o-mini", rate_limiter=rate_limiter)
            | StrOutputParser()
        )

        if self._table_summaries != []:
            print("Loading table summaries...")
            table_summaries = self._table_summaries
        else:
            print("No table summaries found.")
            table_summaries = summary_chain.batch(tables_content, {"max_concurrency": self._max_concurrency})


        entire_text = ""
        for i in text_content:
            entire_text += i
        for file in output_files:
            os.remove(file)
        return entire_text, table_summaries, tables_content

    def extract_text(self, pdf_file):
        reader = PdfReader(pdf_file)
        number_of_pages = len(reader.pages)
        entire_text = ""
        for page_num in range(number_of_pages):
            page = reader.pages[page_num]
            entire_text += page.extract_text() or ""
        return entire_text

    def split_text_into_chunks(self, entire_text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        text_chunks = text_splitter.split_text(entire_text)
        return text_chunks
    
    def embedding_chunks(self, text_chunks):
        if not text_chunks:
            return []
        embeddings = self._model_embedding.encode(text_chunks, show_progress_bar=True)

        embeddings = self._model_embedding.encode(text_chunks, show_progress_bar=True)
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()
        return embeddings
    
    def store_embeddings(self, embeddings, pdf_file, text_chunks, type = "text"):
        suffix = "_TEXT" if type == "text" else "_TABLE"

        # Sử dụng UUID thay vì chuỗi tùy chỉnh làm ID
        ids = [str(uuid.uuid4()) for _ in range(len(text_chunks))]

        payload = [{"source": pdf_file, "content": text, "type": suffix} for text in text_chunks]
        self._client.upload_collection(
            collection_name=self._collection_name,
            vectors=embeddings,
            payload=payload,
            ids=ids,
            batch_size=256,
        )
        count_result = self._client.get_collection(self._collection_name)
        print("Số lượng vector trong collection:", count_result.vectors_count)


    def search(self, question: str):
        query_embedding = self._model_embedding.encode(question).tolist()
        search_result = self._client.search(
            collection_name=self._collection_name,
            query_vector=query_embedding,
            query_filter=None,  
            limit=self._top_k
        )
        print("Kết quả search:")
        for result in search_result:
            print(result)
        return search_result
    
    # def response_generation(self, question):
    #     print(f"\nGenerating response for: {question}") 
    #     results = self.search(question)
    #     user_prompt = """
    #     Question: {question}

    #     Answer:"""
    #     references = [obj.payload["content"] for obj in results]
    #     context_text = "\n\n".join(references)
    #     response = completion(
    #         api_key=self._openai_api_key,
    #         model="gpt-4o-mini",
            # messages=[
            #     {"content": SYSTEM_PROMPT.format(question=question, context=context_text), "role": "system"},
            #     {"content": user_prompt.format(question=question), "role": "user"}
            # ]
    #     )
    #     return response

    def process_pdf_file(self, pdf_file: str):
        print(f"Đang xử lý file: {pdf_file}", flush=True)
        txt_content, table_summaries, tables_content = self.extract_infomation(pdf_file)
        
        if not txt_content and not table_summaries and not tables_content:
            print(f"Không trích xuất được nội dung từ {pdf_file}", flush=True)
            return
        
        text_chunks = self.split_text_into_chunks(txt_content)
        pdf_file_identifier = os.path.join(
            os.path.basename(os.path.dirname(pdf_file)),
            os.path.basename(pdf_file)
        )
        
        embeddings_text = self.embedding_chunks(text_chunks)
        self.store_embeddings(embeddings_text, f"{pdf_file_identifier}_TEXT", text_chunks)
        
        if table_summaries and tables_content:
            if isinstance(table_summaries, str):
                table_summaries = [table_summaries]
            embeddings_table = self.embedding_chunks(table_summaries)
            self.store_embeddings(embeddings_table, f"{pdf_file_identifier}_TABLE", tables_content, type='table')
        print(f"Đã lưu vào vector db thành công file: {pdf_file}")
    
    def initialize_index(self):
        if self._create_collection == True:
            print("Tạo collection mới",  flush=True)
            try:
                self._client.delete_collection(self._collection_name)
            except Exception as e:
                print(f"Không thể xóa collection cũ: {e}",  flush=True)
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
        else:
            print("Không tạo collection mới, thêm vào collection cũ",  flush=True)
        
        if self._input_path:
            if os.path.isdir(self._input_path):
                for stock_codes in os.listdir(self._input_path):
                    print(f"Đang xử lý mã cổ phiếu {stock_codes}", flush=True)
                    stock_path = os.path.join(self._input_path, stock_codes)
                    if os.path.isdir(stock_path):
                        pdf_files = [os.path.join(stock_path, f) for f in os.listdir(stock_path) if f.endswith('.pdf')]
                        print(f"Đang xử lý {pdf_files} tổng số file: {len(pdf_files)}", flush=True)
                        if not pdf_files:
                            raise ValueError(f"Không tìm thấy file PDF nào trong thư mục {stock_path}.")
                        for pdf_file in pdf_files:
                            self.process_pdf_file(pdf_file)
                        if not self._txt_files:
                            raise ValueError("Không thể trích xuất được văn bản từ các file PDF trong thư mục.")
                    else:
                        print(f"{stock_path} không phải là thư mục, bỏ qua.", flush=True)
            elif os.path.isfile(self._input_path) and self._input_path.endswith('.pdf'):
                self.process_pdf_file(self._input_path)
            else:
                raise ValueError("Đường dẫn input_path không hợp lệ. Nó phải là file PDF hoặc thư mục chứa PDF.")

    @staticmethod
    def is_termination_msg(content):
        have_content = content.get("content", None) is not None
        if have_content and "Đã hoàn tất." in content["content"]:
            return True
        return False