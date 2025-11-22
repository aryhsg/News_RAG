import httpx
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from typing import List, Dict, Any
import os
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError
from postgrest.exceptions import APIError as PostgrestAPIError # 捕獲 Supabase 錯誤
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import time
from datetime import datetime, timezone, timedelta


# =======================================================================================
load_dotenv()
# 1. 定義 UTC+8 時區 # 台灣時間是 UTC+8
tz_taipei = timezone(timedelta(hours=8))

# 2. 取得現在時間並指定時區
taipei_date = datetime.now(tz=tz_taipei).date()

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# =======================================================================================
def split_content(news):
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "。", "?", "!", " ", "", "."],
        length_function=len,
        is_separator_regex=False
        )

    chunks = text_spliter.create_documents(
        texts=[news[0]["content"]],
        metadatas=[news[1]]
    )
    return chunks

# =======================================================================================
def transform_datastruct_and_split(news_data):
    news_urls = news_data["data"]["url"]
    news_titles = news_data["data"]["title"]
    news_contents = news_data["data"]["content"]
    news_list = []
    for i in range(len(news_urls)):
        # news = [news:dict, metadata:dict]
        news = [
            {
            "content": news_contents[i]
            },
            {
            "date": taipei_date,
            "url": news_urls[i],
            "title": news_titles[i]
            }
                ]
        news_chunks = split_content(news) # news_chunks = [[page_content,metadata], [page_content,metadata], [page_content,metadata], ...]
        news = {"content": news_chunks[0].page_content, "metadata": news_chunks[1].metadata}
        #news_list.append(news_chunks) # new_list = [news_chunks, news_chunks, news_chunks, ...]
        #all_news_chunks = sum(news_chunks, []) # 所有news_chunks集合成一個列表 [[page_content,metadata], [page_content,metadata], [page_content,metadata], ...]
        news_list.append(news)
    return news_list

# =======================================================================================
class SupabaseUploader:
    """
    負責初始化 Gemini 客戶端和 Supabase 客戶端，
    並執行批量 Embedding 轉換和資料庫寫入的類別。
    """
    def __init__(self, embedding_model: str = 'text-embedding-004'):
        """初始化客戶端並檢查環境變數。"""
        
        # 1. 初始化 Supabase 客戶端
        url: str = os.environ.get("SUPABASE_URL")
        # 假設您的 .env 檔案中的密鑰是叫 SUPABASE_KEY 或 SUPABASE_SERVICE_KEY
        # 請根據您的實際 .env 變數名稱修改下面這行：
        key: str = os.environ.get("SUPABASE_KEY") or os.environ.get("password") 
        
        if not url or not key:
            raise ValueError(
                "Supabase 環境變數 (URL 或 Key) 讀取失敗。請檢查 .env 檔案。"
            )
            
        self.supabase: Client = create_client(url, key)
        self.embedding_model = embedding_model
        self.gemini_client: Optional[genai.Client] = None

        # 2. 初始化 Gemini 客戶端
        try:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                 raise ValueError("GEMINI_API_KEY 變數未設定。")

            self.gemini_client = genai.Client(api_key=api_key)
            print("🟢 Gemini 和 Supabase 客戶端初始化成功。")

        except Exception as e:
            print(f"🔴 錯誤：Gemini 客戶端初始化失敗: {e}")
            self.gemini_client = None


    def _transform_to_vector(self, contents: List[str]) -> List[List[float]]:
        """內部方法：批量呼叫 Gemini API 轉換文本為向量。"""
        if not self.gemini_client:
            return []
            
        print(f"-> 正在呼叫 Gemini API 轉換 {len(contents)} 篇文本...")
        
        try:
            response = self.gemini_client.models.embed_content(
                model=self.embedding_model,
                contents=contents
            )
        except APIError as e:
            print(f"🔴 錯誤：Gemini API 呼叫失敗: {e}")
            return []
        
        # 提取 'values' 列表並回傳
        # 確保回傳的是 List[List[float]]
        vectors: List[List[float]] = [
            e.values for e in response.embeddings if hasattr(e, 'values')
        ]
        
        if len(vectors) != len(contents):
             print(f"⚠️ 警告: 向量數量 ({len(vectors)}) 與輸入文本數量 ({len(contents)}) 不符。")

        print(f"-> 成功轉換 {len(vectors)} 個向量。")
        return vectors

    
    def upload_data(self, news_data: List[list[str, dict]], table_name: str = "news") -> None: # (self, news_data: List[Dict[str, str]], table_name: str = "news")
        """
        主方法：執行整個流程，將新聞資料轉換為向量並插入資料庫。

        參數:
        - news_data: 外部匯入的原始新聞資料列表 [{title, content, url}, ...]
        - table_name: 要寫入的 Supabase 表格名稱 (預設為 'news')
        """
        if not self.gemini_client:
            print("🔴 無法執行上傳，Gemini 客戶端未初始化。")
            return

        # 1. 準備輸入文本列表 (Contents for Embedding)
        contents = [news["content"] for news in news_data]

        # 2. 轉換所有文本為向量
        vectors_list = self._transform_to_vector(contents)
        
        if not vectors_list:
            print("🔴 向量轉換失敗或回傳為空，停止寫入資料庫。")
            return

        # 3. 準備最終插入資料庫的行列表
        insert_rows: List[Dict[str, Any]] = []
        
        for i, news in enumerate(news_data):
            # 確保有對應的向量
            if i < len(vectors_list):
                # 建立要插入的單行資料字典
                insert_row = {
                    "url": news["metadata"]["url"],
                    "content": news["content"],
                    "metadata": news["metadata"],
                    "embedding": vectors_list[i] # 插入 List[float]
                }
                insert_rows.append(insert_row)
            
        # 4. 批量插入 Supabase
        print(f"-> 嘗試批量插入 {len(insert_rows)} 筆資料到表格 '{table_name}'...")
        try:
            supa_response = (
                self.supabase.table(table_name)
                .upsert(insert_rows, on_conflict="url")
                .execute()
            )
            # Supabase SDK 回傳的 response 是一個 PostgrestAPIResponse 物件
            print(f"🟢 資料庫寫入成功！")
            
        except PostgrestAPIError as e:
            # 捕獲常見的 Postgrest 錯誤，例如主鍵衝突 (url unique 限制)
            print(f"🔴 資料庫寫入失敗 (Postgrest Error): {e}")
        except Exception as e:
            print(f"🔴 資料庫寫入失敗 (未知錯誤): {e}")

# =======================================================================================
# Buildpacks 會查找這個 'app' 物件
app = FastAPI()
BASE_PATH = "https://aryhsgsnewsapi.onrender.com/api/scrape-specific-news/?category="
supa_client = SupabaseUploader()
category_list = ["金融","理財","期貨","證券","產業","國際"]

@app.get("/", status_code=200, summary="爬取並上傳財經相關類別的新聞數據")
async def curl_news_and_upload():
    """
    接收 Web 請求後，異步爬取所有指定類別的新聞數據，
    將數據轉換後，上傳到 Supabase 的 'news' 表格中。
    """
    response_list: List[List[Dict[str, Any]]] = []

    async with httpx.AsyncClient(timeout=100.0) as client:
        try:
            for cate in category_list:
                print(f"正在爬取{cate}類文章...")
                url = BASE_PATH+cate
                response = await client.get(url)
                response.raise_for_status()

                transformed_data = transform_datastruct_and_split(response.json())
                response_list.append(transformed_data)
                print(f"{cate}類文章爬取完成!!!")

            #all_response = response_list[0]+response_list[1]+response_list[2]+response_list[3]+response_list[4]+response_list[5]
            all_response = sum(response_list, [])
        
            await run_in_threadpool(supa_client.upload_data,news_data=all_response, table_name="news")
            return {
                "message": "新聞數據爬取、向量轉換和上傳成功",
                "total_records_processed": len(all_response)
            }
            
        except ValueError as e:
            print(f"🔴 致命錯誤：環境設定問題: {e}")

        except Exception as e:
            print(f"🔴 運行時發生未預期錯誤: {e}")

        except IndexError as e:
            print(f"🔴 爬取新聞時發生未預期錯誤: {e}")

        except httpx.HTTPStatusError as e:
            error_message = f"🔴 外部 API 請求失敗 (狀態碼: {e.response.status_code}): {e.response.text}"
            print(error_message)
            return {"error": "HTTP 請求失敗", "details": error_message}




    

    