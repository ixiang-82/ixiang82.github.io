# app.py
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import os
import logging

from search.ai_search import load_data, ai_search_and_rank

# 載入 .env
load_dotenv()

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tire-search")

app = FastAPI()

# 靜態檔案與模板
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 讀取 API Key
API_KEY = os.getenv("OPENAI_API_KEY")
if API_KEY:
    logger.info("✅ OpenAI API Key 已載入")
else:
    logger.warning("❌ OpenAI API Key 未載入，請檢查 .env")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """
    首頁：顯示搜尋表單（frontend templates 負責排版）
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/search", response_class=HTMLResponse)
def search(request: Request, query: str = Form(...)):
    """
    搜尋 endpoint：
      - 讀取本地資料
      - 以 ai_search_and_rank 做本地過濾 + AI 排序
      - 顯示最多 20 筆結果
    """
    # 防呆：API Key 必要時提示（仍允許本地 fallback）
    if not API_KEY:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "OpenAI API Key 未設定，請檢查 .env"}
        )

    # 讀取資料（data/tires.json 必須為 dict，含 keys: tires, keyword_mapping）
    try:
        data = load_data()
        tires = data["tires"]
        keyword_mapping = data["keyword_mapping"]
    except FileNotFoundError:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "載入輪胎資料失敗：data/tires.json 不存在"}
        )
    except Exception as e:
        logger.exception("讀取資料失敗")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": f"讀取輪胎資料失敗：{str(e)}"}
        )

    # 呼叫搜尋與排序（包含品牌偵測邏輯），限制回傳上限
    try:
        results = ai_search_and_rank(
            query=query,
            tires=tires,
            keyword_mapping=keyword_mapping,
            api_key=API_KEY,
            max_results=20
        )
        # 確保 results 是 list（若不是，轉為空列表）
        if not isinstance(results, list):
            logger.warning("AI 回傳格式非 list，使用保底 candidates")
            results = []

        # 若 AI 回傳的項目裡沒有 brand 欄位，要容錯處理也可以在 template 顯示預設值
    except Exception as e:
        logger.exception("AI 搜尋失敗")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": f"AI 搜尋失敗：{str(e)}"}
        )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "results": results,
            "query": query
        }
    )


if __name__ == "__main__":
    import uvicorn
    # 使用 127.0.0.1 可避免某些環境下 0.0.0.0 導致瀏覽器 ERR_ADDRESS_INVALID
    uvicorn.run(app, host="127.0.0.1", port=5000, reload=True)
