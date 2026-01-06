# search/ai_search.py
import json
import re
from openai import OpenAI

DATA_PATH = "data/tires.json"


# ---------- 讀資料 ----------
def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "tires" not in data or "keyword_mapping" not in data:
        raise ValueError("tires.json 必須包含 tires 與 keyword_mapping")

    return data


# ---------- 解析尺寸 ----------
def extract_size(query: str):
    """
    從使用者輸入抓 R16 / R17
    """
    match = re.search(r"r\s*(1[4-9])", query.lower())
    if match:
        return f"R{match.group(1)}"
    return None


def tire_match_size(tire_size: str, target_size: str | None):
    if not target_size:
        return True
    return target_size in tire_size.upper()


# ---------- 關鍵字分類 ----------
def extract_categories(query: str, keyword_mapping: dict):
    matched = set()
    q = query.lower()

    for category, words in keyword_mapping.items():
        if category in ["brand", "size"]:
            continue
        for w in words:
            if w.lower() in q:
                matched.add(category)

    return list(matched)


def extract_brand(query: str, brand_keywords: list):
    q = query.lower()
    for b in brand_keywords:
        if b.lower() in q:
            return b
    return None


# ---------- 核心過濾 ----------
def filter_tires(tires, categories, target_size, target_brand):
    result = []

    for t in tires:
        # 1️⃣ 品牌嚴格過濾
        if target_brand:
            if target_brand not in t.get("brand_cn", "") and target_brand not in t.get("brand_en", ""):
                continue

        # 2️⃣ 尺寸嚴格過濾（直接比對 size 字串）
        if not tire_match_size(t.get("size", ""), target_size):
            continue

        # 3️⃣ 類別過濾
        if categories:
            tire_categories = t.get("categories", [])
            if not any(c in tire_categories for c in categories):
                continue

        result.append(t)

    return result


# ---------- AI 搜尋 ----------
def ai_search_and_rank(
    query: str,
    tires: list,
    keyword_mapping: dict,
    api_key: str,
    max_results: int = 20
):
    # 抓尺寸 / 品牌 / 類別
    target_size = extract_size(query)
    target_brand = extract_brand(query, keyword_mapping.get("brand", []))
    categories = extract_categories(query, keyword_mapping)

    # 本地先嚴格過濾
    candidates = filter_tires(
        tires=tires,
        categories=categories,
        target_size=target_size,
        target_brand=target_brand
    )

    # 沒結果才放寬（防呆）
    if not candidates:
        candidates = tires[:50]

    client = OpenAI(api_key=api_key)

    prompt = f"""
使用者需求：{query}

請從以下輪胎中，依「符合需求程度」排序，回傳前 {max_results} 筆。
只回傳 JSON 陣列，不要任何說明。

輪胎資料：
{json.dumps(candidates, ensure_ascii=False)}
"""

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        parsed = json.loads(res.choices[0].message.content)
        if isinstance(parsed, list):
            return parsed[:max_results]

    except Exception as e:
        print(f"⚠️ AI 排序失敗，使用本地結果: {e}")

    return candidates[:max_results]
