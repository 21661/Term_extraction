# db_interface.py
import pymysql
import json

def get_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='123456',
        db='term_demo',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

def query_term_translation(term):
    """查询本地术语库，返回翻译列表"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT translation FROM term_dictionary WHERE term=%s", (term,))
            res = cur.fetchone()
            if res and res["translation"]:
                return json.loads(res["translation"])
            return []
    finally:
        conn.close()

def save_translation(term, translations, t_type="term", source=None):
    """保存或更新术语翻译"""
    if isinstance(translations, str):
        translations = [t.strip() for t in translations.split(";") if t.strip()]

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT translation FROM term_dictionary WHERE term=%s", (term,))
            existing = cur.fetchone()
            if existing and existing["translation"]:
                existing_list = json.loads(existing["translation"])
                merged = list(dict.fromkeys(existing_list + translations))
            else:
                merged = translations

            merged_json = json.dumps(merged, ensure_ascii=False)

            cur.execute("""
                INSERT INTO term_dictionary (term, translation, type)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE translation=%s, type=%s
            """, (term, merged_json, t_type, merged_json, t_type))
            conn.commit()
    finally:
        conn.close()

