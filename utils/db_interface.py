# db_interface.py (DB-disabled friendly)
import json
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# The path to the SQLite database file. Default: 'term_demo.db' in the project root.
SQLITE_DB_PATH = os.getenv('SQLITE_DB_PATH', 'term_demo.db')
# Allow turning off DB usage entirely for testing. Default: disabled ("1").
DB_DISABLED = os.getenv('DISABLE_DB', '1') == '1'


if DB_DISABLED:
    # ---------------- Disabled mode: provide safe stubs ----------------
    def get_connection():  # pragma: no cover
        raise RuntimeError("Database features are disabled (set DISABLE_DB=0 to enable)")

    def query_term_translation(term):
        """Stub: return empty list when DB is disabled."""
        return []

    def save_translation(term, translations, t_type="term", source=None):
        """Stub: no-op when DB is disabled."""
        return None

    def ensure_db_initialized():
        """Stub: log and do nothing when DB is disabled."""
        logger.info("Database features disabled; skip initialization")
        return None

else:
    # ---------------- Enabled mode: real implementations ----------------
    def get_connection():
        """
        获取数据库连接。如果目标数据库文件不存在则自动创建，并确保 term_dictionary 表存在。
        返回: 已初始化的 sqlite3 连接对象。
        抛出: 连接失败时抛出异常。
        """
        try:
            # connect() will create the file if it doesn't exist
            conn = sqlite3.connect(SQLITE_DB_PATH, timeout=int(os.getenv('DB_CONNECT_TIMEOUT', '5')))
            conn.row_factory = sqlite3.Row  # Return rows as dictionary-like objects
        except sqlite3.Error as e:
            logger.exception("Failed to connect to SQLite database at '%s': %s", SQLITE_DB_PATH, e)
            raise

        # Ensure table exists
        try:
            with conn: # Use connection as a context manager for transactions
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS term_dictionary (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        term TEXT NOT NULL UNIQUE,
                        translation TEXT DEFAULT NULL,
                        type TEXT CHECK(type IN ('term','proper_noun')) DEFAULT 'term'
                    );
                    """
                )
        except sqlite3.Error as e:  # pragma: no cover
            logger.warning("Failed to ensure term_dictionary table exists: %s", e)
        return conn

    def query_term_translation(term: str) -> List[str]:
        """查询本地术语库，返回翻译列表 (list[str])"""
        try:
            conn = get_connection()
            try:
                cur = conn.cursor()
                cur.execute("SELECT translation FROM term_dictionary WHERE term=?", (term,))
                res = cur.fetchone()
                if res and res["translation"]:
                    raw = res["translation"]
                    try:
                        # The data is stored as a JSON string
                        return json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        logger.debug("Failed to parse translation JSON for term '%s'", term)
                        return []
                return []
            finally:
                conn.close()
        except RuntimeError: # DB is disabled
            return []


    def save_translation(term: str, translations: Any, t_type: str = "term", source: Optional[str] = None) -> bool:
        """保存或更新术语翻译。translations 可以是 str (分号分隔) 或 list[str]."""
        if isinstance(translations, str):
            translations = [t.strip() for t in translations.split(";") if t.strip()]
        if not isinstance(translations, list):
            raise TypeError("translations must be list[str] or semicolon-separated string")

        try:
            conn = get_connection()
            try:
                with conn:
                    cur = conn.cursor()
                    cur.execute("SELECT translation FROM term_dictionary WHERE term=?", (term,))
                    existing = cur.fetchone()

                    existing_list: List[str] = []
                    if existing and existing["translation"]:
                        try:
                            existing_list = json.loads(existing["translation"])
                        except (json.JSONDecodeError, TypeError):
                            existing_list = []

                    # merge & deduplicate preserving order
                    merged = list(dict.fromkeys(existing_list + translations))
                    merged_json = json.dumps(merged, ensure_ascii=False)

                    cur.execute(
                        """
                        INSERT INTO term_dictionary (term, translation, type)
                        VALUES (?, ?, ?)
                        ON CONFLICT(term) DO UPDATE SET
                            translation=excluded.translation,
                            type=excluded.type;
                        """,
                        (term, merged_json, t_type),
                    )
                return True
            finally:
                conn.close()
        except RuntimeError: # DB is disabled
            return False


    def ensure_db_initialized():
        """显式确保数据库和表存在。"""
        try:
            conn = get_connection()
            try:
                logger.info("Database initialized (connection successful)")
            finally:
                conn.close()
        except RuntimeError: # DB is disabled
            pass
