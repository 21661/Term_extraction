# db_interface.py (DB-disabled friendly)
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_CHARSET = 'utf8mb4'
DEFAULT_DB_COLLATION = os.getenv('DB_COLLATION', 'utf8mb4_general_ci')
# Allow turning off DB usage entirely for testing. Default: disabled ("1").
DB_DISABLED = os.getenv('DISABLE_DB', '1') == '1'

# Try import pymysql only when DB is enabled
if not DB_DISABLED:
    try:
        import pymysql  # type: ignore
    except Exception as _imp_err:  # pragma: no cover
        logger.warning("pymysql import failed, disabling DB features: %s", _imp_err)
        DB_DISABLED = True
else:
    # If disabled we don't require pymysql; still attempt lazy import in functions if user re-enables.
    pass

if DB_DISABLED:
    # ---------------- Disabled mode: provide safe stubs ----------------
    def _base_conn_params(include_db: bool = True):  # pragma: no cover
        return {}

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
    import pymysql  # type: ignore  # re-import for type checkers
    import pymysql.cursors as pymysql_cursors  # type: ignore

    def _base_conn_params(include_db=True):
        params = dict(
            host=os.getenv('DB_HOST', 'localhost'),
            user=os.getenv('DB_USER', 'root'),
            password=os.getenv('DB_PASSWORD', '123456'),
            charset=DEFAULT_DB_CHARSET,
            cursorclass=pymysql_cursors.DictCursor,
            connect_timeout=int(os.getenv('DB_CONNECT_TIMEOUT', '5')),
        )
        if include_db:
            params['db'] = os.getenv('DB_NAME', 'term_demo')
        return params

    def get_connection():
        """
        获取数据库连接。如果目标数据库不存在则自动创建，并确保 term_dictionary 表存在。
        返回: 已初始化的 pymysql 连接对象。
        抛出: 连接失败时抛出异常。
        """
        db_name = os.getenv('DB_NAME', 'term_demo')

        # First try connect directly
        try:
            conn = pymysql.connect(**_base_conn_params(include_db=True))
        except pymysql.err.OperationalError as e:
            logger.warning("Initial DB connect to '%s' failed: %s", db_name, e)
            # Attempt database creation (unknown database error code is 1049 typically)
            try:
                conn_tmp = pymysql.connect(**_base_conn_params(include_db=False))
                try:
                    with conn_tmp.cursor() as cur:
                        cur.execute(
                            f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET {DEFAULT_DB_CHARSET} COLLATE {DEFAULT_DB_COLLATION};"
                        )
                        conn_tmp.commit()
                        logger.info("Ensured database '%s' exists", db_name)
                finally:
                    conn_tmp.close()
                # Reconnect to created database
                conn = pymysql.connect(**_base_conn_params(include_db=True))
            except Exception as e2:
                logger.exception("Failed to create or connect to database '%s': %s", db_name, e2)
                raise

        # Ensure table exists
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS term_dictionary (
                        id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                        term VARCHAR(255) NOT NULL UNIQUE,
                        translation JSON DEFAULT NULL,
                        type ENUM('term','proper_noun') DEFAULT 'term'
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
                conn.commit()
        except Exception as e:  # pragma: no cover - may fail due to permissions
            logger.warning("Failed to ensure term_dictionary table exists: %s", e)
        return conn

    def query_term_translation(term):
        """查询本地术语库，返回翻译列表 (list[str])"""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT translation FROM term_dictionary WHERE term=%s", (term,))
                res = cur.fetchone()
                if isinstance(res, dict) and res.get("translation"):
                    raw = res["translation"]
                    if isinstance(raw, (list, dict)):
                        return raw if isinstance(raw, list) else []
                    try:
                        return json.loads(raw)
                    except Exception:
                        logger.debug("Failed to parse translation JSON for term '%s'", term)
                        return []
                return []
        finally:
            conn.close()

    def save_translation(term, translations, t_type="term", source=None):  # source kept for API compatibility
        """保存或更新术语翻译。translations 可以是 str (分号分隔) 或 list[str]."""
        if isinstance(translations, str):
            translations = [t.strip() for t in translations.split(";") if t.strip()]
        if not isinstance(translations, list):
            raise TypeError("translations must be list[str] or semicolon-separated string")

        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT translation FROM term_dictionary WHERE term=%s", (term,))
                existing = cur.fetchone()
                existing_list: List[str] = []
                if isinstance(existing, dict) and existing.get("translation"):
                    raw = existing["translation"]
                    try:
                        if isinstance(raw, list):
                            existing_list = raw
                        else:
                            existing_list = json.loads(raw) if isinstance(raw, str) else []
                    except Exception:
                        existing_list = []
                # merge & deduplicate preserving order
                merged = list(dict.fromkeys(existing_list + translations))
                merged_json = json.dumps(merged, ensure_ascii=False)
                cur.execute(
                    """
                    INSERT INTO term_dictionary (term, translation, type)
                    VALUES (%s, %s, %s)
                    ON DUPLICATE KEY UPDATE translation=%s, type=%s
                    """,
                    (term, merged_json, t_type, merged_json, t_type),
                )
                conn.commit()
        finally:
            conn.close()
        return True

    def ensure_db_initialized():
        """显式确保数据库和表存在。"""
        conn = get_connection()
        try:
            logger.info("Database initialized (connection successful)")
        finally:
            conn.close()
