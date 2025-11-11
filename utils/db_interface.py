# db_interface.py
import pymysql
import json
import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_DB_CHARSET = 'utf8mb4'
DEFAULT_DB_COLLATION = 'utf8mb4_general_ci'


def _base_conn_params(include_db=True):
    params = dict(
        host=os.getenv('DB_HOST', 'localhost'),
        user=os.getenv('DB_USER', 'root'),
        password=os.getenv('DB_PASSWORD', '123456'),
        charset=DEFAULT_DB_CHARSET,
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=int(os.getenv('DB_CONNECT_TIMEOUT', '5')),
    )
    if include_db:
        params['db'] = os.getenv('DB_NAME', 'term_demo')
    return params


def get_connection():
    """返回连接。如果目标数据库不存在，则尝试自动创建数据库并继续。

    1) 尝试直接连接到指定数据库；
    2) 如果失败且原因可能是数据库不存在，则连接不带 db，执行 CREATE DATABASE IF NOT EXISTS <db>；
    3) 重新连接并确保 term_dictionary 表存在（CREATE TABLE IF NOT EXISTS ...）。
    """
    db_name = os.getenv('DB_NAME', 'term_demo')

    # First try to connect to the target database
    try:
        conn = pymysql.connect(**_base_conn_params(include_db=True))
    except pymysql.err.OperationalError as e:
        logger.warning("Initial DB connect to '%s' failed: %s", db_name, e)
        # Try to create the database if possible
        try:
            conn_tmp = pymysql.connect(**_base_conn_params(include_db=False))
            try:
                with conn_tmp.cursor() as cur:
                    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET {DEFAULT_DB_CHARSET} COLLATE {DEFAULT_DB_COLLATION};")
                    conn_tmp.commit()
                    logger.info("Ensured database '%s' exists", db_name)
            finally:
                conn_tmp.close()
            # Reconnect to the newly-created (or existing) database
            conn = pymysql.connect(**_base_conn_params(include_db=True))
        except Exception as e2:
            logger.exception("Failed to create or connect to database '%s': %s", db_name, e2)
            # Re-raise to let caller know we couldn't get a usable connection
            raise

    # Ensure the term_dictionary table exists. If not, create it.
    try:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS term_dictionary (
                id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                term VARCHAR(255) NOT NULL UNIQUE,
                translation JSON DEFAULT NULL,
                type ENUM('term','proper_noun') DEFAULT 'term'
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            conn.commit()
    except Exception as e:
        # Log but don't prevent caller from using the connection; some environments may not permit DDL
        logger.warning("Failed to ensure term_dictionary table exists: %s", e)
    return conn


def query_term_translation(term):
    """查询本地术语库，返回翻译列表"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT translation FROM term_dictionary WHERE term=%s", (term,))
            res = cur.fetchone()
            if res and res.get("translation"):
                try:
                    return json.loads(res["translation"]) if isinstance(res["translation"], str) else res["translation"]
                except Exception:
                    # If translation is stored as text, attempt to parse; otherwise return empty
                    try:
                        return json.loads(str(res["translation"]))
                    except Exception:
                        return []
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
            if existing and existing.get("translation"):
                try:
                    existing_list = json.loads(existing["translation"]) if isinstance(existing["translation"], str) else existing["translation"]
                except Exception:
                    existing_list = []
                merged = list(dict.fromkeys(existing_list + translations))
            else:
                merged = translations

            merged_json = json.dumps(merged, ensure_ascii=False)

            # Use INSERT ... ON DUPLICATE KEY UPDATE to upsert
            cur.execute("""
                INSERT INTO term_dictionary (term, translation, type)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE translation=%s, type=%s
            """, (term, merged_json, t_type, merged_json, t_type))
            conn.commit()
    finally:
        conn.close()


def ensure_db_initialized():
    """显式确保数据库和 term_dictionary 表被创建/存在。

    调用此函数会尝试获取连接（如果数据库不存在会创建），并保证 term_dictionary 表存在。
    在无法连接到数据库时会抛出异常。
    """
    conn = get_connection()
    try:
        # get_connection already ensures table exists; just close the connection
        logger.info("Database initialized (connection successful)")
    finally:
        conn.close()
