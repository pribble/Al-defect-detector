"""
SQLite 数据库封装 (defect.db)

线程安全的数据库操作: 建表、插入、查询。
表结构: defect_list(id, uuid, path, name, prediction_time, score, CreatedTime)
"""

import sqlite3
import threading

__all__ = [
    'create_database', 'insert_data', 'query', 'query_value', 'select_day_data',
    'FrameBuffer',
]

_DB_FILE = 'defect.db'

conn = sqlite3.connect(_DB_FILE, check_same_thread=False)
cursor = conn.cursor()
lock = threading.Lock()


def create_database():
    """创建 defect_list 表 (如不存在)"""
    with lock:
        cursor.execute(
            "create table if not exists defect_list ("
            "  id integer primary key,"
            "  uuid char,"
            "  path char(64),"
            "  name char,"
            "  prediction_time char,"
            "  score char,"
            "  [CreatedTime] TimeStamp NOT NULL DEFAULT (datetime('now','localtime')),"
            "  UNIQUE(uuid, path, name)"
            ")"
        )


def insert_data(uid: str, file_name: str, class_name: str, prediction_time, score):
    """插入一条检测记录"""
    with lock:
        cursor.execute(
            "insert or ignore into defect_list(uuid, path, name, prediction_time, score) "
            "values(?, ?, ?, ?, ?)",
            (uid, file_name, class_name, prediction_time, score),
        )
        conn.commit()


def query(columns: str, source: str, condition: str, params=()):
    """
    构建并执行 SELECT, 返回所有结果行.

    Args:
        columns:   选择的字段, 如 ``'*'``, ``'distinct path'``, ``'name, count(1)'``
        source:    数据源, 表名或子查询 ``'(subquery)'``
        condition: WHERE / GROUP BY / ORDER BY / LIMIT 子句
        params:    可选参数元组, 用于 ``?`` 占位符绑定

    Returns: 行列表, 每行为一个 tuple
    """
    sql = "select {} from {} {}".format(columns, source, condition)
    with lock:
        cursor.execute(sql, params)
        rows = cursor.fetchall()
    return rows


def query_value(columns: str, source: str, condition: str, params=()):
    """
    执行 SELECT 并返回第一行第一列的值 (聚合查询快捷方式).

    Args:
        columns:   ``count()``, ``sum(score)``, ``distinct uuid`` 等
        source:    表名或子查询
        condition: WHERE / GROUP BY / ORDER BY / LIMIT
        params:    可选参数元组, 用于 ``?`` 占位符绑定

    Returns: 标量值, 无结果时返回 None
    """
    rows = query(columns, source, condition, params)
    return rows[0][0] if rows else None


class FrameBuffer:
    """固定 2 槽环形帧缓冲, 替代 ``Queue(maxsize=2)``。

    - 预分配 ``[None, None]`` 作为 [oldest, newest]，无运行时分配
    - 线程安全，put 溢出时丢弃最旧帧而非阻塞
    - get 返回 **oldest**, 空时返回 **None** (调用方配合 empty() 使用)
    """

    __slots__ = ('_buf', '_lock')

    def __init__(self):
        self._buf = [None, None]  # [oldest, newest]
        self._lock = threading.Lock()

    def put(self, item):
        with self._lock:
            if self._buf[1] is not None:
                # 双槽已满 → 顶掉 oldest, 原有的 newest 变为 oldest
                self._buf[0] = self._buf[1]
                self._buf[1] = item
            elif self._buf[0] is None:
                self._buf[0] = item
            else:
                self._buf[1] = item

    def get(self):
        with self._lock:
            item = self._buf[0]
            if item is not None and self._buf[1] is not None:
                self._buf[0] = self._buf[1]
                self._buf[1] = None
        return item


def select_day_data(offset_start: str, offset_end: str) -> int:
    """
    查询某天的检测数量.

    Args:
        offset_start: 起始偏移, 如 ＂+0＂ 表示今天, ＂-1＂ 表示昨天
        offset_end:   结束偏移 (排他), 如 ＂+1＂ 表示明天

    Returns: 计数
    """
    return query_value(
        'count()', 'defect_list',
        "where path is not null"
        "  and path != 'detect.jpg'"
        "  and CreatedTime >= datetime('now', 'start of day', ? || ' day')"
        "  and CreatedTime <  datetime('now', 'start of day', ? || ' day')",
        (offset_start, offset_end),
    ) or 0
