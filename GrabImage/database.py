"""
SQLite 数据库封装 (defect.db)

提供线程安全的数据库操作: 建表、插入、查询。
表结构: defect_list(id, uuid, path, name, prediction_time, score, CreatedTime)
"""

import sqlite3
import threading

_DB_FILE = 'defect.db'

conn = sqlite3.connect(_DB_FILE, check_same_thread=False)
cursor = conn.cursor()
lock = threading.Lock()


def _acquire_lock():
    lock.acquire(True)


def _release_lock():
    lock.release()


def create_database():
    """创建 defect_list 表 (如不存在)"""
    _acquire_lock()
    try:
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
    finally:
        _release_lock()


def select_day_data(offset_start: str, offset_end: str) -> int:
    """
    查询某天的检测数量.

    Args:
        offset_start: 起始偏移, 如 "+0" 表示今天, "-1" 表示昨天
        offset_end:   结束偏移 (排他), 如 "+1" 表示明天

    Returns: 计数
    """
    _acquire_lock()
    try:
        cursor.execute(
            "select count() from ("
            "  select * from ("
            "    select * from ("
            "      select * from defect_list where path is not null"
            "    ) where path is not 'detect.jpg'"
            "  ) where CreatedTime >= datetime('now', 'start of day', ? || ' day')"
            "    and CreatedTime <  datetime('now', 'start of day', ? || ' day')"
            ")",
            (offset_start, offset_end),
        )
        values = cursor.fetchall()
    finally:
        _release_lock()
    return values[0][0]


def insert_data(uid: str, file_name: str, class_name: str, prediction_time, score):
    """插入一条检测记录"""
    _acquire_lock()
    try:
        cursor.execute(
            "insert or ignore into defect_list(uuid, path, name, prediction_time, score) "
            "values(?, ?, ?, ?, ?)",
            (uid, file_name, class_name, prediction_time, score),
        )
        conn.commit()
    finally:
        _release_lock()


def select_instructions(columns: str, source: str, condition: str) -> str:
    """
    拼接 SELECT 语句 (不执行).

    Args:
        columns:   选择的字段, 如 `'*'`, `'distinct path'`, `'name, count(1)'`
        source:    数据源, 可以是表名或子查询 `'(subquery)'`
        condition: WHERE / GROUP BY / ORDER BY / LIMIT 子句, 如 `"where path is not null"`

    Returns: SQL 字符串
    """
    return "select {} from {} {}".format(columns, source, condition)


def select_data(sql: str):
    """执行 SQL 并返回所有结果行"""
    _acquire_lock()
    try:
        cursor.execute(sql)
        values = cursor.fetchall()
    finally:
        _release_lock()
    return values
