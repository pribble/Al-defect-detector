import sqlite3
import threading


# 数据库连接
conn = sqlite3.connect('defect.db', check_same_thread = False)
cursor = conn.cursor()
lock = threading.Lock()


# 创建数据库表
def create_database():
    try:
        lock.acquire(True)
        cursor.execute("create table if not exists defect_list (id integer primary key, \
            uuid char, path char(64), name char, prediction_time char, score char, \
            [CreatedTime] TimeStamp NOT NULL DEFAULT (datetime('now','localtime')), UNIQUE(uuid, path, name))")
    finally:
        lock.release()


# 获取每天数据
def select_day_data(key1, key2):
    try:
        lock.acquire(True)
        cursor.execute("select count() from (select * from (select * from (select * from defect_list where path is not null) \
            where path is not 'detect.jpg')  where CreatedTime>=datetime('now','start of day','{} day') \
            and CreatedTime<datetime('now','start of day','{} day'))".format(key1, key2))
        values = cursor.fetchall()
    finally:
        lock.release()
    return values[0][0]


# 插入数据
def insert_data(uid, file_name, class_name, prediction_time, score):
    try:
        lock.acquire(True) 
        cursor.execute("insert or ignore into defect_list(uuid, path, name, prediction_time, score) \
        values(?, ?, ?, ?, ?)", (uid, file_name, class_name, prediction_time, score))
        conn.commit()
    finally:
        lock.release()


# select指令
def select_instructions(key1, key2, key3):
    sql_instructions = "select {} from {} {}".format(key1, key2, key3)
    return sql_instructions


# 获取数据
def select_data(sql):
    try:
        lock.acquire(True)
        cursor.execute(sql)
        values = cursor.fetchall()
    finally:
        lock.release()
    return values
