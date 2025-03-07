from src.database import Database
import moz_sql_parser
import json
import os

db = Database(collect_db_info=False)

def backup_queries():

    cmd = "pg_dump --table='queries' " \
          "--data-only --column-inserts imdb --no-owner > queries.sql"
    try:
        os.system(cmd)
        print("Backup completed")
    except Exception as e:
        print("!!Problem occured!!")
        print(e)

def create_table():
    cursor = db.conn.cursor()
    q = """
    CREATE TABLE queries (
            id SERIAL PRIMARY KEY,
            file_name VARCHAR(10) NOT NULL,
            relations_num REAL,
            query TEXT NOT NULL,
            moz JSON NOT NULL,
            planning REAL,
            execution REAL,
            cost REAL
    )
    """
    cursor.execute(q)
    cursor.close()
    db.conn.commit()

def create_training_file(sql_path, time_path):
    with open(sql_path)as f:
        sql_lines = f.readlines()

    with open(time_path)as f:
        time_lines = f.readlines()

    qid2sql = { line.split('#####')[0]: line.split('#####')[1] for line in sql_lines}
    qid2time = { line.split('#####')[0]: line.split('#####')[1] for line in time_lines}
    cursor = db.conn.cursor()
    
    for qid in qid2sql.keys():
        sql = qid2sql[qid] 
        execution = float(qid2time[qid])
        
        file_name = qid 
        ast = moz_sql_parser.parse(sql)
        relations_num = len(ast["from"])
        
        print(file_name, relations_num, execution)
        
        cursor.execute(
            "INSERT INTO queries (file_name, relations_num, query, moz, execution) VALUES(%s, %s, %s, %s, %s)",
            (file_name, relations_num, sql, json.dumps(ast), execution),
        )
        
    db.conn.commit()
    cursor.close()   

if __name__ == '__main__':
    sql_path = 'data/imdb-test/sql.txt'
    time_path = 'data/imdb-test/time.txt'
    # create_table()
    create_training_file(sql_path, time_path)
