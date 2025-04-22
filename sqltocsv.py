#important libraries to import 
import pandas as pd
import pymysql
import pymysql.cursors as crr
host = input("enter the name of your host")
user = input('enter the name of your user')
password = input("enter the password")
database = input("enter the name of your database")
port = int(input("enter the port number "))
def establish_connection(host: str, user: str, password: str, database: str, port: int) -> pymysql.connections.Connection :
    conn = pymysql.connections
    conn = conn.Connection(
        host = host,
        user = user, 
        password = password,
        database = database,
        port = port
    )
    return conn

table_name = input("enter the name of your table")
def sqlprocess(table_name):
    query = f"SELECT * FROM {table_name};"
    conn = establish_connection(host, user, password, database, port)
    tt = crr.DictCursor(conn)
    tt.execute(f"SELECT * FROM {table_name};")
    oo = tt.fetchall()
    return oo

name = input("enter the name for your csv")
def processtocsv(name):
    oo = sqlprocess(table_name)
    c = pd.DataFrame(oo)
    d = c.to_csv(f"{name}.csv", encoding = "UTF-8")
    return d
processtocsv(name)
def close_connection(name):
    cc = establish_connection(host, user, password, database, port).close()
    return f"your csv file name is {name}.csv, connection is closed with server"

close_connection(name)
