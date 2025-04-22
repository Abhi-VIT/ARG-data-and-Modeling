import pandas as pd
import pymysql
import pymysql.cursors as crr

host = input("Enter the name of your host: ")
user = input("Enter the name of your user: ")
password = input("Enter the password: ")
database = input("Enter the name of your database: ")
port = int(input("Enter the port number: "))

def establish_connection(host: str, user: str, password: str, database: str, port: int) -> pymysql.connections.Connection:
    try:
        conn = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port,
            cursorclass=crr.DictCursor  
        )
        print("Database connection established successfully.")
        return conn
    except pymysql.MySQLError as e:
        print(f"Error connecting to database: {e}")
        return None

def sqlprocess(conn, table_name):
    try:
        with conn.cursor() as cursor:

            cursor.execute("SHOW TABLES LIKE %s", (table_name,))
            table_exists = cursor.fetchone()

            if not table_exists:
                print(f"Error: Table '{table_name}' does not exists in the database.")
                return []
            
            query = f"SELECT * FROM {table_name};"
            cursor.execute(query)
            result = cursor.fetchall()
            return result
    except pymysql.MySQLError as e:
        print(f"Error executing query: {e}")
        return []

def processtocsv(conn, table_name, filename):
    data = sqlprocess(conn, table_name)
    if data:
        df = pd.DataFrame(data)
        df.to_csv(f"{filename}.csv", encoding="UTF-8", index=False)
        print(f"Data successfully exported to {filename}.csv")
    else:
        print("No data found or error in query execution.")

def close_connection(conn):
    if conn:
        conn.close()
        print("Database connection closed.")

if __name__ == "__main__":
    conn = establish_connection(host, user, password, database, port)
    if conn:
        table_name = input("Enter the name of your table: ")
        filename = input("Enter the name for your CSV file: ")
        processtocsv(conn, table_name, filename)
        close_connection(conn)
