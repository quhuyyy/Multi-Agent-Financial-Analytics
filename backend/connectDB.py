import mysql.connector

def get_Connection():
    connection = mysql.connector.connect(
        host='mysql',  
        user='root',  
        password='root',  
        database='Financial',
        charset="utf8mb4",
        collation="utf8mb4_general_ci"
    )
    return connection