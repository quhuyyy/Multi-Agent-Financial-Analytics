import connectDB

con = connectDB.get_Connection()

if con.is_connected():
    print("Kết nối MySQL thành công!")
else:
    print("Kết nối MySQL thất bại!")