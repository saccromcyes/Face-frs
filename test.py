import sqlite3

# connect to the database
conn = sqlite3.connect("faces.db")
cursor = conn.cursor()

# show all rows in the faces table
cursor.execute("SELECT id, name, LENGTH(embedding), image_path FROM faces;")
rows = cursor.fetchall()

print("=== Registered Faces ===")
for row in rows:
    print(f"ID: {row[0]} | Name: {row[1]} | Embedding length: {row[2]} | Image path: {row[3]}")

conn.close()
