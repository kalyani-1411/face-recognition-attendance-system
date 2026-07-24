import mysql.connector
import datetime
from mysql.connector import Error

def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="8669227136Shreya",
            database="attendance"
        )
        return conn
    except Error as e:
        print(f"Error connecting to database: {e}")
        return None

def mark_attendance(name):
    """Mark attendance in the database."""
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database")
        return False
    
    try:
        cursor = conn.cursor()
        # Get current time
        timestamp = datetime.datetime.now()
        query = "INSERT INTO attendance_log (name, timestamp) VALUES (%s, %s)"
        cursor.execute(query, (name, timestamp))
        conn.commit()
        print(f"Attendance marked for {name} at {timestamp}")
        return True
    except Error as e:
        print(f"Error marking attendance: {e}")
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def get_attendance_records():
    """Retrieve attendance records."""
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database")
        return []
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, timestamp FROM attendance_log ORDER BY timestamp DESC")
        records = cursor.fetchall()
        return [{
            "id": record[0],
            "name": record[1],
            "timestamp": record[2].isoformat() if record[2] else None
        } for record in records]
    except Error as e:
        print(f"Error getting attendance records: {e}")
        return []
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def delete_attendance_record(record_id):
    """Delete an attendance record by ID."""
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database")
        return False
    
    try:
        cursor = conn.cursor()
        query = "DELETE FROM attendance_log WHERE id = %s"
        cursor.execute(query, (record_id,))
        conn.commit()
        return True
    except Error as e:
        print(f"Error deleting record: {e}")
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# Create table if not exists
conn = get_db_connection()
if conn:
    try:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance_log (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255),
            timestamp DATETIME
        )
        """)
        conn.commit()
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == "__main__":
    mark_attendance("Test User")
