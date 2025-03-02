import sqlite3
import json
import os

# Database file path
DB_PATH = "glaucoma_analysis.db"

def init_db():
    """Initialize the database with required tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create table for analysis results
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS analysis_results (
        id TEXT PRIMARY KEY,
        filename TEXT,
        result TEXT,
        timestamp TEXT
    )
    ''')
    
    conn.commit()
    conn.close()

def save_analysis_result(image_id, filename, result, timestamp):
    """Save analysis result to database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Convert result dict to JSON string
    result_json = json.dumps(result)
    
    # Insert data
    cursor.execute(
        "INSERT INTO analysis_results (id, filename, result, timestamp) VALUES (?, ?, ?, ?)",
        (image_id, filename, result_json, timestamp)
    )
    
    conn.commit()
    conn.close()

def get_analysis_result(image_id):
    """Retrieve analysis result from database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM analysis_results WHERE id = ?", (image_id,))
    result = cursor.fetchone()
    
    conn.close()
    
    if result:
        return {
            "id": result[0],
            "filename": result[1],
            "result": json.loads(result[2]),
            "timestamp": result[3]
        }
    return None
