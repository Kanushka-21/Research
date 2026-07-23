import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = "tomato_detections.db"

# Shelf life data (days remaining)
SHELF_LIFE = {
    "green": 31,
    "breaker": 29,
    "turning": 24,
    "red": 12,
    "defect": 0
}

def init_database():
    """Initialize database with detection logs table"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id TEXT UNIQUE,
            class_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            shelf_life INTEGER NOT NULL,
            bbox_x1 REAL,
            bbox_y1 REAL,
            bbox_x2 REAL,
            bbox_y2 REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            tab_source TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def save_detection(detection_id, class_name, confidence, bbox=None, tab_source="realtime"):
    """Save a single detection to database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get shelf life based on class
        shelf_life = SHELF_LIFE.get(class_name.lower(), 0)
        
        # Parse bbox
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = None, None, None, None
        if bbox:
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
        
        cursor.execute('''
            INSERT OR IGNORE INTO detections 
            (detection_id, class_name, confidence, shelf_life, bbox_x1, bbox_y1, bbox_x2, bbox_y2, tab_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (detection_id, class_name, confidence, shelf_life, bbox_x1, bbox_y1, bbox_x2, bbox_y2, tab_source))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving detection: {e}")
        return False

def get_statistics(time_window_minutes=None):
    """Get detection statistics"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Build query based on time window
        where_clause = ""
        if time_window_minutes:
            where_clause = f"WHERE timestamp >= datetime('now', '-{time_window_minutes} minutes')"
        
        # Total detections
        cursor.execute(f"SELECT COUNT(*) FROM detections {where_clause}")
        total_count = cursor.fetchone()[0]
        
        # Count by class
        cursor.execute(f"""
            SELECT class_name, COUNT(*) as count FROM detections 
            {where_clause}
            GROUP BY class_name
        """)
        class_counts = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Average confidence
        cursor.execute(f"SELECT AVG(confidence) FROM detections {where_clause}")
        avg_confidence = cursor.fetchone()[0] or 0
        
        # Defect count
        defect_count = class_counts.get("defect", 0)
        defect_ratio = (defect_count / total_count * 100) if total_count > 0 else 0
        
        # Average shelf life by class
        cursor.execute(f"""
            SELECT class_name, AVG(shelf_life) as avg_life FROM detections 
            {where_clause}
            GROUP BY class_name
        """)
        shelf_life_by_class = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Overall average shelf life
        cursor.execute(f"SELECT AVG(shelf_life) FROM detections {where_clause}")
        overall_avg_shelf_life = cursor.fetchone()[0] or 0
        
        # Throughput (detections per minute)
        if time_window_minutes and time_window_minutes > 0:
            throughput = total_count / time_window_minutes
        else:
            throughput = 0
        
        # Recent detections
        cursor.execute(f"""
            SELECT detection_id, class_name, shelf_life, confidence, timestamp FROM detections
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT 20
        """)
        recent = cursor.fetchall()
        
        conn.close()
        
        return {
            "total_count": total_count,
            "class_counts": class_counts,
            "avg_confidence": avg_confidence,
            "defect_count": defect_count,
            "defect_ratio": defect_ratio,
            "shelf_life_by_class": shelf_life_by_class,
            "overall_avg_shelf_life": overall_avg_shelf_life,
            "throughput": throughput,
            "recent": recent
        }
    except Exception as e:
        print(f"Error getting statistics: {e}")
        return None

def get_recent_detections(limit=20):
    """Get recent detections for display"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, class_name, shelf_life, confidence, timestamp FROM detections
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        return results
    except Exception as e:
        print(f"Error getting recent detections: {e}")
        return []

def clear_old_detections(days=7):
    """Clear detections older than specified days"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM detections 
            WHERE timestamp < datetime('now', '-' || ? || ' days')
        ''', (days,))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error clearing old detections: {e}")
        return False

# Initialize database on import
if not Path(DB_PATH).exists():
    init_database()
