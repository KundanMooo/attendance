
import sqlite3
import os
import cv2
from datetime import datetime

def check_database(db_path="embeddings.db"):
    """Check the embeddings database"""
    db_path = os.path.abspath(db_path)
    
    if not os.path.exists(db_path):
        print(f"❌ Database file not found at: {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Check if table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'")
        if not c.fetchone():
            print("❌ Embeddings table not found!")
            conn.close()
            return
        
        # Check available columns
        c.execute("PRAGMA table_info(embeddings)")
        columns = [column[1] for column in c.fetchall()]
        has_person_name = 'person_name' in columns
        
        # Get all records based on available columns
        if has_person_name:
            c.execute("SELECT id, timestamp, person_name FROM embeddings ORDER BY timestamp")
        else:
            c.execute("SELECT id, timestamp FROM embeddings ORDER BY timestamp")
        
        records = c.fetchall()
        
        print(f"📊 Database contains {len(records)} embeddings:")
        print(f"📋 Available columns: {', '.join(columns)}")
        print("-" * 60)
        
        for record in records:
            if has_person_name:
                id_val, timestamp, person_name = record
                print(f"ID: {id_val:3d} | Time: {timestamp} | Name: {person_name}")
            else:
                id_val, timestamp = record
                person_name = f"Person_{id_val}"
                print(f"ID: {id_val:3d} | Time: {timestamp} | Name: {person_name}")
        
        conn.close()
    except Exception as e:
        print(f"❌ Error checking database: {e}")

def check_faces_folder(faces_dir="faces"):
    """Check the faces folder"""
    faces_dir = os.path.abspath(faces_dir)
    
    if not os.path.exists(faces_dir):
        print(f"❌ Faces folder not found at: {faces_dir}")
        return
    
    try:
        face_files = [f for f in os.listdir(faces_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"📁 Faces folder contains {len(face_files)} images:")
        print("-" * 60)
        
        for i, filename in enumerate(sorted(face_files)):
            filepath = os.path.join(faces_dir, filename)
            # Get file size
            size = os.path.getsize(filepath)
            # Get image dimensions
            img = cv2.imread(filepath)
            if img is not None:
                h, w = img.shape[:2]
                print(f"{i+1:3d}. {filename} ({w}x{h}, {size} bytes)")
            else:
                print(f"{i+1:3d}. {filename} (corrupted or unreadable)")
    except Exception as e:
        print(f"❌ Error checking faces folder: {e}")

def display_faces(faces_dir="faces"):
    """Display saved faces"""
    faces_dir = os.path.abspath(faces_dir)
    
    if not os.path.exists(faces_dir):
        print(f"❌ Faces folder not found at: {faces_dir}")
        return
    
    try:
        face_files = [f for f in os.listdir(faces_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not face_files:
            print("❌ No face images found!")
            return
        
        print(f"🖼️ Displaying {len(face_files)} faces. Press any key to continue, 'q' to quit.")
        
        for filename in sorted(face_files):
            filepath = os.path.join(faces_dir, filename)
            img = cv2.imread(filepath)
            
            if img is not None:
                # Resize for display if needed
                scale = min(800/img.shape[1], 600/img.shape[0])
                if scale < 1:
                    img = cv2.resize(img, (0,0), fx=scale, fy=scale)
                
                cv2.imshow(f"Face: {filename}", img)
                key = cv2.waitKey(0) & 0xFF
                cv2.destroyAllWindows()
                
                if key == ord('q'):
                    break
            else:
                print(f"❌ Could not load {filename}")
    except Exception as e:
        print(f"❌ Error displaying faces: {e}")

def main():
    print("🔍 Attendance System - Database and Faces Viewer")
    print("=" * 50)
    
    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    db_path = os.path.join(project_root, "embeddings.db")
    faces_dir = os.path.join(project_root, "faces")
    
    print("\n1. Checking database...")
    check_database(db_path)
    
    print("\n2. Checking faces folder...")
    check_faces_folder(faces_dir)
    
    print("\n3. Would you like to view the saved faces? (y/n)")
    try:
        response = input().lower().strip()
        if response in ['y', 'yes']:
            display_faces(faces_dir)
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main()