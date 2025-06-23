# ============================================================================
# File: src/face_embedder.py
# ============================================================================

import os
import sqlite3
import numpy as np
import cv2
from datetime import datetime
import traceback

class FaceEmbedder:
    def __init__(
        self,
        model_name: str = "buffalo_l",
        providers=None,
        threshold: float = 0.5,
        db_path: str = "embeddings.db",
        debug_dir: str = "debug_faces"
    ):
        # Set default providers if none specified
        if providers is None:
            providers = ["CPUExecutionProvider"]
        
        # Convert to absolute paths
        self.db_path = os.path.abspath(db_path)
        self.debug_dir = os.path.abspath(debug_dir) if debug_dir else None
        
        # Create directories if needed
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)
        
        print(f"📁 Database path: {self.db_path}")
        if self.debug_dir:
            print(f"📁 Debug directory: {self.debug_dir}")
        
        # Initialize InsightFace with error handling
        try:
            from insightface.app import FaceAnalysis
            print("🔄 Initializing InsightFace...")
            
            # Init InsightFace - remove allowed_modules restriction
            # InsightFace needs both detection and recognition modules
            self.app = FaceAnalysis(
                name=model_name,
                providers=providers
            )
            
            print("🔄 Preparing InsightFace model...")
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            print("✅ InsightFace initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize InsightFace: {e}")
            print("📋 Full error traceback:")
            traceback.print_exc()
            raise e
        
        self.threshold = threshold
        print(f"✅ FaceEmbedder initialized with threshold: {threshold}")

        # Database setup with error handling
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._ensure_table()
            self._load_embeddings()
            print(f"✅ Loaded {len(self.embeddings)} embeddings from database")
        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
            traceback.print_exc()
            raise e

    def _ensure_table(self):
        """Create the embeddings table if it doesn't exist"""
        try:
            c = self.conn.cursor()
            
            # Check if table exists
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'")
            table_exists = c.fetchone() is not None
            
            if not table_exists:
                # Create new table without person_name column
                c.execute("""
                    CREATE TABLE embeddings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        embedding BLOB NOT NULL,
                        timestamp TEXT NOT NULL
                    )
                """)
                print("✅ Created new embeddings table")
            else:
                # Check if person_name column exists
                c.execute("PRAGMA table_info(embeddings)")
                columns = [column[1] for column in c.fetchall()]
                self.has_person_name = 'person_name' in columns
                print(f"✅ Using existing table (person_name column: {self.has_person_name})")
            
            self.conn.commit()
            print("✅ Database table ensured")
        except Exception as e:
            print(f"❌ Failed to create database table: {e}")
            raise e

    def _load_embeddings(self):
        """Load existing embeddings from database"""
        try:
            c = self.conn.cursor()
            
            # Check if person_name column exists
            c.execute("PRAGMA table_info(embeddings)")
            columns = [column[1] for column in c.fetchall()]
            self.has_person_name = 'person_name' in columns
            
            # Query based on available columns
            if self.has_person_name:
                c.execute("SELECT id, embedding, person_name FROM embeddings")
            else:
                c.execute("SELECT id, embedding FROM embeddings")
            
            self.embeddings = []
            self.embedding_ids = []
            self.embedding_names = []
            
            for row in c.fetchall():
                emb_id = row[0]
                emb = np.frombuffer(row[1], dtype=np.float32)
                name = row[2] if self.has_person_name else f"Person_{emb_id}"
                
                self.embeddings.append(emb)
                self.embedding_ids.append(emb_id)
                self.embedding_names.append(name)
            
            print(f"✅ Loaded {len(self.embeddings)} embeddings from database")
        except Exception as e:
            print(f"❌ Failed to load embeddings: {e}")
            # Initialize empty lists if loading fails
            self.embeddings = []
            self.embedding_ids = []
            self.embedding_names = []
            self.has_person_name = False

    def get_embedding(self, face_img: np.ndarray) -> np.ndarray:
        """Extract embedding from face image using YOLO-detected face"""
        try:
            # Validate input
            if face_img is None or face_img.size == 0:
                print("❌ Invalid face image - empty or None")
                return None
            
            # Handle grayscale images
            if len(face_img.shape) == 2:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
            elif face_img.shape[2] == 1:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
            
            # Ensure minimum size
            if face_img.shape[0] < 32 or face_img.shape[1] < 32:
                print(f"❌ Face image too small: {face_img.shape}")
                return None
                
            # Convert to RGB (InsightFace requirement)
            rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Since we already have a cropped face from YOLO, we'll use InsightFace
            # to detect faces within this cropped region and extract embeddings
            faces = self.app.get(rgb_img)
            
            if len(faces) < 1:
                print("❌ No face detected by InsightFace in the cropped region")
                if self.debug_dir:
                    fname = f"fail_{datetime.now().strftime('%H%M%S_%f')}.jpg"
                    save_path = os.path.join(self.debug_dir, fname)
                    cv2.imwrite(save_path, face_img)
                    print(f"💾 Saved failed face to: {save_path}")
                return None
                
            # Return normalized embedding from the first (and likely only) face
            embedding = faces[0].normed_embedding
            return embedding.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Error in get_embedding: {e}")
            if self.debug_dir:
                try:
                    fname = f"error_{datetime.now().strftime('%H%M%S_%f')}.jpg"
                    save_path = os.path.join(self.debug_dir, fname)
                    cv2.imwrite(save_path, face_img)
                    print(f"💾 Saved error face to: {save_path}")
                except:
                    pass
            return None

    def find_similar(self, emb: np.ndarray):
        """Find most similar embedding in database"""
        if not self.embeddings:
            return None, None, 0.0

        best_sim = -1.0
        best_id = None
        best_name = None
        
        try:
            for emb_id, stored_emb, name in zip(self.embedding_ids, self.embeddings, self.embedding_names):
                sim = np.dot(emb, stored_emb)  # Cosine similarity
                if sim > best_sim:
                    best_sim = sim
                    best_id = emb_id
                    best_name = name
        except Exception as e:
            print(f"❌ Error in find_similar: {e}")
            return None, None, 0.0
                
        return best_id, best_name, best_sim

    def register(self, emb: np.ndarray, person_name: str = "Unknown"):
        """Register new embedding in database"""
        try:
            blob = emb.tobytes()
            ts = datetime.utcnow().isoformat(timespec="seconds")
            c = self.conn.cursor()
            
            # Insert based on available columns
            if hasattr(self, 'has_person_name') and self.has_person_name:
                c.execute("""
                    INSERT INTO embeddings (embedding, timestamp, person_name)
                    VALUES (?, ?, ?)
                """, (blob, ts, person_name))
            else:
                c.execute("""
                    INSERT INTO embeddings (embedding, timestamp)
                    VALUES (?, ?)
                """, (blob, ts))
            
            self.conn.commit()
            
            # Update in-memory data
            new_id = c.lastrowid
            self.embeddings.append(emb)
            self.embedding_ids.append(new_id)
            
            # Use person_name if column exists, otherwise generate from ID
            display_name = person_name if hasattr(self, 'has_person_name') and self.has_person_name else f"Person_{new_id}"
            self.embedding_names.append(display_name)
            
            print(f"🗄 Registered new embedding for '{display_name}' (ID: {new_id}) at {ts}")
            return new_id
        except Exception as e:
            print(f"❌ Failed to register embedding: {e}")
            return None

    def get_stats(self):
        """Get database statistics"""
        try:
            c = self.conn.cursor()
            c.execute("SELECT COUNT(*) FROM embeddings")
            count = c.fetchone()[0]
            return f"Database contains {count} embeddings"
        except Exception as e:
            print(f"❌ Failed to get stats: {e}")
            return "Database stats unavailable"

    def close(self):
        """Close database connection"""
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
        except Exception as e:
            print(f"❌ Error closing database: {e}")