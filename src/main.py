
import os
import cv2
import numpy as np
import traceback
from datetime import datetime
from yolo_face_detector import YOLOFaceDetector
from face_embedder import FaceEmbedder

def main():
    print("🚀 Starting Face Recognition System")
    
    # Get current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Create directories relative to project root
    faces_dir = os.path.join(project_root, "faces")
    debug_dir = os.path.join(project_root, "debug_faces")
    models_dir = os.path.join(project_root, "models")
    
    os.makedirs(faces_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"📁 Project root: {project_root}")
    print(f"📁 Faces will be saved to: {faces_dir}")
    print(f"📁 Debug images in: {debug_dir}")
    print(f"📁 Models in: {models_dir}")

    # Initialize models
    print("🔄 Loading models...")
    
    # Initialize face detector
    try:
        detector = YOLOFaceDetector(
            model_source=os.path.join(models_dir, "yolov8n-face.pt"),
            conf_threshold=0.7  # Higher confidence for better faces
        )
        print("✅ Face detector loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load face detector: {e}")
        traceback.print_exc()
        return

    # Initialize face embedder
    try:
        print("🔄 Initializing face embedder...")
        embedder = FaceEmbedder(
            threshold=0.6,
            db_path=os.path.join(project_root, "embeddings.db"),
            debug_dir=debug_dir
        )
        print(f"✅ Face embedder loaded successfully")
        print(embedder.get_stats())
    except Exception as e:
        print(f"❌ Failed to load face embedder: {e}")
        traceback.print_exc()
        return

    # Open camera
    print("📹 Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera.")
        return

    print("✅ Camera opened successfully!")
    print("🎯 Press 'q' to quit, 's' to show stats")
    
    frame_idx = 0
    faces_processed = 0
    unique_faces_saved = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break

            # Detect faces
            boxes = detector.detect_faces(frame)
            
            # Draw rectangles and info
            annotated = detector.draw_rectangles(frame, boxes)
            
            # Add frame info
            info_text = f"Frame: {frame_idx} | Faces: {len(boxes)} | Saved: {unique_faces_saved}"
            cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show frame
            cv2.imshow("Face Recognition System", annotated)

            # Process each detected face
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for i, box in enumerate(boxes):
                faces_processed += 1
                
                # Extract face coordinates
                if len(box) == 5:  # x, y, w, h, confidence
                    x, y, w, h, conf = box
                else:  # x, y, w, h format
                    x, y, w, h = box[:4]
                    conf = 0.0
                
                # Add some padding (20% of face size)
                pad_w = int(w * 0.2)
                pad_h = int(h * 0.2)
                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(frame.shape[1], x + w + pad_w)
                y2 = min(frame.shape[0], y + h + pad_h)
                
                # Extract face image
                face_img = frame[y1:y2, x1:x2]
                
                # Skip empty or tiny faces
                if face_img.size == 0:
                    print(f"⚠️ Face {i} empty - skipping")
                    continue
                    
                if face_img.shape[0] < 50 or face_img.shape[1] < 50:
                    print(f"⚠️ Face {i} too small ({face_img.shape}) - skipping")
                    continue
                
                print(f"🔍 Processing face {i} (size: {face_img.shape}, conf: {conf:.2f})")
                
                # Get embedding
                emb = embedder.get_embedding(face_img)
                if emb is None:
                    print(f"❌ Could not get embedding for face {i}")
                    continue

                print(f"✅ Got embedding for face {i} (shape: {emb.shape})")
                
                # Check similarity
                best_id, best_name, similarity = embedder.find_similar(emb)
                
                if similarity < embedder.threshold:
                    # New face
                    person_name = f"Person_{unique_faces_saved + 1}"
                    person_id = embedder.register(emb, person_name)
                    
                    if person_id is not None:
                        # Save face image
                        fname = f"face_{person_id}_{timestamp}_{i}.jpg"
                        face_path = os.path.join(faces_dir, fname)
                        cv2.imwrite(face_path, face_img)
                        
                        unique_faces_saved += 1
                        print(f"🗂 NEW FACE! Saved as {fname}")
                        print(f"📊 Total unique faces saved: {unique_faces_saved}")
                    else:
                        print(f"❌ Failed to register face {i}")
                else:
                    print(f"🔁 Face {i} matches '{best_name}' (ID: {best_id}, similarity: {similarity:.3f})")

            frame_idx += 1
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 Quit requested")
                break
            elif key == ord('s'):
                print(f"📊 STATS: Frames: {frame_idx}, Faces processed: {faces_processed}, Unique saved: {unique_faces_saved}")
                print(embedder.get_stats())

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error in main loop: {e}")
        traceback.print_exc()
    finally:
        print("🧹 Cleaning up...")
        if 'embedder' in locals():
            embedder.close()
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        print(f"✅ Session complete! Processed {faces_processed} faces, saved {unique_faces_saved} unique faces")
        print(f"📁 Check the 'faces' folder for saved images")
        print(f"📁 Check 'debug_faces' for problematic images")
        print(f"🗄 Check 'embeddings.db' for the database")

if __name__ == "__main__":
    main()

