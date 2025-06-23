
import os
import cv2
import torch
from ultralytics import YOLO

class YOLOFaceDetector:
    def __init__(
        self,
        model_source: str = "models/yolov8n-face.pt",
        conf_threshold: float = 0.5
    ):
        # Check if model exists
        if not os.path.exists(model_source):
            print(f"⚠️ Model not found at {model_source}, downloading...")
            try:
                # Create models directory if doesn't exist
                os.makedirs(os.path.dirname(model_source), exist_ok=True)
                # Download model
                model = YOLO("yolov8n-face.pt")
                model.export(format="pt")
                if os.path.exists("yolov8n-face.pt"):
                    os.rename("yolov8n-face.pt", model_source)
                    print(f"✅ Downloaded model to {model_source}")
                else:
                    print("❌ Failed to save downloaded model")
                    model_source = "yolov8n-face.pt"
            except Exception as e:
                print(f"❌ Failed to download model: {e}")
                # Use pretrained model as fallback
                print("🔄 Using pretrained model from Ultralytics hub")
                model_source = "yolov8n-face.pt"
        
        print(f"🔄 Loading YOLO model: {model_source}")
        
        # Load with explicit weights_only=False
        try:
            # First try standard loading
            self.model = YOLO(model_source)
        except Exception as e:
            print(f"⚠️ Standard load failed: {e}")
            print("🔄 Trying alternative loading method")
            try:
                # Alternative loading method
                ckpt = torch.load(model_source, map_location="cpu", weights_only=False)
                self.model = YOLO(model=ckpt)
            except Exception as e2:
                print(f"❌ Alternative load failed: {e2}")
                print("🔄 Trying default model")
                self.model = YOLO("yolov8n.pt")
        
        self.conf_threshold = conf_threshold
        print(f"✅ YOLO model loaded with confidence threshold: {conf_threshold}")

    def detect_faces(self, frame):
        results = self.model.predict(
            frame,
            imgsz=640,
            conf=self.conf_threshold,
            verbose=False
        )
        
        boxes = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = float(box.conf[0])
                
                # Convert to x, y, w, h format
                boxes.append((x1, y1, x2 - x1, y2 - y1, confidence))
        
        return boxes

    def draw_rectangles(self, frame, boxes):
        annotated = frame.copy()
        for box in boxes:
            if len(box) == 5:  # x, y, w, h, confidence
                x, y, w, h, conf = box
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Add confidence text
                text = f"Face: {conf:.2f}"
                cv2.putText(annotated, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:  # x, y, w, h format (backward compatibility)
                x, y, w, h = box[:4]
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return annotated
