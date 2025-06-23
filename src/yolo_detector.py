class YoloDetector:
    def __init__(self, model_path='yolov8_face.pt'):
        from ultralytics import YOLO
        self.model = YOLO(model_path)

    def detect_faces(self, frame):
        results = self.model(frame)
        boxes = []
        for result in results:
            for box in result.boxes:
                boxes.append(box.xyxy[0].tolist())  # Get bounding box coordinates
        return boxes