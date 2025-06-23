import cv2
from yolo_detector import YoloDetector
from mtcnn_cropper import MTCNNCropper
from database import Database
from utils import load_settings

def main():
    settings = load_settings()
    camera_source = settings['camera_source']
    
    # Initialize the YOLO detector and MTCNN cropper
    yolo_detector = YoloDetector()
    mtcnn_cropper = MTCNNCropper()
    
    # Initialize the database
    db = Database()
    db.initialize()

    # Open the camera
    cap = cv2.VideoCapture(camera_source)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Detect faces using YOLO
        boxes = yolo_detector.detect_faces(frame)

        # Crop faces using MTCNN
        cropped_faces = mtcnn_cropper.crop_faces(frame, boxes)

        # Process the cropped faces (e.g., save to database, display, etc.)
        for face in cropped_faces:
            # Here you can add code to save the face or perform other actions
            pass

        # Display the frame with detected faces
        for box in boxes:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()