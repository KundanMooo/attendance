class MTCNNCropper:
    def __init__(self):
        from mtcnn import MTCNN
        self.detector = MTCNN()

    def crop_faces(self, frame, boxes):
        cropped_faces = []
        for box in boxes:
            x1, y1, x2, y2 = box
            cropped_face = frame[y1:y2, x1:x2]
            cropped_faces.append(cropped_face)
        return cropped_faces