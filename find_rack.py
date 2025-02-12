import sys
from PyQt5 import QtWidgets, QtGui, QtCore, uic
import cv2
from ultralytics import YOLO


class RackDetectionApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(RackDetectionApp, self).__init__()
        # Load the UI file
        uic.loadUi('find_rack.ui', self)  # Replace 'find_rack.ui' with your actual file path

        # Load YOLO model
        self.model = YOLO('models/yolov8s.pt')

        # Define manual rack mapping (excluding 'person')
        self.rack_mapping = {
            'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5,
            'bus': 6, 'train': 7, 'truck': 8, 'boat': 9, 'traffic light': 10,
            'fire hydrant': 11, 'stop sign': 12, 'parking meter': 13, 'bench': 14,
            'bird': 15, 'cat': 16, 'dog': 17, 'horse': 18, 'sheep': 19,
            'cow': 20, 'elephant': 21, 'bear': 22, 'zebra': 23, 'giraffe': 24,
            'backpack': 25, 'umbrella': 26, 'handbag': 27, 'tie': 28, 'suitcase': 29,
            'frisbee': 30, 'skis': 31, 'snowboard': 32, 'sports ball': 33, 'kite': 34,
            'baseball bat': 35, 'baseball glove': 36, 'skateboard': 37, 'surfboard': 38,
            'tennis racket': 39, 'bottle': 40, 'wine glass': 41, 'cup': 42, 'fork': 43,
            'knife': 44, 'spoon': 45, 'bowl': 46, 'banana': 47, 'apple': 48, 'sandwich': 49,
            'orange': 50, 'broccoli': 51, 'carrot': 52, 'hot dog': 53, 'pizza': 54,
            'donut': 55, 'cake': 56, 'chair': 57, 'couch': 58, 'potted plant': 59,
            'bed': 60, 'dining table': 61, 'toilet': 62, 'tv': 63, 'laptop': 64,
            'mouse': 65, 'remote': 66, 'keyboard': 67, 'cell phone': 68, 'microwave': 69,
            'oven': 70, 'toaster': 71, 'sink': 72, 'refrigerator': 73, 'book': 74,
            'clock': 75, 'vase': 76, 'scissors': 77, 'teddy bear': 78, 'hair drier': 79,
            'toothbrush': 80
        }

        # Start video capture
        self.cap = cv2.VideoCapture(0)

        # Create a timer to periodically update the frame
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30 ms interval
        self.lbl_results = self.findChild(QtWidgets.QLabel, "lbl_results")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Run YOLO model on frame
        results = self.model(frame)
        detections = results[0].boxes.data

        detected_objects = []  # List to store detected object names with rack numbers

        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection.tolist()
            class_id = int(class_id)
            class_name = self.model.names.get(class_id, "Unknown")

            # Skip detection if it's a 'person'
            if class_name == 'person':
                continue

            rack_number = self.rack_mapping.get(class_name, "Unknown")

            # Add detected object and rack info to list
            detected_objects.append(f"{class_name} (Rack {rack_number})")

            # Draw bounding box and label on frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{class_name} (Rack {rack_number})"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Update lbl_results with detected objects
        self.lbl_results.setText("\n".join(detected_objects) if detected_objects else "No objects detected.")

        # Convert frame to QImage for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_frame.shape
        bytes_per_line = channel * width
        q_img = QtGui.QImage(rgb_frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)

        # Set the QImage on the camera_feed_lbl
        self.camera_feed_lbl.setPixmap(QtGui.QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        # Release camera and stop timer on close
        self.cap.release()
        self.timer.stop()
        super(RackDetectionApp, self).closeEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = RackDetectionApp()
    main_window.show()
    sys.exit(app.exec_())
