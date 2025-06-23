# Face Detection Application

This project implements a face detection application using YOLOv8 for face detection and MTCNN for cropping faces from video frames. The application also utilizes SQLite for data storage, allowing for efficient management of visit records and fingerprints.

## Project Structure

```
face-detection-app
├── src
│   ├── main.py            # Entry point of the application
│   ├── yolo_detector.py    # YOLOv8 face detection implementation
│   ├── mtcnn_cropper.py    # MTCNN face cropping implementation
│   ├── database.py         # SQLite database management
│   └── utils.py           # Utility functions for logging and configuration
├── requirements.txt       # List of dependencies
├── README.md              # Project documentation
└── .env.example           # Template for environment variables
```

## Setup Instructions

To set up the environment on a Windows PC, follow these steps:

1. **Install Python**: Download and install Python (version 3.7 or higher) from the official website.

2. **Open Command Prompt**: Navigate to the project directory where the `face-detection-app` folder is located.

3. **Create a Virtual Environment**: Run the following command to create a virtual environment:
   ```
   python -m venv venv
   ```

4. **Activate the Virtual Environment**: Use the command below to activate the virtual environment:
   ```
   venv\Scripts\activate
   ```

5. **Install Required Packages**: Install the necessary packages by running:
   ```
   pip install -r requirements.txt
   ```

6. **Configure Environment Variables**: Rename `.env.example` to `.env` and fill in the required settings such as camera source, frame interval, save paths, and timeout durations.

7. **Run the Application**: Start the application by executing:
   ```
   python src/main.py
   ```

This will initiate the face detection application using YOLOv8 and MTCNN, with SQLite for data storage.

## Main Components

- **YOLOv8 Detector**: The `yolo_detector.py` file contains the `YoloDetector` class, which is responsible for detecting faces in video frames using the YOLOv8 model.

- **MTCNN Cropper**: The `mtcnn_cropper.py` file includes the `MTCNNCropper` class, which crops detected faces from the frames using the MTCNN algorithm.

- **Database Management**: The `database.py` file handles all SQLite database operations, including initializing the database, inserting visit records, and checking for existing fingerprints.

- **Utility Functions**: The `utils.py` file provides various utility functions for logging events, reading settings from the configuration file, and managing file paths for saving images and logs.

## License

This project is licensed under the MIT License.