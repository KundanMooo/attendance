# 🎯 Face Attendance System - Single File Solution

A complete, optimized real-time face attendance system in one Python file with automatic database management, face detection, recognition, and attendance tracking.

## ✨ Features

- **Single File**: Everything in one optimized Python file
- **Multiple Detectors**: YOLO, MTCNN, or OpenCV Haar Cascades
- **Face Recognition**: InsightFace embeddings for accurate identification
- **Auto Database**: SQLite with automatic schema creation
- **Smart Attendance**: Automatic check-in/check-out with timeout handling
- **Real-time Logging**: Comprehensive logging and statistics
- **Configurable**: JSON configuration with sensible defaults
- **Robust**: Graceful fallbacks when dependencies are missing

## 🚀 Quick Start

### Method 1: Automated Setup (Windows)

1. **Download Files**:
   ```bash
   # Download these files to a new folder:
   # - face_attendance_system.py
   # - setup.bat
   # - requirements.txt
   # - config.json
   ```

2. **Run Setup**:
   ```bash
   # Double-click setup.bat or run in PowerShell:
   .\setup.bat
   ```

3. **Start System**:
   ```bash
   # In the same directory:
   venv\Scripts\activate
   python face_attendance_system.py
   ```

### Method 2: Manual Setup

1. **Create Environment**:
   ```bash
   mkdir face_attendance
   cd face_attendance
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

2. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install "numpy>=1.24.0,<2.0"
   pip install opencv-python>=4.8.0
   pip install ultralytics>=8.0.0
   pip install insightface>=0.7.3
   pip install facenet-pytorch>=2.5.0
   pip install Pillow>=9.0.0
   pip install onnxruntime>=1.15.0
   ```

3. **Run System**:
   ```bash
   python face_attendance_system.py
   ```

## 🎮 Usage

### Basic Commands

```bash
# Run the system
python face_attendance_system.py

# Initialize database only
python face_attendance_system.py --init-db

# Show statistics
python face_attendance_system.py --stats

# Use custom config
python face_attendance_system.py --config my_config.json
```

### Controls During Runtime

- **`q`**: Quit the system
- **`s`**: Show current statistics
- **`Ctrl+C`**: Emergency stop

## ⚙️ Configuration

Edit `config.json` to customize behavior:

```json
{
  "video_source": 0,              // 0=webcam, "video.mp4"=file
  "detector": "yolo",             // "yolo", "mtcnn", "haar"
  "similarity_threshold": 0.65,   // Face matching threshold
  "confidence_threshold": 0.7,    // Detection confidence
  "min_face_size": 50,           // Minimum face size in pixels
  "exit_timeout_minutes": 5,     // Auto checkout timeout
  "display_video": true,         // Show video window
  "save_crops": true,           // Save face crops to disk
  "skip_frames": 3              // Process every Nth frame
}
```

### Video Sources

```json
"video_source": 0           // Default webcam
"video_source": 1           // Second camera
"video_source": "video.mp4" // Video file
"video_source": "rtsp://..." // IP camera
```

## 📊 Database Schema

The system automatically creates SQLite tables:

### Persons Table
- `person_id`: Unique identifier
- `name`: Optional name (NULL for auto-registered)
- `embedding`: Face embedding blob
- `registered_at`: Registration timestamp
- `last_seen`: Last detection timestamp
- `visit_count`: Total visit count

### Attendance Table
- `attendance_id`: Unique record ID
- `person_id`: Reference to person
- `check_in`: Entry timestamp
- `check_out`: Exit timestamp (NULL if still present)
- `session_duration`: Session length in seconds

## 🛠️ Troubleshooting

### Common Issues

**1. ImportError: numpy.core.multiarray failed to import**
```bash
pip install --force-reinstall "numpy>=1.24.0,<2.0"
pip install --force-reinstall opencv-python
```

**2. YOLO model not found**
- The system will auto-download YOLOv8 on first run
- Or manually download: `yolo export model=yolov8n-face.pt`

**3. No camera detected**
- Check `video_source` in config.json
- Try different camera indices (0, 1, 2...)
- Ensure camera permissions are granted

**4. Poor face detection**
- Adjust `confidence_threshold` (lower = more detections)
- Change `detector` type in config
- Ensure good lighting conditions

**5. False face matches**
- Increase `similarity_threshold` (higher = stricter matching)
- Check face crop quality in logs/faces/

### Performance Optimization

**For faster processing**:
```json
{
  "skip_frames": 5,           // Process fewer frames
  "resolution": [320, 240],   // Lower resolution
  "detector": "haar",         // Faster detector
  "save_crops": false         // Disable crop saving
}
```

**For better accuracy**:
```json
{
  "skip_frames": 1,           // Process all frames
  "resolution": [1280, 720],  // Higher resolution
  "detector": "yolo",         // Better detector
  "min_face_size": 80         // Larger minimum face size
}
```

## 📁 Output Structure

```
face_attendance/
├── face_attendance_system.py   # Main system file
├── config.json                 # Configuration
├── data/
│   └── attendance.db           # SQLite database
├── logs/
│   ├── system.log             # System logs
│   └── faces/                 # Face crops by date
│       ├── 2025-06-23/
│       │   ├── person_1_10-30-15-123.jpg
│       │   └── person_2_10-31-22-456.jpg
│       └── 2025-06-24/
└── venv/                      # Virtual environment
```

## 🔧 Advanced Usage

### Custom Names for Persons

```python
# You can manually assign names in the database
import sqlite3
conn = sqlite3.connect('data/attendance.db')
cursor = conn.cursor()
cursor.execute("UPDATE persons SET name = ? WHERE person_id = ?", ("John Doe", 1))
conn.commit()
```

### Export Attendance Data

```sql
-- Daily attendance report
SELECT p.person_id, p.name, a.check_in, a.check_out, a.session_duration
FROM persons p
JOIN attendance a ON p.person_id = a.person_id
WHERE DATE(a.check_in) = '2025-06-23';
```

### Integration with Other Systems

The system exposes a simple API through the DatabaseManager class for integration with web applications, HR systems, etc.

## 📋 System Requirements

- **Python**: 3.8+
- **OS**: Windows 10+, Linux, macOS
- **RAM**: 4GB minimum, 8GB recommended
- **Camera**: USB webcam or built-in camera
- **Storage**: 1GB for dependencies, varies for face crops and logs

## 🤝 Support

If you encounter issues:

1. Check the log file at `logs/system.log`
2. Verify your camera works with other applications
3. Try different detector types in config
4. Ensure all dependencies are correctly installed
5. Check Python and pip versions

## 📄 License

This project is provided as-is for educational and commercial use. Ensure compliance with local privacy laws when using facial recognition technology.