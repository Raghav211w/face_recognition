# KIET Face Attendance System

A comprehensive web-based face recognition attendance system designed for KIET college.

## Features

- **Live Face Detection**: Real-time face detection and recognition from webcam
- **Student Management**: Add and manage student records with personal details
- **Training System**: Automated face data collection with guided poses
- **Attendance Tracking**: Automatic attendance marking with confidence scores
- **Admin Dashboard**: Complete administrative interface
- **Progressive Training**: Captures 50+ photos across multiple poses for better accuracy

## Setup Instructions

### Prerequisites

- Python 3.7+
- Webcam access
- Modern web browser with camera permissions

### Installation

1. Clone or download the project:
```bash
cd face_attendance_project
```

2. Install dependencies:
```bash
pip3 install -r requirements.txt
```

3. Run the application:
```bash
python3 app.py
```

4. Open your web browser and go to: `http://localhost:5000`

## Usage Guide

### Admin Login
- Username: `admin`
- Password: `admin123`

### Adding Students
1. Log in as admin
2. Click on "Student" -> "Add Student"
3. Fill in student details (Name, Roll No, Department, Year)
4. Save the student record

### Training Face Data
1. From admin dashboard, click "Train Data"
2. Select a student from the dropdown
3. Click "Start Training"
4. Click "Start Auto Capture"
5. Follow the on-screen commands:
   - Look straight ahead (8 photos)
   - Look up (6 photos)
   - Look down (6 photos)
   - Look left (6 photos)
   - Look right (6 photos)
   - Tilt head left (4 photos)
   - Tilt head right (4 photos)
   - Smile (4 photos)
   - Neutral expression (4 photos)
   - Various angles (2 photos each)

6. The system will automatically capture 50 photos total
7. Training will complete automatically and update the model

### Live Attendance
1. Go to the home page
2. Click "Start Detection"
3. Position yourself in front of the camera
4. The system will detect faces and show student information
5. Attendance is automatically recorded (once per day per student)

### Viewing Attendance
1. From admin dashboard, click "Check Attendance"
2. Use filters to view by date or department
3. Export or view attendance records

## Technical Details

- **Face Recognition**: Uses `face_recognition` library with dlib backend
- **Database**: SQLite for data storage
- **Web Framework**: Flask with responsive Bootstrap UI
- **Camera**: WebRTC for real-time video processing
- **Confidence Threshold**: 60% minimum for recognition
- **Training Photos**: 50 photos across 13 different poses

## Security Features

- Face detection validation (single face only)
- Confidence scoring for accuracy
- Duplicate attendance prevention
- Secure admin authentication

## File Structure

```
face_attendance_project/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/            # HTML templates
│   ├── base.html         # Base template
│   ├── index.html        # Home page with live detection
│   ├── admin_login.html  # Admin login page
│   ├── admin_dashboard.html # Admin dashboard
│   ├── add_student.html  # Add student form
│   ├── train_data.html   # Training interface
│   └── check_attendance.html # Attendance records
└── face_attendance.db    # SQLite database (created automatically)
```

## Browser Compatibility

- Chrome (recommended)
- Firefox
- Safari
- Edge

Note: Camera access permissions must be granted for the system to work properly.

## Troubleshooting

1. **Camera not working**: Check browser permissions and ensure camera is not being used by another application
2. **Face not detected**: Ensure good lighting and face is clearly visible
3. **Training fails**: Make sure only one face is visible during training
4. **Database errors**: Delete `face_attendance.db` to reset the database

## Support

For issues or questions, please ensure you have the latest version of the dependencies installed and check the browser console for any error messages.
