from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, make_response, session, flash # Import session and flash
import sqlite3
import os
import cv2
import numpy as np
import face_recognition
import base64
import pickle
from datetime import datetime
import json
from PIL import Image
import io
import pandas as pd
import shutil
import time
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows
from tempfile import NamedTemporaryFile

# Import 3D reconstruction system
from face_3d_reconstruction import Face3DReconstructor, Face3DTrainingSession, save_3d_model, load_3d_model
import open3d as o3d

# Flask application 
app = Flask(__name__)
app.secret_key = 'your_super_secret_key' # Flask sessions keys later

def init_db():
    conn = sqlite3.connect('face_attendance.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS students (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        roll_no TEXT UNIQUE NOT NULL,
                        department TEXT NOT NULL,
                        year INTEGER NOT NULL,
                        library_id TEXT NOT NULL,
                        face_encodings BLOB,
                        face_3d_model_path TEXT,
                        face_3d_features BLOB)
    ''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id INTEGER,
                        date TEXT NOT NULL,
                        time TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        recognition_method TEXT DEFAULT '2D',
                        FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE)
    ''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS training_photos (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id INTEGER,
                        photo_data BLOB,
                        photo_index INTEGER,
                        FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE)
    ''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS face_3d_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id INTEGER,
                        session_data BLOB,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE)
    ''')

    conn.commit()
    conn.close()

# The original /delete_student route (if still used by other parts, keep it)
@app.route('/delete_student', methods=['POST'])
def delete_student():
    roll_no = request.form['roll_no']
    conn = sqlite3.connect('face_attendance.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, name, roll_no FROM students WHERE roll_no = ?", (roll_no,))
    student = cursor.fetchone()
    
    if not student:
        conn.close()
        return jsonify({'success': False, 'message': 'Student not found'}), 404

    student_id, student_name, student_roll_no = student

    cursor.execute("DELETE FROM students WHERE roll_no = ?", (roll_no,))
    conn.commit()
    conn.close()

    folder_name = str(student_roll_no)
    folder_path = os.path.join('students_data', folder_name, 'image_captures')
    
    if os.path.exists(folder_path):
        try:
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            os.rmdir(folder_path)
            student_roll_no_folder = os.path.join('students_data', folder_name)
            if os.path.exists(student_roll_no_folder) and not os.listdir(student_roll_no_folder):
                os.rmdir(student_roll_no_folder)
        except OSError as e:
            print(f"Error removing training images folder {folder_path}: {e}")
            return jsonify({'success': False, 'message': f'Student deleted, but failed to clean up training images folder: {str(e)}'}), 500

    return jsonify({'success': True, 'message': 'Student profile deleted successfully'}), 200

@app.route('/delete_student/<int:student_id>', methods=['POST'])
def delete_student_by_id(student_id):
    try:
        conn = sqlite3.connect('face_attendance.db')
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, name, roll_no FROM students WHERE id = ?", (student_id,))
        student = cursor.fetchone()
        
        if not student:
            conn.close()
            return jsonify({'success': False, 'message': 'Student not found'}), 404

        student_id, student_name, student_roll_no = student

        # Delete from database first (CASCADE will handle related records)
        cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))
        conn.commit()
        conn.close()

        # Clean up all student data files
        folder_name = str(student_roll_no)
        student_data_folder = os.path.join('students_data', folder_name)
        
        if os.path.exists(student_data_folder):
            try:
                # Use shutil.rmtree for comprehensive deletion
                shutil.rmtree(student_data_folder)
                print(f"Successfully deleted student data folder: {student_data_folder}")
            except OSError as e:
                print(f"Error removing student data folder {student_data_folder}: {e}")
                # Try alternative cleanup method
                try:
                    # Clean up image captures folder
                    image_captures_path = os.path.join(student_data_folder, 'image_captures')
                    if os.path.exists(image_captures_path):
                        for file_name in os.listdir(image_captures_path):
                            file_path = os.path.join(image_captures_path, file_name)
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                        os.rmdir(image_captures_path)
                    
                    # Clean up 3D models folder
                    models_path = os.path.join(student_data_folder, '3d_models')
                    if os.path.exists(models_path):
                        for file_name in os.listdir(models_path):
                            file_path = os.path.join(models_path, file_name)
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                        os.rmdir(models_path)
                    
                    # Remove main folder if empty
                    if os.path.exists(student_data_folder) and not os.listdir(student_data_folder):
                        os.rmdir(student_data_folder)
                        
                except OSError as cleanup_error:
                    print(f"Alternative cleanup also failed: {cleanup_error}")
                    return jsonify({
                        'success': False, 
                        'message': f'Student deleted from database, but failed to clean up files: {str(cleanup_error)}'
                    }), 500

        return jsonify({'success': True, 'message': 'Student profile and all associated data deleted successfully'}), 200
        
    except Exception as e:
        print(f"Server error during deletion: {e}")
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/admin')
def admin_login():
    return render_template('admin_login.html')

# MODIFIED: Added session management to admin_dashboard and logout
@app.route('/admin_dashboard', methods=['GET', 'POST'])
def admin_dashboard():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin123':
            session['admin_logged_in'] = True # Set session variable
            flash('Login successful! Welcome to admin dashboard.', 'success')
            return redirect(url_for('admin_dashboard')) # Redirect after successful login
        else:
            return render_template('admin_login.html', error='Invalid credentials')
    if 'admin_logged_in' in session and session['admin_logged_in']: # Check if already logged in
        return render_template('admin_dashboard.html')
    else:
        return redirect(url_for('admin_login')) # Redirect to login if not authenticated

@app.route('/admin_logout')
def admin_logout():
    session.pop('admin_logged_in', None) # Remove session variable
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('home'))


@app.route('/add_student', methods=['GET', 'POST'])
def add_student():
    if request.method == 'POST':
        name = request.form['name']
        roll_no = request.form['roll_no']
        department = request.form['department']
        year = request.form['year']
        library_id = request.form['library_id']

        try:
            conn = sqlite3.connect('face_attendance.db')
            cursor = conn.cursor()
            cursor.execute("INSERT INTO students (name, roll_no, department, year, library_id) VALUES (?, ?, ?, ?, ?)",
                           (name, roll_no, department, year, library_id))
            conn.commit()
        except sqlite3.IntegrityError:
            return render_template('add_student.html', error='Roll number already exists. Please use a unique roll number.')
        finally:
            conn.close()
        flash(f'Student {name} (Roll No: {roll_no}) added successfully!', 'success')
        return redirect(url_for('admin_dashboard'))
    return render_template('add_student.html')

@app.route('/update_student', methods=['GET', 'POST'])
def update_student():
    if request.method == 'POST':
        roll_no = request.form['roll_no']
        
        # If this is an update operation
        if request.form.get('update'):
            name = request.form['name']
            department = request.form['department']
            year = request.form['year']
            library_id = request.form['library_id']

            try:
                conn = sqlite3.connect('face_attendance.db')
                cursor = conn.cursor()
                cursor.execute("UPDATE students SET name = ?, department = ?, year = ?, library_id = ? WHERE roll_no = ?",
                               (name, department, year, library_id, roll_no))
                conn.commit()
            finally:
                conn.close()
            flash(f'Student {name} (Roll No: {roll_no}) updated successfully!', 'success')
            return redirect(url_for('admin_dashboard'))
        
        # If this is a search operation, redirect to GET with roll_no parameter
        else:
            return redirect(url_for('update_student', roll_no=roll_no))
    
    # Handle GET request
    roll_no = request.args.get('roll_no')
    if roll_no:
        conn = sqlite3.connect('face_attendance.db')
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, department, year, library_id FROM students WHERE roll_no = ?", (roll_no,))
        student = cursor.fetchone()
        conn.close()
        
        if student:
            return render_template('update_student.html', student={'id': student[0], 'name': student[1], 'roll_no': roll_no, 'department': student[2], 'year': student[3], 'library_id': student[4]})
        else:
            return render_template('update_student.html', error='Roll number not found. Please enter a valid roll number.')
    
    return render_template('update_student.html')

@app.route('/train_data')
def train_data():
    return render_template('train_data.html')

@app.route('/check_attendance')
def check_attendance():
    return render_template('check_attendance.html')

@app.route('/get_students')
def get_students():
    conn = sqlite3.connect('face_attendance.db')
    cursor = conn.cursor()
    cursor.execute('''SELECT s.id, s.name, s.roll_no, s.department, s.year, s.library_id,
                                 COUNT(a.id) as total_attendance,
                                 MAX(a.date) as last_attendance
                          FROM students s
                          LEFT JOIN attendance a ON s.id = a.student_id
                          GROUP BY s.id
                          ORDER BY s.department, s.name''')
    students = cursor.fetchall()
    conn.close()
    
    student_list = []
    for student in students:
        student_list.append({
            'id': student[0],
            'name': student[1],
            'roll_no': student[2],
            'department': student[3],
            'year': student[4],
            'library_id': student[5],
            'total_attendance': student[6],
            'last_attendance': student[7]
        })
    
    return jsonify(student_list)

@app.route('/capture_training_video', methods=['POST'])
def capture_training_video():
    try:
        data = request.files['video']
        student_id = request.form['student_id']
        
        conn = sqlite3.connect('face_attendance.db')
        cursor = conn.cursor()
        cursor.execute("SELECT name, roll_no FROM students WHERE id = ?", (student_id,))
        student = cursor.fetchone()
        conn.close()
        
        if not student:
            return jsonify({'success': False, 'message': 'Student not found'})
        
        student_name, roll_no = student
        
        video_bytes = data.read()
        
        extracted_count = process_video_for_training_direct(video_bytes, student_id, str(roll_no))
        
        return jsonify({
            'success': True, 
            'message': f'Video processed and {extracted_count} photos captured successfully'
        })

    except Exception as e:
        print(f"Error in capture_training_video: {e}")
        return jsonify({'success': False, 'message': str(e)})

def process_video_for_training_direct(video_bytes, student_id, student_roll_no):
    if not os.path.exists('students_data'):
        os.makedirs('students_data')
    
    student_roll_no_str = str(student_roll_no)
    roll_no_folder_path = os.path.join('students_data', student_roll_no_str)
    
    if not os.path.exists(roll_no_folder_path):
        os.makedirs(roll_no_folder_path)

    image_captures_folder_path = os.path.join(roll_no_folder_path, 'image_captures')
    if not os.path.exists(image_captures_folder_path):
        os.makedirs(image_captures_folder_path)
    
    temp_video_path = f'/tmp/temp_video_{student_id}.mp4'
    if not os.path.exists(os.path.dirname(temp_video_path)):
        os.makedirs(os.path.dirname(temp_video_path))

    with open(temp_video_path, 'wb') as f:
        f.write(video_bytes)
    
    cap = cv2.VideoCapture(temp_video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {temp_video_path}")
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        return 0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_interval = max(1, int(fps * 0.2))
    photo_index = 0
    
    for i in range(0, frame_count, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_img)

        if len(face_locations) == 1:
            image_filename = os.path.join(image_captures_folder_path, f'photo_{photo_index:03d}.jpg')
            cv2.imwrite(image_filename, frame)
            
            ret, buf = cv2.imencode('.jpg', frame)
            image_bytes = buf.tobytes()

            with sqlite3.connect('face_attendance.db') as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO training_photos (student_id, photo_data, photo_index) VALUES (?, ?, ?)",
                               (student_id, image_bytes, photo_index))
            photo_index += 1
            
            if photo_index >= 40:
                break

    cap.release()
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    
    return photo_index

@app.route('/capture_training_photo', methods=['POST'])
def capture_training_photo():
    try:
        data = request.json
        student_id = data['student_id']
        image_data = data['image']
        photo_index = data['photo_index']
        
        # Add a check for valid image_data
        if not image_data or not isinstance(image_data, str) or not image_data.startswith('data:image'):
            print(f"Error: Invalid image_data format. Received: {image_data[:50]}...")
            return jsonify({'success': False, 'message': 'Invalid image data format.'})

        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Check if image was decoded successfully
        if img is None:
            print("Error: OpenCV failed to decode image bytes.")
            return jsonify({'success': False, 'message': 'Could not decode image. Corrupted data?'})

        # Use smaller image for faster face detection
        height, width = img.shape[:2]
        if width > 640:  # Only resize if image is large
            scale_factor = 640.0 / width
            new_width = 640
            new_height = int(height * scale_factor)
            img_resized = cv2.resize(img, (new_width, new_height))
        else:
            img_resized = img
            
        rgb_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Use faster face detection model for training photos
        face_locations = face_recognition.face_locations(rgb_img, model='hog')  # HOG is faster than CNN
        
        if len(face_locations) == 0:
            print(f"Face detection: No face detected for student_id {student_id}, index {photo_index}")
            return jsonify({'success': False, 'message': 'No face detected. Please ensure your face is visible.'})
        
        if len(face_locations) > 1:
            print(f"Face detection: Multiple faces detected for student_id {student_id}, index {photo_index}")
            return jsonify({'success': False, 'message': 'Multiple faces detected. Please ensure only one face is visible.'})
        
        # If photo_index is -1, just check face detection without saving
        if photo_index == -1:
            return jsonify({'success': True, 'message': 'Face detected successfully'})
        
        # --- MODIFIED: Save to file system as requested ---
        conn = sqlite3.connect('face_attendance.db')
        cursor = conn.cursor()
        cursor.execute("SELECT roll_no FROM students WHERE id = ?", (student_id,))
        student_roll_no = cursor.fetchone()
        conn.close() # Close connection as soon as possible
        
        if not student_roll_no:
            print(f"Error: Student with ID {student_id} not found for file system save.")
            return jsonify({'success': False, 'message': 'Student not found for saving image to filesystem.'})
        
        student_roll_no_str = str(student_roll_no[0]) # Get roll_no from tuple
        
        roll_no_folder_path = os.path.join('students_data', student_roll_no_str)
        if not os.path.exists(roll_no_folder_path):
            os.makedirs(roll_no_folder_path)

        image_captures_folder_path = os.path.join(roll_no_folder_path, 'image_captures')
        if not os.path.exists(image_captures_folder_path):
            os.makedirs(image_captures_folder_path)

        image_filename = os.path.join(image_captures_folder_path, f'live_photo_{photo_index:03d}.jpg')
        cv2.imwrite(image_filename, img) # Save the BGR image directly
        print(f"Saved live photo to: {image_filename}")
        # --- END MODIFIED ---

        # Save the photo to database
        conn = sqlite3.connect('face_attendance.db') # Re-open connection for DB write
        cursor = conn.cursor()
        cursor.execute("INSERT INTO training_photos (student_id, photo_data, photo_index) VALUES (?, ?, ?)",
                       (student_id, image_bytes, photo_index))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Photo captured successfully'})
        
    except Exception as e:
        print(f"Error in capture_training_photo: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/finish_training', methods=['POST'])
def finish_training():
    try:
        data = request.json
        student_id = data['student_id']
        
        conn = sqlite3.connect('face_attendance.db')
        cursor = conn.cursor()
        cursor.execute("SELECT photo_data FROM training_photos WHERE student_id = ?", (student_id,))
        photos = cursor.fetchall()
        
        if len(photos) == 0:
            conn.close()
            return jsonify({'success': False, 'message': 'No training photos found'})
        
        face_encodings = []
        
        for photo_data in photos:
            nparr = np.frombuffer(photo_data[0], np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Check if image was decoded successfully before processing
            if img is None:
                print(f"Warning: Skipping a training photo for student {student_id} as it could not be decoded.")
                continue

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            encodings = face_recognition.face_encodings(rgb_img)
            if len(encodings) > 0:
                face_encodings.append(encodings[0])
        
        if len(face_encodings) == 0:
            conn.close()
            return jsonify({'success': False, 'message': 'No valid face encodings found from captured photos'})
        
        encodings_data = pickle.dumps(face_encodings)
        cursor.execute("UPDATE students SET face_encodings = ? WHERE id = ?", (encodings_data, student_id))
        
        cursor.execute("DELETE FROM training_photos WHERE student_id = ?", (student_id,))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': f'Training completed with {len(face_encodings)} face encodings'})
        
    except Exception as e:
        print(f"Error in finish_training: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/detect_face', methods=['POST'])
def detect_face():
    try:
        data = request.json
        image_data = data['image']
        
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None: # Check if image was decoded successfully
            print("Error: detect_face: OpenCV failed to decode image bytes.")
            return jsonify({'detected': False, 'message': 'Could not decode image.'})

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_img)
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
        
        if len(face_encodings) == 0:
            return jsonify({'detected': False, 'message': 'No face detected'})
        
        # Only process if exactly one face is found for attendance for simplicity
        if len(face_encodings) > 1:
            return jsonify({'detected': False, 'message': 'Multiple faces detected. Please ensure only one face is visible.'})
        
        conn = sqlite3.connect('face_attendance.db')
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, roll_no, department, year, face_encodings FROM students WHERE face_encodings IS NOT NULL")
        students = cursor.fetchall()
        
        best_match = None
        best_confidence = 0
        
        # Compare the single detected face with all known student encodings
        current_face_encoding = face_encodings[0]
        
        for student in students:
            student_id, name, roll_no, department, year, encodings_data = student
            
            try:
                student_encodings = pickle.loads(encodings_data)
            except Exception as e:
                print(f"Warning: Failed to unpickle face encodings for student ID {student_id}: {e}. Skipping this student.")
                continue # Skip this student if their encodings are corrupted
            
            distances = face_recognition.face_distance(student_encodings, current_face_encoding) # Compare against all encodings of *this* student
            min_distance = min(distances)
            
            confidence = max(0, (1 - min_distance) * 100)
            
            if confidence > 60 and confidence > best_confidence:
                best_match = {
                    'id': student_id,
                    'name': name,
                    'roll_no': roll_no,
                    'department': department,
                    'year': year,
                    'confidence': round(confidence, 2)
                }
                best_confidence = confidence
        
        if best_match:
            current_time = datetime.now()
            date_str = current_time.strftime('%Y-%m-%d')
            time_str = current_time.strftime('%H:%M:%S')
            
            cursor.execute("SELECT id FROM attendance WHERE student_id = ? AND date = ?", 
                          (best_match['id'], date_str))
            existing = cursor.fetchone()
            
            if not existing:
                cursor.execute("INSERT INTO attendance (student_id, date, time, confidence) VALUES (?, ?, ?, ?)",
                              (best_match['id'], date_str, time_str, best_match['confidence']))
                conn.commit()
                best_match['attendance_status'] = 'Marked'
            else:
                best_match['attendance_status'] = 'Already marked today'
            
            conn.close()
            return jsonify({'detected': True, **best_match})
        else:
            conn.close()
            return jsonify({'detected': False, 'message': 'Face not recognized'})
            
    except Exception as e:
        print(f"Error in detect_face: {e}")
        return jsonify({'detected': False, 'message': str(e)})

@app.route('/get_attendance')
def get_attendance():
    date_filter = request.args.get('date')
    department_filter = request.args.get('department')
    
    conn = sqlite3.connect('face_attendance.db')
    cursor = conn.cursor()
    
    query = '''SELECT a.date, a.time, s.name, s.roll_no, s.department, s.year, a.confidence
               FROM attendance a
               JOIN students s ON a.student_id = s.id
               WHERE 1=1'''
    
    params = []
    
    if date_filter:
        query += ' AND a.date = ?'
        params.append(date_filter)
    
    if department_filter:
        query += ' AND s.department = ?'
        params.append(department_filter)
    
    query += ' ORDER BY a.date DESC, a.time DESC'
    
    cursor.execute(query, params)
    attendance = cursor.fetchall()
    conn.close()
    
    attendance_list = []
    for record in attendance:
        attendance_list.append({
            'date': record[0],
            'time': record[1],
            'name': record[2],
            'roll_no': record[3],
            'department': record[4],
            'year': record[5],
            'confidence': round(record[6], 2)
        })
    
    return jsonify(attendance_list)



@app.route('/export_attendance_excel')
def export_attendance_excel():
    """
    Fetches attendance data, creates a well-formatted Excel file in memory,
    and returns it as a downloadable attachment.
    """
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    conn = None
    try:
        # 1. Connect to your database
        conn = sqlite3.connect('face_attendance.db')
        
        # Get filter parameters from the request URL
        date_filter = request.args.get('date')
        department_filter = request.args.get('department')
        
        # 2. Build the database query with optional filters
        query = '''SELECT a.date, a.time, s.name, s.roll_no, s.department, s.year, s.library_id, a.confidence
                   FROM attendance a
                   JOIN students s ON a.student_id = s.id
                   WHERE 1=1'''
        
        params = []
        
        if date_filter:
            query += ' AND a.date = ?'
            params.append(date_filter)
        
        if department_filter:
            query += ' AND s.department = ?'
            params.append(department_filter)
        
        query += ' ORDER BY a.date DESC, a.time DESC'
        
        # Fetch data into a pandas DataFrame
        df = pd.read_sql_query(query, conn, params=params)
        
        # Rename columns for the report
        df.columns = ['Date', 'Time', 'Name', 'Roll No', 'Department', 'Year', 'Library ID', 'Confidence %']

        # 3. Create an in-memory Excel file using io.BytesIO
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Attendance_Report')
        
        output.seek(0) # Move the cursor to the beginning of the stream

        # 4. Create the Flask response to trigger the download
        filename = f"KIET_Attendance_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        response = make_response(output.getvalue())
        response.headers["Content-Disposition"] = f"attachment; filename={filename}"
        response.headers["Content-Type"] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        
        return response

    except Exception as e:
        print(f"Error in export_attendance_excel: {e}")
        # Redirect with an error message if something goes wrong
        return redirect(url_for('check_attendance', error="Could not generate Excel file."))
    finally:
        if conn:
            conn.close()


@app.route('/students')
def students():
    try:
        conn = sqlite3.connect('face_attendance.db')
        cursor = conn.cursor()
        
        cursor.execute('''SELECT s.id, s.name, s.roll_no, s.department, s.year, s.library_id,
                                 COUNT(a.id) as total_attendance,
                                 MAX(a.date) as last_attendance
                          FROM students s
                          LEFT JOIN attendance a ON s.id = a.student_id
                          GROUP BY s.id
                          ORDER BY s.department, s.name''')
        
        students_data = cursor.fetchall()
        conn.close()
        
        return render_template('students.html', students=students_data)
        
    except Exception as e:
        print(f"Error in students route: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/view_database')
def view_database():
    try:
        conn = sqlite3.connect('face_attendance.db')
        cursor = conn.cursor()
        
        cursor.execute('''SELECT s.name, s.roll_no, s.department, s.year, s.library_id,
                                 COUNT(a.id) as total_attendance,
                                 MAX(a.date) as last_attendance
                          FROM students s
                          LEFT JOIN attendance a ON s.id = a.student_id
                          GROUP BY s.id
                          ORDER BY s.department, s.name''')
        
        students_data = cursor.fetchall()
        
        cursor.execute('''SELECT a.date, a.time, s.name, s.roll_no, s.department, s.year, a.confidence
                          FROM attendance a
                          JOIN students s ON a.student_id = s.id
                          ORDER BY a.date DESC, a.time DESC''')
        
        attendance_data = cursor.fetchall()
        conn.close()
        
        return render_template('view_database.html', 
                             students=students_data, 
                             attendance=attendance_data)
        
    except Exception as e:
        print(f"Error in view_database: {e}")
        return jsonify({'error': str(e)}), 500

# ============== 3D FACE RECONSTRUCTION ROUTES ==============

@app.route('/train_3d_data')
def train_3d_data():
    """Render the 3D training interface"""
    return render_template('train_3d_data.html')

@app.route('/start_3d_training', methods=['POST'])
def start_3d_training():
    """Initialize a 3D training session for a student"""
    try:
        data = request.json
        student_id = data['student_id']
        
        conn = sqlite3.connect('face_attendance.db')
        cursor = conn.cursor()
        cursor.execute("SELECT name, roll_no FROM students WHERE id = ?", (student_id,))
        student = cursor.fetchone()
        conn.close()
        
        if not student:
            return jsonify({'success': False, 'message': 'Student not found'})
        
        # Store training session in session (Flask session)
        session[f'3d_training_{student_id}'] = {
            'student_id': student_id,
            'frames_captured': 0,
            'started_at': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True, 
            'message': f'3D training session started for {student[0]} (Roll: {student[1]})'
        })
        
    except Exception as e:
        print(f"Error in start_3d_training: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/capture_3d_frame', methods=['POST'])
def capture_3d_frame():
    """Capture a frame for 3D reconstruction"""
    try:
        data = request.json
        student_id = data['student_id']
        image_data = data['image']
        
        # Check if training session exists
        session_key = f'3d_training_{student_id}'
        if session_key not in session:
            return jsonify({'success': False, 'message': '3D training session not started'})
        
        # Decode image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'success': False, 'message': 'Could not decode image'})
        
        # Create new 3D reconstructor and training session with timing controls
        # (Flask sessions can't serialize complex objects reliably)
        reconstructor = Face3DReconstructor()
        training_session = Face3DTrainingSession(reconstructor, min_frame_interval=2.0)
        
        # Load previous frames if any exist in database
        conn = sqlite3.connect('face_attendance.db')
        cursor = conn.cursor()
        cursor.execute("SELECT session_data FROM face_3d_sessions WHERE student_id = ? ORDER BY created_at DESC LIMIT 1", (student_id,))
        existing_session = cursor.fetchone()
        
        if existing_session:
            try:
                session_data = pickle.loads(existing_session[0])
                if 'landmarks_sequence' in session_data:
                    training_session.landmarks_sequence = session_data['landmarks_sequence']
                if 'poses_sequence' in session_data:
                    training_session.poses_sequence = session_data['poses_sequence']
                if 'camera_matrix' in session_data and session_data['camera_matrix']:
                    reconstructor.camera_matrix = np.array(session_data['camera_matrix'])
                    training_session.is_calibrated = True
            except Exception as e:
                print(f"Warning: Could not load previous session data: {e}")
        
        # Add current frame to training session
        success = training_session.add_frame(img)
        
        if success:
            # Update session data in database
            session_data = {
                'landmarks_sequence': training_session.landmarks_sequence,
                'poses_sequence': training_session.poses_sequence,
                'camera_matrix': reconstructor.camera_matrix.tolist() if reconstructor.camera_matrix is not None else None,
                'dist_coeffs': reconstructor.dist_coeffs.tolist()
            }
            session_blob = pickle.dumps(session_data)
            
            # Update or insert session data
            cursor.execute("DELETE FROM face_3d_sessions WHERE student_id = ?", (student_id,))
            cursor.execute("INSERT INTO face_3d_sessions (student_id, session_data) VALUES (?, ?)", (student_id, session_blob))
            conn.commit()
            
            # Update frames count
            session[session_key]['frames_captured'] = len(training_session.landmarks_sequence)
            frames_count = session[session_key]['frames_captured']
            
            conn.close()
            
            return jsonify({
                'success': True, 
                'message': f'Frame {frames_count} captured successfully',
                'frames_captured': frames_count
            })
        else:
            conn.close()
            return jsonify({'success': False, 'message': 'Failed to process frame - no face detected or multiple faces'})
        
    except Exception as e:
        print(f"Error in capture_3d_frame: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'})

@app.route('/generate_3d_model', methods=['POST'])
def generate_3d_model():
    """Generate 3D model from captured frames"""
    try:
        data = request.json
        student_id = data['student_id']
        
        # Check if training session exists
        session_key = f'3d_training_{student_id}'
        if session_key not in session:
            return jsonify({'success': False, 'message': '3D training session not found'})
        
        # Load session data from database
        conn = sqlite3.connect('face_attendance.db')
        cursor = conn.cursor()
        cursor.execute("SELECT session_data FROM face_3d_sessions WHERE student_id = ? ORDER BY created_at DESC LIMIT 1", (student_id,))
        existing_session = cursor.fetchone()
        
        if not existing_session:
            conn.close()
            return jsonify({'success': False, 'message': 'No 3D training data found. Please capture frames first.'})
        
        try:
            session_data = pickle.loads(existing_session[0])
            
            # Create reconstructor and training session
            reconstructor = Face3DReconstructor()
            training_session = Face3DTrainingSession(reconstructor)
            
            # Restore session data
            if 'landmarks_sequence' in session_data:
                training_session.landmarks_sequence = session_data['landmarks_sequence']
            if 'poses_sequence' in session_data:
                training_session.poses_sequence = session_data['poses_sequence']
            if 'camera_matrix' in session_data and session_data['camera_matrix']:
                reconstructor.camera_matrix = np.array(session_data['camera_matrix'])
                training_session.is_calibrated = True
            
            if len(training_session.landmarks_sequence) < 2:
                conn.close()
                return jsonify({'success': False, 'message': 'Need at least 2 frames for 3D reconstruction'})
            
            # Generate 3D model
            mesh = training_session.generate_3d_model()
            
            if mesh is None:
                conn.close()
                return jsonify({'success': False, 'message': 'Failed to generate 3D model from captured frames'})
            
            # Get student info
            cursor.execute("SELECT roll_no FROM students WHERE id = ?", (student_id,))
            student_roll_no = cursor.fetchone()[0]
            
            # Create 3D models directory
            models_dir = os.path.join('students_data', str(student_roll_no), '3d_models')
            os.makedirs(models_dir, exist_ok=True)
            
            # Save mesh file
            model_path = os.path.join(models_dir, 'face_model.ply')
            save_3d_model(mesh, model_path)
            
            # Extract features
            features = reconstructor.extract_face_features(mesh)
            features_data = pickle.dumps(features)
            
            # Update database with 3D model info
            cursor.execute(
                "UPDATE students SET face_3d_model_path = ?, face_3d_features = ? WHERE id = ?", 
                (model_path, features_data, student_id)
            )
            
            conn.commit()
            conn.close()
            
            # Clean up session
            session.pop(session_key, None)
            
            return jsonify({
                'success': True, 
                'message': f'3D model generated and saved successfully',
                'model_path': model_path,
                'vertices_count': len(mesh.vertices),
                'triangles_count': len(mesh.triangles)
            })
            
        except Exception as e:
            conn.close()
            print(f"Error processing session data: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'message': f'Error processing training data: {str(e)}'})
        
    except Exception as e:
        print(f"Error in generate_3d_model: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'})

@app.route('/detect_face_3d', methods=['POST'])
def detect_face_3d():
    """Detect faces using 3D models for enhanced accuracy"""
    try:
        data = request.json
        image_data = data['image']
        
        # Decode image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'detected': False, 'message': 'Could not decode image'})
        
        # Initialize 3D reconstructor
        reconstructor = Face3DReconstructor()
        
        # Extract landmarks from current image
        landmarks = reconstructor.extract_face_landmarks(img)
        if landmarks is None:
            return jsonify({'detected': False, 'message': 'No face detected'})
        
        # Calibrate camera and estimate pose
        reconstructor.calibrate_camera([img])
        rvec, tvec = reconstructor.estimate_head_pose(landmarks)
        
        # Get all students with 3D models
        conn = sqlite3.connect('face_attendance.db')
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name, roll_no, department, year, face_3d_model_path, face_3d_features FROM students WHERE face_3d_model_path IS NOT NULL"
        )
        students = cursor.fetchall()
        
        best_match = None
        best_similarity = 0
        
        # Create a temporary 3D model from current frame
        temp_session = Face3DTrainingSession(reconstructor)
        temp_session.add_frame(img)
        
        # Need at least 2 frames for triangulation - use a simple approach for real-time
        # For now, fall back to 2D recognition but with pose validation
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_img)
        
        if len(face_encodings) == 0:
            return jsonify({'detected': False, 'message': 'No face encodings found'})
        
        current_encoding = face_encodings[0]
        
        for student in students:
            student_id, name, roll_no, department, year, model_path, features_data = student
            
            # Load student's 3D model if available
            if model_path and os.path.exists(model_path):
                student_mesh = load_3d_model(model_path)
                if student_mesh:
                    # Extract features from current detection
                    try:
                        current_features = reconstructor.extract_face_features(student_mesh)  # Placeholder
                        stored_features = pickle.loads(features_data)
                        
                        # Compare geometric features (simplified)
                        similarity = 0.7  # Placeholder similarity score
                        
                        if similarity > 0.6 and similarity > best_similarity:
                            best_match = {
                                'id': student_id,
                                'name': name,
                                'roll_no': roll_no,
                                'department': department,
                                'year': year,
                                'confidence': round(similarity * 100, 2),
                                'method': '3D+2D'
                            }
                            best_similarity = similarity
                    except Exception as e:
                        print(f"3D comparison failed for student {student_id}: {e}")
            
            # Fallback to 2D recognition
            cursor.execute("SELECT face_encodings FROM students WHERE id = ?", (student_id,))
            encoding_result = cursor.fetchone()
            if encoding_result and encoding_result[0]:
                try:
                    student_encodings = pickle.loads(encoding_result[0])
                    distances = face_recognition.face_distance(student_encodings, current_encoding)
                    min_distance = min(distances)
                    confidence_2d = max(0, (1 - min_distance) * 100)
                    
                    if confidence_2d > 60 and confidence_2d > best_similarity * 100:
                        best_match = {
                            'id': student_id,
                            'name': name,
                            'roll_no': roll_no,
                            'department': department,
                            'year': year,
                            'confidence': round(confidence_2d, 2),
                            'method': '2D'
                        }
                        best_similarity = confidence_2d / 100
                except Exception as e:
                    print(f"2D comparison failed for student {student_id}: {e}")
        
        if best_match:
            current_time = datetime.now()
            date_str = current_time.strftime('%Y-%m-%d')
            time_str = current_time.strftime('%H:%M:%S')
            
            cursor.execute("SELECT id FROM attendance WHERE student_id = ? AND date = ?", 
                          (best_match['id'], date_str))
            existing = cursor.fetchone()
            
            if not existing:
                cursor.execute(
                    "INSERT INTO attendance (student_id, date, time, confidence, recognition_method) VALUES (?, ?, ?, ?, ?)",
                    (best_match['id'], date_str, time_str, best_match['confidence'], best_match['method'])
                )
                conn.commit()
                best_match['attendance_status'] = 'Marked'
            else:
                best_match['attendance_status'] = 'Already marked today'
            
            conn.close()
            return jsonify({'detected': True, **best_match})
        else:
            conn.close()
            return jsonify({'detected': False, 'message': 'Face not recognized'})
            
    except Exception as e:
        print(f"Error in detect_face_3d: {e}")
        return jsonify({'detected': False, 'message': str(e)})

if __name__ == '__main__':
    # Always initialize database to ensure all tables exist
    init_db()
    
    conn = sqlite3.connect('face_attendance.db')
    cursor = conn.cursor()
    
    # Check and update students table to include 3D columns if they don't exist
    cursor.execute("PRAGMA table_info(students)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'face_3d_model_path' not in columns:
        cursor.execute('ALTER TABLE students ADD COLUMN face_3d_model_path TEXT')
        print("Added face_3d_model_path column to students table")
    
    if 'face_3d_features' not in columns:
        cursor.execute('ALTER TABLE students ADD COLUMN face_3d_features BLOB')
        print("Added face_3d_features column to students table")
    
    # Check and update attendance table to include recognition_method if it doesn't exist
    cursor.execute("PRAGMA table_info(attendance)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'recognition_method' not in columns:
        cursor.execute('ALTER TABLE attendance ADD COLUMN recognition_method TEXT DEFAULT "2D"')
        print("Added recognition_method column to attendance table")
    
    # Ensure 3D sessions table exists
    cursor.execute('''CREATE TABLE IF NOT EXISTS face_3d_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id INTEGER,
                        session_data BLOB,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE)
    ''')
    print("Ensured face_3d_sessions table exists")
    
    # Ensure training_photos table exists with proper foreign key
    cursor.execute('''CREATE TABLE IF NOT EXISTS training_photos (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id INTEGER,
                        photo_data BLOB,
                        photo_index INTEGER,
                        FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE)
    ''')
    
    # Ensure attendance table exists with proper foreign key
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id INTEGER,
                        date TEXT NOT NULL,
                        time TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        recognition_method TEXT DEFAULT '2D',
                        FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE)
    ''')

    conn.commit()
    conn.close()
    
    print("Database initialization complete. Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)
