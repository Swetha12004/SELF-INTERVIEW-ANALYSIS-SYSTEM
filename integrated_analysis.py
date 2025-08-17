from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import librosa
import soundfile as sf
from deepface import DeepFace
import os
from flask_cors import CORS
import pyaudio
import wave
import time
import tempfile
from textblob import TextBlob
import google.generativeai as genai
from dotenv import load_dotenv
import re
from werkzeug.utils import secure_filename
import atexit

app = Flask(__name__, 
    template_folder='templates',
    static_folder='static'
)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'mp4', 'avi', 'mov'}
MAX_RECORDING_TIME = 180  # 3 minutes

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv('GEN_AI_KEY'))
model = genai.GenerativeModel('gemini-pro')

# Global variables
camera = None
is_recording = False
emotion_history = {'facial': [], 'voice': []}
start_time = None

# Resume Analysis Keywords
TECHNICAL_KEYWORDS = [
    'python', 'java', 'javascript', 'machine learning', 'sql', 'aws',
    'docker', 'kubernetes', 'react', 'angular', 'node.js', 'devops'
]

SOFT_SKILLS = [
    'leadership', 'communication', 'teamwork', 'problem-solving',
    'analytical', 'project management', 'time management'
]

# Job Emotion Mapping
JOB_EMOTION_MAPPING = {
    'Software Developer': {
        'happy': 90, 'neutral': 80, 'calm': 85, 'focused': 95
    },
    'Data Scientist': {
        'neutral': 90, 'focused': 95, 'analytical': 95
    },
    'Project Manager': {
        'confident': 90, 'neutral': 85, 'focused': 90
    }
}

def init_camera():
    global camera
    try:
        if camera is not None:
            camera.release()
        
        for index in range(3):
            camera = cv2.VideoCapture(index)
            if camera.isOpened():
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                camera.set(cv2.CAP_PROP_FPS, 30)
                print(f"Camera initialized on index {index}")
                return camera
                
        raise RuntimeError("No working camera found")
    except Exception as e:
        print(f"Camera initialization error: {e}")
        return None

def analyze_facial_emotions(frame):
    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'
        )
        return result[0]['dominant_emotion']
    except Exception as e:
        print(f"Emotion detection error: {e}")
        return "neutral"

def analyze_speech(audio_path):
    try:
        audio, sr = librosa.load(audio_path)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr)
        
        # Simple emotion mapping based on audio features
        energy = np.mean(librosa.feature.rms(y=audio))
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        
        if energy > 0.1 and tempo > 120:
            emotion = "excited"
        elif energy > 0.05:
            emotion = "confident"
        else:
            emotion = "calm"
            
        return {
            'emotion': emotion,
            'confidence': float(energy * 100)
        }
    except Exception as e:
        print(f"Speech analysis error: {e}")
        return {'emotion': 'neutral', 'confidence': 0}

def analyze_resume(text):
    technical_skills = [skill for skill in TECHNICAL_KEYWORDS 
                       if skill in text.lower()]
    soft_skills = [skill for skill in SOFT_SKILLS 
                   if skill in text.lower()]
    
    # Extract years of experience
    exp_match = re.search(r'(\d+)\s*years?\s*(?:of)?\s*experience', text.lower())
    years_exp = int(exp_match.group(1)) if exp_match else 0
    
    return {
        'technical_skills': technical_skills,
        'soft_skills': soft_skills,
        'experience_years': years_exp,
        'level': 'Senior' if years_exp > 5 else 'Mid' if years_exp > 2 else 'Junior'
    }

def get_job_suitability(emotions):
    scores = {}
    for job, criteria in JOB_EMOTION_MAPPING.items():
        score = sum(criteria.get(emotion, 0) for emotion in emotions) / len(emotions)
        scores[job] = min(100, score)
    return scores

@app.route('/')
def index():
    return render_template('integrated.html')

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        global camera, is_recording, start_time
        
        if camera is None:
            camera = init_camera()
        
        while True:
            try:
                success, frame = camera.read()
                if not success:
                    continue
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Add recording time indicator
                if is_recording:
                    elapsed = int(time.time() - start_time)
                    remaining = MAX_RECORDING_TIME - elapsed
                    cv2.putText(
                        frame,
                        f"Time remaining: {remaining}s",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
                
                # Detect emotion
                emotion = analyze_facial_emotions(frame)
                if emotion:
                    cv2.putText(
                        frame,
                        f"Emotion: {emotion}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
            except Exception as e:
                print(f"Frame generation error: {e}")
                time.sleep(0.1)

    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_analysis', methods=['POST'])
def start_analysis():
    global is_recording, start_time
    try:
        is_recording = True
        start_time = time.time()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stop_analysis', methods=['POST'])
def stop_analysis():
    global is_recording
    try:
        is_recording = False
        return jsonify({
            "facial_emotions": emotion_history['facial'],
            "voice_emotions": emotion_history['voice']
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analyze_resume', methods=['POST'])
def analyze_resume_endpoint():
    if 'resume' not in request.files:
        return jsonify({"error": "No resume file provided"}), 400
        
    file = request.files['resume']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    if file and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            analysis = analyze_resume(text)
            os.remove(filepath)
            
            return jsonify(analysis)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/chat', methods=['POST'])
def chat():
    try:
        message = request.json.get('message')
        response = model.generate_content(message)
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@atexit.register
def cleanup():
    global camera
    if camera is not None:
        camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("Starting integrated analysis system...")
    try:
        app.run(debug=True, port=5000)
    except Exception as e:
        print(f"Failed to start server: {e}")
        cleanup() 