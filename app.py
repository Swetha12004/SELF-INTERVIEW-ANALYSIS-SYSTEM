from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import librosa
import soundfile as sf
from deepface import DeepFace
import os
from flask_cors import CORS
from collections import Counter
import pyaudio
import wave
import time
import tempfile
import google.generativeai as genai
from dotenv import load_dotenv
from speech_analysis_.speech_analysis import AnalysisSystem
import atexit
from Resume.Resume_analyze.resume_analyzer import ResumeAnalyzer
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Define directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
AUDIO_DIR = os.path.join(STATIC_DIR, 'audio')
IMAGES_DIR = os.path.join(STATIC_DIR, 'images')

# Create directories if they don't exist
for directory in [STATIC_DIR, AUDIO_DIR, IMAGES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Audio recording constants
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100

# Global variables
recording = False
audio = pyaudio.PyAudio()
stream = None
audio_frames = []
start_time = 0
emotion_history = {'facial': [], 'voice': []}

# Initialize camera
camera = None
is_recording = False

# Add this job suitability mapping dictionary
JOB_EMOTION_MAPPING = {
    'Software Developer': {
        'happy': 90,
        'neutral': 80,
        'calm': 85,
        'focused': 95,
        'stressed': 60
    },
    'Customer Service Representative': {
        'happy': 95,
        'neutral': 85,
        'calm': 90,
        'empathetic': 95,
        'stressed': 50
    },
    'Data Scientist': {
        'neutral': 90,
        'focused': 95,
        'calm': 85,
        'analytical': 95,
        'stressed': 70
    },
    'Sales Representative': {
        'happy': 95,
        'enthusiastic': 90,
        'confident': 95,
        'neutral': 75,
        'stressed': 60
    },
    'Project Manager': {
        'calm': 90,
        'confident': 90,
        'neutral': 85,
        'focused': 90,
        'stressed': 70
    }
}

# Load environment variables
load_dotenv()

# Configure Google AI
genai.configure(api_key=os.getenv('GEN_AI_KEY'))
model = genai.GenerativeModel('gemini-pro')

# Add this after the Google AI configuration
CHAT_PROMPT = """You are an AI interview preparation assistant. Your role is to:
1. Help candidates prepare for job interviews
2. Provide relevant advice based on their emotional state
3. Give constructive feedback
4. Suggest improvements based on their responses

Current emotional state: {emotion}
Job suitability scores: {job_scores}

Please provide helpful and encouraging responses."""

# Add these global variables at the top after imports
global_camera = None
is_recording = False

# Add at the top with other constants
MAX_RECORDING_TIME = 180  # 3 minutes in seconds

# Add these configurations
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_camera():
    global camera
    try:
        # Release existing camera if any
        if camera is not None:
            camera.release()
        
        # Try different camera indices
        for index in range(3):  # Try first 3 camera indices
            camera = cv2.VideoCapture(index)
            if camera.isOpened():
                # Set camera properties
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                camera.set(cv2.CAP_PROP_FPS, 30)
                print(f"Camera initialized successfully on index {index}")
                return camera
                
        raise RuntimeError("No working camera found")
    except Exception as e:
        print(f"Camera initialization error: {e}")
        return None

def process_audio(audio_path):
    """Process audio using librosa instead of torchaudio"""
    try:
        # Load audio file
        audio, sr = librosa.load(audio_path)
        
        # Process audio (example features)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
        
        return {
            'mfccs': mfccs.tolist(),
            'spectral': spectral_centroids.tolist()
        }
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def analyze_facial_emotions(frame):
    """Analyze facial emotions using DeepFace"""
    try:
        result = DeepFace.analyze(frame, 
                                 actions=['emotion'],
                                 enforce_detection=False)
        return result[0]['dominant_emotion']
    except Exception as e:
        print(f"Error analyzing facial emotion: {e}")
        return "neutral"

def get_job_suitability(emotion):
    """Calculate job suitability scores based on detected emotion"""
    job_scores = {}
    
    # Default score for neutral emotion
    base_score = 75 if emotion == 'neutral' else 60
    
    for job, emotion_scores in JOB_EMOTION_MAPPING.items():
        # Get the score for the detected emotion, or use a default value
        score = emotion_scores.get(emotion, base_score)
        
        # Apply some variance based on the emotion context
        if emotion in ['happy', 'calm', 'focused']:
            score = min(score + 10, 100)
        elif emotion in ['angry', 'sad', 'fearful']:
            score = max(score - 10, 0)
            
        job_scores[job] = score
    
    return {
        "detected_emotion": emotion,
        "job_scores": job_scores
    }

def calculate_emotion_percentages(emotions_list):
    if not emotions_list:
        return {}
    emotion_counts = Counter(emotions_list)
    total = len(emotions_list)
    return {emotion: (count/total) * 100 for emotion, count in emotion_counts.items()}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/face')
def face_analysis():
    return render_template('face.html')

@app.route('/speech')
def speech_analysis():
    return render_template('speech_index.html')

@app.route('/resume')
def resume_analysis():
    return render_template('resume_index.html')

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        global camera, is_recording
        
        if camera is None:
            camera = init_camera()
            if camera is None:
                print("Failed to initialize camera")
                return
        
        start_time = time.time()
        
        while True:
            try:
                if not camera.isOpened():
                    camera = init_camera()
                    if camera is None:
                        time.sleep(1)
                        continue
                
                ret, frame = camera.read()
                if not ret:
                    print("Failed to grab frame")
                    continue
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Add recording time indicator
                if is_recording:
                    elapsed_time = int(time.time() - start_time)
                    remaining_time = MAX_RECORDING_TIME - elapsed_time
                    time_text = f"Time remaining: {remaining_time}s"
                    cv2.putText(
                        frame,
                        time_text,
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
                
                # Detect emotion
                try:
                    result = DeepFace.analyze(
                        frame,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='opencv'
                    )
                    emotion = result[0]['dominant_emotion']
                    
                    # Draw emotion text
                    cv2.putText(
                        frame,
                        f"Emotion: {emotion}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    
                    if is_recording:
                        emotion_history['facial'].append(emotion)
                except Exception as e:
                    print(f"Emotion detection error: {e}")
                
                # Convert frame to jpg
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
            except Exception as e:
                print(f"Frame generation error: {e}")
                time.sleep(0.1)
                continue

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/analyze_emotions')
def analyze_emotions():
    try:
        if camera is None:
            return jsonify({"emotion": "neutral", "confidence": 0})
            
        success, frame = camera.read()
        if not success:
            return jsonify({"emotion": "neutral", "confidence": 0})
        
        # Detect emotion using DeepFace
        try:
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )
            
            # Get emotion and confidence
            emotions = result[0]['emotion']
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            confidence = emotions[dominant_emotion]
            
            return jsonify({
                "emotion": dominant_emotion,
                "confidence": confidence
            })
        except Exception as e:
            print(f"DeepFace error: {e}")
            return jsonify({"emotion": "neutral", "confidence": 0})
            
    except Exception as e:
        print(f"Emotion analysis error: {e}")
        return jsonify({"emotion": "neutral", "confidence": 0})

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global is_recording, emotion_history, camera, start_time
    try:
        # Initialize camera
        camera = init_camera()
        if camera is None or not camera.isOpened():
            return jsonify({"error": "Could not access camera"}), 500

        # Reset variables
        is_recording = True
        emotion_history = {'facial': [], 'voice': []}
        start_time = time.time()
        
        return jsonify({
            "status": "success",
            "max_time": MAX_RECORDING_TIME
        })
    except Exception as e:
        print(f"Error in start_recording: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/stop_recording', methods=['GET'])
def stop_recording():
    global is_recording, emotion_history, start_time
    try:
        is_recording = False
        elapsed_time = time.time() - start_time
        
        if elapsed_time >= MAX_RECORDING_TIME:
            message = "Recording completed - 3 minute limit reached"
        else:
            message = "Recording stopped by user"

        # Get current emotion
        ret, frame = camera.read()
        if ret:
            facial_emotion = analyze_facial_emotions(frame)
            emotion_history['facial'].append(facial_emotion)
        else:
            facial_emotion = "neutral"

        # Save audio file for speech analysis
        audio_path = os.path.join(AUDIO_DIR, f'temp_audio_{time.time()}.wav')
        wf = wave.open(audio_path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(audio_frames))
        wf.close()

        # Process speech
        speech_text, job_matches = analysis_system.process_speech(audio_path)
        
        # Get combined analysis
        voice_emotion = "neutral"  # Default voice emotion
        combined_job_scores = analysis_system.analyze_combined_factors(
            speech_text, 
            voice_emotion, 
            facial_emotion
        )

        # Calculate results
        facial_percentages = calculate_emotion_percentages(emotion_history['facial'])
        voice_percentages = {'neutral': 100}  # Default voice emotion
        
        # Generate visualization
        analysis_system.generate_visualization(combined_job_scores)
        
        # Clean up temporary audio file
        try:
            os.remove(audio_path)
        except Exception as e:
            print(f"Error removing temp audio: {e}")

        result = {
            "emotion_percentages": {
                "facial": facial_percentages,
                "voice": voice_percentages
            },
            "most_common_emotions": {
                "facial": facial_emotion,
                "voice": voice_emotion
            },
            "job_suitability": {
                "detected_emotion": facial_emotion,
                "job_scores": combined_job_scores
            },
            "speech_text": speech_text,
            "job_matches": job_matches,
            "recording_info": {
                "duration": elapsed_time,
                "message": message
            }
        }

        return jsonify(result)

    except Exception as e:
        print(f"Error in stop_recording: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/chat_message', methods=['POST'])
def chat_message():
    try:
        message = request.json.get('message')
        
        # Get current emotion state
        ret, frame = camera.read()
        facial_emotion = "neutral"
        if ret:
            facial_emotion = analyze_facial_emotions(frame)
        
        # Get job suitability
        job_suitability = get_job_suitability(facial_emotion)
        
        # Create contextualized prompt
        prompt = CHAT_PROMPT.format(
            emotion=facial_emotion,
            job_scores=job_suitability['job_scores']
        )
        
        # Generate response using Google AI
        chat = model.start_chat(history=[])
        response = chat.send_message(f"{prompt}\nUser: {message}")
        
        return jsonify({"response": response.text})
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500

# Add a camera test route
@app.route('/test_camera')
def test_camera():
    try:
        camera = init_camera()
        if camera is not None and camera.isOpened():
            camera.release()
            return jsonify({"status": "success", "message": "Camera is accessible"})
        else:
            return jsonify({"status": "error", "message": "Camera not accessible"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Add cleanup function
@atexit.register
def cleanup():
    global camera
    if camera is not None:
        camera.release()
    cv2.destroyAllWindows()

@app.route('/demo')
def demo():
    return render_template('face.html')

@app.route('/analyze_resume', methods=['POST'])
def analyze_resume():
    if 'resume' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['resume']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        try:
            # Save file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Analyze resume
            analyzer = ResumeAnalyzer()
            with open(filepath, 'rb') as f:
                text = f.read().decode('utf-8', errors='ignore')
            
            # Get analysis results
            analysis = analyzer.analyze_resume(text)
            experience = analyzer.analyze_experience_level(text)
            
            # Clean up
            os.remove(filepath)
            
            return jsonify({
                'technical_skills_found': analysis['technical_skills_found'],
                'experience_level': experience,
                'job_matches': analyzer._get_suitable_roles(experience['years'], experience['level'])
            })
            
        except Exception as e:
            print(f"Resume analysis error: {e}")  # Add this for debugging
            return jsonify({'error': str(e)})
            
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    try:
        app.run(debug=True, port=5000)
    except Exception as e:
        print(f"Server error: {e}")
        cleanup()
