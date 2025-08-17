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
import openai
from dotenv import load_dotenv
import re
from werkzeug.utils import secure_filename
import atexit
from collections import Counter
import whisper
from scipy.io import wavfile

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'mp4', 'avi', 'mov'}
MIN_RECORDING_TIME = 10  # Minimum recording time in seconds
MAX_RECORDING_TIME = 180  # Maximum recording time in seconds

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load environment variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
whisper_model = whisper.load_model("base")

# Global variables
camera = None
is_recording = False
emotion_history = {'facial': [], 'voice': []}
start_time = None
audio = pyaudio.PyAudio()
audio_stream = None
audio_frames = []
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100

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
        # First, release any existing camera
        if camera is not None:
            camera.release()
            cv2.destroyAllWindows()
            time.sleep(1)  # Give time for camera to release
        
        # Try different backends
        backends = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_V4L2]
        
        for backend in backends:
            for index in range(2):  # Try first two camera indices
                try:
                    camera = cv2.VideoCapture(index, backend)
                    if camera.isOpened():
                        # Set camera properties
                        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        camera.set(cv2.CAP_PROP_FPS, 30)
                        print(f"Camera initialized successfully with backend {backend} on index {index}")
                        return camera
                except Exception as e:
                    print(f"Failed to initialize camera with backend {backend} on index {index}: {e}")
                    continue
        
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
        emotion = result[0]['dominant_emotion']
        emotion_scores = result[0]['emotion']
        
        # Add to emotion history
        if emotion:
            emotion_history['facial'].append(emotion)
        
        return emotion, emotion_scores
    except Exception as e:
        print(f"Emotion analysis error: {e}")
        return None, None

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

def get_job_suitability(facial_emotions, voice_emotions, speech_text):
    try:
        job_scores = {}
        
        # Calculate emotion-based scores
        for job, criteria in JOB_EMOTION_MAPPING.items():
            score = 0
            total_weight = 0
            
            # Weight facial emotions (50%)
            for emotion, count in facial_emotions:
                if emotion in criteria:
                    score += criteria[emotion] * count * 0.5
                    total_weight += count * 0.5
            
            # Weight voice emotions (30%)
            for emotion, count in voice_emotions:
                if emotion in criteria:
                    score += criteria[emotion] * count * 0.3
                    total_weight += count * 0.3
            
            # Analyze speech content (20%)
            speech_score = analyze_speech_content(speech_text, job)
            score += speech_score * 0.2
            total_weight += 0.2
            
            # Calculate final score
            job_scores[job] = round((score / total_weight if total_weight > 0 else 50), 2)
        
        return job_scores
    except Exception as e:
        print(f"Job suitability calculation error: {e}")
        return {'error': str(e)}

def analyze_speech_content(text, job):
    # Job-specific keywords
    keywords = {
        'Software Developer': [
            'code', 'develop', 'programming', 'software', 'technical',
            'problem-solving', 'algorithm', 'engineering'
        ],
        'Data Scientist': [
            'data', 'analysis', 'statistics', 'machine learning', 'research',
            'analytical', 'model', 'prediction'
        ],
        'Project Manager': [
            'manage', 'team', 'leadership', 'coordinate', 'planning',
            'communication', 'organization', 'strategy'
        ]
    }
    
    # Count keyword matches
    text = text.lower()
    job_keywords = keywords.get(job, [])
    matches = sum(1 for keyword in job_keywords if keyword in text)
    
    # Calculate score (0-100)
    return min(100, (matches / len(job_keywords) if job_keywords else 0) * 100)

def calculate_overall_confidence():
    try:
        facial_emotions = Counter(emotion_history['facial'])
        voice_emotions = Counter(emotion_history['voice'])
        
        # Calculate confidence metrics
        emotional_stability = calculate_emotional_stability(facial_emotions)
        communication_clarity = calculate_communication_clarity(voice_emotions)
        professional_demeanor = calculate_professional_demeanor(facial_emotions, voice_emotions)
        
        # Overall score
        overall_score = (emotional_stability + communication_clarity + professional_demeanor) / 3
        
        return {
            "score": round(overall_score, 2),
            "factors": {
                "emotional_stability": get_rating(emotional_stability),
                "communication_clarity": get_rating(communication_clarity),
                "professional_demeanor": get_rating(professional_demeanor)
            }
        }
    except Exception as e:
        print(f"Confidence calculation error: {e}")
        return {"score": 0, "factors": {}}

def calculate_emotional_stability(emotions):
    positive = sum(emotions.get(e, 0) for e in ['happy', 'calm', 'neutral'])
    negative = sum(emotions.get(e, 0) for e in ['angry', 'sad', 'fear'])
    total = sum(emotions.values())
    return (positive / total * 100) if total > 0 else 50

def calculate_communication_clarity(emotions):
    clear = sum(emotions.get(e, 0) for e in ['confident', 'calm', 'neutral'])
    unclear = sum(emotions.get(e, 0) for e in ['nervous', 'uncertain', 'stressed'])
    total = sum(emotions.values())
    return (clear / total * 100) if total > 0 else 50

def calculate_professional_demeanor(facial, voice):
    professional = sum(facial.get(e, 0) for e in ['neutral', 'calm']) + \
                  sum(voice.get(e, 0) for e in ['confident', 'calm'])
    total = sum(facial.values()) + sum(voice.values())
    return (professional / total * 100) if total > 0 else 50

def get_rating(score):
    if score >= 90: return "Excellent"
    elif score >= 75: return "Good"
    elif score >= 60: return "Fair"
    else: return "Needs Improvement"

def generate_job_recommendation(job_scores):
    sorted_jobs = sorted(job_scores.items(), key=lambda x: x[1], reverse=True)
    top_job = sorted_jobs[0]
    
    return {
        "best_match": top_job[0],
        "match_score": top_job[1],
        "explanation": f"Based on your emotional responses and communication style, "
                      f"you show strong alignment with the role of {top_job[0]}."
    }

@app.route('/')
def index():
    return render_template('integrated.html')

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        global camera, is_recording, start_time
        
        if camera is None or not camera.isOpened():
            camera = init_camera()
            if camera is None:
                return b''
        
        while True:
            try:
                success, frame = camera.read()
                if not success:
                    print("Failed to grab frame")
                    time.sleep(0.1)
                    continue
                
                # Flip frame horizontally for mirror effect
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
                try:
                    result = DeepFace.analyze(
                        frame,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='opencv'
                    )
                    emotion = result[0]['dominant_emotion']
                    cv2.putText(
                        frame,
                        f"Emotion: {emotion}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
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

    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_analysis', methods=['POST'])
def start_analysis():
    global is_recording, start_time, camera, audio_stream, audio_frames
    try:
        if camera is None or not camera.isOpened():
            camera = init_camera()
            if camera is None:
                return jsonify({"error": "Failed to initialize camera"}), 500
        
        # Reset audio state
        if audio_stream is not None:
            audio_stream.stop_stream()
            audio_stream.close()
        audio_frames = []
        
        is_recording = True
        start_time = time.time()
        start_audio_recording()
        
        return jsonify({"status": "success"})
    except Exception as e:
        print(f"Start analysis error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/stop_analysis', methods=['POST'])
def stop_analysis():
    global is_recording, emotion_history, start_time
    try:
        if not is_recording or start_time is None:
            return jsonify({"error": "Recording not started"}), 400
            
        recording_duration = time.time() - start_time
        
        if recording_duration < MIN_RECORDING_TIME:
            return jsonify({
                "error": f"Please record for at least {MIN_RECORDING_TIME} seconds",
                "recording_time": recording_duration
            }), 400
        
        is_recording = False
        speech_text = get_speech_text()
        
        # Analyze voice emotion from speech
        voice_emotion = analyze_voice_emotion(speech_text)
        
        # Calculate statistics
        facial_emotions = Counter(emotion_history['facial']).most_common()
        voice_emotions = Counter(emotion_history['voice']).most_common()
        
        # Calculate job suitability with speech content
        job_scores = get_job_suitability(facial_emotions, voice_emotions, speech_text)
        
        analysis_result = {
            "facial_analysis": {
                "dominant_emotion": facial_emotions[0][0] if facial_emotions else "neutral",
                "emotion_distribution": dict(facial_emotions)
            },
            "voice_analysis": {
                "dominant_emotion": voice_emotions[0][0] if voice_emotions else "neutral",
                "emotion_distribution": dict(voice_emotions),
                "speech_text": speech_text
            },
            "job_suitability": {
                "best_matches": job_scores,
                "recommendation": generate_job_recommendation(job_scores)
            },
            "overall_confidence": calculate_overall_confidence(),
            "recording_time": recording_duration
        }
        
        # Reset histories
        emotion_history = {'facial': [], 'voice': []}
        
        return jsonify(analysis_result)
        
    except Exception as e:
        print(f"Analysis error: {e}")
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
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI interviewer. Ask relevant questions based on the candidate's resume and previous responses."},
                {"role": "user", "content": message}
            ]
        )
        return jsonify({"response": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def start_audio_recording():
    global audio_stream, audio_frames
    audio_frames = []
    try:
        audio_stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=audio_callback
        )
        audio_stream.start_stream()
        print("Audio recording started")
    except Exception as e:
        print(f"Error starting audio recording: {e}")

def audio_callback(in_data, frame_count, time_info, status):
    global audio_frames
    audio_frames.append(in_data)
    return (in_data, pyaudio.paContinue)

def stop_audio_recording():
    global audio_stream, audio_frames
    if audio_stream:
        try:
            audio_stream.stop_stream()
            audio_stream.close()
            
            # Convert float32 to int16 for better compatibility
            audio_data = b''.join(audio_frames)
            audio_np = np.frombuffer(audio_data, dtype=np.float32)
            audio_int16 = (audio_np * 32767).astype(np.int16)
            
            # Save audio to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)  # 2 bytes for int16
                wf.setframerate(RATE)
                wf.writeframes(audio_int16.tobytes())
            
            return temp_file.name
        except Exception as e:
            print(f"Error stopping audio recording: {e}")
    return None

def get_speech_text():
    try:
        audio_file = stop_audio_recording()
        if not audio_file:
            print("No audio file created")
            return "No audio recorded"
            
        print(f"Processing audio file: {audio_file}")
        
        # Use Whisper for speech recognition
        result = whisper_model.transcribe(audio_file)
        text = result["text"]
        print(f"Recognized text: {text}")
        return text
            
    except Exception as e:
        print(f"Speech recognition error: {e}")
        return f"Speech recognition failed: {str(e)}"
    finally:
        if audio_file and os.path.exists(audio_file):
            try:
                os.remove(audio_file)
            except Exception as e:
                print(f"Error removing temporary file: {e}")

def analyze_voice_emotion(text):
    try:
        # Use TextBlob for sentiment analysis
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity
        
        # Map sentiment to emotions
        if polarity > 0.5:
            emotion = 'confident' if subjectivity > 0.5 else 'happy'
        elif polarity > 0:
            emotion = 'calm' if subjectivity > 0.5 else 'neutral'
        elif polarity > -0.5:
            emotion = 'uncertain' if subjectivity > 0.5 else 'nervous'
        else:
            emotion = 'stressed' if subjectivity > 0.5 else 'negative'
            
        emotion_history['voice'].append(emotion)
        return emotion
    except Exception as e:
        print(f"Voice emotion analysis error: {e}")
        return 'neutral'

@atexit.register
def cleanup():
    global camera, audio, audio_stream
    if camera is not None:
        camera.release()
    if audio_stream is not None:
        audio_stream.stop_stream()
        audio_stream.close()
    audio.terminate()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("Starting integrated analysis system...")
    try:
        app.run(debug=True, port=5000)
    except Exception as e:
        print(f"Failed to start server: {e}")
        cleanup()