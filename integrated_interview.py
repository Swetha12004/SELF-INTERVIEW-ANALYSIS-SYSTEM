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
from collections import Counter
import re
import speech_recognition as sr
from gtts import gTTS

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
MAX_RECORDING_TIME = 180  # 3 minutes

# Global variables
camera = None
is_recording = False
start_time = None

# Interview questions by stage
INTRO_QUESTIONS = [
    "Hello! I'm your AI interviewer today. Could you please introduce yourself?",
    "What role are you interviewing for today?",
    "Which technical domain are you most interested in?"
]

TECHNICAL_QUESTIONS = {
    'python': [
        "Can you explain the difference between lists and tuples in Python?",
        "How does Python handle memory management?",
        "What are decorators in Python and how do you use them?"
    ],
    'java': [
        "What is the difference between HashMap and HashTable in Java?",
        "Explain the concept of Java Virtual Machine.",
        "How does garbage collection work in Java?"
    ],
    'web_development': [
        "Explain the difference between GET and POST requests.",
        "What are the main features of RESTful APIs?",
        "How does HTTPS work?"
    ]
}

# Speech Analysis System
class AnalysisSystem:
    def __init__(self):
        self.job_keywords = {
            'software_developer': ['python', 'java', 'coding', 'development', 'algorithms'],
            'data_scientist': ['machine learning', 'data', 'analysis', 'statistics'],
            'project_manager': ['management', 'leadership', 'agile', 'scrum']
        }
        self.technical_terms = set([word for keywords in self.job_keywords.values() 
                                  for word in keywords])

    def analyze_speech(self, audio_data):
        try:
            # Convert audio to text using speech recognition
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_data) as source:
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio)

            # Analyze speech content
            job_matches = self.analyze_job_matches(text)
            sentiment = TextBlob(text).sentiment
            
            return {
                'speech_text': text,
                'job_matches': job_matches,
                'confidence_score': sentiment.subjectivity * 100,
                'sentiment_score': (sentiment.polarity + 1) * 50
            }
        except Exception as e:
            print(f"Speech analysis error: {e}")
            return None

    def analyze_job_matches(self, text):
        text = text.lower()
        matches = {}
        for job, keywords in self.job_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            matches[job] = (score / len(keywords)) * 100
        return matches

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
                return camera
                
        raise RuntimeError("No working camera found")
    except Exception as e:
        print(f"Camera initialization error: {e}")
        return None

@app.route('/')
def index():
    return render_template('interview_index.html')

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        global camera, is_recording, start_time
        
        if camera is None:
            camera = init_camera()
        
        while True:
            success, frame = camera.read()
            if not success:
                continue
                
            try:
                # Analyze facial emotions
                result = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=False
                )
                emotion = result[0]['dominant_emotion']
                
                # Add emotion text to frame
                cv2.putText(
                    frame,
                    f"Emotion: {emotion}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                # Convert frame to jpg
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
            except Exception as e:
                print(f"Frame processing error: {e}")
                time.sleep(0.1)

    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_interview', methods=['POST'])
def start_interview():
    global is_recording, start_time
    try:
        is_recording = True
        start_time = time.time()
        return jsonify({
            "status": "success",
            "question": INTRO_QUESTIONS[0]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stop_interview', methods=['POST'])
def stop_interview():
    global is_recording
    try:
        is_recording = False
        return jsonify({
            "status": "success",
            "message": "Interview completed"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analyze_response', methods=['POST'])
def analyze_response():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
        
    try:
        audio_file = request.files['audio']
        analysis_system = AnalysisSystem()
        
        # Analyze speech
        results = analysis_system.analyze_speech(audio_file)
        
        if not results:
            return jsonify({"error": "Failed to analyze speech"}), 500
            
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting interview system...")
    try:
        app.run(debug=True, port=5000)
    except Exception as e:
        print(f"Failed to start server: {e}") 