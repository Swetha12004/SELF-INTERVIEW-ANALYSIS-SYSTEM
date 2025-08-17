from flask import Flask, request, jsonify, render_template
import os
import time
import numpy as np
from textblob import TextBlob
from collections import Counter
import re

class AnalysisSystem:
    def __init__(self):
        self.job_keywords = {
            'software_developer': ['python', 'java', 'coding', 'development', 'algorithms'],
            'data_scientist': ['machine learning', 'data', 'analysis', 'statistics', 'python'],
            'project_manager': ['management', 'leadership', 'agile', 'scrum', 'coordination']
        }
        
        self.technical_terms = set([word for keywords in self.job_keywords.values() for word in keywords])

    def process_speech(self, audio_path):
        """Process speech from audio file"""
        try:
            # Placeholder for speech-to-text
            speech_text = """
            Hi, I'm a software engineer with 5 years of experience in Python development and machine learning. 
            I've worked extensively with data analysis and algorithms, developing scalable solutions using Java 
            and Python frameworks. I've led several agile development teams and managed complex projects using 
            scrum methodology. My recent work involves implementing machine learning models for data analysis 
            and statistical modeling. I'm particularly interested in combining software development with 
            data science techniques to build intelligent systems. I have strong leadership skills and experience 
            in project coordination, always focusing on efficient development and team management.
            
            Some of my key technical skills include:
            - Python programming and frameworks (Django, Flask)
            - Machine learning and statistical analysis
            - Java development and algorithms
            - Agile project management
            - Data analysis and visualization
            
            I'm looking for opportunities where I can apply both my coding expertise and analytical skills 
            while taking on leadership responsibilities in development projects.
            """
            
            # Analyze various aspects
            job_matches = self.analyze_job_matches(speech_text)
            keywords, counts = self.analyze_keywords(speech_text)
            performance_metrics = self.analyze_performance(speech_text)
            
            return {
                'speech_text': speech_text,
                'job_matches': job_matches,
                'keywords': keywords,
                'keyword_counts': counts,
                'technical_score': performance_metrics['technical'],
                'communication_score': performance_metrics['communication'],
                'confidence_score': performance_metrics['confidence'],
                'domain_knowledge': performance_metrics['domain_knowledge'],
                'overall_match': performance_metrics['overall']
            }
            
        except Exception as e:
            print(f"Error processing speech: {e}")
            return {
                'speech_text': "Error processing speech",
                'job_matches': {},
                'keywords': [],
                'keyword_counts': [],
                'technical_score': 0,
                'communication_score': 0,
                'confidence_score': 0,
                'domain_knowledge': 0,
                'overall_match': 0
            }

    def analyze_job_matches(self, text):
        """Analyze text for job keyword matches"""
        text = text.lower()
        matches = {}
        
        for job, keywords in self.job_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            matches[job] = (score / len(keywords)) * 100
            
        return matches

    def analyze_keywords(self, text):
        """Analyze keyword frequency"""
        words = re.findall(r'\b\w+\b', text.lower())
        technical_words = [word for word in words if word in self.technical_terms]
        counter = Counter(technical_words)
        keywords = list(counter.keys())
        counts = list(counter.values())
        return keywords, counts

    def analyze_performance(self, text):
        """Analyze various performance metrics"""
        # Calculate technical score
        technical_words = sum(1 for word in text.lower().split() if word in self.technical_terms)
        technical_score = min(100, technical_words * 20)
        
        # Simple metrics for demonstration
        word_count = len(text.split())
        communication_score = min(100, word_count * 2)
        confidence_score = 75  # Placeholder
        domain_knowledge = technical_score
        
        overall_match = (technical_score + communication_score + confidence_score + domain_knowledge) / 4
        
        return {
            'technical': technical_score,
            'communication': communication_score,
            'confidence': confidence_score,
            'domain_knowledge': domain_knowledge,
            'overall': overall_match
        }

    def analyze_combined_factors(self, speech_text, voice_emotion, facial_emotion):
        """Combine different factors for overall analysis"""
        job_scores = {
            'software_developer': 0,
            'data_scientist': 0,
            'project_manager': 0
        }
        
        # Analyze text content
        text_scores = self.analyze_job_matches(speech_text)
        
        # Weight different factors
        for job in job_scores:
            text_weight = 0.6
            emotion_weight = 0.4
            
            base_score = text_scores.get(job, 0)
            emotion_score = 70 if facial_emotion in ['happy', 'neutral'] else 50
            
            job_scores[job] = (base_score * text_weight) + (emotion_score * emotion_weight)
        
        return job_scores

# Initialize Flask app
app = Flask(__name__)
analysis_system = AnalysisSystem()

@app.route('/')
def index():
    return render_template('speech_index.html')

@app.route('/analyze_speech', methods=['POST'])
def analyze_speech():
    try:
        # Get audio file
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
            
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        # Process speech
        speech_data = analysis_system.process_speech(audio_file)
        
        return jsonify(speech_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)