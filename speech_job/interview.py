from flask import Flask, request, jsonify, render_template
import os
import random
import time
import re
import statistics
from PIL import Image
import io
import cv2
import numpy as np
from gtts import gTTS
import speech_recognition as sr

app = Flask(__name__, 
    template_folder='Templates',  # Specify template folder
    static_folder='static'        # Specify static folder
)

print("Starting interview system...")

# Store conversation history
conversations = {}

# Interview questions by stage
INTRO_QUESTIONS = [
    "Hello! I'm your AI interviewer today. Could you please introduce yourself?",
    "What role are you interviewing for today?",
    "Which technical domain are you most interested in? (Python/Java/Web/Data Science/DevOps/Cloud)"
]

# Expanded technical questions
TECHNICAL_QUESTIONS = {
    'python': [
        "Can you explain the difference between lists and tuples in Python?",
        "How does Python handle memory management?",
        "What are decorators in Python and how do you use them?",
        "Explain the concept of generators in Python.",
        "How do you handle exceptions in Python?",
        "What is the GIL in Python?",
        "Explain Python's garbage collection mechanism.",
        "What are context managers in Python?"
    ],
    'java': [
        "What is the difference between HashMap and HashTable in Java?",
        "Explain the concept of Java Virtual Machine.",
        "How does garbage collection work in Java?",
        "What are the differences between interface and abstract class?",
        "Explain the concept of multithreading in Java.",
        "What are Java memory areas?",
        "Explain the Spring Framework architecture."
    ],
    'web_development': [
        "Explain the difference between GET and POST requests.",
        "What are the main features of RESTful APIs?",
        "How does HTTPS work?",
        "Explain the concept of CORS.",
        "What is the difference between localStorage and sessionStorage?",
        "Explain how React's Virtual DOM works.",
        "What are Web Workers?"
    ],
    'data_science': [
        "Explain the difference between supervised and unsupervised learning.",
        "What is overfitting and how do you prevent it?",
        "Explain the concept of feature scaling.",
        "What is the difference between bias and variance?",
        "How does random forest work?"
    ],
    'devops': [
        "Explain the CI/CD pipeline.",
        "What is Docker and how does it work?",
        "Explain the concept of Infrastructure as Code.",
        "How do you handle container orchestration?",
        "What are the key components of Kubernetes?"
    ],
    'cloud': [
        "Explain cloud service models (IaaS, PaaS, SaaS).",
        "What are the key AWS services you've worked with?",
        "How do you handle cloud security?",
        "Explain the concept of auto-scaling.",
        "What is serverless computing?"
    ]
}

# Add follow-up questions
FOLLOW_UP_QUESTIONS = {
    'python': {
        'list': "How would you choose between a list and a tuple?",
        'decorator': "Can you give an example of a practical decorator use case?",
        'generator': "When would you use a generator instead of a list?",
        'exception': "What's the difference between try-except and try-finally?"
    },
    'java': {
        'hashmap': "How do you handle collisions in HashMap?",
        'jvm': "What are the main components of JVM?",
        'interface': "When would you choose an interface over an abstract class?",
        'thread': "How do you handle thread synchronization?"
    },
    'web_development': {
        'api': "What are the best practices for API security?",
        'http': "What are the common HTTP status codes?",
        'cors': "How do you handle CORS in your applications?",
        'storage': "When would you use localStorage vs cookies?"
    }
}

# Add scoring and feedback
SCORING_CRITERIA = {
    'python': {
        'keywords': ['decorator', 'generator', 'gil', 'context manager', 'exception handling', 
                    'memory management', 'garbage collection', 'list comprehension'],
        'concepts': ['threading', 'async', 'oop', 'functional programming']
    },
    'java': {
        'keywords': ['jvm', 'garbage collection', 'multithreading', 'spring', 'hibernate',
                    'interface', 'abstract class', 'hashmap'],
        'concepts': ['polymorphism', 'inheritance', 'encapsulation', 'dependency injection']
    }
    # Add for other domains...
}

# Add these new constants after your existing ones

INTERVIEW_MAX_DURATION = 1800  # 30 minutes in seconds
SUSPICIOUS_PATTERNS = {
    'copy_paste': [
        r'copied from',
        r'stackoverflow\.com',
        r'github\.com',
        r'geeksforgeeks\.org',
    ],
    'keyword_stuffing': r'(\b\w+\b)(?:\s+\1\b)+',  # Repeated words
    'off_topic': [
        r'unrelated content',
        r'irrelevant information',
    ]
}

RESPONSE_METRICS = {
    'min_length': 50,  # Minimum expected response length
    'max_typing_speed': 200,  # characters per second
    'min_response_time': 5,  # seconds
}

# Add these constants at the top with other constants
TIME_LIMITS = {
    'total_interview': 1800,  # 30 minutes
    'per_question': 300,      # 5 minutes per question
    'warning_threshold': 300  # Show warning when 5 minutes remaining
}

# Add these constants at the top
INTERVIEW_TERMINATION_CONDITIONS = {
    'max_empty_responses': 2,  # Maximum number of empty/invalid responses
    'min_response_length': 20,  # Minimum characters for a valid response
    'max_malpractice_score': 5  # Maximum allowed malpractice incidents
}

# Add this new class for analysis
class InterviewAnalyzer:
    def __init__(self):
        self.malpractice_incidents = []
        self.response_times = []
        self.response_lengths = []
        self.keywords_used = set()
        self.start_time = time.time()

    def analyze_response(self, text, response_time):
        issues = []
        
        # Minimum response check
        if len(text.strip()) < 20:  # Minimum 20 characters
            issues.append({
                'type': 'insufficient_response',
                'detail': 'Response is too brief. Please provide more detailed answers.'
            })
            return issues  # Return early for very short responses
        
        # Check response time
        if response_time < RESPONSE_METRICS['min_response_time']:
            issues.append({
                'type': 'suspicious_timing',
                'detail': f'Very quick response ({response_time:.1f}s)'
            })
        
        # Check response length
        if len(text) < RESPONSE_METRICS['min_length']:
            issues.append({
                'type': 'insufficient_detail',
                'detail': 'Response too brief'
            })
        
        # Check for copy-paste patterns
        for pattern in SUSPICIOUS_PATTERNS['copy_paste']:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append({
                    'type': 'potential_plagiarism',
                    'detail': f'Found pattern indicating copied content: {pattern}'
                })
        
        # Check for keyword stuffing
        if re.search(SUSPICIOUS_PATTERNS['keyword_stuffing'], text, re.IGNORECASE):
            issues.append({
                'type': 'keyword_stuffing',
                'detail': 'Suspicious repetition of words'
            })
        
        return issues

    def generate_report(self, conversation_data):
        total_time = time.time() - self.start_time
        
        report = {
            'duration': f"{total_time/60:.1f} minutes",
            'total_questions': len(conversation_data['questions_asked']),
            'technical_score': conversation_data['score'],
            'response_quality': self._analyze_response_quality(conversation_data),
            'malpractice_incidents': self.malpractice_incidents,
            'recommendations': self._generate_recommendations(conversation_data)
        }
        
        return report

    def _analyze_response_quality(self, conversation_data):
        return {
            'keyword_coverage': len(conversation_data['keywords_mentioned']),
            'average_response_length': sum(self.response_lengths) / len(self.response_lengths) if self.response_lengths else 0,
            'response_time_consistency': self._calculate_time_consistency()
        }

    def _calculate_time_consistency(self):
        if len(self.response_times) < 2:
            return "Insufficient data"
        
        avg_time = sum(self.response_times) / len(self.response_times)
        variance = sum((t - avg_time) ** 2 for t in self.response_times) / len(self.response_times)
        return "Consistent" if variance < 100 else "Inconsistent"

    def _generate_recommendations(self, conversation_data):
        recommendations = []
        
        if conversation_data['score'] < 10:
            recommendations.append("Focus on using more technical terms and concepts")
        
        if len(self.malpractice_incidents) > 0:
            recommendations.append("Avoid copy-pasting and provide original answers")
        
        # Add check for empty response_lengths
        if self.response_lengths:  # Only calculate mean if there's data
            if statistics.mean(self.response_lengths) < 100:
                recommendations.append("Provide more detailed explanations")
        else:
            recommendations.append("Provide more detailed explanations")
        
        return recommendations

# Add this new class for time management
class InterviewTimer:
    def __init__(self):
        self.start_time = time.time()
        self.question_start_time = time.time()
        self.warnings_sent = set()
    
    def get_time_status(self):
        current_time = time.time()
        elapsed_total = current_time - self.start_time
        elapsed_question = current_time - self.question_start_time
        remaining_total = TIME_LIMITS['total_interview'] - elapsed_total
        
        return {
            'elapsed_total': int(elapsed_total),
            'elapsed_question': int(elapsed_question),
            'remaining_total': int(remaining_total),
            'time_warning': self._get_time_warning(remaining_total)
        }
    
    def _get_time_warning(self, remaining):
        if remaining <= TIME_LIMITS['warning_threshold'] and 'final_warning' not in self.warnings_sent:
            self.warnings_sent.add('final_warning')
            return f"⚠️ Warning: {int(remaining/60)} minutes remaining in the interview"
        return None
    
    def new_question(self):
        self.question_start_time = time.time()

def determine_domain(text):
    """Determine the technical domain based on user response"""
    text = text.lower()
    domain_keywords = {
        'python': ['python', 'django', 'flask'],
        'java': ['java', 'spring', 'hibernate'],
        'web_development': ['web', 'frontend', 'backend', 'javascript', 'react', 'angular'],
        'data_science': ['data', 'machine learning', 'ai', 'analytics'],
        'devops': ['devops', 'ci/cd', 'docker', 'kubernetes'],
        'cloud': ['cloud', 'aws', 'azure', 'gcp']
    }
    
    for domain, keywords in domain_keywords.items():
        if any(keyword in text for keyword in keywords):
            return domain
    return 'python'  # default domain

def get_follow_up_question(text, domain):
    """Get relevant follow-up question based on keywords"""
    text = text.lower()
    follow_ups = FOLLOW_UP_QUESTIONS.get(domain, {})
    for keyword, question in follow_ups.items():
        if keyword in text:
            return question
    return None

def analyze_response(text, domain):
    """Analyze user response for keywords and concepts"""
    text = text.lower()
    score = 0
    feedback = []
    
    if domain in SCORING_CRITERIA:
        # Check for keywords
        for keyword in SCORING_CRITERIA[domain]['keywords']:
            if keyword in text:
                score += 1
                feedback.append(f"Good use of term: {keyword}")
        
        # Check for concepts
        for concept in SCORING_CRITERIA[domain]['concepts']:
            if concept in text:
                score += 2
                feedback.append(f"Excellent explanation of {concept}")
    
    return score, feedback

def should_terminate_interview(conv_data, text_input):
    """Check if interview should be terminated"""
    reasons = []
    
    # Check for empty or too short responses
    if len(text_input.strip()) < INTERVIEW_TERMINATION_CONDITIONS['min_response_length']:
        conv_data['empty_responses'] = conv_data.get('empty_responses', 0) + 1
        if conv_data['empty_responses'] >= INTERVIEW_TERMINATION_CONDITIONS['max_empty_responses']:
            reasons.append("Multiple insufficient responses detected")
    
    # Check malpractice score
    if len(conv_data['analyzer'].malpractice_incidents) >= INTERVIEW_TERMINATION_CONDITIONS['max_malpractice_score']:
        reasons.append("Excessive suspicious activities detected")
    
    # Check for off-topic responses
    if conv_data['stage'] == 'technical' and not any(keyword in text_input.lower() 
        for keyword in SCORING_CRITERIA.get(conv_data['domain'], {}).get('keywords', [])):
        conv_data['off_topic_responses'] = conv_data.get('off_topic_responses', 0) + 1
        if conv_data['off_topic_responses'] >= 2:
            reasons.append("Multiple off-topic responses")
    
    return bool(reasons), reasons

def generate_response(text_input, conversation_id):
    """Generate interview response based on conversation stage"""
    if conversation_id not in conversations:
        conversations[conversation_id] = {
            'stage': 'intro',
            'stage_index': 0,
            'domain': None,
            'questions_asked': set(),
            'score': 0,
            'keywords_mentioned': set(),
            'feedback': [],
            'analyzer': InterviewAnalyzer(),
            'timer': InterviewTimer(),
            'empty_responses': 0,
            'off_topic_responses': 0
        }
    
    conv_data = conversations[conversation_id]
    
    # Check for termination conditions
    should_terminate, reasons = should_terminate_interview(conv_data, text_input)
    if should_terminate:
        return generate_final_report(conversation_id, termination_reasons=reasons)
    
    analyzer = conv_data['analyzer']
    timer = conv_data['timer']
    time_status = timer.get_time_status()
    
    # Check if total time exceeded
    if time_status['remaining_total'] <= 0:
        return generate_final_report(conversation_id)
    
    # Get time warning if any
    time_warning = time_status['time_warning']
    
    # Analyze response for malpractice
    response_time = time.time() - conv_data.get('last_question_time', time.time())
    issues = analyzer.analyze_response(text_input, response_time)
    
    if issues:
        analyzer.malpractice_incidents.extend(issues)
    
    # Update metrics
    analyzer.response_times.append(response_time)
    analyzer.response_lengths.append(len(text_input))
    
    # Check if interview time exceeded
    if time.time() - analyzer.start_time > INTERVIEW_MAX_DURATION:
        return generate_final_report(conversation_id)
    
    # Add scoring for technical stage
    if conv_data['stage'] == 'technical':
        score, feedback = analyze_response(text_input, conv_data['domain'])
        conv_data['score'] += score
        conv_data['feedback'].extend(feedback)
    
    # Introduction stage
    if conv_data['stage'] == 'intro':
        if conv_data['stage_index'] < len(INTRO_QUESTIONS) - 1:
            conv_data['stage_index'] += 1
            return INTRO_QUESTIONS[conv_data['stage_index']]
        else:
            conv_data['stage'] = 'technical'
            conv_data['domain'] = determine_domain(text_input)
            first_question = random.choice(TECHNICAL_QUESTIONS[conv_data['domain']])
            conv_data['questions_asked'].add(first_question)
            return f"Great! Let's start with some {conv_data['domain']} questions.\n\n{first_question}"
    
    # Technical stage
    else:
        domain = conv_data['domain']
        
        # Check for follow-up question based on user's response
        follow_up = get_follow_up_question(text_input, domain)
        if follow_up and follow_up not in conv_data['questions_asked']:
            conv_data['questions_asked'].add(follow_up)
            return f"Interesting point. {follow_up}"
        
        # If no follow-up, ask next technical question
        available_questions = [q for q in TECHNICAL_QUESTIONS[domain] 
                             if q not in conv_data['questions_asked']]
        
        if available_questions:
            next_question = random.choice(available_questions)
            conv_data['questions_asked'].add(next_question)
            return f"Thank you for your response.\n\n{next_question}"
        else:
            total_score = conv_data['score']
            feedback_summary = "\n".join(conv_data['feedback'])
            response = f"""
Thank you for completing the interview!

Your performance summary:
Score: {total_score}
Key strengths:
{feedback_summary}

Areas for improvement:
- Consider discussing more practical examples
- Explain concepts in more detail
- Share specific project experiences
"""
            
            # Add time warning to response if exists
            if time_warning:
                response = f"{time_warning}\n\n{response}"
            
            # Update question timer when moving to next question
            timer.new_question()
            
            return response

def generate_final_report(conversation_id, termination_reasons=None):
    conv_data = conversations[conversation_id]
    analyzer = conv_data['analyzer']
    timer = conv_data['timer']
    time_status = timer.get_time_status()
    
    report = analyzer.generate_report(conv_data)
    
    # Add termination reason if interview was terminated early
    termination_info = ""
    if termination_reasons:
        termination_info = "\n⚠️ Interview Terminated Early\nReasons:\n" + "\n".join(f"- {reason}" for reason in termination_reasons)
    
    return f"""
INTERVIEW COMPLETION REPORT
{termination_info}

Time Analysis:
- Total Duration: {time_status['elapsed_total'] // 60} minutes {time_status['elapsed_total'] % 60} seconds
- Average Time per Question: {time_status['elapsed_total'] / max(len(conv_data['questions_asked']), 1):.1f} seconds
- Time Management: {"Good" if time_status['elapsed_total'] < TIME_LIMITS['total_interview'] else "Time Exceeded"}

Technical Score: {report['technical_score']}/20
Final Grade: {"Failed" if termination_reasons else "Passed" if report['technical_score'] > 10 else "Failed"}

Performance Analysis:
- Questions Answered: {report['total_questions']}
- Questions Attempted: {len(conv_data['questions_asked'])}
- Response Quality: {report['response_quality']['average_response_length']:.0f} chars average
- Timing Consistency: {report['response_quality']['response_time_consistency']}

{'🚩 Suspicious Activities Detected:' if report['malpractice_incidents'] else 'No suspicious activities detected'}
{chr(10).join(f"- {incident['type']}: {incident['detail']}" for incident in report['malpractice_incidents'])}

Interview Status: {"Terminated Early" if termination_reasons else "Completed"}
{"Termination Reasons:" if termination_reasons else ""}
{chr(10).join(f"- {reason}" for reason in (termination_reasons or []))}

Recommendations:
{chr(10).join(f"- {rec}" for rec in report['recommendations'])}

Thank you for participating in the interview.
"""

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/start_interview', methods=['POST'])
def start_interview():
    """Start a new interview session"""
    conversation_id = str(time.time())
    conversations[conversation_id] = {
        'stage': 'intro',
        'stage_index': 0,
        'domain': None,
        'questions_asked': set(),
        'score': 0,
        'keywords_mentioned': set(),
        'feedback': [],
        'analyzer': InterviewAnalyzer(),
        'timer': InterviewTimer(),
        'empty_responses': 0,
        'off_topic_responses': 0
    }
    
    return jsonify({
        'conversation_id': conversation_id,
        'message': INTRO_QUESTIONS[0],
        'time_status': conversations[conversation_id]['timer'].get_time_status()
    })

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message', '').strip()
    conversation_id = data.get('conversation_id')
    
    if not conversation_id:
        return jsonify({'error': 'Missing conversation ID'}), 400
    
    try:
        # Handle empty or very short responses
        if not user_input or len(user_input) < INTERVIEW_TERMINATION_CONDITIONS['min_response_length']:
            conv_data = conversations.get(conversation_id, {})
            conv_data['empty_responses'] = conv_data.get('empty_responses', 0) + 1
            if conv_data['empty_responses'] >= INTERVIEW_TERMINATION_CONDITIONS['max_empty_responses']:
                response = generate_final_report(conversation_id, ["Interview terminated due to insufficient responses"])
                return jsonify({
                    'response': response,
                    'conversation_id': conversation_id,
                    'is_final_report': True,
                    'terminated_early': True
                })
        
        response = generate_response(user_input, conversation_id)
        time_status = conversations[conversation_id]['timer'].get_time_status()
        
        # Check if this is a final report
        is_final_report = "INTERVIEW COMPLETION REPORT" in response
        was_terminated = "Interview Terminated Early" in response
        
        return jsonify({
            'response': response,
            'conversation_id': conversation_id,
            'is_final_report': is_final_report,
            'terminated_early': was_terminated,
            'score': conversations[conversation_id]['score'],
            'feedback': conversations[conversation_id]['feedback'][-3:] if not is_final_report else [],
            'time_status': time_status
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Get the image from the request
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Load face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Analyze attention and confidence
        if len(faces) > 0:
            attention_level = "Good"
            confidence_level = "High"
        else:
            attention_level = "Low - Face not detected"
            confidence_level = "Low"
        
        return jsonify({
            'attention_level': attention_level,
            'confidence_level': confidence_level
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting interview system...")
    try:
        print("Starting server on http://127.0.0.1:3000")
        app.run(host='127.0.0.1', port=3000, debug=True)
    except Exception as e:
        print(f"Failed to start on port 3000, trying port 8090...")
        try:
            app.run(host='127.0.0.1', port=8090, debug=True)
        except Exception as e2:
            print(f"Failed to start server: {e2}")