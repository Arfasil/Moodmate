import os
import cv2
from flask import Flask, request, jsonify, render_template, Response, send_from_directory
from deepface import DeepFace
import numpy as np
import tempfile
import logging
import base64
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_migrate import Migrate
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
import speech_recognition as sr
import pyttsx3
import time
load_dotenv()


app = Flask(__name__)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///psychologists.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'ab663ac0e322099b21c2ccb909231506901f38466789e478'  # Important for migrations

db = SQLAlchemy(app)
migrate = Migrate(app, db)
CORS(app)
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel("gemini-2.5-flash")
conversation_history = []

# Initialize text-to-speech engine
engine = pyttsx3.init()

def create_mental_health_prompt(user_input, chat_history=None):
    """Create a context-aware prompt for mental health responses."""
    system_prompt = """You are an empathetic and supportive mental health companion. Your responses should:
    1. Show genuine understanding and validate feelings
    2. Offer practical, actionable coping strategies when appropriate
    3. Maintain a warm, non-judgmental tone
    4. Encourage professional help when needed
    5. Never provide medical diagnosis or treatment advice
    6. Use person-centered language and active listening techniques
    7. Be concise but thoughtful
    8. Focus on empowerment and resilience
    
    Previous conversation context (if any):
    {chat_history}
    
    Please respond to the following message with empathy and care: {user_input}
    """
    
    # Format chat history if available
    history_text = ""
    if chat_history:
        history_text = "\n".join([f"User: {msg['user']}\nAssistant: {msg['bot']}" 
                                for msg in chat_history[-3:]])  # Include last 3 messages for context
    
    return system_prompt.format(chat_history=history_text, user_input=user_input)

def get_safety_disclaimer():
    """Return safety disclaimer for crisis situations."""
    return """
    If you're having thoughts of self-harm or suicide, please know that help is available 24/7:
    - National Crisis Helpline (US): 988
    - Crisis Text Line: Text HOME to 741741
    - Emergency Services: 911
    
    You're not alone, and professional help is available to support you through this difficult time.
    """

def contains_crisis_keywords(text):
    """Check if the text contains crisis-related keywords."""
    crisis_keywords = [
        'suicide', 'kill myself', 'die', 'end it all', 'self-harm', 'hurt myself',
        'hopeless', 'can\'t go on', 'better off dead', 'no reason to live',
        'want to die', 'end my life', 'give up'
    ]
    return any(keyword in text.lower() for keyword in crisis_keywords)

def text_to_speech(text):
    """Convert text to speech."""
    try:
        engine.say(text)
        engine.runAndWait()
        return True
    except Exception as e:
        print(f"Text-to-speech error: {str(e)}")
        return False

def speech_to_text():
    """Convert speech to text using Google Speech Recognition with a time limit."""
    recognizer = sr.Recognizer()
    max_duration = 30  # Maximum recording duration in seconds
    
    try:
        with sr.Microphone() as source:
            print("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            print("Listening... (Max duration: 30 seconds)")
            start_time = time.time()
            
            # Set dynamic energy threshold
            recognizer.dynamic_energy_threshold = True
            recognizer.energy_threshold = 4000
            
            # Listen with timeout and phrase time limit
            audio = recognizer.listen(
                source,
                timeout=5,  # Wait up to 5 seconds for the phrase to start
                phrase_time_limit=max_duration  # Stop listening after max_duration
            )
            
            # Check if maximum duration exceeded
            if time.time() - start_time > max_duration:
                return "Recording time limit exceeded. Please try again with a shorter message."
            
            text = recognizer.recognize_google(audio)
            print(f"Recognized text: {text}")
            return text
            
    except sr.WaitTimeoutError:
        return "No speech detected within timeout period"
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand that. Please speak clearly and try again"
    except sr.RequestError as e:
        return f"Sorry, there was an error with the speech recognition service: {str(e)}"
    except Exception as e:
        print(f"Speech recognition error: {str(e)}")
        return "An error occurred during speech recognition"
PHQ9_SEVERITY = {
    "minimal": (0, 4),
    "mild": (5, 9),
    "moderate": (10, 14),
    "moderately_severe": (15, 19),
    "severe": (20, 27)
}

# Emotion weights for risk calculation (negative emotions increase risk)
EMOTION_WEIGHTS = {
    "sadness": 0.8,
    "anger": 0.5,
    "fear": 0.7,
    "disgust": 0.4,
    "surprise": 0.1,
    "happiness": -0.9  # Reduces risk
}

# Intervention recommendations based on risk level
INTERVENTIONS = {
    "low": [
        "Practice 5 minutes of mindful breathing",
        "Take a short walk outside",
        "Write down three things you're grateful for"
    ],
    "medium": [
        "Try a 10-minute guided meditation",
        "Reach out to a friend or family member",
        "Practice progressive muscle relaxation",
        "Consider scheduling a check-in with a counselor"
    ],
    "high": [
        "Please consider contacting a mental health professional",
        "Call or text a crisis helpline: 988",
        "Practice grounding technique: identify 5 things you see, 4 things you feel, 3 things you hear, 2 things you smell, 1 thing you taste",
        "Reach out to a trusted support person"
    ]
}



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/emotion')
def MentalStateDetection():
    return render_template('emotion.html')

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        file.save(temp_file.name)
    
    try:
        result = DeepFace.analyze(img_path=temp_file.name, actions=['emotion'], enforce_detection=True)
        os.unlink(temp_file.name)
        
        emotions = result[0]['emotion']
        dominant_emotion = result[0]['dominant_emotion']
        confidence = emotions[dominant_emotion] / sum(emotions.values())
        
        confidence = float(confidence)
        
        insights = get_mental_health_insights(dominant_emotion)
        
        return jsonify({
            'emotion': dominant_emotion,
            'confidence': confidence,
            'insights': insights
        })
    
    except Exception as e:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        app.logger.error(f"Emotion detection error: {e}")
        return jsonify({'error': str(e), 'emotion': 'Unknown', 'confidence': 0, 'insights': {}}), 500

@app.route('/resources')
def mental_health_resources():
    resources = [
        {'name': 'National Suicide Prevention Lifeline', 'description': 'Free, confidential support for people in distress', 'contact': '988', 'website': 'https://988lifeline.org/'},
        {'name': 'Crisis Text Line', 'description': 'Free 24/7 support via text', 'text': 'HOME to 741741', 'website': 'https://www.crisistextline.org/'},
        {'name': 'SAMHSA National Helpline', 'description': 'Treatment referral and information service', 'phone': '1-800-662-HELP (4357)', 'website': 'https://www.samhsa.gov/find-help/national-helpline'}
    ]
    return render_template('resources.html', resources=resources)

def get_mental_health_insights(emotion):
    insights = {
        'happy': {'insight': 'It\'s great that you\'re feeling positive! Maintain this emotional state by practicing gratitude and engaging in activities you enjoy.', 'risk_level': 'Low', 'recommendation': 'Continue your current positive habits and self-care routine.'},
        'sad': {'insight': 'You seem to be experiencing sadness. It\'s important to acknowledge your feelings and seek support if needed.', 'risk_level': 'Medium', 'recommendation': 'Consider talking to a trusted friend, family member, or mental health professional.'},
        'angry': {'insight': 'Anger can be a challenging emotion. It\'s important to find healthy ways to process and express your feelings.', 'risk_level': 'Medium', 'recommendation': 'Try stress-management techniques like deep breathing, meditation, or physical exercise.'},
        'fear': {'insight': 'Feelings of fear or anxiety can be overwhelming. It\'s okay to seek help and develop coping strategies.', 'risk_level': 'High', 'recommendation': 'Practice grounding techniques, consider professional counseling.'},
        'surprise': {'insight': 'Surprise can trigger various emotional responses. Take a moment to understand and process your feelings.', 'risk_level': 'Low', 'recommendation': 'Reflect on the source of surprise and your emotional reaction.'},
        'disgust': {'insight': 'Feelings of disgust might indicate underlying stress or discomfort. It\'s important to identify the root cause.', 'risk_level': 'Medium', 'recommendation': 'Practice self-reflection and consider talking to a counselor.'},
        'neutral': {'insight': 'A neutral emotional state can be a good baseline for emotional stability.', 'risk_level': 'Low', 'recommendation': 'Continue practicing emotional awareness and self-care.'}
    }
    emotion = emotion.lower()
    return insights.get(emotion, {'insight': 'Your emotions are unique and valid. Consider exploring them further.', 'risk_level': 'Undefined', 'recommendation': 'Seek professional guidance for personalized support.'})

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            app.logger.error("Failed to capture frame from camera.")
            break

        dominant_emotion = analyze_emotion_from_frame(frame)  # Assuming this is defined elsewhere.
        frame_base64 = encode_frame_to_base64(frame)  # Assuming this is defined elsewhere.
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + base64.b64decode(frame_base64) + b'\r\n')

    cap.release()
    
# Global variable to store current emotion
current_emotion = "No emotion detected"

def generate_frames():
    global current_emotion
    cap = cv2.VideoCapture(0)
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Convert grayscale frame to RGB format
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = rgb_frame[y:y + h, x:x + w]
            
            try:
                # Perform emotion analysis on the face ROI
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                
                # Determine the dominant emotion
                current_emotion = result[0]['dominant_emotion']
                
                # Draw rectangle around face and label with predicted emotion
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, current_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            except Exception as e:
                print(f"Error in emotion detection: {e}")
        
        # Encode frame to base64 for sending to frontend
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + base64.b64decode(frame_base64) + b'\r\n')
    
    cap.release()
@app.route('/calculate_risk', methods=['POST'])
def calculate_risk():
    try:
        data = request.json
        
        # Extract PHQ-9 responses (values should be 0-3 for each question)
        phq9_responses = data.get('phq9_responses', [])
        if not phq9_responses or len(phq9_responses) != 9:
            return jsonify({"error": "Invalid PHQ-9 data"}), 400
            
        # Extract emotion probabilities (values should be 0-1 for each emotion)
        emotion_probs = data.get('emotion_probs', {})
        required_emotions = set(EMOTION_WEIGHTS.keys())
        if not all(emotion in emotion_probs for emotion in required_emotions):
            return jsonify({"error": "Missing emotion data"}), 400
        
        # Calculate PHQ-9 score
        phq9_score = sum(phq9_responses)
        
        # Determine PHQ-9 severity category
        phq9_severity = "minimal"
        for severity, (min_score, max_score) in PHQ9_SEVERITY.items():
            if min_score <= phq9_score <= max_score:
                phq9_severity = severity
                break
        
        # Calculate emotion component of risk
        emotion_risk = sum(emotion_probs[emotion] * weight for emotion, weight in EMOTION_WEIGHTS.items())
        
        # Normalize emotion risk to 0-1 scale
        max_possible_emotion_risk = sum(abs(weight) for weight in EMOTION_WEIGHTS.values())
        normalized_emotion_risk = (emotion_risk + 1.5) / (max_possible_emotion_risk)
        normalized_emotion_risk = max(0, min(normalized_emotion_risk, 1))
        
        # Normalize PHQ-9 score to 0-1 scale
        normalized_phq9 = phq9_score / 27
        
        # Calculate composite risk score (70% PHQ-9, 30% emotion)
        composite_risk = (0.7 * normalized_phq9) + (0.3 * normalized_emotion_risk)
        
        # Determine risk level
        if composite_risk < 0.3:
            risk_level = "low"
        elif composite_risk < 0.6:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        # Get appropriate interventions
        recommended_interventions = INTERVENTIONS[risk_level]
        
        # Special case: Question 9 of PHQ-9 is about self-harm, handle separately
        if phq9_responses[8] >= 1:  # If scored 1 or higher on question 9
            risk_level = "high"
            recommended_interventions = INTERVENTIONS["high"]
            recommended_interventions.insert(0, "Your response indicates thoughts of self-harm. Please seek immediate support.")
        
        return jsonify({
            "phq9_score": phq9_score,
            "phq9_severity": phq9_severity,
            "emotion_risk": round(normalized_emotion_risk, 2),
            "composite_risk": round(composite_risk, 2),
            "risk_level": risk_level,
            "interventions": recommended_interventions
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/face')
def faceEmotionDetection():
    return render_template('face.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_emotion')
def get_emotion():
    return jsonify({"emotion": current_emotion})

@app.route('/Zoom')
def doctalk():
    return render_template('talkToHeel.html')
@app.route('/Depression Detction')
def depression():
    return render_template('depression.html')
class Psychologist(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    specialization = db.Column(db.String(200), nullable=False)
    active_hours_start = db.Column(db.String(20), nullable=False)
    active_hours_end = db.Column(db.String(20), nullable=False)
    contact_email = db.Column(db.String(100), nullable=False)
    availability = db.Column(db.Boolean, default=True)
    zoom_link = db.Column(db.String(200))


class ZoomRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_name = db.Column(db.String(100), nullable=False)
    patient_email = db.Column(db.String(100), nullable=False)
    psychologist_id = db.Column(db.Integer, db.ForeignKey('psychologist.id'), nullable=False)
    request_time = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default='Pending')  # Pending, Accepted, Rejected


# Seed psychologists if the table is empty
def seed_psychologists():
    if Psychologist.query.count() > 0:
        return  # Skip if data already exists
    
    psychologists_data = [
        {
            'name': 'Dr. Emily Chen',
            'specialization': 'Anxiety and Depression',
            'active_hours_start': '09:00 AM',
            'active_hours_end': '4:00 PM',
            'contact_email': 'emily.chen@example.com',
            'availability': True,
            'zoom_link': 'https://us05web.zoom.us/j/89621517906?pwd=qbXGnB7wlzuUZighk3VujNO7tLRqdn.1'
        },
        {
            'name': 'Dr. Michael Rodriguez',
            'specialization': 'Relationship Counseling',
            'active_hours_start': '10:00',
            'active_hours_end': '18:00',
            'contact_email': 'michael.rodriguez@example.com',
            'availability': True,
            'zoom_link': 'https://us05web.zoom.us/j/88943987130?pwd=4YETDaTqZRPzYKbGBWYag1rxeOka2k.1'
        },
        {
            'name': 'Dr. Sarah Patel',
            'specialization': 'Trauma and PTSD',
            'active_hours_start': '08:00',
            'active_hours_end': '16:00',
            'contact_email': 'sarah.patel@example.com',
            'availability': True,
            'zoom_link': 'https://us05web.zoom.us/j/88943987130?pwd=4YETDaTqZRPzYKbGBWYag1rxeOka2k.1'
        },
        {
            'name': 'Dr. John Kim',
            'specialization': 'Family Therapy',
            'active_hours_start': '11:00',
            'active_hours_end': '19:00',
            'contact_email': 'john.kim@example.com',
            'availability': True,
            'zoom_link': 'https://us05web.zoom.us/j/88943987130?pwd=4YETDaTqZRPzYKbGBWYag1rxeOka2k.1'
        },
        {
            'name': 'Dr. Lisa Martinez',
            'specialization': 'Child Psychology',
            'active_hours_start': '07:00',
            'active_hours_end': '15:00',
            'contact_email': 'https://us05web.zoom.us/j/88943987130?pwd=4YETDaTqZRPzYKbGBWYag1rxeOka2k.1',
            'availability': True,
            'zoom_link': 'https://zoom.us/j/5678901234'
        }
    ]
    
    for psych_data in psychologists_data:
        psychologist = Psychologist(**psych_data)
        db.session.add(psychologist)
    
    db.session.commit()


# Routes
@app.route('/psychologists', methods=['GET'])
def get_psychologists():
    psychologists = Psychologist.query.all()
    return jsonify([
        {
            'id': p.id,
            'name': p.name,
            'specialization': p.specialization,
            'active_hours_start': p.active_hours_start,
            'active_hours_end': p.active_hours_end,
            'contact_email': p.contact_email,
            'availability': p.availability
        }
        for p in psychologists
    ])


@app.route('/submit-request', methods=['POST'])
def submit_zoom_request():
    data = request.json
    
    # Validate request data
    if not all(key in data for key in ['patient_name', 'patient_email', 'psychologist_id']):
        return jsonify({'error': 'Missing required fields'}), 400
    
    # Validate psychologist ID
    psychologist = Psychologist.query.get(data['psychologist_id'])
    if not psychologist:
        return jsonify({'error': 'Psychologist not found'}), 404
    
    # Create Zoom request
    new_request = ZoomRequest(
        patient_name=data['patient_name'],
        patient_email=data['patient_email'],
        psychologist_id=data['psychologist_id']
    )
    db.session.add(new_request)
    db.session.commit()
    
    return jsonify({
        'message': 'Request submitted successfully',
        'zoom_link': psychologist.zoom_link
    })


@app.before_request
def initialize_database():
    # Ensure the database tables exist
    db.create_all()
    seed_psychologists()
    
@app.route('/Mental Health Chat bot')
def chatbot():
    """Render the home page."""
    return render_template('chatbot.html')

@app.route('/text_chat', methods=['POST'])
def text_chat():
    """Handle text-based chat interactions."""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Create contextual prompt with conversation history
        prompt = create_mental_health_prompt(user_message, conversation_history)
        
        # Generate response using Gemini
        response = model.generate_content(prompt)
        bot_response = response.text
        
        # Check for crisis keywords and append disclaimer if needed
        if contains_crisis_keywords(user_message):
            bot_response += get_safety_disclaimer()
        
        # Store conversation history
        conversation_history.append({
            'user': user_message,
            'bot': bot_response
        })
        
        return jsonify({
            'response': bot_response,
            'user_message': user_message
        })
    
    except Exception as e:
        print(f"Text chat error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/voice_chat', methods=['POST'])
def voice_chat():
    """Handle voice-based chat interactions."""
    try:
        # Convert speech to text
        user_message = speech_to_text()
        
        if user_message in ["No speech detected", "Sorry, I couldn't understand that",
                          "Sorry, there was an error with the speech recognition service"]:
            return jsonify({
                'user_message': user_message,
                'response': "I'm having trouble hearing you. Could you please try again?"
            })
        
        # Create contextual prompt
        prompt = create_mental_health_prompt(user_message, conversation_history)
        
        # Generate response using Gemini
        response = model.generate_content(prompt)
        bot_response = response.text
        
        # Add crisis disclaimer if needed
        if contains_crisis_keywords(user_message):
            bot_response += get_safety_disclaimer()
        
        # Convert response to speech
        text_to_speech(bot_response)
        
        # Store conversation history
        conversation_history.append({
            'user': user_message,
            'bot': bot_response
        })
        
        return jsonify({
            'response': bot_response,
            'user_message': user_message
        })
    
    except Exception as e:
        print(f"Voice chat error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
     # Check for required environment variables
    if not os.getenv('GEMINI_API_KEY'):
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    app.run(debug=True)
