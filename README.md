"🧠 Moodmate - The Mental Health Companion"

description: |
  Flask-based AI-powered mental health companion that integrates emotion detection, 
  PHQ-9 depression assessment, real-time face emotion analysis, voice + text chatbot 
  interactions, and psychologist session scheduling. Designed to provide empathetic, 
  supportive, and non-judgmental interactions while promoting emotional well-being 
  and timely interventions.

features:
  - Real-time Emotion Detection:
      - Detects emotions from images and live webcam feed using DeepFace & OpenCV.
      - Provides mental health insights and coping strategies.
  - Voice & Text Chatbot:
      - AI-powered empathetic conversation using Google Gemini API.
      - Supports both speech-to-text and text-to-speech interactions.
  - Depression Risk Assessment (PHQ-9):
      - Calculates depression severity using PHQ-9 questionnaire.
      - Combines emotion analysis with PHQ-9 for personalized risk scoring.
      - Suggests coping strategies and interventions based on risk level.
  - Psychologist Booking System:
      - SQLite database with seeded psychologist data.
      - Patients can submit Zoom requests for sessions.
      - Automatic availability and session link handling.
  - Mental Health Resources:
      - Provides crisis helplines and self-help resources.
      - Includes safety disclaimers for sensitive situations.

tech_stack:
  backend: 
    - Flask
    - SQLAlchemy
    - Flask-Migrate
    - Flask-CORS
  database: SQLite
  ai_models: 
    - Google Gemini API
    - DeepFace (emotion detection)
  speech_processing:
    - SpeechRecognition
    - Pyttsx3
  frontend: 
    - HTML
    - CSS
    - JavaScript
  others: 
    - dotenv (environment variables)
    - OpenCV (face detection)

project_structure:
  - app.py: Main Flask application
  - templates/:
      - index.html
      - emotion.html
      - face.html
      - chatbot.html
      - depression.html
      - resources.html
      - talkToHeel.html
  - psychologists.db: SQLite database
  - requirements.txt: Python dependencies
  - .env: Environment variables (API keys)
  - README.md: Documentation

installation:
  steps:
    - Clone the repository:
        command: |
          git clone https://github.com/your-username/mental-health-companion.git
          cd mental-health-companion
    - Create and activate a virtual environment:
        mac_linux: |
          python -m venv env
          source env/bin/activate
        windows: |
          python -m venv env
          env\Scripts\activate
    - Install dependencies:
        command: pip install -r requirements.txt
    - Set environment variables:
        create_file: .env
        content: |
          GEMINI_API_KEY=your_api_key_here
    - Run the Flask app:
        command: python app.py
        visit: "http://127.0.0.1:5000/"

phq9_severity_levels:
  minimal: 0-4
  mild: 5-9
  moderate: 10-14
  moderately_severe: 15-19
  severe: 20-27
  risk_score_formula: "70% PHQ-9 + 30% Emotion Analysis"

safety_disclaimer: |
  This project is not a substitute for professional mental health care.
  If you or someone you know is struggling with self-harm or suicidal thoughts:
    - US: Dial 988 for the National Suicide Prevention Lifeline
    - Text HOME to 741741 (Crisis Text Line)
    - Call 911 for emergencies

author: "✨ A R Mohammed Fasil"

license:
  type: "MIT License"
  permissions: "Free to use and modify"

about: |
  An AI-powered Mental Health Support Web App built with Flask, DeepFace, 
  Google Gemini AI, and SQLAlchemy. It provides real-time emotion detection 
  from facial expressions, PHQ-9 based depression risk assessment, empathetic 
  AI chatbot (text & voice), crisis intervention resources, and psychologist 
  consultation via Zoom.


