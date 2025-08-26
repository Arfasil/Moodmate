# 🧠 Moodmate -the mental Health Campanion

This project is a **Flask-based AI-powered mental health companion** that integrates **emotion detection, PHQ-9 depression assessment, real-time face emotion analysis, voice + text chatbot interactions, and psychologist session scheduling**.  
It is designed to provide **empathetic, supportive, and non-judgmental interactions** while promoting emotional well-being and timely interventions.

---

## 🚀 Features

- **Real-time Emotion Detection**
  - Detects emotions from images and live webcam feed using DeepFace & OpenCV.
  - Provides mental health insights and coping strategies.

- **Voice & Text Chatbot**
  - AI-powered empathetic conversation using **Google Gemini API**.
  - Supports both **speech-to-text** and **text-to-speech** interactions.

- **Depression Risk Assessment (PHQ-9)**
  - Calculates depression severity using PHQ-9 questionnaire.
  - Combines emotion analysis with PHQ-9 for **personalized risk scoring**.
  - Suggests coping strategies and interventions based on risk level.

- **Psychologist Booking System**
  - SQLite database with seeded psychologist data.
  - Patients can **submit Zoom requests** for sessions.
  - Automatic availability and session link handling.

- **Mental Health Resources**
  - Provides crisis helplines and self-help resources.
  - Includes safety disclaimers for sensitive situations.

---

## 🛠️ Tech Stack

- **Backend:** Flask, SQLAlchemy, Flask-Migrate, Flask-CORS  
- **Database:** SQLite  
- **AI Models:** Google Gemini API, DeepFace (for emotion detection)  
- **Speech Processing:** SpeechRecognition, Pyttsx3  
- **Frontend:** HTML, CSS, JS (templates integrated via Flask)  
- **Others:** dotenv for environment variables, OpenCV for face detection  

---

## 📂 Project Structure
📦 project-root
├── app.py # Main Flask application
├── templates/ # HTML frontend templates
│ ├── index.html
│ ├── emotion.html
│ ├── face.html
│ ├── chatbot.html
│ ├── depression.html
│ ├── resources.html
│ └── talkToHeel.html
├── psychologists.db # SQLite database
├── requirements.txt # Python dependencies
├── .env # Environment variables (API keys)
└── README.md # Documentation


---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/mental-health-companion.git
cd mental-health-companion


2. Create and activate a virtual environment

python -m venv env
source env/bin/activate   # On Mac/Linux
env\Scripts\activate      # On Windows

3. Install dependencies

pip install -r requirements.txt


4. Set environment variables

Create a .env file in the project root and add your Google Gemini API key:


GEMINI_API_KEY=your_api_key_here


5. Run the Flask app

python app.py


Visit: http://127.0.0.1:5000/
 in your browser.

📊 PHQ-9 Severity Levels

Minimal: 0–4

Mild: 5–9

Moderate: 10–14

Moderately Severe: 15–19

Severe: 20–27

Risk score = 70% PHQ-9 + 30% Emotion Analysis

⚠️ Safety Disclaimer

This project is not a substitute for professional mental health care.
If you or someone you know is struggling with self-harm or suicidal thoughts:

US: Dial 988 for the National Suicide Prevention Lifeline

Text HOME to 741741 (Crisis Text Line)

Call 911 for emergencies

✨ Author

A R Mohammed Fasil


📜 License

This project is licensed under the MIT License – feel free to use and modify it.


