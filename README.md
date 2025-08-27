# 🧠 Moodmate - The Mental Health Companion

**Moodmate** is a Flask-based AI-powered mental health companion that integrates real-time emotion detection, PHQ-9 depression assessment, an AI-powered voice + text chatbot, and psychologist session scheduling.  
It is designed to provide **empathetic, supportive, and non-judgmental interactions** while promoting emotional well-being and timely interventions.

---

## 🚀 Features

### 🔹 Real-time Emotion Detection
- Detects emotions from images and live webcam feed using **DeepFace & OpenCV**.
- Provides mental health insights and coping strategies.

### 🔹 Voice & Text Chatbot
- AI-powered empathetic conversation using **Google Gemini API**.
- Supports both **speech-to-text** and **text-to-speech** interactions.

### 🔹 Depression Risk Assessment (PHQ-9)
- Calculates depression severity using the **PHQ-9 questionnaire**.
- Combines **emotion analysis + PHQ-9** for personalized risk scoring.
- Suggests coping strategies and interventions based on risk level.

### 🔹 Psychologist Booking System
- **SQLite database** with seeded psychologist data.
- Patients can submit **Zoom requests** for sessions.
- Handles availability and session links automatically.

### 🔹 Mental Health Resources
- Provides **crisis helplines and self-help resources**.
- Includes **safety disclaimers** for sensitive situations.

---

## 🛠 Tech Stack

- **Backend**: Flask, SQLAlchemy, Flask-Migrate, Flask-CORS  
- **Database**: SQLite  
- **AI Models**: Google Gemini API, DeepFace  
- **Speech Processing**: SpeechRecognition, Pyttsx3  
- **Frontend**: HTML, CSS, JavaScript  
- **Others**: dotenv, OpenCV  

---

## 📂 Project Structure

```bash
mental-health-companion/
│── app.py                # Main Flask application
│── psychologists.db      # SQLite database
│── requirements.txt      # Python dependencies
│── .env                  # Environment variables (API keys)
│── README.md             # Documentation
│
├── templates/            # HTML templates
│   ├── index.html
│   ├── emotion.html
│   ├── face.html
│   ├── chatbot.html
│   ├── depression.html
│   ├── resources.html
│   └── talkToHeel.html

⚙️ Installation & Setup
1. Clone the repository
git clone https://github.com/Arfasil/Moodmate.git
cd Moodmate

2. Create and activate a virtual environment

Mac/Linux

python -m venv env
source env/bin/activate


Windows

python -m venv env
env\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Set environment variables

Create a .env file in the root folder and add:

GEMINI_API_KEY=your_api_key_here

5. Run the Flask app
python app.py


Visit the app in your browser:
👉 http://127.0.0.1:5000/

🧩 PHQ-9 Severity Levels
Severity Level	Score Range
Minimal	0 - 4
Mild	5 - 9
Moderate	10 - 14
Moderately Severe	15 - 19
Severe	20 - 27

Risk Score Formula:
70% PHQ-9 + 30% Emotion Analysis

⚠️ Safety Disclaimer

This project is not a substitute for professional mental health care.
If you or someone you know is struggling with self-harm or suicidal thoughts:

US: Dial 988 for the National Suicide Prevention Lifeline

Text: HOME to 741741 (Crisis Text Line)

Emergency: Call 911 immediately

👨‍💻 Author

✨ A R Mohammed Fasil

📜 License

This project is licensed under the MIT License – free to use and modify.

💡 About

Moodmate is an AI-powered Mental Health Support Web App built with Flask, DeepFace, Google Gemini AI, and SQLAlchemy.
It provides:

Real-time emotion detection from facial expressions

PHQ-9 based depression risk assessment

Empathetic AI chatbot (text & voice)

Crisis intervention resources

Psychologist consultation via Zoom
