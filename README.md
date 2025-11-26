# MedicareAgent 💊

A real-time, streaming pharmacy-chatbot built with CrewAI + FastAPI + WebSockets.  
It helps users check medicine availability (e.g. “Where can I get Paracetamol near me?”) and returns a clean list of nearby pharmacies via live streaming.

---

## 🚀 Project Overview

- **Goal:** Provide a light, real-time AI assistant that can fetch and relay pharmacy information (medicine availability) via a backend API, and stream responses to clients like a chat bot.  
- **How it works:**  
  1. User sends a query via WebSocket (e.g. “Do you have Paracetamol?”)  
  2. CrewAI agent interprets the query, calls the `medicine_tool` as needed  
  3. Server streams back the results token-by-token over WebSocket — user sees the answer live as it’s generated  

This makes the experience fast, interactive, and feels like a live chat.

---

## 📁 Repository Structure




**`tools/medicine.py`** — core tool that fetches pharmacy data (medicine availability) from your backend API.  
**`crew.py`** — defines the `Crew` with a polite, concise agent, strict prompt to avoid hallucinations, and streaming-enabled configuration.  
**`server.py`** — initializes FastAPI, accepts WebSocket connections, uses `crew.py` to handle user messages, and streams responses back to clients.  
**`utils.py`** — helper functions, e.g. a streaming worker that bridges blocking tool/LLM calls to asynchronous WebSocket streaming.  
**`client.html`** — a minimal HTML + JS front-end to test the chatbot without needing a full React or production UI.  
**`console.py`** — (optional) CLI-based interface to test the bot directly from terminal.  

---

## 🛠️ Getting Started — Run Locally

1. **Clone the repository**

   ```bash
   git clone https://github.com/sashanth17/MedicareAgent.git
   cd MedicareAgent

   python -m venv .venv

   source .venv/bin/activate   # (Unix / Mac)
   pip install -r requirements.txt

   python -m venv .venv
   source .venv/bin/activate   # (Unix / Mac)
   pip install -r requirements.txt

   GEMINI_API_KEY=your_gemini_api_key
   BACKEND_API_URL=https://your-backend-url

   python server.py

5.	Test with the HTML client
	•	Open client.html in your browser
	•	Connect to the WebSocket server (should be ws://localhost:8001/ws)
	•	Send queries like "Where can I get paracetamol near me?"
	•	Bot will stream replies in real-time

⸻
📚 Documentation

➡️ Source Code Documentation: [Click Here](https://docs.google.com/document/d/173P9j6MIzoyzQ7YYAmoSGX-8cQG-eBEobwQFWqlqYTo/edit?tab=t.0)￼

⸻

🔮 Future Enhancements
	•	Distance-based sorting of pharmacies
	•	Automatic location detection
	•	Full UI in React / Flutter
	•	Authentication + saved user preferences
	•	Real-time stock updates + maps integration

⸻

⭐ Support

If this project helps you, please ⭐ the repo!
Contributions, bug reports, ideas — all are welcome! 😄
