# ReSage AI — AI-Powered Resume Screener

> Research-grade resume screening for students and job seekers.

ReSage AI evaluates your resume using Google Gemini AI and rule-based analysis — scoring structure, flagging grammar issues, recommending improvements, and matching you to job roles, all in seconds. Files are deleted immediately after analysis.

---

## Features

- **AI-assisted scoring** (0–100 Sage Score) powered by Google Gemini
- **Profile completeness** check — summary, skills, education, experience, projects, contact
- **Grammar & writing quality** checks
- **Personalised improvement suggestions**
- **Job role recommendations** based on your resume content
- **Zero data retention** — files deleted immediately after analysis
- **No sign-up required**

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python 3.11, Flask 3.1 |
| AI | Google Gemini API (via `google-generativeai`) |
| PDF parsing | `pdfplumber` |
| DOCX parsing | `python-docx` |
| Production server | Gunicorn |
| Frontend | Vanilla HTML/CSS/JS (Jinja2 templates) |

---

## Local Development

### Prerequisites
- Python 3.11+
- A [Gemini API key](https://aistudio.google.com/apikey) (free)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ReSageAI.git
cd ReSageAI

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Run the development server
python backend/app.py
```

Then open [http://localhost:5000](http://localhost:5000).

---

## Deployment

> **Important:** This is a full Python Flask application. It **cannot** be deployed directly to Vercel or Netlify (those platforms are for static sites / serverless JS). Use **Railway** or **Render** instead — both have free tiers.

### Deploy to Railway (Recommended)

1. Push your code to GitHub
2. Go to [railway.app](https://railway.app) and create a new project
3. Select **Deploy from GitHub repo**
4. Add environment variables in the Railway dashboard:
   - `GEMINI_API_KEY` → your Gemini API key
   - `FLASK_SECRET_KEY` → a strong random string (`python -c "import secrets; print(secrets.token_hex(32))"`)
5. Railway will auto-detect the `Procfile` and deploy. Done!

### Deploy to Render

1. Push your code to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Render will auto-detect the `render.yaml` blueprint
5. Add `GEMINI_API_KEY` in environment variables
6. Click **Deploy**

### Required Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | ✅ Yes | Google Gemini API key from [aistudio.google.com](https://aistudio.google.com/apikey) |
| `FLASK_SECRET_KEY` | ✅ Yes (production) | Strong random secret for Flask sessions |
| `APP_DEBUG` | No | Set `false` in production (default) |
| `GEMINI_MODEL` | No | Override model, default `gemini-2.5-flash` |

---

## Project Structure

```
ReSageAI/
├── backend/
│   ├── app.py          # Flask application factory
│   ├── config.py       # All configuration constants
│   ├── routes.py       # HTTP routes and error handlers
│   ├── extractor.py    # PDF/DOCX text extraction
│   ├── analyser.py     # Scoring, suggestions, job matching
│   ├── ai_client.py    # Gemini AI client wrapper
│   └── requirements.txt
├── templates/
│   ├── index.html      # Landing/upload page
│   ├── analysis.html   # Loading animation page
│   └── results.html    # Results display page
├── static/
│   ├── css/style.css
│   ├── js/script.js
│   └── images/logo.png
├── Procfile            # Railway/Render start command
├── runtime.txt         # Python version pin
├── railway.json        # Railway configuration
├── render.yaml         # Render deployment blueprint
└── .env.example        # Environment variable template
```

---

## Notes

- `.doc` (legacy Word) files are not supported on cloud deployments due to native library requirements. Please use `.pdf` or `.docx`.
- The app uses an in-process cache for AI responses (1-hour TTL) to avoid redundant API calls.
- Uploaded files are deleted from disk immediately after text extraction.

---

*Built for a research project · © 2026 ReSage AI*
