# ReSage AI

ReSage AI is a resume screening and evaluation web application built with Flask. It extracts resume text from PDF, DOC, and DOCX files, evaluates structure and keywords, and provides actionable feedback.

## Features

- Resume upload and secure file handling
- AI-assisted scoring and feedback (Gemini-compatible)
- Resume section detection and quality checks
- Responsive, minimal UI with accessibility improvements
- Secure HTTP headers and file size limits

## Requirements

- Python 3.11+
- Flask
- python-dotenv
- pdfplumber
- python-docx
- textract (optional for `.doc` files)
- OpenCV optional for future photo detection
- Gemini API key stored in `.env`

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
```

Create a `.env` file in the repository root with:

```env
GEMINI_API_KEY=your_api_key_here
FLASK_SECRET_KEY=some_random_secret
```

## Run locally

```bash
python backend/app.py
```

Then open `http://127.0.0.1:5000`.

## Deployment

For cloud deployment, use a WSGI server such as Gunicorn or a platform that supports Python/Flask.

Example with Gunicorn:

```bash
gunicorn -w 3 -b 0.0.0.0:8000 backend.app:app
```

## Notes

- The app will keep uploads in `backend/uploads` and removes them after analysis.
- If AI dependencies are unavailable, the app falls back to deterministic scoring and suggestions.
- Add CI/CD or GitHub Actions for production deployment.