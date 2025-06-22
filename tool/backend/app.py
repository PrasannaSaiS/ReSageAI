import os
import re
import cv2

from flask import Flask, render_template, request, redirect, url_for, abort
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

import pdfplumber
import docx
import textract

import google.generativeai as genai

# ── Load environment variables ────────────────────────────────────
basedir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(basedir, '..'))
load_dotenv(os.path.join(project_root, '.env'))

API_KEY = os.getenv("GEMINI_API_KEY")  # put your API key in .env as GEMINI_API_KEY
if not API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in .env")

# ── Configure Gemini API ──────────────────────────────────────────
genai.configure(api_key=API_KEY)

generation_config = {
    "temperature": 0.4,
    "top_p": 1.0,
    "top_k": 40,
    "max_output_tokens": 2048,
}

model = genai.GenerativeModel(
    model_name="models/gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="You are a professional AI resume evaluator who provides honest scores, feedback, and grammar/spelling correction."
)

# ── Flask setup ───────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(project_root, 'templates'),
    static_folder=os.path.join(project_root, 'static')
)

UPLOAD_FOLDER = os.path.join(basedir, 'uploads')
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ── Helpers ────────────────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_and_photo(path):
    ext = path.rsplit('.', 1)[1].lower()
    text = ""
    has_photo = False

    if ext == 'pdf':
        with pdfplumber.open(path) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
            for page in pdf.pages:
                if page.images:
                    has_photo = True
                    break
    elif ext == 'docx':
        doc = docx.Document(path)
        text = "\n".join(p.text for p in doc.paragraphs)
        if doc.inline_shapes:
            has_photo = True
    else:
        text = textract.process(path).decode('utf-8', errors='ignore')

    # split camelCase for better readability
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    return text, has_photo

def detect_criteria(text, has_photo):
    t = text.lower()
    return {
        'Profile Photo': has_photo,
        'Summary': bool(re.search(r'\b(objective|summary)\b', t)),
        'Skills': bool(re.search(r'\bskills?\b', t)),
        'Education': bool(re.search(r'\b(education|b\.?tech|bachelor|master|phd)\b', t)),
        'Projects': bool(re.search(r'\bprojects?\b', t)),
        'Contact Info': bool(
            re.search(r'[\w.+-]+@[\w-]+\.[\w.-]+', text) or
            re.search(r'\+?\d[\d ()-]{7,}', text)
        )
    }

def ask_gemini(prompt):
    try:
        response = model.generate_content(prompt)   
        return response.text
    except Exception as e:
        return f"❌ Gemini API Error: {e}"

def score_and_suggest(text):
    prompt = (
        "You are an expert resume screener. Return exactly:\n"
        "1. A single line “Score: XX/100” (give some ample marks)\n"
        "2. Up to 5 bullet-point suggestions prefixed with “- ”(Give it as line statements without bold fonts)\n\n"
        f"{text[:3000]}"
    )
    raw = ask_gemini(prompt)
    lines = raw.splitlines()
    score = 0
    suggestions = []

    for ln in lines:
        m = re.search(r"Score\s*:\s*(\d{1,3})\s*/\s*100", ln, re.IGNORECASE)
        if m:
            score = int(m.group(1))
            break
    for ln in lines:
        if ln.strip().startswith('-'):
            suggestions.append(ln.strip('- ').strip())

    return score, suggestions[:5]

def grammar_errors(text):
    prompt = (
        "List grammatical and spelling errors in the following resume text. Cross check before listing the errors. Please don't give capitalization issues and inconsistency errors, kindly ignore them since you don't know to identify it perfectly  example: Key Achievement: should be Key Achievement- this was the silly error you made on testing avaoid them please. Also no need to give useless or errors without much impact when there is not big errors in the resume.\n"
        "Output each error as a separate line prefixed with “- ”.\n\n\n"
        f"{text[:3000]}"
    )
    raw = ask_gemini(prompt)
    return [ln.strip('- ').strip() for ln in raw.splitlines() if ln.strip().startswith('-')]

def recommend_jobs(text):
    prompt = (
        "You are a career guidance expert. Based on the resume text below, suggest:\n"
        "1. 3-5 suitable job roles (in bullet points prefixed by '- ')\n"
        "2. The field/industry the candidate is most suited for\n"
        "Be precise and professional. Avoid unnecessary sentences.\n\n"
        f"{text[:3000]}"
    )
    raw = ask_gemini(prompt)
    job_roles = []
    field = "Not identified"
    
    for ln in raw.splitlines():
        if ln.strip().startswith('-'):
            job_roles.append(ln.strip('- ').strip())
        elif 'industry' in ln.lower() or 'field' in ln.lower():
            field = ln.strip()

    return job_roles[:5], field

# ── Routes ─────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    f = request.files.get('resume')
    if not f or f.filename == '':
        abort(400, "No file selected")
    if not allowed_file(f.filename):
        abort(400, "Invalid file type")
    fname = secure_filename(f.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    f.save(path)
    return redirect(url_for('analysis', filename=fname))

@app.route('/analysis')
def analysis():
    fname = request.args.get('filename')
    if not fname:
        abort(400, "Filename missing")

    path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    text, has_photo = extract_text_and_photo(path)
    criteria = detect_criteria(text, has_photo)

    if sum(criteria.values()) < 3:
        abort(400, "Uploaded file doesn’t appear to be a resume.")

    try:
        score, suggestions = score_and_suggest(text)
        errors = grammar_errors(text)
        job_roles, job_field = recommend_jobs(text)

    except Exception as e:
        abort(502, f"AI model error: {e}")

    return render_template(
        'results.html',
        score=score,
        criteria=criteria,
        errors=errors[:10],
        suggestions=suggestions,
        job_roles=job_roles,
        job_field=job_field
    )

@app.errorhandler(400)
@app.errorhandler(502)
def handle_errors(e):
    return render_template(
        'results.html',
        score=0,
        criteria={},
        errors=[str(e)],
        suggestions=[]
    ), e.code

if __name__ == '__main__':
    app.run(debug=True)
