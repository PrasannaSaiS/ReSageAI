# ReSage AI — Comprehensive Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Core Features](#core-features)
3. [How It Works](#how-it-works)
4. [Directory Structure](#directory-structure)
5. [Technical Stack](#technical-stack)
6. [Backend Architecture](#backend-architecture)
7. [Frontend Architecture](#frontend-architecture)
8. [Detailed File Descriptions](#detailed-file-descriptions)
9. [Data Flow](#data-flow)
10. [Configuration & Deployment](#configuration--deployment)

---

## Project Overview

**ReSage AI** is a comprehensive resume screening and evaluation web application built with Flask. It leverages AI (Gemini API) to intelligently analyze resumes, extract critical information, detect structural issues, provide actionable feedback, and recommend suitable job roles.

### Purpose
The application serves as a professional resume evaluator that:
- Analyzes resume structure and content quality
- Scores resumes on a 0-100 scale based on competitiveness
- Detects missing critical resume sections
- Identifies grammar and writing errors
- Recommends suitable job roles based on extracted keywords
- Provides AI-assisted suggestions for improvement
- Maintains security through file validation, sandboxing, and secure headers

### Key Innovation
ReSage AI combines **AI-assisted analysis** with **rule-based fallbacks**, ensuring functionality even when AI services are unavailable. It uses intelligent caching, format-agnostic extraction, and research-backed scoring methodologies.

---

## Core Features

### 1. **Multi-Format Resume Support**
- **Supported Formats**: PDF, DOC, DOCX
- **File Validation**: Magic byte verification (binary signature validation)
- **File Size Limit**: 6 MB maximum upload size
- **Automatic Cleanup**: Uploaded files are deleted immediately after analysis for privacy

### 2. **Resume Text Extraction**
- **PDF Extraction**: Uses `pdfplumber` for pixel-perfect text extraction
- **DOCX Extraction**: Parses Office Open XML format with `python-docx`
- **DOC Extraction**: Legacy format support via `textract`
- **Text Normalization**: 
  - CamelCase splitting for extracted PDF text
  - Technology terminology standardization (e.g., "fastapi" → "FastAPI")
  - Preserves URLs, emails, and technical names

### 3. **Structural Analysis**
- **Section Detection**: Identifies 8 critical resume sections:
  - Profile Photo presence
  - Summary/Objective
  - Skills section
  - Education credentials
  - Work Experience
  - Projects/Portfolio
  - Contact Information
  - Certifications
- **Rule-Based Detection**: Uses regex patterns to find sections without AI
- **Photo Detection**: Identifies if resume includes a profile photograph

### 4. **AI-Assisted Scoring & Feedback**
- **Resume Scoring**: 0-100 point scale indicating resume competitiveness
- **Smart Suggestions**: 5 actionable improvement recommendations
- **Error Detection**: Grammar and spelling error identification (excludes URLs, emails, technical terms)
- **Job Role Recommendation**: Suggests 3-5 most suitable job positions
- **Career Field Identification**: Categorizes resume into relevant career domain
- **Fallback Mechanism**: Deterministic scoring when AI is unavailable

### 5. **Security Features**
- **Content Security Policy (CSP)**: Strict CSP headers prevent XSS attacks
- **CORS & Frame Protection**: X-Frame-Options: DENY prevents clickjacking
- **File Type Validation**: Binary signature verification prevents file extension spoofing
- **Path Traversal Prevention**: Secure filename handling and path validation
- **HTTP-Only Cookies**: Prevents JavaScript access to session cookies
- **HTTPS Enforcement**: HSTS header in production mode

### 6. **User Interface**
- **Drag-and-Drop Upload**: Intuitive file upload experience
- **Progress Animation**: 4-step loading animation during analysis
- **Results Dashboard**: Comprehensive results view with visual score indicator
- **Profile Overview**: Color-coded checklist of resume sections
- **Responsive Design**: Mobile-friendly interface using modern CSS

---

## How It Works

### Complete User Journey

```
USER UPLOADS RESUME
        ↓
   FILE VALIDATION
   - Extension check
   - Magic byte verification
   - File size check
        ↓
   SECURE STORAGE
   - UUID-based filename
   - Temporary upload folder
        ↓
   REDIRECT TO ANALYSIS PAGE
   - Shows 4-step progress animation
        ↓
   TEXT EXTRACTION & NORMALIZATION
   - Determine file format (PDF/DOC/DOCX)
   - Extract raw text using appropriate library
   - Fix camelCase splitting
   - Normalize technology terms
        ↓
   STRUCTURAL ANALYSIS
   - Detect resume sections (regex-based)
   - Identify profile photo
        ↓
   AI ANALYSIS (Parallel with fallback)
   - Send text to Gemini API
   - Parse response for: score, suggestions, errors, roles, field
   - Cache result for 1 hour
   - On AI failure: Use deterministic scoring
        ↓
   RESULTS COMPILATION
   - Score with color-coded badge (green/amber/red)
   - Section presence checklist
   - Grammar error list
   - Job role recommendations
   - Career field classification
        ↓
   FILE CLEANUP
   - Delete uploaded resume from disk
        ↓
   DISPLAY RESULTS
   - Show all analysis data
   - Enable new upload option
```

### Analysis Decision Tree

```
RECEIVE TEXT
    ↓
CHECK CACHE (1-hour TTL)
    ├─ FOUND → Return cached result
    └─ NOT FOUND → Continue
        ↓
    IS AI AVAILABLE?
        ├─ YES → Send single combined prompt to Gemini
        │         Parse response for: score, suggestions, errors, roles, field
        │         Cache result
        │         Return AI result
        │
        └─ NO → Use Rule-Based Fallback:
                 - Calculate score: 40 base + (7 × sections present)
                 - Generate generic suggestions based on missing sections
                 - List only known typos from dictionary
                 - Recommend roles based on keyword patterns
                 - Return rule-based result
```

---

## Directory Structure

```
ReSageAI/
├── README.md                      # Project overview and setup instructions
├── PROJECT_DOCUMENTATION.md       # This file (comprehensive documentation)
│
├── backend/
│   ├── app.py                     # Flask application factory and entry point
│   ├── config.py                  # Configuration management (env variables, constants)
│   ├── routes.py                  # HTTP route handlers, error handlers, file validation
│   ├── extractor.py               # Resume text extraction for PDF/DOC/DOCX
│   ├── analyser.py                # Analysis logic: scoring, suggestions, job matching
│   ├── ai_client.py               # Gemini API client and wrapper
│   ├── requirements.txt            # Python dependencies
│   └── uploads/                   # Temporary folder for uploaded resumes (auto-cleaned)
│
├── static/
│   ├── css/
│   │   └── style.css              # Main stylesheet (design tokens, responsive layout)
│   ├── js/
│   │   └── script.js              # Client-side logic (file upload, animations)
│   └── images/
│       └── logo.png               # Brand logo asset
│
└── templates/
    ├── index.html                 # Upload page (initial landing)
    ├── analysis.html              # Loading animation page
    └── results.html               # Results display page
```

### Directory Purposes

| Directory | Purpose |
|-----------|---------|
| `backend/` | All Python server logic and API endpoints |
| `backend/uploads/` | Temporary storage for uploaded resume files (cleaned after analysis) |
| `static/` | Client-side assets (CSS, JavaScript, images) |
| `static/css/` | Stylesheets with design tokens and responsive components |
| `static/js/` | Client-side JavaScript for interactivity and animations |
| `static/images/` | Brand assets and UI images |
| `templates/` | Jinja2 HTML templates rendered server-side |

---

## Technical Stack

### Backend
- **Framework**: Flask (Python micro web framework)
- **Python Version**: 3.11+
- **API Integration**: Google Gemini API (with fallback support)
- **File Processing**:
  - `pdfplumber`: PDF text extraction and image detection
  - `python-docx`: Microsoft Word (.docx) document parsing
  - `textract`: Legacy .doc file conversion
- **Environment**: `python-dotenv` for configuration management

### Frontend
- **Markup**: HTML5 with Jinja2 templating
- **Styling**: CSS3 (custom design system with CSS variables)
- **JavaScript**: Vanilla JavaScript (no framework dependencies)
- **Interactivity**: Drag-and-drop file upload, progress animations

### Security Libraries
- **Werkzeug**: Secure filename generation, file upload handling
- **Flask Built-in**: Session management, CSRF protection

### Deployment
- **WSGI Server**: Gunicorn (recommended for production)
- **Platforms**: Cloud-ready (AWS, GCP, Heroku, Azure, etc.)
- **Environment Variables**: Managed via `.env` file

---

## Backend Architecture

### Design Principles

The codebase follows **SOLID principles** and **clean architecture**:

1. **Single Responsibility**: Each module owns one concern
2. **Open/Closed Principle**: Easy to extend (e.g., add new file formats, analysis dimensions)
3. **Dependency Inversion**: High-level modules depend on abstractions, not implementations
4. **DRY (Don't Repeat Yourself)**: Shared logic centralized in utility functions

### Module Relationships

```
app.py (factory)
    ├── imports: config, routes
    └── creates Flask app with config + blueprints
        
config.py (constants & env)
    └── imported by: app, routes, extractor, analyser, ai_client
    └── loads .env file once at startup
    └── defines all paths, API keys, limits, models
        
routes.py (HTTP handling)
    ├── imports: config, extractor, analyser, ai_client
    ├── handles: uploads, file validation, magic bytes, security headers
    └── orchestrates: extraction → analysis → results
        
extractor.py (text extraction)
    ├── imports: config (file extensions)
    ├── uses: pdfplumber, python-docx, textract
    └── exports: extract_text_and_photo(path) → tuple[str, bool]
        
analyser.py (analysis & scoring)
    ├── imports: config, ai_client
    ├── functions: detect_criteria(), score_and_suggest(), grammar_errors(), recommend_jobs()
    └── internal cache: _cache[hash] → (result, timestamp)
        
ai_client.py (Gemini integration)
    ├── imports: config
    ├── exports: ask_gemini(prompt), check_ai_status()
    └── handles: SDK detection, client init, fallback models
```

---

## Frontend Architecture

### Page Structure

#### 1. **Index Page** (`index.html`)
- **Purpose**: Initial landing page and resume upload interface
- **Features**:
  - Hero section with value proposition
  - Feature list highlighting key capabilities
  - Drag-and-drop file upload card
  - File type/size validation feedback
  - AI availability indicator
  - Error message display
- **Interactions**: File selection, drag-and-drop, form submission

#### 2. **Analysis Page** (`analysis.html`)
- **Purpose**: Loading/progress screen during analysis
- **Features**:
  - Animated spinner
  - 4-step progress tracker with timing:
    - Step 1 (0ms): Extracting resume content
    - Step 2 (800ms): Analyzing structure
    - Step 3 (1800ms): Running AI assessment
    - Step 4 (2800ms): Generating recommendations
  - Auto-redirect to results after animation completes
- **Interactions**: Auto-progression, JavaScript-driven timeline

#### 3. **Results Page** (`results.html`)
- **Purpose**: Display comprehensive analysis results
- **Sections**:
  - **Score Panel**: Visual 0-100 score ring, badge (Strong/Average/Needs Work), assessment text
  - **Profile Overview**: Checklist of 8 resume sections with presence indicators
  - **Job Recommendations**: List of suggested roles and career field
  - **Improvements Needed**: Grammar/spelling errors with explanations
  - **AI Suggestions**: 5 actionable improvement recommendations
  - **Analysis Mode**: Indicator showing if results are AI-assisted or rule-based
- **Styling**: Color-coded by score:
  - Green (#16a34a) for scores ≥80
  - Amber (#d97706) for scores 60-79
  - Red (#dc2626) for scores <60

### Design System

**CSS Design Tokens** (defined in `:root`):
```css
Colors:
  --primary: #4f46e5 (Indigo for primary CTA)
  --primary-dark: #3730a3 (Darker shade for hover)
  --success: #16a34a (Green for positive indicators)
  --warning: #d97706 (Amber for caution)
  --danger: #dc2626 (Red for errors)
  --surface: #ffffff (Main background)
  
Spacing:
  --radius-sm: 10px
  --radius: 16px
  --radius-lg: 22px
  
Effects:
  --shadow-sm, --shadow, --shadow-lg
  --transition: 0.2s ease
```

### Client-Side Logic

**Key JavaScript Functions** (in `script.js`):

1. **File Upload Handler**
   - Validates file type (MIME + extension)
   - Shows/updates selected filename
   - Prevents invalid submissions

2. **Drag-and-Drop**
   - Detects dragover/dragleave events
   - Visual feedback via CSS class toggling
   - Prevents default browser behavior

3. **Analysis Progress Animation**
   - Extracts filename from page data attribute
   - Sequences 4 steps with 800ms intervals
   - Toggles step visibility: inactive → active → done
   - Redirects to `/results` after final step

---

## Detailed File Descriptions

### 1. **backend/app.py** — Flask Application Factory
**Role**: Entry point and Flask app configuration

**Key Responsibilities**:
- Creates Flask app instance with proper template/static folders
- Applies security configuration:
  - MAX_CONTENT_LENGTH: 6 MB upload limit
  - SECRET_KEY: For session signing (from env or random)
  - SESSION_COOKIE_HTTPONLY: Prevents JavaScript access
  - SESSION_COOKIE_SECURE: HTTPS-only in production
  - SESSION_COOKIE_SAMESITE: "Lax" for CSRF protection
- Creates `uploads/` directory if missing
- Configures logging (DEBUG if `APP_DEBUG=true`)
- Registers blueprint (routes)

**Design Pattern**: **Factory Pattern** — separates app creation from configuration

**Entry Point**:
```python
if __name__ == "__main__":
    app.run(debug=config.APP_DEBUG, host="0.0.0.0", port=5000)
```

**Dependencies**:
- Flask
- config (constants)
- routes (blueprint)

---

### 2. **backend/config.py** — Environment & Constants Management
**Role**: Single source of truth for all configuration

**Key Sections**:

**Paths**:
- `BASE_DIR`: Path to `backend/` folder
- `ROOT_DIR`: Project root (parent of `backend/`)
- `UPLOAD_DIR`: `backend/uploads/`

**Flask Configuration**:
- `SECRET_KEY`: Session signing key (from env or random UUID)
- `APP_DEBUG`: Debug mode (from `APP_DEBUG` env var)
- `MAX_UPLOAD_BYTES`: 6 MB file size limit

**Gemini API**:
- `GEMINI_API_KEY`: API key from `.env`
- `GEMINI_MODEL`: Primary model name (default: "gemini-2.5-flash")
- `GEMINI_FALLBACKS`: Fallback models if primary fails
- `GEMINI_MAX_TOKENS`: Response length limit (2048)
- `GEMINI_TEMPERATURE`: Creativity/determinism (0.4 = moderate)

**File Validation**:
- `ALLOWED_EXTENSIONS`: {"pdf", "doc", "docx"}
- `MAGIC_BYTES`: Binary signatures for file validation:
  - PDF: `%PDF`
  - DOCX: `PK\x03\x04` (ZIP header)
  - DOC: `\xd0\xcf\x11\xe0` (OLE2 header)

**Caching**:
- `ANALYSIS_CACHE_TTL`: 3600 seconds (1 hour)
- `AI_STATUS_CACHE_TTL`: 300 seconds (5 minutes)

**Design Pattern**: **Configuration Object** — centralizes all constants

**Usage**: Imported by all modules; loaded once at startup

---

### 3. **backend/routes.py** — HTTP Routes & Request Handling
**Role**: All request/response handling and file validation

**Key Routes**:

#### `GET /`
- **Purpose**: Render upload landing page
- **Logic**:
  - Check AI availability
  - Render `index.html` with AI status
  - Display error message if present

#### `POST /upload`
- **Purpose**: Handle file upload and validation
- **Validation Steps**:
  1. File exists and is selected
  2. Extension in ALLOWED_EXTENSIONS
  3. Save to `uploads/` with UUID filename
  4. Validate magic bytes (binary signature)
  5. Redirect to `/analysis`
- **Error Handling**: 400 errors for validation failures, 413 for oversized files
- **Helper Functions**:
  - `_allowed_file(filename)`: Extension validation
  - `_make_filename(filename)`: UUID + extension filename
  - `_validate_magic(path, ext)`: Binary signature verification

#### `GET /analysis`
- **Purpose**: Show loading/progress animation
- **Logic**:
  - Validate filename parameter (no path traversal)
  - Render `analysis.html` with filename
  - HTML triggers JavaScript redirect to `/results` after animation

#### `GET /results`
- **Purpose**: Orchestrate analysis and display results
- **Workflow**:
  1. Validate filename (security checks)
  2. Extract text and detect photo: `extract_text_and_photo(path)`
  3. Detect sections: `detect_criteria(text, has_photo)`
  4. Get score & suggestions: `score_and_suggest(text)`
  5. Get grammar errors: `grammar_errors(text)`
  6. Get job recommendations: `recommend_jobs(text)`
  7. Delete file from disk
  8. Render `results.html` with all data
- **Error Handling**: 400 errors for missing/invalid files or unrecognizable content

#### `GET /ai_status`
- **Purpose**: Return JSON with AI availability (for client-side checks)
- **Response**: `{"ai_available": bool, "message": str}`

**Security Features**:

**Error Handlers**:
- `handle_too_large()`: 413 for > 6 MB files
- `handle_400()`: 400 errors with fallback results
- `handle_500()`: 500 errors with generic message

**Security Headers** (`set_security_headers` middleware):
```python
Headers:
  Content-Security-Policy: Strict CSP preventing inline scripts
  X-Content-Type-Options: nosniff (prevents MIME sniffing)
  X-Frame-Options: DENY (prevents clickjacking)
  X-XSS-Protection: 1; mode=block (XSS filter)
  Referrer-Policy: strict-origin-when-cross-origin
  Permissions-Policy: Disable camera, microphone, geolocation
  Strict-Transport-Security: (HTTPS enforcement in production)
```

**Design Pattern**: **Dependency Inversion** — routes depend on abstractions (extractor, analyser, ai_client)

---

### 4. **backend/extractor.py** — Resume Text & Photo Extraction
**Role**: Multi-format text extraction with normalization

**Main Function**:
```python
extract_text_and_photo(path: Path) -> tuple[str, bool]
```

**Supported Formats**:

1. **PDF** (`_extract_pdf`):
   - Uses `pdfplumber`
   - Extracts text from each page
   - Detects images on each page for photo presence

2. **DOCX** (`_extract_docx`):
   - Uses `python-docx`
   - Extracts text from all paragraphs
   - Checks for inline shapes (images)

3. **DOC** (`_extract_doc`):
   - Uses `textract`
   - Converts legacy OLE2 format to UTF-8 text
   - No photo detection (backward compatibility)

**Text Normalization**:

1. **CamelCase Fixing** (`_fix_camel_case`):
   - Splits camelCase: "JavaSpring" → "Java Spring"
   - Preserves URLs, emails, paths: "github.com/myrepo" unchanged
   - Fixes PDF extraction artifacts

2. **Technology Normalization** (`_normalise_tech`):
   - 140+ regex patterns mapping common variations to standard terms
   - Examples:
     - "fast api" → "FastAPI"
     - "node j s" → "Node.js"
     - "mongo db" → "MongoDB"
   - **Open/Closed Principle**: Extend `TECH_NORMALIZATION` dict to add new terms

**TECH_NORMALIZATION Dictionary**:
Contains 140+ technology terms with regex patterns covering:
- Languages: Python, JavaScript, TypeScript, Java, C++, Rust, etc.
- Frameworks: FastAPI, SpringBoot, Django, Next.js, React, Vue, etc.
- Databases: MongoDB, PostgreSQL, MySQL, DynamoDB, Cassandra, etc.
- Cloud: AWS, GCP, Azure, Kubernetes, Docker, Terraform, etc.
- DevOps: Jenkins, CircleCI, GitHub Actions, ArgoCD, etc.
- Data Tools: TensorFlow, PyTorch, Pandas, Spark, Snowflake, etc.
- Security: Snyk, Veracode, Fortify, Checkmarx, etc.

**Design Pattern**: **Strategy Pattern** — format-specific extractors, same interface

**Error Handling**:
- Raises `FileNotFoundError` if file missing
- Raises `ValueError` for unsupported formats
- Raises `RuntimeError` if required library missing

---

### 5. **backend/analyser.py** — Resume Analysis & Scoring
**Role**: All analysis logic: scoring, suggestions, error detection, job matching

**Core Functions**:

#### 1. **`detect_criteria(text: str, has_photo: bool) → dict[str, bool]`**
**Purpose**: Rule-based detection of 8 critical resume sections

**Sections Detected**:
```
Profile Photo: has_photo parameter
Summary: regex for "objective|summary|professional summary|profile"
Skills: regex for "skills?"
Education: regex for degree keywords (B.Tech, MBA, bachelor, university, college, etc.)
Experience: regex for "experience|work history|employment|internship"
Projects: regex for "projects?"
Contact Info: regex for emails or phone numbers
Certifications: regex for "certif|aws|gcp|azure|pmp|cpa|cfa|comptia"
```

**Returns**: Dict with 8 boolean values indicating section presence

#### 2. **`_ai_analyse(text: str) → dict`**
**Purpose**: Single combined AI call for all analysis dimensions

**Cache Mechanism**:
- Hash first 4000 chars using SHA-256
- TTL: 3600 seconds (1 hour)
- Returns cached result if hit
- On miss: calls `ask_gemini()` once

**Prompt Structure** (sent to Gemini):
```
[System instructions about being resume evaluator]
[Today's date]

Analyze resume and respond in EXACTLY this format:

SCORE: <0-100>
SUGGESTIONS:
- <suggestion 1>
... (up to 5)
ERRORS:
- <grammar error 1>
... (up to 8; exclude URLs, emails, tech names)
FIELD: <career field 3-5 words>
ROLES:
- <job role 1>
... (3-5 roles)

[First 3500 chars of resume text]
```

**Response Parsing**:
- Extracts score, suggestions, errors, field, roles
- Filters out fake errors (handles URLs, emails, tech names)
- Caps lists at max counts
- Returns dict: `{"score": int, "suggestions": list, "errors": list, "field": str, "roles": list}`

**Fallback**: Returns empty dict if AI unavailable

#### 3. **`score_and_suggest(text: str) → tuple[int, list[str]]`**
**Purpose**: Get resume score (0-100) and 5 improvement suggestions

**Logic**:
```
IF AI available and cached:
  Return AI score + AI suggestions (up to 5)
ELSE (rule-based fallback):
  Base score = 40
  Add 7 points per section present (max 8 sections)
  Generate suggestions based on missing sections:
    - Missing Summary → "Add professional summary"
    - Missing Skills → "Add dedicated skills section"
    - Missing Projects → "Describe key projects"
    - Missing Contact Info → "Add email/phone"
    - Short text (<600 chars) → "Expand with metrics"
  Cap score between 40-92
  Return (score, suggestions[:5])
```

**Scoring Range**:
- Minimum: 40 (very poor structure)
- Maximum: 92 (excellent but leaves room for improvement)
- Rule-based: 40 + (sections × 7) = 40 to 96

#### 4. **`grammar_errors(text: str) → list[str]`**
**Purpose**: Detect grammar and spelling errors

**Logic**:
```
IF AI available and cached:
  Return AI-detected errors (up to 8)
ELSE (rule-based fallback):
  Check for common typos: "teh", "recieve", "adress", "acheivement", etc.
  Return matching errors
```

**AI Error Filtering**:
- Filters out false positives:
  - URLs (https://, github.com, etc.)
  - Technical terms
  - CamelCase identifiers
  - Usernames/emails/handles
- Limits to real writing errors

#### 5. **`recommend_jobs(text: str) → tuple[list[str], str]`**
**Purpose**: Recommend job roles and identify career field

**Logic**:
```
IF AI available and cached:
  Return AI roles + field
ELSE (rule-based fallback):
  IF text contains research keywords:
    Return ["Research Analyst", "Research Scientist"], "Research & Development"
  ELIF text contains data keywords:
    Return ["Data Analyst", "ML Engineer"], "Data Science & Analytics"
  ELIF text contains software keywords:
    Return ["Software Engineer", "Backend Developer"], "Software Engineering"
  ELIF text contains design keywords:
    Return ["UI/UX Designer", "Frontend Developer"], "Design & Product"
  ELSE:
    Return ["Project Coordinator", "Operations Analyst"], "Business Operations"
```

**Returns**: Tuple of (list of 3-5 job roles, career field string)

**Internal Cache** (`_cache`):
- Structure: `dict[text_hash: str, tuple[result: dict, timestamp: float]]`
- Populated by `_ai_analyse()`
- Read by public functions
- TTL checked on every access

**Design Pattern**: **Template Method** — `_ai_analyse` template, specific functions extract fields

---

### 6. **backend/ai_client.py** — Gemini API Integration
**Role**: Gemini client initialization and low-level API wrapper

**SDK Detection**:
```
Try:
  from google import genai (NEW SDK - preferred)
  from google.genai import types
  _SDK = "new"
Catch ImportError:
  Try:
    import google.generativeai (LEGACY SDK)
    _SDK = "legacy"
  Catch ImportError:
    AI_AVAILABLE = False
```

**Client Initialization**:
```python
if AI_AVAILABLE:
  if _SDK == "new":
    _client = genai.Client(api_key=GEMINI_API_KEY)
  else:
    generativeai.configure(api_key=GEMINI_API_KEY)
    _client = generativeai
```

**Status Caching**:
- Function: `check_ai_status() → tuple[bool, str]`
- Cache TTL: 300 seconds (5 minutes)
- Returns: (is_available, message)
- No live API ping — configuration-only

**Core Function**:

```python
ask_gemini(prompt: str) → str
```

**Logic**:
1. Build model chain: [primary model] + [fallbacks]
2. For each model:
   - Build system prompt (includes today's date for context)
   - Call Gemini API with:
     - model_name
     - prompt
     - temperature: 0.4 (low randomness for consistent scoring)
     - max_output_tokens: 2048
     - top_p: 1.0
   - Handle different SDK versions:
     - NEW SDK: `client.models.generate_content()`
     - LEGACY SDK: `client.GenerativeModel().generate_content()`
3. On quota/429 error: Try next fallback model
4. On other errors: Return empty string (analyser handles fallback)
5. Return response text or empty string

**System Prompt** (dynamic):
```
You are a professional AI resume evaluator.
Provide honest, structured scoring and actionable feedback.
Today's date is [date]. 
Use this date when reasoning about timelines, experience durations, 
and whether resume dates are past, present, or future.
```

**Error Handling**:
- Logs all API errors
- Detects quota errors ("quota" in message or "429" in error)
- Continues to fallback models on quota exhaustion
- Returns "" on total failure (analyser falls back to rule-based)

**Design Pattern**: **Adapter Pattern** — abstracts SDK differences, same interface to caller

---

### 7. **backend/requirements.txt** — Python Dependencies

```
Flask                      # Web framework
python-dotenv             # Environment variable loading
pdfplumber                # PDF text extraction
python-docx               # DOCX file parsing
textract                  # Legacy DOC file conversion
opencv-python             # Optional for photo detection
google-generativeai       # Gemini API (legacy SDK)
```

**Note**: Code supports both old & new Google genai SDK

---

## Frontend Files

### 8. **templates/index.html** — Upload Landing Page

**Structure**:
```html
<header class="topbar">
  - Brand logo + title
  - AI availability indicator (green/red dot)
</header>
<main class="hero-panel">
  <section class="hero-copy">
    - Headline: "Upload your resume..."
    - Feature list (4 items)
  </section>
  <section class="upload-card">
    - File input (accept: .pdf, .doc, .docx)
    - Drag-and-drop zone
    - File preview
    - Submit button
    - Upload constraints note
    - Error display (if any)
  </section>
</main>
```

**Features**:
- Drag-and-drop file upload with visual feedback
- File type validation (client + server)
- File preview display
- Error messaging with icon
- Responsive design

**Form Action**: `POST /upload`

---

### 9. **templates/analysis.html** — Loading Animation Page

**Structure**:
```
Centered loader card with:
  - Animated spinner (CSS keyframe rotation)
  - Heading: "Analyzing your resume..."
  - Subtext
  - 4-step progress tracker
  - Brand footer
```

**Progress Steps**:
1. Extracting resume content (0ms)
2. Analyzing structure (800ms)
3. Running AI assessment (1800ms)
4. Generating recommendations (2800ms)

**Auto-Redirect**:
- JavaScript monitors step completions
- After step 4 completes (2800ms), redirects to `/results`
- Filename passed via data attribute

---

### 10. **templates/results.html** — Results Display Page

**Layout** (CSS grid):

**Header**:
- Back link to home
- Brand
- Analysis mode indicator (AI-assisted/Rule-based)

**Main Results Grid**:

1. **Score Panel** (score-ring-wrap):
   - Circular progress ring (SVG-like CSS)
   - Score number: 0-100
   - Contextual badge: "Strong" (green), "Average" (amber), "Needs Work" (red)
   - Assessment paragraph based on score

2. **Profile Overview** (criteria-list):
   - Checklist of 8 sections
   - ✓ Present / ✗ Missing indicators
   - Green/Red color coding

3. **Improvements Needed**:
   - List of grammar/spelling errors
   - Up to 10 errors displayed
   - Error message format: "word" should be "correction"

4. **AI Suggestions** (or Recommendations):
   - List of 5 actionable suggestions
   - Human-readable bullet points

5. **Job Recommendations**:
   - Strongest career field (3-5 word description)
   - List of 3-5 recommended job roles

**Color Coding** (server-side logic):
```python
score >= 80:  green (#16a34a) + Strong label
score >= 60:  amber (#d97706) + Average label
score < 60:   red (#dc2626) + Needs Work label
```

---

### 11. **static/css/style.css** — Design System & Styling

**Design Tokens** (CSS variables):
```css
Primary Colors:
  --primary: #4f46e5 (Indigo)
  --primary-dark: #3730a3
  --primary-pale: #eef2ff

Surfaces:
  --surface: #ffffff
  --surface-alt: #f8fafc

Borders & Text:
  --border: #e2e8f0
  --text: #1e293b
  --text-mid: #475569
  --muted: #64748b

Status Colors:
  --success: #16a34a (Green)
  --warning: #d97706 (Amber)
  --danger: #dc2626 (Red)

Radius:
  --radius-sm: 10px
  --radius: 16px
  --radius-lg: 22px

Shadows:
  --shadow-sm, --shadow, --shadow-lg

Transitions:
  --transition: 0.2s ease
```

**Component Classes**:
- `.topbar`: Header navigation
- `.upload-card`: File upload container
- `.file-control`: Custom file input styling
- `.primary-button`: Main action button
- `.score-ring`: Circular score visualization
- `.panel`: Card container
- `.badge-ok`, `.badge-missing`: Status indicators
- `.dragover`: Drag-and-drop visual feedback

---

### 12. **static/js/script.js** — Client-Side Interactivity

**File Upload Handler**:
```javascript
handleFiles(files):
  - Validates MIME type
  - Validates file extension
  - Updates file preview
  - Sets fileInput.files
```

**Drag-and-Drop**:
```javascript
dropArea.addEventListener('dragover') → Add 'dragover' class
dropArea.addEventListener('dragleave') → Remove 'dragover' class
dropArea.addEventListener('drop') → handleFiles() + prevent default
```

**Analysis Progress Animation**:
```javascript
On /analysis page:
  - Get filename from data-filename attribute
  - Build /results URL with filename parameter
  - Get 4 step elements by ID
  - On intervals [0, 800, 1800, 2800]:
    - Add 'active' class to current step
    - Move previous step from 'active' to 'done'
  - After final step (350ms after 2800):
    - window.location.href = resultsUrl
```

---

## Data Flow

### Complete Resume Analysis Pipeline

```
USER BROWSER
    │
    ├─ [1] Selects/drags resume file
    │       └─> handleFiles() validates type
    │
    ├─ [2] POST /upload
    │       │
    │       └─> routes.py:upload()
    │           ├─ Check: file exists
    │           ├─ Check: extension in ALLOWED_EXTENSIONS
    │           ├─ Check: file size < 6 MB
    │           ├─ Save: to uploads/ with UUID name
    │           ├─ Check: magic bytes match file type
    │           └─ Redirect: GET /analysis?filename=xyz
    │
    ├─ [3] GET /analysis
    │       │
    │       └─> routes.py:analysis()
    │           └─ Render analysis.html with filename
    │           └─ Browser loads, JavaScript starts
    │
    ├─ [4] Analysis Page (client-side)
    │       └─> script.js progresses through 4 steps
    │           └─ After animation, redirects to /results
    │
    ├─ [5] GET /results?filename=xyz
    │       │
    │       └─> routes.py:results()
    │           │
    │           ├─ [5a] extractor.extract_text_and_photo()
    │           │       ├─ Determine file format (PDF/DOC/DOCX)
    │           │       ├─ Load file from uploads/
    │           │       ├─ Extract raw text (format-specific)
    │           │       ├─ Fix camelCase splitting
    │           │       ├─ Normalize tech terms (140+ patterns)
    │           │       ├─ Detect photo presence
    │           │       └─ Return: (text, has_photo)
    │           │
    │           ├─ [5b] analyser.detect_criteria()
    │           │       └─ Run 8 regex patterns
    │           │       └─ Return: dict of section presence
    │           │
    │           ├─ [5c] analyser.score_and_suggest()
    │           │       │
    │           │       ├─ Call: analyser._ai_analyse()
    │           │       │   ├─ Hash text (SHA-256)
    │           │       │   ├─ Check cache (3600s TTL)
    │           │       │   ├─ If miss: ai_client.ask_gemini()
    │           │       │   │   ├─ Try primary Gemini model
    │           │       │   │   ├─ On fail: try fallback models
    │           │       │   │   ├─ Parse response (score, suggestions, errors, roles, field)
    │           │       │   │   ├─ Cache result (1 hour)
    │           │       │   │   └─ Return parsed dict
    │           │       │   └─ Return cached result if hit
    │           │       │
    │           │       ├─ Extract: AI score + suggestions
    │           │       └─ Return: (score, suggestions[:5])
    │           │           OR (rule-based score, generated suggestions)
    │           │
    │           ├─ [5d] analyser.grammar_errors()
    │           │       └─ Return: AI errors OR typo dictionary matches
    │           │
    │           ├─ [5e] analyser.recommend_jobs()
    │           │       └─ Return: (job_roles, career_field)
    │           │           From AI OR keyword-based matching
    │           │
    │           ├─ [5f] Delete uploaded file from disk
    │           │
    │           └─> Render results.html with:
    │               - score (0-100)
    │               - criteria (section presence dict)
    │               - suggestions (list)
    │               - errors (list)
    │               - job_roles (list)
    │               - job_field (string)
    │               - ai_enabled (boolean)
    │
    └─ [6] Results Page (rendered HTML + Jinja2)
            ├─ Score ring (CSS with --pct custom property)
            ├─ Color-coded badge based on score
            ├─ Section checklist with ✓/✗ indicators
            ├─ Grammar errors list
            ├─ AI suggestions list
            ├─ Recommended job roles
            └─ Analysis mode indicator
```

### Caching Strategy

**Two-level caching**:

1. **AI Analysis Cache** (`analyser._cache`):
   - Key: SHA-256 hash of first 4000 chars
   - Value: (result_dict, timestamp)
   - TTL: 3600 seconds
   - Populated by: `_ai_analyse()`
   - Used by: `score_and_suggest()`, `grammar_errors()`, `recommend_jobs()`
   - Benefit: Identical resumes analyzed only once

2. **AI Status Cache** (`ai_client._status_cache`):
   - Key: None (single global cache)
   - Value: (is_available: bool, message: str, timestamp: float)
   - TTL: 300 seconds
   - Populated by: `check_ai_status()`
   - Benefit: Avoid repeated config checks

---

## Configuration & Deployment

### Environment Variables (`.env` file)

**Required**:
```env
GEMINI_API_KEY=sk-... # Your Gemini API key
FLASK_SECRET_KEY=... # Random secret for sessions (auto-generated if missing)
```

**Optional**:
```env
APP_DEBUG=true                           # Enable debug mode (default: false)
GEMINI_MODEL=gemini-2.5-flash           # Primary model (default: gemini-2.5-flash)
PORT=5000                                # Port number (default: 5000)
```

### Installation & Setup

**1. Create Virtual Environment**:
```bash
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
# or
.venv\Scripts\activate       # Windows
```

**2. Install Dependencies**:
```bash
pip install -r backend/requirements.txt
```

**3. Create `.env` File**:
```bash
# In project root
echo 'GEMINI_API_KEY=your_api_key_here' > .env
echo 'FLASK_SECRET_KEY=some_random_secret' >> .env
```

**4. Run Locally**:
```bash
python backend/app.py
```
Navigate to: `http://127.0.0.1:5000`

### Production Deployment

**Using Gunicorn**:
```bash
# Install
pip install gunicorn

# Run
gunicorn -w 3 -b 0.0.0.0:8000 backend.app:app
```

**Parameters**:
- `-w 3`: 3 worker processes
- `-b 0.0.0.0:8000`: Bind to all interfaces, port 8000

**Cloud Platforms**:

**Heroku**:
```bash
# Procfile
web: gunicorn -w 3 backend.app:app

# Deploy
git push heroku main
```

**AWS (Elastic Beanstalk)**:
```bash
eb create resage-ai-env
eb deploy
```

**Google Cloud Run**:
```bash
gcloud run deploy resage-ai \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

**Azure App Service**:
```bash
az webapp up --name resage-ai --runtime "PYTHON:3.11"
```

### Security Checklist

- [ ] `.env` file NOT in version control (use `.gitignore`)
- [ ] `FLASK_SECRET_KEY` is strong random secret (not default)
- [ ] `APP_DEBUG=false` in production
- [ ] HTTPS enabled (via reverse proxy or platform SSL)
- [ ] File uploads directory has restricted permissions
- [ ] Regular dependency updates (`pip install --upgrade -r requirements.txt`)
- [ ] Monitor API quotas (Gemini API rate limits)
- [ ] Log file access and errors
- [ ] Implement rate limiting for `/upload` endpoint
- [ ] Regular security audits of input validation

### Performance Optimization

**Current Optimizations**:
1. Single AI call per resume (combines score, suggestions, errors, roles)
2. 1-hour caching of identical resume analyses
3. LazyLoading of optional dependencies (pdfplumber, textract)
4. 3500-char text limit to API (reduces token usage)
5. Multiprocessing with Gunicorn (3 workers)

**Future Improvements**:
- Async file uploads with Celery + Redis
- CDN for static assets
- Database for analysis history
- User authentication and dashboard
- Batch resume processing
- PDF image optimization

---

## Troubleshooting

### Common Issues

**"AI client not configured"**:
- Solution: Set `GEMINI_API_KEY` in `.env`
- Check: `echo $GEMINI_API_KEY`

**"File too large (413)"**:
- Solution: Reduce file size or increase `MAX_UPLOAD_BYTES` in `config.py`

**"Unsupported format"**:
- Solution: Ensure file is valid PDF/DOC/DOCX
- Check: Run `file resume.pdf` to verify file type

**"Resume not found"**:
- Solution: Re-upload the resume (files auto-deleted after analysis)

**"textract not installed"**:
- Solution: `pip install textract` (required for .doc files)

**API Quota Exceeded**:
- Issue: Too many Gemini API calls
- Solution: Implement rate limiting or upgrade Gemini plan

---

## Architecture Diagrams

### Module Dependency Graph
```
app.py
├── config.py ◄─┬─ routes.py
│               ├─ extractor.py
│               ├─ analyser.py
│               └─ ai_client.py
├── routes.py ◄─┬─ extractor.py (import)
│               ├─ analyser.py (import)
│               └─ ai_client.py (import)
```

### Request-Response Cycle
```
Browser
   │
   ├─ GET / ───────────┐
   ├─ POST /upload ────┼─> Flask app (app.py)
   ├─ GET /analysis ───┤   │
   ├─ GET /results ────┤   ├─> routes.py (HTTP handling)
   └─ GET /ai_status ──┤   │   │
                       │   │   ├─> extractor.py (file parsing)
                       │   │   ├─> analyser.py (analysis)
                       │   │   └─> ai_client.py (Gemini API)
                       │   │
                       └── Gemini API
                           (Optional AI service)
```

---

## Summary

**ReSage AI** is a production-ready resume analysis platform that combines:
- **Robust file handling**: Multi-format support with security validation
- **Smart analysis**: AI-powered insights with intelligent fallbacks
- **Clean architecture**: SOLID principles, clear separation of concerns
- **Security-first design**: CSP headers, magic byte validation, path traversal protection
- **Scalable infrastructure**: Gunicorn-compatible, cloud-deployable

The codebase is well-structured for maintenance, extensibility, and future enhancements while maintaining strict security standards and professional user experience.

---

**Document Generated**: 2024
**Project**: ReSage AI Resume Screening
**Version**: 1.0
