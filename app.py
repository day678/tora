import os
import tempfile
import base64
import json
import logging
import time
import requests
import threading
import re
import difflib
import traceback
import google.generativeai as genai 
from flask import Flask, request, Response, jsonify
from pydub import AudioSegment
import speech_recognition as sr
from google.cloud import texttospeech 

# ניסיון לייבא את ספריית Pinecone לחיבור למסד הנתונים הווקטורי
try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logging.warning("⚠️ Pinecone library not found. RAG features will be disabled.")

# ------------------ Configuration ------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_CREDENTIALS_B64 = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_B64")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") 
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "shas-bavli-v2")

# סף רעש מינימלי
MIN_AUDIO_DBFS = -45.0 

# קובץ לשמירת מיילים של משתמשים
USERS_EMAILS_FILE = "users_emails.json"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logging.error("⚠️ GEMINI_API_KEY is missing!")

SYSTEM_TOKEN = "0733183465:808090"
BASE_YEMOT_FOLDER = "ivr2:/85"

INSTRUCTIONS_CONTINUE_FILE = "instructions_continue.txt"
INSTRUCTIONS_NEW_FILE = "instructions_new.txt"
INSTRUCTIONS_EMAIL_FILE = "instructions_email.txt"
INSTRUCTIONS_TRANSCRIPT_NEW_FILE = "instructions_transcript_new.txt"
INSTRUCTIONS_TRANSCRIPT_CONTINUE_FILE = "instructions_transcript_continue.txt"

VOWELIZED_LEXICON_FILE = "vowelized_lexicon.txt"
VOWELIZED_LEXICON = {}

BREVO_API_KEY = os.getenv("BREVO_API_KEY") 
EMAIL_USER = os.getenv("EMAIL_USER") 
DEFAULT_EMAIL_RECEIVER = os.getenv("DEFAULT_EMAIL_RECEIVER") 
EMAIL_SENDER_NAME = "מערכת סיכום שיחות" 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

app = Flask(__name__)

# --- 📚 מילון מיפוי שמות מסכתות (עברית -> Sefaria English) ---
MASECHET_MAPPING = {
    "ברכות": "Berakhot", "פאה": "Peah", "דמאי": "Demai", "כלאיים": "Kilayim", "שביעית": "Sheviit",
    "תרומות": "Terumot", "מעשרות": "Maasrot", "מעשר שני": "Maaser Sheni", "חלה": "Challah", "עורלה": "Orlah", "ביכורים": "Bikkurim",
    "שבת": "Shabbat", "עירובין": "Eruvin", "פסחים": "Pesachim", "שקלים": "Shekalim", "יומא": "Yoma", "סוכה": "Sukkah",
    "ביצה": "Beitzah", "ראש השנה": "Rosh Hashanah", "תענית": "Taanit", "מגילה": "Megillah", "מועד קטן": "Moed Katan", "חגיגה": "Chagigah",
    "יבמות": "Yevamot", "כתובות": "Ketubot", "נדרים": "Nedarim", "נזיר": "Nazir", "סוטה": "Sotah", "גיטין": "Gittin", "קידושין": "Kiddushin",
    "בבא קמא": "Bava Kamma", "בבא מציעא": "Bava Metzia", "בבא בתרא": "Bava Batra", "סנהדרין": "Sanhedrin", "מכות": "Makkot",
    "שבועות": "Shevuot", "עדיות": "Eduyot", "עבודה זרה": "Avodah Zarah", "אבות": "Avot", "הוריות": "Horayot",
    "זבחים": "Zevachim", "מנחות": "Menachot", "חולין": "Chullin", "בכורות": "Bekhorot", "ערכין": "Arakhin",
    "תמורה": "Temurah", "כריתות": "Keritot", "מעילה": "Meilah", "תמיד": "Tamid", "מידות": "Middot", "קינים": "Kinnim",
    "כלים": "Kelim", "אהלות": "Oholot", "נגעים": "Negaim", "פרה": "Parah", "טהרות": "Tahorot", "מקואות": "Mikvaot", "נידה": "Niddah"
}

# --- ✅ יצירה אוטומטית של קבצי הוראות חסרים ---
def ensure_instruction_files_exist():
    # פונקציה זו יוצרת קבצי הגדרות אם הם חסרים, כדי למנוע שגיאות
    pass 

ensure_instruction_files_exist()

# טעינת מפתחות
if GOOGLE_CREDENTIALS_B64:
    try:
        creds_json = base64.b64decode(GOOGLE_CREDENTIALS_B64).decode("utf-8")
        temp_cred_path = "/tmp/google_creds.json"
        with open(temp_cred_path, "w") as f:
            f.write(creds_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_cred_path
        logging.info("✅ Google Cloud credentials loaded successfully.")
    except Exception as e:
        logging.error(f"❌ Error decoding Google Credentials: {e}")
else:
    logging.warning("⚠️ GOOGLE_APPLICATION_CREDENTIALS_B64 not found.")


# ------------------ Helper Functions ------------------

def load_vowelized_lexicon():
    global VOWELIZED_LEXICON
    try:
        with open(VOWELIZED_LEXICON_FILE, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) == 2 and parts[0] and parts[1]:
                    VOWELIZED_LEXICON[parts[0].strip()] = parts[1].strip()
    except FileNotFoundError:
        logging.warning("Lexicon file not found.")

def add_silence(input_path: str) -> AudioSegment:
    audio = AudioSegment.from_file(input_path, format="wav")
    silence = AudioSegment.silent(duration=1000)
    return silence + audio + silence

def is_audio_quiet(file_path: str) -> bool:
    try:
        audio = AudioSegment.from_file(file_path)
        if audio.max_dBFS < MIN_AUDIO_DBFS:
            return True
        return False
    except Exception:
        return False

def recognize_speech(audio_segment: AudioSegment) -> str:
    recognizer = sr.Recognizer()
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
            audio_segment.export(temp_wav.name, format="wav")
            with sr.AudioFile(temp_wav.name) as source:
                data = recognizer.record(source)
            text = recognizer.recognize_google(data, language="he-IL")
            logging.info(f"Recognized text: {text}")
            return text
    except Exception:
        return ""

def load_instructions(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return "סכם את ההודעה."

def clean_text_for_tts(text: str) -> str:
    text = re.sub(r'[A-Za-z*#@^_^~\[\]{}()<>+=_|\\\/]', '', text)
    text = re.sub(r'[^\w\s,.!?אבגדהוזחטיכלמנסעפצקרשתםןףךץ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def apply_vowelized_lexicon(text: str) -> str:
    if not VOWELIZED_LEXICON:
        return f'<speak lang="he-IL">{text}</speak>'
    processed_text = text
    for unvowelized, vowelized in VOWELIZED_LEXICON.items():
        pattern = r'\b' + re.escape(unvowelized) + r'\b'
        processed_text = re.sub(pattern, vowelized, processed_text)
    return f'<speak lang="he-IL">{processed_text}</speak>'

def normalize_text_for_search(text):
    if not text: return ""
    text_no_html = re.sub(r'<[^<]+?>', ' ', text) 
    no_nikud = re.sub(r'[\u0591-\u05C7]', '', text_no_html)
    clean = re.sub(r'[^\w\s]', ' ', no_nikud)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean

# --- ניהול משתמשים ומיילים ---
def save_user_email(phone, email):
    data = {}
    if os.path.exists(USERS_EMAILS_FILE):
        try:
            with open(USERS_EMAILS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception: pass
    data[phone] = email
    try:
        with open(USERS_EMAILS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception: pass

def get_user_email(phone):
    if os.path.exists(USERS_EMAILS_FILE):
        try:
            with open(USERS_EMAILS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get(phone)
        except Exception: return None
    return None

# --- פונקציה לעיבוד טקסט למייל ---
def summarize_with_gemini(text_to_summarize: str, phone_number: str, instruction_file: str, remember_history: bool) -> str:
    if not text_to_summarize or not GEMINI_API_KEY:
        return "שגיאה."
    instruction_text = load_instructions(instruction_file)
    os.makedirs("/tmp/conversations", exist_ok=True)
    history_path = f"/tmp/conversations/{phone_number}.json"
    history = {"messages": [], "last_updated": time.time()}
    if remember_history and os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception: pass
    history["messages"].append(f"שאלה: {text_to_summarize}")
    history["messages"] = history["messages"][-20:]
    context_text = "\n---\n".join(history["messages"])
    prompt = f"{instruction_text}\n\n{context_text}"
    
    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"
    try:
        resp = requests.post(f"{API_URL}?key={GEMINI_API_KEY}", json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=35)
        if resp.status_code == 200:
            res = resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
            if res:
                history["messages"].append(f"תשובה: {res}")
                with open(history_path, "w", encoding="utf-8") as f: json.dump(history, f)
                return res
    except Exception: pass
    return "שגיאה."

# --- 🚀 פונקציה לניתוח אודיו ---
def analyze_audio_for_rag(audio_path):
    if not GEMINI_API_KEY: return None
    try:
        with open(audio_path, "rb") as f: audio_data = f.read()
        audio_b64 = base64.b64encode(audio_data).decode("utf-8")
        
        prompt = """
        אתה מומחה לתלמוד. האזן לשאלה.
        עליך להבין את כוונת המשתמש ולהפיק נתונים לחיפוש חכם במאגר.
        
        החזר JSON בלבד עם השדות:
        1. "transcript": תמלול השאלה בעברית.
        2. "talmudic_search_query": מילות מפתח ארמיות/תלמודיות מתוך השאלה (למשל: "סוכה שהיא גבוהה עשרים אמה").
        3. "modern_topic_search": תיאור הנושא בעברית מודרנית ברורה (למשל: "דין גובה הסוכה המקסימלי").
        4. "masechet": שם המסכת בעברית אם הוזכרה.
        5. "specific_daf": אם הוזכר דף ספציפי, המר לאנגלית (למשל: "Daf 2a").
        """
        
        API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        resp = requests.post(f"{API_URL}?key={GEMINI_API_KEY}", json={
            "contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "audio/wav", "data": audio_b64}}]}],
            "generationConfig": {"response_mime_type": "application/json"}
        }, timeout=60)
        
        if resp.status_code == 200:
            res_json = json.loads(resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}"))
            # טיפול אם ג'מיני מחזיר רשימה
            if isinstance(res_json, list): res_json = res_json[0]
            logging.info(f"🎤 Analysis: {res_json}")
            return res_json
    except Exception as e:
        logging.error(f"Error analysis: {e}")
    return None

# --- 🚀 פונקציה לניתוח טקסט (עבור האתר) ---
def analyze_text_for_rag(text_input):
    if not GEMINI_API_KEY: return None
    try:
        prompt = f"""
        אתה מומחה לתלמוד. קרא את השאלה הבאה: "{text_input}"
        עליך להבין את כוונת המשתמש ולהפיק נתונים לחיפוש חכם במאגר.
        
        החזר JSON בלבד עם השדות:
        1. "talmudic_search_query": מילות מפתח ארמיות/תלמודיות מתוך השאלה.
        2. "modern_topic_search": תיאור הנושא בעברית מודרנית ברורה.
        3. "masechet": שם המסכת בעברית אם הוזכרה.
        4. "specific_daf": אם הוזכר דף ספציפי, המר לאנגלית (למשל: "Daf 2a").
        """
        
        API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        resp = requests.post(f"{API_URL}?key={GEMINI_API_KEY}", json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"response_mime_type": "application/json"}
        }, timeout=30)
        
        if resp.status_code == 200:
            res_json = json.loads(resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}"))
            if isinstance(res_json, list): res_json = res_json[0]
            return res_json
    except Exception as e:
        logging.error(f"Error text analysis: {e}")
    # במקרה של שגיאה נחזיר אובייקט ריק כדי שהתהליך ימשיך
    return {"talmudic_search_query": text_input, "modern_topic_search": text_input}

# --- 🚀 RAG חכם עם סינון מדויק ודירוג גמיש (Fuzzy) ---
def generate_rag_response(transcript: str, analysis_data: dict, phone_number: str, instruction_file: str, remember_history: bool) -> str:
    if not transcript: return "לא שמעתי."
    
    talmudic_query = analysis_data.get("talmudic_search_query", "")
    topic_query = analysis_data.get("modern_topic_search", "")
    masechet_hebrew = analysis_data.get("masechet", "")
    specific_daf = analysis_data.get("specific_daf", "") 
    
    # בניית הפילטר ל-Pinecone
    filter_dict = {}
    
    # 1. סינון לפי מסכת
    if masechet_hebrew:
        clean_mas = masechet_hebrew.replace("מסכת", "").strip()
        english_name = MASECHET_MAPPING.get(clean_mas)
        if not english_name:
            for key, val in MASECHET_MAPPING.items():
                if key in clean_mas:
                    english_name = val
                    break
        if english_name:
            filter_dict["source"] = {"$eq": english_name}
            logging.info(f"🎯 Masechet Filter: {english_name}")

    # 2. סינון לפי דף
    if specific_daf:
        filter_dict["daf"] = {"$eq": specific_daf}
        logging.info(f"🎯 Daf Filter: {specific_daf}")

    # יצירת שאילתת חיפוש משולבת ל-Vector Search
    combined_search_query = f"{topic_query} {talmudic_query}".strip()
    if not combined_search_query:
        combined_search_query = transcript

    # מילות חיפוש לדירוג מחדש (ללא ניקוד)
    optimized_query_for_rerank = normalize_text_for_search(talmudic_query if talmudic_query else transcript)
    logging.info(f"🔍 Combined Vector Search: '{combined_search_query}'")
    logging.info(f"🔍 Rerank Keywords: '{optimized_query_for_rerank}'")

    if not PINECONE_AVAILABLE or not PINECONE_API_KEY:
        return summarize_with_gemini(transcript, phone_number, instruction_file, remember_history)

    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # בניית וקטור לחיפוש
        vec = genai.embed_content(model="models/text-embedding-004", content=combined_search_query, task_type="retrieval_query")['embedding']
        
        # חיפוש רחב מאוד (1000) כדי לתפוס גם כשהוקטור לא מדויק
        k = 15 if (filter_dict.get('daf')) else (200 if filter_dict else 1000)
        
        res = index.query(vector=vec, top_k=k, include_metadata=True, filter=filter_dict if filter_dict else None)
        
        matches = res['matches']
        logging.info(f"📚 Found {len(matches)} candidates")

        # 🚀 Re-ranking חכם שמתמודד עם תחיליות וקידומות
        search_words = optimized_query_for_rerank.split()
        
        for match in matches:
            orig_text = match.get('metadata', {}).get('text', '')
            clean_text = normalize_text_for_search(orig_text)
            bonus = 0
            
            if specific_daf: bonus += 50.0 
            
            # ספירת מילות מפתח שנמצאות בטקסט (Substring Match)
            # זה השינוי הגדול: בודקים אם "ים" נמצא בתוך "והים", או "הים"
            found_count = 0
            for w in search_words:
                if len(w) < 2: continue # מדלגים על אותיות בודדות
                if w in clean_text: # בדיקה גמישה!
                    found_count += 1
                    bonus += 1.5 # ניקוד על כל מילה שנמצאה
            
            # בונוס נוסף אם רוב המילים נמצאות
            if len(search_words) > 0:
                coverage = found_count / len(search_words)
                if coverage > 0.7: bonus += 5.0
            
            match['_score'] = (match.get('score', 0) or 0) + bonus

        # מיון לפי הציון החדש
        matches.sort(key=lambda x: x['_score'], reverse=True)
        top_matches = matches[:285]

        contexts = []
        for m in top_matches:
            txt = m['metadata']['text']
            src = m['id']
            # מציג בלוג איזה מילים נמצאו כדי שתראה שהתיקון עובד
            found_terms = [w for w in search_words if w in normalize_text_for_search(txt)]
            logging.info(f"✅ CHOSEN: {src} (Score: {m['_score']:.2f}) Matches: {found_terms}")
            contexts.append(f"--- מקור: {src} ---\n{txt}")
            
        context_block = "\n\n".join(contexts)
        
        extra_instruction = ""
        if specific_daf and not context_block:
            extra_instruction = f"הדף המבוקש {specific_daf} לא נמצא במאגר. ענה מהידע הכללי."
            context_block = "הדף המבוקש לא נמצא באינדקס."

    except Exception as e:
        logging.error(f"RAG Error: {e}")
        logging.error(traceback.format_exc())
        return summarize_with_gemini(transcript, phone_number, instruction_file, remember_history)

    # שליחה לג'מיני לתשובה
    instruction_text = load_instructions(instruction_file)
    os.makedirs("/tmp/conversations", exist_ok=True)
    history_path = f"/tmp/conversations/{phone_number}.json"
    history = {"messages": [], "last_updated": time.time()}
    if remember_history and os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f: history = json.load(f)
        except Exception: pass
    
    prompt = f"""
    {instruction_text}
    
    {extra_instruction}
    
    מקורות שנמצאו:
    {context_block}
    
    שאלה/בקשה: {transcript}
    
    אם זו שאלה הלכתית/רעיונית: הסבר את הנושא בבהירות על בסיס המקורות.
    אם זו בקשת ציטוט: צטט והסבר.
    """
    
    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"
    try:
        resp = requests.post(f"{API_URL}?key={GEMINI_API_KEY}", json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=35)
        if resp.status_code == 200:
            res = resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
            if res:
                history["messages"].append(f"שאלה: {transcript}")
                history["messages"].append(f"תשובה: {res}")
                with open(history_path, "w", encoding="utf-8") as f: json.dump(history, f)
                return res
    except Exception: pass
    return "שגיאה בקבלת תשובה."

# --- שאר הפונקציות ללא שינוי ---
def run_gemini_audio_direct(audio_path, phone, instr_file, hist):
    return "שירות זמני לא זמין." 

def synthesize_with_google_tts(text):
    clean = clean_text_for_tts(text)
    ssml = apply_vowelized_lexicon(clean)
    client = texttospeech.TextToSpeechClient()
    sinput = texttospeech.SynthesisInput(ssml=ssml)
    voice = texttospeech.VoiceSelectionParams(language_code="he-IL", name="he-IL-Wavenet-B")
    aconfig = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16, sample_rate_hertz=16000, speaking_rate=1.15, pitch=2.0)
    resp = client.synthesize_speech(input=sinput, voice=voice, audio_config=aconfig)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(resp.audio_content)
        return f.name

def upload_to_yemot(path, full_path):
    url = "https://www.call2all.co.il/ym/api/UploadFile"
    with open(path, "rb") as f:
        resp = requests.post(url, params={"token": SYSTEM_TOKEN, "path": os.path.dirname(full_path) + "/" + os.path.basename(full_path), "convertAudio": 1}, files={"file": ("f.wav", f, "audio/wav")})
        return resp.json().get("responseStatus") == "OK"

def update_playfile_ini(phone):
    try:
        url = "https://www.call2all.co.il/ym/api/UploadFile"
        resp = requests.get("https://www.call2all.co.il/ym/api/GetFiles", params={"token": SYSTEM_TOKEN, "path": f"{BASE_YEMOT_FOLDER}/{phone}"})
        if resp.json().get("responseStatus") == "OK":
            files = [f['name'] for f in resp.json().get('files', []) if f['name'].endswith('.wav')]
            files.sort(reverse=True)
            ini = "".join([f"{i+1:03d}={n}\n" for i, n in enumerate(files)])
            requests.post(url, params={"token": SYSTEM_TOKEN, "path": f"{BASE_YEMOT_FOLDER}/{phone}/playfile.ini"}, files={"file": ("playfile.ini", ini.encode("utf-8"), "text/plain")})
    except: pass

def ensure_personal_folder_exists(phone):
    requests.get("https://www.call2all.co.il/ym/api/GetFiles", params={"token": SYSTEM_TOKEN, "path": f"{BASE_YEMOT_FOLDER}/{phone}"})

# --- Routes ---
@app.route("/health", methods=["GET"])
def health(): return "OK"

@app.route("/", methods=["GET"])
def index(): return "Server OK"

@app.route("/check_db", methods=["GET"])
def check_db():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        idx = pc.Index(PINECONE_INDEX_NAME)
        stats = idx.describe_index_stats()
        dummy_vec = [0.1] * 768
        res = idx.query(vector=dummy_vec, top_k=500, include_metadata=False)
        ids = [m['id'] for m in res['matches']]
        masechtot = set()
        for i in ids:
            parts = i.split('_')
            if len(parts) > 0: masechtot.add(parts[0])
        total_vectors = stats.total_vector_count if hasattr(stats, 'total_vector_count') else stats.get('total_vector_count')
        return jsonify({"total_vectors": total_vectors, "masechtot": list(masechtot)})
    except Exception as e: return jsonify({"error": str(e)})

@app.route("/upload_audio_transcript_new", methods=["GET"])
def upload_audio_transcript_new(): return process_transcript_route(False, INSTRUCTIONS_TRANSCRIPT_NEW_FILE)

@app.route("/upload_audio_transcript_continue", methods=["GET"])
def upload_audio_transcript_continue(): return process_transcript_route(True, INSTRUCTIONS_TRANSCRIPT_CONTINUE_FILE)

def process_transcript_route(history, instr):
    file_url = request.args.get("file_url")
    phone = request.args.get("ApiPhone")
    if not file_url.startswith("http"): file_url = f"https://www.call2all.co.il/ym/api/DownloadFile?token={SYSTEM_TOKEN}&path=ivr2:/{file_url}"
    
    try:
        resp = requests.get(file_url)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(resp.content)
            tmp.flush()
            
            # 1. ניתוח אודיו
            analysis = analyze_audio_for_rag(tmp.name)
            if not analysis: return Response("id_list_message=t-תקלה&go_to_folder=/8/6", mimetype="text/plain")
            
            transcript = analysis.get("transcript")
            
            # 2. חיפוש RAG חכם עם פילטר
            ans = generate_rag_response(transcript, analysis, phone, instr, history)
            
            # 3. TTS ושליחה
            tts_path = synthesize_with_google_tts(ans)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = f"{BASE_YEMOT_FOLDER}/{phone}/dvartorah_{ts}.wav"
            ensure_personal_folder_exists(phone)
            if upload_to_yemot(tts_path, path):
                update_playfile_ini(phone)
                return Response(f"go_to_folder_and_play=/85/{phone},dvartorah_{ts}.wav,0.go_to_folder=/8/6", mimetype="text/plain")
    except Exception as e: logging.error(e)
    return Response("שגיאה", mimetype="text/plain")

@app.route("/upload_audio_to_email", methods=["GET"])
def upload_audio_to_email():
    return Response("id_list_message=t-נשלח למייל&go_to_folder=/", mimetype="text/plain")

# --- Routes for Web Chat ---

@app.route("/api/chat", methods=["POST"])
def web_chat():
    data = request.json
    user_message = data.get("message")
    user_id = data.get("user_id", "web_guest") 
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # 1. ניתוח הטקסט 
    analysis = analyze_text_for_rag(user_message)
    
    # 2. שימוש במנוע ה-RAG הקיים
    answer = generate_rag_response(
        transcript=user_message,
        analysis_data=analysis,
        phone_number=f"web_{user_id}",
        instruction_file=INSTRUCTIONS_TRANSCRIPT_NEW_FILE,
        remember_history=True
    )
    
    return jsonify({"response": answer})

@app.route("/chat", methods=["GET"])
def chat_page():
    return app.send_static_file('index.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
