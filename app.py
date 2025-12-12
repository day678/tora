import os
import tempfile
import base64
import json
import logging
import time
import requests
import threading
import time
import re
import subprocess 
import google.generativeai as genai 
from flask import Flask, request, Response
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
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") # 🆕 מפתח ל-Pinecone
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "shas-bavli-v2") # 🆕 שם האינדקס שלך

# סף רעש מינימלי (בדציבלים). אם הקובץ שקט מזה, הוא ייחשב כשקט מדי.
MIN_AUDIO_DBFS = -45.0 

# קובץ לשמירת מיילים של משתמשים
USERS_EMAILS_FILE = "users_emails.json"

# 🛠 הגדרת Gemini API Key למודול החדש
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logging.error("⚠️ GEMINI_API_KEY is missing!")

SYSTEM_TOKEN = "0733183465:808090"
BASE_YEMOT_FOLDER = "ivr2:/85"  # שלוחה ראשית לכל הקבצים

INSTRUCTIONS_CONTINUE_FILE = "instructions_continue.txt"
INSTRUCTIONS_NEW_FILE = "instructions_new.txt"
# --- קובץ הנחיות למייל ---
INSTRUCTIONS_EMAIL_FILE = "instructions_email.txt"
# --- קבצי הנחיות חדשים לתמלול ---
INSTRUCTIONS_TRANSCRIPT_NEW_FILE = "instructions_transcript_new.txt"
INSTRUCTIONS_TRANSCRIPT_CONTINUE_FILE = "instructions_transcript_continue.txt"

VOWELIZED_LEXICON_FILE = "vowelized_lexicon.txt"
VOWELIZED_LEXICON = {}

# --- הגדרות חדשות לשליחת מייל (Brevo) ---
BREVO_API_KEY = os.getenv("BREVO_API_KEY") 
EMAIL_USER = os.getenv("EMAIL_USER") 
DEFAULT_EMAIL_RECEIVER = os.getenv("DEFAULT_EMAIL_RECEIVER") 
EMAIL_SENDER_NAME = "מערכת סיכום שיחות" 

# ------------------ Logging ------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

app = Flask(__name__)

# יצירת קובץ זמני למפתח של Google Cloud
if GOOGLE_CREDENTIALS_B64:
    creds_json = base64.b64decode(GOOGLE_CREDENTIALS_B64).decode("utf-8")
    temp_cred_path = "/tmp/google_creds.json"
    with open(temp_cred_path, "w") as f:
        f.write(creds_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_cred_path
    logging.info("✅ Google Cloud credentials loaded successfully.")
else:
    logging.warning("⚠️ GOOGLE_APPLICATION_CREDENTIALS_B64 not found. TTS will fail.")


# ------------------ Helper Functions ------------------

def load_vowelized_lexicon():
    """טוען את קובץ המילים המנוקדות לזיכרון."""
    global VOWELIZED_LEXICON
    try:
        with open(VOWELIZED_LEXICON_FILE, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) == 2 and parts[0] and parts[1]:
                    VOWELIZED_LEXICON[parts[0].strip()] = parts[1].strip()
        logging.info(f"✅ Loaded {len(VOWELIZED_LEXICON)} words into the vowelized lexicon.")
    except FileNotFoundError:
        logging.warning(f"⚠️ Lexicon file {VOWELIZED_LEXICON_FILE} not found. Running without custom pronunciation.")
    except Exception as e:
        logging.error(f"❌ Error loading lexicon: {e}")


def add_silence(input_path: str) -> AudioSegment:
    audio = AudioSegment.from_file(input_path, format="wav")
    silence = AudioSegment.silent(duration=1000)
    return silence + audio + silence

# ✅ פונקציה לבדיקת עוצמת השמע
def is_audio_quiet(file_path: str) -> bool:
    """בודק אם קובץ האודיו שקט מדי (מתחת לסף שהוגדר)."""
    try:
        audio = AudioSegment.from_file(file_path)
        logging.info(f"🎤 Audio max dBFS: {audio.max_dBFS}")
        if audio.max_dBFS < MIN_AUDIO_DBFS:
            return True
        return False
    except Exception as e:
        logging.error(f"⚠️ Error checking audio volume: {e}")
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
    except sr.UnknownValueError:
        return ""
    except Exception as e:
        logging.error(f"Speech recognition error: {e}")
        return ""


def load_instructions(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        logging.warning(f"⚠️ Instruction file {file_path} not found or unreadable.")
        return "סכם את ההודעה הבאה בצורה ברורה ותמציתית."


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


# --- 🆕 ניהול משתמשים ומיילים ---
def save_user_email(phone, email):
    """שומר את המייל של המשתמש בקובץ JSON."""
    data = {}
    if os.path.exists(USERS_EMAILS_FILE):
        try:
            with open(USERS_EMAILS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logging.error(f"Error reading email file: {e}")
            
    data[phone] = email
    
    try:
        with open(USERS_EMAILS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"📧 Saved email for {phone}: {email}")
    except Exception as e:
        logging.error(f"Error saving email file: {e}")

def get_user_email(phone):
    """מחזיר את המייל השמור של המשתמש, או None אם לא קיים."""
    if os.path.exists(USERS_EMAILS_FILE):
        try:
            with open(USERS_EMAILS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get(phone)
        except Exception:
            return None
    return None


# --- פונקציה לעיבוד ישיר של אודיו מול ג'מיני (Direct Audio) - עבור ה-Route הרגיל ---
def run_gemini_audio_direct(audio_path: str, phone_number: str, instruction_file: str, remember_history: bool) -> str:
    if not GEMINI_API_KEY:
        logging.error("Missing GEMINI_API_KEY")
        return "שגיאה: חסר מפתח API."

    # 1. קריאת קובץ האודיו וקידוד ל-Base64
    try:
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        audio_b64 = base64.b64encode(audio_data).decode("utf-8")
    except Exception as e:
        logging.error(f"Error reading audio file: {e}")
        return "שגיאה בקריאת קובץ השמע."

    # 2. טעינת הנחיות והיסטוריה
    instruction_text = load_instructions(instruction_file)
    os.makedirs("/tmp/conversations", exist_ok=True)
    history_path = f"/tmp/conversations/{phone_number}.json"
    history = {"messages": [], "last_updated": time.time()}

    context_parts = []
    
    # אם יש היסטוריה
    if remember_history and os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
            if time.time() - history.get("last_updated", 0) > 1 * 3600:
                history = {"messages": [], "last_updated": time.time()}
            
            if history["messages"]:
                history_context = "היסטוריית השיחה עד כה (תשובות קודמות):\n" + "\n---\n".join(history["messages"])
                context_parts.append({"text": history_context})
        except Exception:
            pass

    # הוספת ההנחיה הראשית
    context_parts.append({"text": f"{instruction_text}\n\nהנה ההודעה הקולית החדשה של המשתמש, ענה עליה בקצרה:"})
    
    # הוספת קובץ האודיו עצמו
    context_parts.append({
        "inline_data": {
            "mime_type": "audio/wav", 
            "data": audio_b64
        }
    })

    # 3. הכנת הבקשה ל-Gemini
    payload = {
        "contents": [{"parts": context_parts}],
        "generationConfig": {"temperature": 0.6, "max_output_tokens": 800}
    }

    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
    for attempt in range(3):
        try:
            response = requests.post(f"{API_URL}?key={GEMINI_API_KEY}", json=payload, timeout=60)
            
            if response.status_code == 429:
                wait_time = 5 * (attempt + 1)
                time.sleep(wait_time)
                continue
                
            response.raise_for_status()
            data = response.json()
            result_text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
            
            if result_text:
                if remember_history:
                    history["messages"].append(f"תשובה: {result_text}")
                    history["messages"] = history["messages"][-20:]
                    history["last_updated"] = time.time()
                    with open(history_path, "w", encoding="utf-8") as f:
                        json.dump(history, f, ensure_ascii=False, indent=2)
                
                return result_text
                
        except Exception as e:
            logging.error(f"Gemini Direct Audio API error (attempt {attempt+1}): {e}")
            time.sleep(2) 
            
    return "שגיאה: עומס חריג בשרתי הבינה המלאכותית."


# --- פונקציה לעיבוד טקסט (משמשת למייל) ---
def summarize_with_gemini(text_to_summarize: str, phone_number: str, instruction_file: str, remember_history: bool) -> str:
    if not text_to_summarize or not GEMINI_API_KEY:
        return "שגיאה: לא ניתן לנסח תשובה."

    instruction_text = load_instructions(instruction_file)
    os.makedirs("/tmp/conversations", exist_ok=True)
    history_path = f"/tmp/conversations/{phone_number}.json"
    history = {"messages": [], "last_updated": time.time()}

    # טעינת היסטוריה
    if remember_history and os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            pass
        if time.time() - history.get("last_updated", 0) > 1 * 3600:
            history = {"messages": [], "last_updated": time.time()}
        
        # מוסיפים את הטקסט הנוכחי להיסטוריה
        history["messages"].append(f"שאלה: {text_to_summarize}")
        history["messages"] = history["messages"][-20:]
        history["last_updated"] = time.time()
        context_text = "\n---\n".join(history["messages"])
    else:
        # אם אין היסטוריה או שזו שיחה חדשה
        history = {"messages": [f"שאלה: {text_to_summarize}"], "last_updated": time.time()}
        context_text = f"שאלה: {text_to_summarize}"

    prompt = f"{instruction_text}\n\nהנה הטקסט שהתקבל:\n{context_text}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.6, "max_output_tokens": 2900}
    }

    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"
    
    for attempt in range(3):
        try:
            response = requests.post(f"{API_URL}?key={GEMINI_API_KEY}", json=payload, timeout=35)
            
            if response.status_code == 429:
                wait_time = 5 * (attempt + 1)
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            data = response.json()
            result = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
            
            if result:
                # עדכון ההיסטוריה עם התשובה
                if remember_history:
                    history["messages"].append(f"תשובה: {result}")
                    history["messages"] = history["messages"][-20:] # שומר על גודל סביר
                    with open(history_path, "w", encoding="utf-8") as f:
                        json.dump(history, f, ensure_ascii=False, indent=2)
                return result
        except Exception as e:
            logging.error(f"Gemini API error (attempt {attempt+1}): {e}")
            time.sleep(1)
            
    return "שגיאה בקבלת תשובה מג'מיני."

# --- 🆕 פונקציה חדשה: RAG (חיפוש וקטורי + תשובה) עבור המסלול החדש ---
def generate_rag_response(user_query: str, phone_number: str, instruction_file: str, remember_history: bool) -> str:
    """
    1. הופך את השאלה לווקטור (Embedding).
    2. מחפש את הקטעים הכי רלוונטיים ב-Pinecone (ההחלטה איזה וקטורים יישלחו נעשית כאן).
    3. שולח לג'מיני את השאלה + המקורות שנמצאו.
    """
    if not user_query or not GEMINI_API_KEY:
        return "שגיאה: חסר טקסט או מפתח API."

    # אם אין Pinecone מוגדר, חוזרים לשיטה הישנה (גיבוי)
    if not PINECONE_AVAILABLE or not PINECONE_API_KEY:
        logging.warning("⚠️ RAG skipped: Pinecone not configured. Falling back to standard Gemini.")
        return summarize_with_gemini(user_query, phone_number, instruction_file, remember_history)

    try:
        # שלב א: יצירת וקטור לשאלה (Embedding)
        # זה השלב שבו המערכת מבינה את המשמעות של השאלה, גם אם התמלול לא 100%
        embedding_result = genai.embed_content(
            model="models/text-embedding-004",
            content=user_query,
            task_type="retrieval_query"
        )
        query_vector = embedding_result['embedding']

        # שלב ב: חיפוש במסד הנתונים (Retrieval)
        # כאן מתבצעת ההחלטה איזה מידע לשלוף על סמך קרבה מתמטית
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        
        search_results = index.query(
            vector=query_vector,
            top_k=4,  # מספר הקטעים לשליפה (אפשר לשנות ל-3 או 5)
            include_metadata=True
        )

        # שלב ג: בניית ההקשר (Context) מתוך התוצאות
        retrieved_contexts = []
        for match in search_results['matches']:
            # הנחה: הטקסט המקורי שמור בתוך metadata תחת שדה 'text'
            if 'metadata' in match and 'text' in match['metadata']:
                source_text = match['metadata']['text']
                # אם יש מזהה מקור (כמו שם מסכת ודף), נוסיף אותו
                source_id = match['id'] if 'id' in match else "מקור"
                retrieved_contexts.append(f"--- מקור ({source_id}) ---\n{source_text}")

        context_block = "\n\n".join(retrieved_contexts)
        
        if not context_block:
             logging.info("ℹ️ No relevant context found in DB for this query.")
             context_block = "לא נמצאו מקורות ישירים במאגר, ענה על בסיס הידע הכללי שלך."

    except Exception as e:
        logging.error(f"❌ RAG Error (Embedding/Pinecone): {e}")
        # במקרה של תקלה בחיפוש, עדיין ננסה לענות רגיל
        return summarize_with_gemini(user_query, phone_number, instruction_file, remember_history)

    # שלב ד: הכנת הפרומפט המלא לג'מיני (עם המקורות)
    instruction_text = load_instructions(instruction_file)
    
    # ניהול היסטוריה (זהה לפונקציות הקודמות)
    os.makedirs("/tmp/conversations", exist_ok=True)
    history_path = f"/tmp/conversations/{phone_number}.json"
    history = {"messages": [], "last_updated": time.time()}

    if remember_history and os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
            if time.time() - history.get("last_updated", 0) > 1 * 3600:
                history = {"messages": [], "last_updated": time.time()}
        except Exception:
            pass

    # בניית ההודעה לג'מיני
    history_str = ""
    if history["messages"]:
        history_str = "היסטוריית שיחה קודמת:\n" + "\n".join(history["messages"][-6:]) # לוקחים רק את האחרונות כדי לחסוך מקום

    # הפרומפט החדש: הנחיות + מקורות מהמאגר + היסטוריה + השאלה החדשה
    final_prompt = f"""
{instruction_text}

📚 **מקורות מידע (מהתלמוד/מאגר המידע) שיש להתבסס עליהם בתשובה:**
{context_block}

💬 {history_str}

❓ **שאלה חדשה:**
{user_query}

אנא ענה על השאלה בהתבסס על המקורות המצורפים לעיל. אם התמלול נראה שגוי, נסה להבין את הכוונה לפי המקורות.
"""

    # שלב ה: שליחה לג'מיני
    payload = {
        "contents": [{"parts": [{"text": final_prompt}]}],
        "generationConfig": {"temperature": 0.5, "max_output_tokens": 2000} # טמפרטורה נמוכה יותר לדיוק
    }

    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"
    
    for attempt in range(3):
        try:
            response = requests.post(f"{API_URL}?key={GEMINI_API_KEY}", json=payload, timeout=40)
            
            if response.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue

            response.raise_for_status()
            data = response.json()
            result = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
            
            if result:
                if remember_history:
                    history["messages"].append(f"שאלה: {user_query}")
                    history["messages"].append(f"תשובה: {result}")
                    history["messages"] = history["messages"][-20:]
                    history["last_updated"] = time.time()
                    with open(history_path, "w", encoding="utf-8") as f:
                        json.dump(history, f, ensure_ascii=False, indent=2)
                return result
                
        except Exception as e:
            logging.error(f"Gemini RAG API error (attempt {attempt+1}): {e}")
            time.sleep(1)

    return "שגיאה בקבלת תשובה מהמערכת החכמה."


# --- הפונקציה המקורית (משתמשת בגוגל) ---
def synthesize_with_google_tts(text: str) -> str:
    cleaned_text = clean_text_for_tts(text)
    ssml_text = apply_vowelized_lexicon(cleaned_text)
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        raise EnvironmentError("Google Cloud credentials not configured for TTS.")
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)
    voice = texttospeech.VoiceSelectionParams(language_code="he-IL", name="he-IL-Wavenet-B")
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16, sample_rate_hertz=16000, speaking_rate=1.15, pitch=2.0)
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    with open(output_path, "wb") as out:
        out.write(response.audio_content)
    logging.info(f"✅ Google TTS file created: {output_path}")
    return output_path


def upload_to_yemot(audio_path: str, yemot_full_path: str):
    url = "https://www.call2all.co.il/ym/api/UploadFile"
    path_no_file = os.path.dirname(yemot_full_path)
    file_name = os.path.basename(yemot_full_path)
    with open(audio_path, "rb") as f:
        files = {"file": (file_name, f, "audio/wav")}
        params = {"token": SYSTEM_TOKEN, "path": f"{path_no_file}/{file_name}", "convertAudio": 1}
        response = requests.post(url, params=params, files=files)
        data = response.json()
        if data.get("responseStatus") == "OK":
            logging.info(f"✅ File uploaded successfully to {yemot_full_path}")
            return True
        else:
            logging.error(f"❌ Upload failed: {data}")
            return False


# --- פונקציה לעדכון קובץ playfile.ini ---
def update_playfile_ini(phone_number: str):
    folder_path = f"{BASE_YEMOT_FOLDER}/{phone_number}"
    url_get_files = "https://www.call2all.co.il/ym/api/GetFiles"
    url_upload = "https://www.call2all.co.il/ym/api/UploadFile"

    try:
        response = requests.get(url_get_files, params={"token": SYSTEM_TOKEN, "path": folder_path})
        data = response.json()
        
        if data.get("responseStatus") != "OK":
            return

        files_list = []
        if "files" in data:
            for file_info in data["files"]:
                file_name = file_info.get("name", "")
                if file_name.endswith(".wav") and not file_name.startswith("ext") and not file_name.startswith("playfile"):
                    files_list.append(file_name)

        files_list.sort(reverse=True)

        ini_content = ""
        for index, file_name in enumerate(files_list):
            ini_content += f"{index + 1:03d}={file_name}\n"

        if not ini_content:
            return

        files = {"file": ("playfile.ini", ini_content.encode("utf-8"), "text/plain")}
        params = {"token": SYSTEM_TOKEN, "path": f"{folder_path}/playfile.ini"}
        requests.post(url_upload, params=params, files=files)

    except Exception as e:
        logging.error(f"❌ Error updating playfile.ini: {e}")


# ✅ פונקציה לווידוא יצירת תיקייה אישית
def ensure_personal_folder_exists(phone_number: str):
    folder_path = f"{BASE_YEMOT_FOLDER}/{phone_number}"
    url_check = "https://www.call2all.co.il/ym/api/GetFiles"
    url_upload = "https://www.call2all.co.il/ym/api/UploadFile"

    try:
        response = requests.get(url_check, params={"token": SYSTEM_TOKEN, "path": folder_path})
        data = response.json()
        if data.get("responseStatus") == "OK":
            return
    except Exception:
        pass

    ext_ini_content = """type=playfile
sayfile=yes
allow_download=yes
after_play_tfr=tfr_more_options
control_after_play_moreA1=minus
control_after_play_moreA2=go_to_folder
control_after_play_moreA3=restart
control_after_play_moreA4=add_to_playlist
playfile_control_play_goto=/1
playfile_end_goto=/11
"""
    files = {"file": ("ext.ini", ext_ini_content.encode("utf-8"), "text/plain")}
    params = {"token": SYSTEM_TOKEN, "path": f"{folder_path}/ext.ini"}
    requests.post(url_upload, params=params, files=files)


# --- פונקציית עזר לשליחת מייל ---
def send_email(to_address: str, subject: str, body: str) -> bool:
    BREVO_API_KEY = os.getenv("BREVO_API_KEY")
    if not all([BREVO_API_KEY, EMAIL_USER, to_address]):
        return False
    
    try:
        api_url = "https://api.brevo.com/v3/smtp/email"
        nl = '\n'
        html_body = body.replace(nl, '<br>')
        fixed_footer = "<br><br>---<br><b>תודה על השימוש בשירות.</b>"
        
        html_content = f"""
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ direction: rtl; font-family: Arial, sans-serif; text-align: right; font-size: 16px; }}
                h2 {{ color: #333; font-size: 20px; margin-top: 15px; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
            </style>
        </head>
        <body dir="rtl">
            <p>שלום,</p>
            <p>התקבל תמלול וסיכום משיחה נכנסת.</p>
            <h2>**פרטי השיחה**</h2>
            <p>{html_body}</p>
            {fixed_footer}
        </body>
        </html>
        """

        body_for_html = body
        body_for_html = body_for_html.replace('-----------------------------------', '<hr>')
        body_for_html = body_for_html.replace('תמלול ההקלטה האחרונה:', '<h2>תמלול ההקלטה האחרונה:</h2>')
        body_for_html = body_for_html.replace('סיכום מלא:', '<h2>סיכום מלא:</h2>')
        
        html_content = f"""
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ direction: rtl; font-family: Arial, sans-serif; text-align: right; font-size: 16px; }}
                h2 {{ color: #004d99; font-size: 20px; margin-top: 20px; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
                b {{ font-weight: bold; }}
            </style>
        </head>
        <body dir="rtl">
            <p>שלום,</p>
            <p>התקבל תמלול וסיכום משיחה נכנסת.</p>
            {body_for_html.replace(nl, '<br>')}
            <hr>
            <p><b>תודה על השימוש בשירות.</b></p>
        </body>
        </html>
        """

        payload = {
            "sender": {"email": EMAIL_USER, "name": EMAIL_SENDER_NAME},
            "to": [{"email": to_address}],
            "subject": subject,
            "htmlContent": html_content
        }
        
        headers = {
            "accept": "application/json",
            "api-key": BREVO_API_KEY,
            "content-type": "application/json"
        }
        
        response = requests.post(api_url, json=payload, headers=headers)
        if response.status_code == 201:
            return True
        else:
            return False
            
    except Exception:
        return False


load_vowelized_lexicon()

# ------------------ Routes ------------------

@app.route("/health", methods=["GET"])
def health():
    return Response("OK", status=200, mimetype="text/plain")


@app.route("/update_email", methods=["GET"])
def update_email():
    phone = request.args.get("ApiPhone")
    new_email = request.args.get("USER_EMAIL")
    
    if phone and new_email:
        new_email = new_email.strip()
        save_user_email(phone, new_email)
        return Response("id_list_message=t-המייל עודכן בהצלחה במערכת&go_to_folder=/8", mimetype="text/plain")
    
    return Response("id_list_message=t-אירעה שגיאה בקליטת המייל&go_to_folder=/8", mimetype="text/plain")


@app.route("/check_email_exists", methods=["GET"])
def check_email_exists():
    phone = request.args.get("ApiPhone")
    email = get_user_email(phone)
    RECORDING_FOLDER = "/9715"
    EMAIL_SETUP_FOLDER = "/8"

    if email:
        return Response(f"go_to_folder={RECORDING_FOLDER}", mimetype="text/plain")
    else:
        return Response(f"id_list_message=t-לא מוגדרת כתובת מייל עבור הטלפון שלכם. הנכם מועברים להגדרת הכתובת.&go_to_folder={EMAIL_SETUP_FOLDER}", mimetype="text/plain")


# --- הפונקציה המקורית: שיחה קולית (אודיו לאודיו) ---
def process_audio_request(request, remember_history: bool, instruction_file: str):
    file_url = request.args.get("file_url")
    phone_number = request.args.get("ApiPhone", "unknown")

    if not remember_history:
        history_path = f"/tmp/conversations/{phone_number}.json"
        if os.path.exists(history_path):
            try:
                os.remove(history_path)
            except Exception:
                pass
        remember_history = True

    if not file_url.startswith("http"):
        file_url = f"https://www.call2all.co.il/ym/api/DownloadFile?token={SYSTEM_TOKEN}&path=ivr2:/{file_url}"

    try:
        response = requests.get(file_url, timeout=20)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_input:
            temp_input.write(response.content)
            temp_input.flush()
            
            if is_audio_quiet(temp_input.name):
                return Response("id_list_message=t-הקובץ שקט מדי, אנא נסו להקליט שוב&go_to_folder=/8/6", mimetype="text/plain")

            gemini_result_text = ""
            def run_gemini():
                nonlocal gemini_result_text
                gemini_result_text = run_gemini_audio_direct(temp_input.name, phone_number, instruction_file, remember_history)
            
            gemini_thread = threading.Thread(target=run_gemini)
            gemini_thread.start()
            gemini_thread.join()

            final_dvartorah = gemini_result_text
            
            tts_path = synthesize_with_google_tts(final_dvartorah)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            personal_folder = f"{BASE_YEMOT_FOLDER}/{phone_number}"
            yemot_full_path = f"{personal_folder}/dvartorah_{timestamp}.wav"

            ensure_personal_folder_exists(phone_number)

            upload_success = upload_to_yemot(tts_path, yemot_full_path)
            os.remove(tts_path)

            if upload_success:
                update_playfile_ini(phone_number)
                playback_command = f"go_to_folder_and_play=/85/{phone_number},dvartorah_{timestamp}.wav,0.go_to_folder=/8/6"
                return Response(playback_command, mimetype="text/plain")
            else:
                return Response("שגיאה בהעלאת הקובץ לשרת.", mimetype="text/plain")
    except Exception as e:
        logging.error(f"Critical error: {e}")
        return Response(f"שגיאה קריטית בעיבוד: {e}", mimetype="text/plain")


# --- ✅ פונקציה חדשה: תמלול טקסט (Google STT -> RAG -> Google TTS) ---
def process_audio_request_transcript(request, remember_history: bool, instruction_file: str):
    file_url = request.args.get("file_url")
    phone_number = request.args.get("ApiPhone", "unknown")

    # ניהול היסטוריה: אם זו שיחה חדשה, נמחק היסטוריה ישנה
    if not remember_history:
        history_path = f"/tmp/conversations/{phone_number}.json"
        if os.path.exists(history_path):
            try:
                os.remove(history_path)
                logging.info(f"🗑️ נמחקה היסטוריה ישנה עבור {phone_number} (נושא חדש - תמלול).")
            except Exception:
                pass
        remember_history = True # ועכשיו נזכור את ההמשך

    if not file_url.startswith("http"):
        file_url = f"https://www.call2all.co.il/ym/api/DownloadFile?token={SYSTEM_TOKEN}&path=ivr2:/{file_url}"

    logging.info(f"Downloading audio from: {file_url} (For Transcript RAG)")
    try:
        response = requests.get(file_url, timeout=20)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_input:
            temp_input.write(response.content)
            temp_input.flush()
            
            # בדיקת שקט
            if is_audio_quiet(temp_input.name):
                return Response("id_list_message=t-הקובץ שקט מדי, אנא נסו להקליט שוב&go_to_folder=/8/6", mimetype="text/plain")

            processed_audio = add_silence(temp_input.name)
            
            # 1. תמלול (STT) - המרת שאלת המאזין לטקסט
            recognized_text = recognize_speech(processed_audio)
            
            if not recognized_text:
                 return Response("id_list_message=t-לא הצלחתי להבין את הנאמר, אנא נסו שוב.&go_to_folder=/8/6", mimetype="text/plain")

            # 2. שליחת הטקסט למנגנון ה-RAG החדש (חיפוש בבסיס ידע + ג'מיני)
            gemini_result_text = ""
            def run_rag_logic():
                nonlocal gemini_result_text
                # ✅ כאן השינוי הגדול: שימוש בפונקציה החדשה generate_rag_response
                gemini_result_text = generate_rag_response(recognized_text, phone_number, instruction_file, remember_history)
            
            gemini_thread = threading.Thread(target=run_rag_logic)
            gemini_thread.start()
            gemini_thread.join()

            final_response = gemini_result_text
            
            # 3. המרת תשובת ג'מיני לקובץ שמע (TTS) של גוגל
            tts_path = synthesize_with_google_tts(final_response)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            personal_folder = f"{BASE_YEMOT_FOLDER}/{phone_number}"
            yemot_full_path = f"{personal_folder}/dvartorah_{timestamp}.wav"

            ensure_personal_folder_exists(phone_number)

            upload_success = upload_to_yemot(tts_path, yemot_full_path)
            os.remove(tts_path)

            if upload_success:
                update_playfile_ini(phone_number)
                playback_command = f"go_to_folder_and_play=/85/{phone_number},dvartorah_{timestamp}.wav,0.go_to_folder=/8/6"
                return Response(playback_command, mimetype="text/plain")
            else:
                return Response("שגיאה בהעלאת הקובץ לשרת.", mimetype="text/plain")
    except Exception as e:
        logging.error(f"Critical error (Transcript Route): {e}")
        return Response(f"שגיאה קריטית בעיבוד: {e}", mimetype="text/plain")


@app.route("/upload_audio_continue", methods=["GET"])
def upload_audio_continue():
    return process_audio_request(request, remember_history=True, instruction_file=INSTRUCTIONS_CONTINUE_FILE)


@app.route("/upload_audio_new", methods=["GET"])
def upload_audio_new():
    return process_audio_request(request, remember_history=False, instruction_file=INSTRUCTIONS_NEW_FILE)


# --- ✅ Routes חדשים עבור תמלול טקסט (שיחה מבוססת טקסט) ---
@app.route("/upload_audio_transcript_new", methods=["GET"])
def upload_audio_transcript_new():
    """מסלול לשיחה חדשה מבוססת תמלול טקסט (עם חיבור לידע)"""
    return process_audio_request_transcript(request, remember_history=False, instruction_file=INSTRUCTIONS_TRANSCRIPT_NEW_FILE)

@app.route("/upload_audio_transcript_continue", methods=["GET"])
def upload_audio_transcript_continue():
    """מסלול להמשך שיחה מבוססת תמלול טקסט (עם חיבור לידע)"""
    return process_audio_request_transcript(request, remember_history=True, instruction_file=INSTRUCTIONS_TRANSCRIPT_CONTINUE_FILE)


def process_audio_for_email(request):
    file_url = request.args.get("file_url")
    phone_number = request.args.get("ApiPhone", "unknown")
    
    if not file_url:
        return Response("id_list_message=t-שגיאת הגדרה חמורה במערכת, הקלטה לא התקבלה. אנא פנה למנהל.go_to_folder=/8/6", mimetype="text/plain")

    saved_email = get_user_email(phone_number)
    if saved_email:
        email_to = saved_email
    else:
        email_to = request.args.get("ApiEmail", DEFAULT_EMAIL_RECEIVER)

    if not email_to:
        return Response("id_list_message=t-שגיאה, לא הוגדרה כתובת מייל לשליחה.go_to_folder=/8/6", mimetype="text/plain")

    if not file_url.startswith("http"):
        file_url = f"https://www.call2all.co.il/ym/api/DownloadFile?token={SYSTEM_TOKEN}&path=ivr2:/{file_url}"

    try:
        response = requests.get(file_url, timeout=20)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_input:
            temp_input.write(response.content)
            temp_input.flush()
            
            if is_audio_quiet(temp_input.name):
                return Response("id_list_message=t-הקובץ שקט מדי, אנא נסו להקליט שוב&go_to_folder=/8/6", mimetype="text/plain")

            processed_audio = add_silence(temp_input.name)
            recognized_text = recognize_speech(processed_audio)
            if not recognized_text:
                return Response("לא זוהה דיבור ברור. אנא נסה שוב.", mimetype="text/plain")

            gemini_result = {}
            def run_gemini():
                # שימוש בפונקציה הרגילה עבור מיילים (ללא מאגר ידע, רק סיכום)
                gemini_result["text"] = summarize_with_gemini(recognized_text, phone_number, INSTRUCTIONS_EMAIL_FILE, remember_history=False)
            gemini_thread = threading.Thread(target=run_gemini)
            gemini_thread.start()
            gemini_thread.join()

            final_dvartorah_summary = gemini_result.get("text", "לא נוצר סיכום.")

            subject = f"סיכום שיחה חדש:"
            body_content = f"""
<hr>
<h2>תמלול ההקלטה האחרונה:</h2>
<p>{recognized_text}</p>
<hr>
<h2>סיכום מלא:</h2>
<p>{final_dvartorah_summary}</p>
"""
            email_success = send_email(email_to, subject, body_content)

            if email_success:
                return Response("id_list_message=t-ההודעה נשלחה בהצלחה למייל&go_to_folder=/", mimetype="text/plain")
            else:
                return Response("id_list_message=t-שגיאה בשליחת המייל, אנא נסה שוב.go_to_folder=/", mimetype="text/plain")

    except Exception as e:
        logging.error(f"Critical error in email processing: {e}")
        return Response(f"שגיאה קריטית בעיבוד למייל: {e}", mimetype="text/plain")

@app.route("/upload_audio_to_email", methods=["GET"])
def upload_audio_to_email():
    return process_audio_for_email(request)


# ------------------ Run ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
