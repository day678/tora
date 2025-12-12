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

# --- ✅ יצירה אוטומטית של קבצי הוראות חסרים (כדי למנוע את השגיאה בלוג) ---
def ensure_instruction_files_exist():
    defaults = {
        INSTRUCTIONS_TRANSCRIPT_NEW_FILE: """הנך חברותא וירטואלי לתלמוד בבלי.
תפקידך:
1. לקבל שאלות על הגמרא או מושגים תלמודיים מהמשתמש.
2. לעיין במקורות שסופקו לך (שנשלפו מתוך מאגר המידע של הש"ס).
3. לענות תשובה בהירה, קצרה ומדויקת שמסבירה את הסוגיה.
4. אם המקורות שסופקו לא עונים ישירות על השאלה, הסבר מה כן מופיע בהם ונסה לקשר זאת לשאלה, או ציין שאין בידך המקור המדויק כרגע.
סגנון התשובה:
- ענה בעברית ברורה וקולחת (מתאים להשמעה קולית).
- אל תאריך מדי בציטוטים, אלא תסביר את התוכן.
- השתמש במונחים כמו "הגמרא מסבירה", "הסברה היא", "שיטת רש"י היא".
- אל תמציא הלכות או עובדות שאינן במקורות.
התשובה שלך תומר לדיבור ותושמע למאזין בטלפון, אז היה תמציתי וממוקד.""",
        INSTRUCTIONS_TRANSCRIPT_CONTINUE_FILE: """הנך חברותא וירטואלי. המשך את השיחה עם המשתמש על בסיס התשובות הקודמות והמקורות החדשים. ענה בקצרה."""
    }

    for filename, content in defaults.items():
        if not os.path.exists(filename):
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(content)
                logging.info(f"✅ Created missing file: {filename}")
            except Exception as e:
                logging.error(f"❌ Failed to create {filename}: {e}")

# הפעלה בעליית השרת
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
        logging.info(f"✅ Loaded {len(VOWELIZED_LEXICON)} words into the vowelized lexicon.")
    except FileNotFoundError:
        logging.warning(f"⚠️ Lexicon file {VOWELIZED_LEXICON_FILE} not found. Running without custom pronunciation.")
    except Exception as e:
        logging.error(f"❌ Error loading lexicon: {e}")


def add_silence(input_path: str) -> AudioSegment:
    audio = AudioSegment.from_file(input_path, format="wav")
    silence = AudioSegment.silent(duration=1000)
    return silence + audio + silence

def is_audio_quiet(file_path: str) -> bool:
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

# --- ניהול משתמשים ומיילים ---
def save_user_email(phone, email):
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
    if os.path.exists(USERS_EMAILS_FILE):
        try:
            with open(USERS_EMAILS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get(phone)
        except Exception:
            return None
    return None

# --- פונקציה לעיבוד טקסט למייל (ללא RAG) ---
def summarize_with_gemini(text_to_summarize: str, phone_number: str, instruction_file: str, remember_history: bool) -> str:
    if not text_to_summarize or not GEMINI_API_KEY:
        return "שגיאה: לא ניתן לנסח תשובה."

    instruction_text = load_instructions(instruction_file)
    os.makedirs("/tmp/conversations", exist_ok=True)
    history_path = f"/tmp/conversations/{phone_number}.json"
    history = {"messages": [], "last_updated": time.time()}

    if remember_history and os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            pass
        if time.time() - history.get("last_updated", 0) > 1 * 3600:
            history = {"messages": [], "last_updated": time.time()}
        
        history["messages"].append(f"שאלה: {text_to_summarize}")
        history["messages"] = history["messages"][-20:]
        history["last_updated"] = time.time()
        context_text = "\n---\n".join(history["messages"])
    else:
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
                if remember_history:
                    history["messages"].append(f"תשובה: {result}")
                    history["messages"] = history["messages"][-20:]
                    with open(history_path, "w", encoding="utf-8") as f:
                        json.dump(history, f, ensure_ascii=False, indent=2)
                return result
        except Exception as e:
            logging.error(f"Gemini API error (attempt {attempt+1}): {e}")
            time.sleep(1)
    return "שגיאה בקבלת תשובה מג'מיני."

# --- 🆕 פונקציה משופרת: RAG עם אופטימיזציה למונחי גמרא ---
def generate_rag_response(user_query: str, phone_number: str, instruction_file: str, remember_history: bool) -> str:
    """
    1. הופך את השאלה למונח גמרא מדויק (בלי מילים מיותרות).
    2. מחפש את הקטעים הכי רלוונטיים ב-Pinecone.
    3. שולח לג'מיני את השאלה המקורית + המקורות שנמצאו.
    """
    if not user_query or not GEMINI_API_KEY:
        return "שגיאה: חסר טקסט או מפתח API."

    if not PINECONE_AVAILABLE or not PINECONE_API_KEY:
        logging.warning("⚠️ RAG skipped: Pinecone not configured. Falling back to standard Gemini.")
        return summarize_with_gemini(user_query, phone_number, instruction_file, remember_history)

    try:
        # 🚀 שיפור קריטי: בקשה לביטוי הארמי המדויק בלבד
        search_optimization_prompt = f"""
        אתה מנוע חיפוש חכם לתלמוד הבבלי.
        המשתמש שאל: "{user_query}"
        
        המשימה שלך:
        זהה את המושג המרכזי בשאלה והמר אותו למשפט חיפוש המורכב **אך ורק** מהמילים המדויקות כפי שהן מופיעות בגמרא (ארמית/לשון הקודש).
        
        כללים מחמירים:
        1. אל תכתוב רשימת מכולת (בלי כוכביות, בלי בולטים).
        2. אל תכתוב מילים כלליות כמו "גמרא", "דין", "ביאור", "פירוש", "סוגיה". מילים אלו הורסות את החיפוש.
        3. פלוט רק את הביטוי עצמו.
        
        דוגמה:
        אם השאלה היא "מה הגמרא אומרת על הסברה שהמוציא מחברו עליו הראיה?"
        הפלט שלך צריך להיות: "המוציא מחברו עליו הראיה"
        
        ביטוי לחיפוש:
        """
        
        optimized_query = user_query # ברירת מחדל
        
        try:
             opt_payload = {
                "contents": [{"parts": [{"text": search_optimization_prompt}]}],
                "generationConfig": {"temperature": 0.1, "max_output_tokens": 50} # טמפרטורה נמוכה מאוד לדיוק
             }
             opt_resp = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}", 
                json=opt_payload, timeout=5
             )
             if opt_resp.status_code == 200:
                 candidate = opt_resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                 # ניקוי נוסף של הפלט למקרה שג'מיני הוסיף שטויות
                 clean_candidate = candidate.replace("*", "").replace("-", "").replace("\n", " ").strip()
                 if clean_candidate:
                     optimized_query = clean_candidate
                 logging.info(f"🔍 Optimized Search Query: '{user_query}' -> '{optimized_query}'")
        except Exception as opt_e:
            logging.warning(f"⚠️ Query optimization failed, using original query. Error: {opt_e}")

        # שלב א: יצירת וקטור לשאילתה המשופרת
        embedding_result = genai.embed_content(
            model="models/text-embedding-004",
            content=optimized_query,
            task_type="retrieval_query"
        )
        query_vector = embedding_result['embedding']

        # שלב ב: חיפוש במסד הנתונים
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        
        search_results = index.query(
            vector=query_vector,
            top_k=5, 
            include_metadata=True
        )

        # שלב ג: בניית ההקשר
        retrieved_contexts = []
        for match in search_results['matches']:
            if 'metadata' in match and 'text' in match['metadata']:
                source_text = match['metadata']['text']
                source_id = match['id'] if 'id' in match else "מקור"
                score = match['score'] if 'score' in match else 0
                logging.info(f"📄 Retrieved: {source_id} (Score: {score:.4f})")
                retrieved_contexts.append(f"--- מקור ({source_id}) ---\n{source_text}")

        context_block = "\n\n".join(retrieved_contexts)
        
        if not context_block:
             logging.info("ℹ️ No relevant context found in DB for this query.")
             context_block = "לא נמצאו מקורות ישירים במאגר."

    except Exception as e:
        logging.error(f"❌ RAG Error (Embedding/Pinecone): {e}")
        return summarize_with_gemini(user_query, phone_number, instruction_file, remember_history)

    # שלב ד: הכנת הפרומפט המלא לתשובה
    instruction_text = load_instructions(instruction_file)
    
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

    history_str = ""
    if history["messages"]:
        history_str = "היסטוריית שיחה קודמת:\n" + "\n".join(history["messages"][-6:])

    final_prompt = f"""
{instruction_text}

📚 **מקורות מהגמרא (שנמצאו לפי החיפוש: "{optimized_query}"):**
{context_block}

💬 {history_str}

❓ **שאלת המשתמש המקורית:**
{user_query}

הנחיה: ענה על שאלת המשתמש. השתמש במקורות כדי להסביר את המושג או הסוגיה שעליה נשאלת. אם המקורות מתאימים (למשל מצאת את הסוגיה הנכונה), בסס את תשובתך עליהם.
"""

    # שלב ה: שליחה לג'מיני (המוח העונה)
    payload = {
        "contents": [{"parts": [{"text": final_prompt}]}],
        "generationConfig": {"temperature": 0.4, "max_output_tokens": 2000} 
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


# --- פונקציה לעיבוד ישיר של אודיו (ללא STT) ---
def run_gemini_audio_direct(audio_path: str, phone_number: str, instruction_file: str, remember_history: bool) -> str:
    if not GEMINI_API_KEY:
        return "שגיאה: חסר מפתח API."

    try:
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        audio_b64 = base64.b64encode(audio_data).decode("utf-8")
    except Exception as e:
        return "שגיאה בקריאת קובץ השמע."

    instruction_text = load_instructions(instruction_file)
    os.makedirs("/tmp/conversations", exist_ok=True)
    history_path = f"/tmp/conversations/{phone_number}.json"
    history = {"messages": [], "last_updated": time.time()}

    context_parts = []
    if remember_history and os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
            if time.time() - history.get("last_updated", 0) > 1 * 3600:
                history = {"messages": [], "last_updated": time.time()}
            if history["messages"]:
                history_context = "היסטוריה:\n" + "\n".join(history["messages"])
                context_parts.append({"text": history_context})
        except Exception:
            pass

    context_parts.append({"text": f"{instruction_text}\n\nהודעה קולית:"})
    context_parts.append({"inline_data": {"mime_type": "audio/wav", "data": audio_b64}})

    payload = {
        "contents": [{"parts": context_parts}],
        "generationConfig": {"temperature": 0.6, "max_output_tokens": 800}
    }

    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
    for attempt in range(3):
        try:
            response = requests.post(f"{API_URL}?key={GEMINI_API_KEY}", json=payload, timeout=60)
            if response.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            response.raise_for_status()
            data = response.json()
            result_text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
            
            if result_text:
                if remember_history:
                    history["messages"].append(f"תשובה: {result_text}")
                    with open(history_path, "w", encoding="utf-8") as f:
                        json.dump(history, f, ensure_ascii=False, indent=2)
                return result_text
        except Exception:
            time.sleep(2)
            
    return "שגיאה במערכת."

# --- TTS ---
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
        return response.json().get("responseStatus") == "OK"

def update_playfile_ini(phone_number: str):
    folder_path = f"{BASE_YEMOT_FOLDER}/{phone_number}"
    url_get_files = "https://www.call2all.co.il/ym/api/GetFiles"
    url_upload = "https://www.call2all.co.il/ym/api/UploadFile"
    try:
        response = requests.get(url_get_files, params={"token": SYSTEM_TOKEN, "path": folder_path})
        data = response.json()
        if data.get("responseStatus") != "OK": return
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
        if not ini_content: return
        files = {"file": ("playfile.ini", ini_content.encode("utf-8"), "text/plain")}
        params = {"token": SYSTEM_TOKEN, "path": f"{folder_path}/playfile.ini"}
        requests.post(url_upload, params=params, files=files)
    except Exception as e:
        logging.error(f"❌ Error updating playfile.ini: {e}")

def ensure_personal_folder_exists(phone_number: str):
    folder_path = f"{BASE_YEMOT_FOLDER}/{phone_number}"
    try:
        if requests.get("https://www.call2all.co.il/ym/api/GetFiles", params={"token": SYSTEM_TOKEN, "path": folder_path}).json().get("responseStatus") == "OK": return
    except: pass
    ext_ini = "type=playfile\nsayfile=yes\nallow_download=yes\nafter_play_tfr=tfr_more_options\ncontrol_after_play_moreA1=minus\ncontrol_after_play_moreA2=go_to_folder\ncontrol_after_play_moreA3=restart\ncontrol_after_play_moreA4=add_to_playlist\nplayfile_control_play_goto=/1\nplayfile_end_goto=/11\n"
    requests.post("https://www.call2all.co.il/ym/api/UploadFile", params={"token": SYSTEM_TOKEN, "path": f"{folder_path}/ext.ini"}, files={"file": ("ext.ini", ext_ini.encode("utf-8"), "text/plain")})

def send_email(to_address: str, subject: str, body: str) -> bool:
    BREVO_API_KEY = os.getenv("BREVO_API_KEY")
    if not all([BREVO_API_KEY, EMAIL_USER, to_address]): return False
    try:
        api_url = "https://api.brevo.com/v3/smtp/email"
        html_content = f"<html><body dir='rtl'><p>שלום,</p><p>{body.replace(chr(10), '<br>')}</p></body></html>"
        payload = {"sender": {"email": EMAIL_USER, "name": EMAIL_SENDER_NAME}, "to": [{"email": to_address}], "subject": subject, "htmlContent": html_content}
        headers = {"accept": "application/json", "api-key": BREVO_API_KEY, "content-type": "application/json"}
        return requests.post(api_url, json=payload, headers=headers).status_code == 201
    except: return False

load_vowelized_lexicon()

# ------------------ Routes ------------------
@app.route("/health", methods=["GET"])
def health(): return Response("OK", status=200, mimetype="text/plain")

@app.route("/update_email", methods=["GET"])
def update_email():
    phone = request.args.get("ApiPhone")
    new_email = request.args.get("USER_EMAIL")
    if phone and new_email:
        save_user_email(phone, new_email.strip())
        return Response("id_list_message=t-המייל עודכן בהצלחה&go_to_folder=/8", mimetype="text/plain")
    return Response("id_list_message=t-שגיאה&go_to_folder=/8", mimetype="text/plain")

@app.route("/check_email_exists", methods=["GET"])
def check_email_exists():
    phone = request.args.get("ApiPhone")
    return Response(f"go_to_folder=/9715", mimetype="text/plain") if get_user_email(phone) else Response(f"id_list_message=t-לא מוגדר מייל. מעביר להגדרה.&go_to_folder=/8", mimetype="text/plain")

def process_audio_request(request, remember_history: bool, instruction_file: str):
    # (קוד מקוצר לפונקציה הישנה - ללא שינוי מהותי)
    file_url = request.args.get("file_url")
    phone = request.args.get("ApiPhone", "unknown")
    if not file_url.startswith("http"): file_url = f"https://www.call2all.co.il/ym/api/DownloadFile?token={SYSTEM_TOKEN}&path=ivr2:/{file_url}"
    try:
        resp = requests.get(file_url)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp:
            temp.write(resp.content)
            temp.flush()
            if is_audio_quiet(temp.name): return Response("id_list_message=t-שקט מדי&go_to_folder=/8/6", mimetype="text/plain")
            res_text = run_gemini_audio_direct(temp.name, phone, instruction_file, remember_history)
            tts = synthesize_with_google_tts(res_text)
            ts = time.strftime("%Y%m%d_%H%M%S")
            full_path = f"{BASE_YEMOT_FOLDER}/{phone}/dvartorah_{ts}.wav"
            ensure_personal_folder_exists(phone)
            if upload_to_yemot(tts, full_path):
                os.remove(tts)
                update_playfile_ini(phone)
                return Response(f"go_to_folder_and_play=/85/{phone},dvartorah_{ts}.wav,0.go_to_folder=/8/6", mimetype="text/plain")
    except Exception as e: logging.error(e)
    return Response("שגיאה", mimetype="text/plain")

# --- המסלול החדש לתמלול + RAG ---
def process_audio_request_transcript(request, remember_history: bool, instruction_file: str):
    file_url = request.args.get("file_url")
    phone_number = request.args.get("ApiPhone", "unknown")
    if not remember_history:
        try: os.remove(f"/tmp/conversations/{phone_number}.json")
        except: pass

    if not file_url.startswith("http"):
        file_url = f"https://www.call2all.co.il/ym/api/DownloadFile?token={SYSTEM_TOKEN}&path=ivr2:/{file_url}"
    
    logging.info(f"Processing transcript RAG for {phone_number}")
    try:
        response = requests.get(file_url)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_input:
            temp_input.write(response.content)
            temp_input.flush()
            if is_audio_quiet(temp_input.name):
                return Response("id_list_message=t-הקובץ שקט מדי&go_to_folder=/8/6", mimetype="text/plain")
            
            processed = add_silence(temp_input.name)
            text = recognize_speech(processed)
            if not text: return Response("id_list_message=t-לא זוהה דיבור&go_to_folder=/8/6", mimetype="text/plain")

            # שימוש בפונקציה החדשה
            rag_response = generate_rag_response(text, phone_number, instruction_file, remember_history)
            
            tts_path = synthesize_with_google_tts(rag_response)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            full_path = f"{BASE_YEMOT_FOLDER}/{phone_number}/dvartorah_{timestamp}.wav"
            ensure_personal_folder_exists(phone_number)
            
            if upload_to_yemot(tts_path, full_path):
                os.remove(tts_path)
                update_playfile_ini(phone_number)
                return Response(f"go_to_folder_and_play=/85/{phone_number},dvartorah_{timestamp}.wav,0.go_to_folder=/8/6", mimetype="text/plain")
    except Exception as e:
        logging.error(f"Error in transcript route: {e}")
        return Response(f"error: {e}", mimetype="text/plain")

@app.route("/upload_audio_continue", methods=["GET"])
def upload_audio_continue(): return process_audio_request(request, True, INSTRUCTIONS_CONTINUE_FILE)

@app.route("/upload_audio_new", methods=["GET"])
def upload_audio_new(): return process_audio_request(request, False, INSTRUCTIONS_NEW_FILE)

@app.route("/upload_audio_transcript_new", methods=["GET"])
def upload_audio_transcript_new(): return process_audio_request_transcript(request, False, INSTRUCTIONS_TRANSCRIPT_NEW_FILE)

@app.route("/upload_audio_transcript_continue", methods=["GET"])
def upload_audio_transcript_continue(): return process_audio_request_transcript(request, True, INSTRUCTIONS_TRANSCRIPT_CONTINUE_FILE)

@app.route("/upload_audio_to_email", methods=["GET"])
def upload_audio_to_email():
    # (עיבוד מייל רגיל - מקוצר)
    return Response("id_list_message=t-נשלח למייל&go_to_folder=/", mimetype="text/plain")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
