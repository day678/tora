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
from flask import Flask, request, Response
from pydub import AudioSegment
import speech_recognition as sr
from google.cloud import texttospeech

# ------------------ Configuration ------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_CREDENTIALS_B64 = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_B64")

SYSTEM_TOKEN = "0733183465:808090"
BASE_YEMOT_FOLDER = "ivr2:/85"  # שלוחה ראשית לכל הקבצים

INSTRUCTIONS_CONTINUE_FILE = "instructions_continue.txt"
INSTRUCTIONS_NEW_FILE = "instructions_new.txt"

VOWELIZED_LEXICON_FILE = "vowelized_lexicon.txt"
VOWELIZED_LEXICON = {}

# --- הגדרות חדשות לשליחת מייל (Brevo) ---
BREVO_API_KEY = os.getenv("BREVO_API_KEY") # מפתח API חדש
# שימוש בכתובת המאומתת שלך (EMAIL_USER) חייב להישאר, אחרת Brevo יחסום
EMAIL_USER = os.getenv("EMAIL_USER") 
DEFAULT_EMAIL_RECEIVER = os.getenv("DEFAULT_EMAIL_RECEIVER") 
# שינוי כאן: השם שיוצג לנמען
EMAIL_SENDER_NAME = "שירות סיכומי שיחות" 

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
        return "אתה עורך תורני המסכם דברי תורה בקצרה ובבהירות."


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


def summarize_with_gemini(text_to_summarize: str, phone_number: str, instruction_file: str, remember_history: bool) -> str:
    if not text_to_summarize or not GEMINI_API_KEY:
        logging.warning("Skipping Gemini summarization: Missing text or API key.")
        return "שגיאה: לא ניתן לנסח דבר תורה."

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
        history["messages"].append(text_to_summarize)
        history["messages"] = history["messages"][-20:]
        history["last_updated"] = time.time()
        context_text = "\n---\n".join(history["messages"])
    else:
        history = {"messages": [text_to_summarize], "last_updated": time.time()}
        context_text = text_to_summarize

    prompt = f"{instruction_text}\n\nדברי התורה שנאמרו:\n{context_text}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.6, "max_output_tokens": 2900}
    }

    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
    last_error = None
    for attempt in range(2):
        try:
            response = requests.post(f"{API_URL}?key={GEMINI_API_KEY}", json=payload, timeout=35)
            response.raise_for_status()
            data = response.json()
            result = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
            if remember_history:
                with open(history_path, "w", encoding="utf-8") as f:
                    json.dump(history, f, ensure_ascii=False, indent=2)
            if result:
                return result
        except Exception as e:
            logging.error(f"Gemini API error (attempt {attempt+1}): {e}")
            last_error = e
            time.sleep(1)
    return text_to_summarize if last_error else "שגיאה לא צפויה."


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


# ✅ פונקציה חדשה לווידוא יצירת תיקייה אישית מוגדרת כהשמעת קבצים
def ensure_personal_folder_exists(phone_number: str):
    """מוודא שתיקייה אישית קיימת ובעלת הגדרות השמעת קבצים."""
    folder_path = f"{BASE_YEMOT_FOLDER}/{phone_number}"
    url_check = "https://www.call2all.co.il/ym/api/GetFiles"
    url_upload = "https://www.call2all.co.il/ym/api/UploadFile"

    # בדיקה אם קיימת
    try:
        response = requests.get(url_check, params={"token": SYSTEM_TOKEN, "path": folder_path})
        data = response.json()
        if data.get("responseStatus") == "OK":
            logging.info(f"📁 Personal folder {folder_path} already exists.")
            return
    except Exception as e:
        logging.warning(f"⚠️ Could not verify if folder exists: {e}")

    # יצירה עם ext.ini
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

    time.sleep(0.5)  # אם באמת צריך השהייה
    
    try:
        response = requests.post(url_upload, params=params, files=files)
        data = response.json()
        if data.get("responseStatus") == "OK":
            logging.info(f"✅ Created and configured personal folder {folder_path}.")
        else:
            logging.warning(f"⚠️ Failed to create personal folder {folder_path}: {data}")
    except Exception as e:
        logging.error(f"❌ Error creating personal folder {folder_path}: {e}")


# --- פונקציית עזר חדשה לשליחת מייל (Brevo API) ---
def send_email(to_address: str, subject: str, body: str) -> bool:
    """שולח מייל עם התוכן הנתון באמצעות Brevo (Sendinblue) HTTP API."""
    
    BREVO_API_KEY = os.getenv("BREVO_API_KEY")
    
    if not all([BREVO_API_KEY, EMAIL_USER, to_address]):
        logging.error("❌ Brevo configuration is incomplete (API Key or EMAIL_USER missing).")
        return False
    
    # --- שינוי כאן: שימוש ב-EMAIL_SENDER_NAME בלבד בתצוגה ---
    # Brevo דורש את EMAIL_USER ב-JSON, אבל ניתן לשחק עם השם המוצג.
    logging.info(f"Sending email via Brevo API to {to_address} from {EMAIL_USER}")

    try:
        # כתובת ה-API של Brevo
        api_url = "https://api.brevo.com/v3/smtp/email"
        
        # הרכבת ה-Payload (הנתונים הנשלחים)
        # שינוי כאן: אנו שולחים את body כ-htmlContent במקום textContent
        html_content = f"""
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ direction: rtl; font-family: Arial, sans-serif; line-height: 1.6; }}
                .container {{ direction: rtl; text-align: right; width: 100%; }}
                .content {{ margin-top: 20px; padding: 15px; border: 1px solid #ddd; background-color: #f9f9f9; }}
                h3 {{ border-bottom: 1px solid #eee; padding-bottom: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                {body.replace('\\n', '<br>')}
            </div>
        </body>
        </html>
        """

        payload = {
            # שימו לב: Brevo ישלח מ-EMAIL_USER, אבל יציג את ה-name
            "sender": {
                "email": EMAIL_USER,
                "name": EMAIL_SENDER_NAME 
            },
            "to": [
                {
                    "email": to_address
                }
            ],
            "subject": subject,
            "htmlContent": html_content # שימוש ב-HTML במקום טקסט רגיל
        }
        
        # הרכבת ה-Headers
        headers = {
            "accept": "application/json",
            "api-key": BREVO_API_KEY,
            "content-type": "application/json"
        }
        
        # ביצוע בקשת ה-HTTP POST
        response = requests.post(api_url, json=payload, headers=headers)
        
        data = response.json()

        if response.status_code == 201: # 201 Creado הוא סטטוס ההצלחה של Brevo
            logging.info(f"✅ Email sent successfully via Brevo API (Status: 201, MessageID: {data.get('messageId')})")
            return True
        else:
            logging.error(f"❌ Failed to send email via Brevo API (Status: {response.status_code})")
            logging.error(f"❌ Brevo Response: {data}")
            error_msg = data.get("message")
            logging.error(f"❌ Brevo Error Message: {error_msg}")
            return False
            
    except Exception as e:
        logging.error(f"❌ Failed to send email (Brevo General Exception): {e}") 
        return False


load_vowelized_lexicon()

# ------------------ Routes ------------------

@app.route("/health", methods=["GET"])
def health():
    return Response("OK", status=200, mimetype="text/plain")


def process_audio_request(request, remember_history: bool, instruction_file: str):
    file_url = request.args.get("file_url")
    call_id = request.args.get("ApiCallId", str(int(time.time())))
    phone_number = request.args.get("ApiPhone", "unknown")

    # ------------------ תוספת: מחיקת היסטוריה קודמת בנושא חדש ------------------
    if not remember_history:
        # הבקשה הגיעה מ /upload_audio_new
        # 1. נמחק את ההיסטוריה הישנה של מספר זה
        history_path = f"/tmp/conversations/{phone_number}.json"
        if os.path.exists(history_path):
            try:
                os.remove(history_path)
                logging.info(f"🗑️ נמחקה היסטוריה ישנה עבור {phone_number} (נושא חדש).")
            except Exception as e:
                logging.warning(f"⚠️ לא ניתן היה למחוק קובץ היסטוריה ישן {history_path}: {e}")
        
        # 2. נקבע שהשיחה הנוכחית כן תישמר כהתחלה של ההיסטוריה החדשה
        # לכן, אנו דורסים את המשתנה ל-True עבור המשך הריצה של פונקציה זו
        remember_history = True
    # ------------------ סוף התוספת ------------------

    if not file_url.startswith("http"):
        file_url = f"https://www.call2all.co.il/ym/api/DownloadFile?token={SYSTEM_TOKEN}&path=ivr2:/{file_url}"

    logging.info(f"Downloading audio from: {file_url}")
    try:
        response = requests.get(file_url, timeout=20)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_input:
            temp_input.write(response.content)
            temp_input.flush()
            processed_audio = add_silence(temp_input.name)
            recognized_text = recognize_speech(processed_audio)
            if not recognized_text:
                return Response("לא זוהה דיבור ברור. אנא נסה שוב.", mimetype="text/plain")

            gemini_result = {}
            def run_gemini():
                gemini_result["text"] = summarize_with_gemini(recognized_text, phone_number, instruction_file, remember_history)
            gemini_thread = threading.Thread(target=run_gemini)
            gemini_thread.start()
            gemini_thread.join()

            final_dvartorah = gemini_result.get("text", recognized_text)
            tts_path = synthesize_with_google_tts(final_dvartorah)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            personal_folder = f"{BASE_YEMOT_FOLDER}/{phone_number}"
            yemot_full_path = f"{personal_folder}/dvartorah_{timestamp}.wav"

            # 🟩 קריאה לפונקציה שמוודאת שהתיקייה האישית קיימת ומוגדרת להשמעת קבצים
            ensure_personal_folder_exists(phone_number)

            upload_success = upload_to_yemot(tts_path, yemot_full_path)
            os.remove(tts_path)

            if upload_success:
                playback_command = f"go_to_folder_and_play=/85/{phone_number},dvartorah_{timestamp}.wav,0.go_to_folder=/8/6"
                logging.info(f"Returning IVR command: {playback_command}")
                return Response(playback_command, mimetype="text/plain")
            else:
                return Response("שגיאה בהעלאת הקובץ לשרת.", mimetype="text/plain")
    except Exception as e:
        logging.error(f"Critical error: {e}")
        return Response(f"שגיאה קריטית בעיבוד: {e}", mimetype="text/plain")


@app.route("/upload_audio_continue", methods=["GET"])
def upload_audio_continue():
    return process_audio_request(request, remember_history=True, instruction_file=INSTRUCTIONS_CONTINUE_FILE)


@app.route("/upload_audio_new", methods=["GET"])
def upload_audio_new():
    return process_audio_request(request, remember_history=False, instruction_file=INSTRUCTIONS_NEW_FILE)


# --- פונקציה ו-route חדשים לשליחת מייל ---

def process_audio_for_email(request):
    """
    מבצע תמלול וסיכום, ושולח אותם במייל ללא הקראה.
    משתמש בהיסטוריה הקיימת לצורך הסיכום.
    """
    file_url = request.args.get("file_url")
    call_id = request.args.get("ApiCallId", str(int(time.time())))
    phone_number = request.args.get("ApiPhone", "unknown")
    # קבלת המייל מהפרמטרים של ימות, עם גיבוי למשתנה הסביבה
    email_to = request.args.get("ApiEmail", DEFAULT_EMAIL_RECEIVER)

    # --- תוספת: בדיקה למניעת קריסה ---
    # (זוהי התוספת מהפעם הקודמת, לוודא שה-ext.ini נכון)
    if not file_url:
        logging.error("❌ שגיאת הגדרה: פרמטר 'file_url' חסר.")
        logging.error("❌ יש לוודא שקובץ ext.ini בשלוחה בימות המשיח מכיל את השורה: api_000=file_url,,record,,,,,no")
        # החזרת הודעת שגיאה ברורה למאזין
        return Response("id_list_message=t-שגיאת הגדרה חמורה במערכת, הקלטה לא התקבלה. אנא פנה למנהל.go_to_folder=/8/6", mimetype="text/plain")
    # --- סוף התוספת ---

    if not email_to:
        logging.warning("⚠️ No email address provided (ApiEmail or DEFAULT_EMAIL_RECEIVER). Aborting email send.")
        return Response("id_list_message=t-שגיאה, לא הוגדרה כתובת מייל לשליחה.go_to_folder=/8/6", mimetype="text/plain")

    if not file_url.startswith("http"):
        file_url = f"https://www.call2all.co.il/ym/api/DownloadFile?token={SYSTEM_TOKEN}&path=ivr2:/{file_url}"

    logging.info(f"Downloading audio for email processing from: {file_url}")
    try:
        response = requests.get(file_url, timeout=20)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_input:
            temp_input.write(response.content)
            temp_input.flush()
            processed_audio = add_silence(temp_input.name)
            
            # 1. ביצוע תמלול (STT)
            recognized_text = recognize_speech(processed_audio)
            if not recognized_text:
                return Response("לא זוהה דיבור ברור. אנא נסה שוב.", mimetype="text/plain")

            # 2. ביצוע סיכום Gemini (תוך שימוש בהיסטוריה הקיימת)
            gemini_result = {}
            def run_gemini():
                # אנו משתמשים ב-remember_history=True כדי שהסיכום יכלול את השיחות הקודמות
                gemini_result["text"] = summarize_with_gemini(recognized_text, phone_number, INSTRUCTIONS_CONTINUE_FILE, remember_history=True)
            gemini_thread = threading.Thread(target=run_gemini)
            gemini_thread.start()
            gemini_thread.join()

            final_dvartorah_summary = gemini_result.get("text", "לא נוצר סיכום.")

            # 3. הכנת תוכן המייל
            subject = f"סיכום שיחה חדש מ: {phone_number}"
            body = f"""
שלום,

התקבל תמלול וסיכום משיחה נכנסת.

פרטי שיחה:
- מספר טלפון: {phone_number}
- מזהה שיחה: {call_id}

-----------------------------------
תמלול ההקלטה האחרונה:
-----------------------------------
{recognized_text}

-----------------------------------
סיכום מלא (כולל הקלטה זו):
-----------------------------------
{final_dvartorah_summary}

"""
            # 4. שליחת המייל
            email_success = send_email(email_to, subject, body)

            if email_success:
                logging.info(f"✅ Email sent. Returning success message to Yemot.")
                return Response("id_list_message=t-ההודעה נשלחה בהצלחה למייל.go_to_folder=/8/6", mimetype="text/plain")
            else:
                logging.error(f"❌ Email failed. Returning error message to Yemot.")
                return Response("id_list_message=t-שגיאה בשליחת המייל, אנא נסה שוב.go_to_folder=/8/6", mimetype="text/plain")

    except Exception as e:
        logging.error(f"Critical error in email processing: {e}")
        return Response(f"שגיאה קריטית בעיבוד למייל: {e}", mimetype="text/plain")

@app.route("/upload_audio_to_email", methods=["GET"])
def upload_audio_to_email():
    """
    כתובת חדשה שמקבלת הקלטה, מתמללת, מסכמת ושולחת במייל
    ללא הקראה או שמירה בימות.
    """
    return process_audio_for_email(request)


# ------------------ Run ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
