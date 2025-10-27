import os
import tempfile
import base64
import json
import logging
import time
import requests
import threading
import re
from flask import Flask, request, Response
from pydub import AudioSegment
# הוספת ייבוא של ספריית ה-Speech הראשית של גוגל
from google.cloud import texttospeech, speech
# הסרת הייבוא של speech_recognition as sr

# ------------------ Configuration ------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_CREDENTIALS_B64 = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_B64")

SYSTEM_TOKEN = "0733181406:80809090"
BASE_YEMOT_FOLDER = "ivr2:/85"  # שלוחה ראשית לכל הקבצים

INSTRUCTIONS_CONTINUE_FILE = "instructions_continue.txt"
INSTRUCTIONS_NEW_FILE = "instructions_new.txt"

VOWELIZED_LEXICON_FILE = "vowelized_lexicon.txt"
VOWELIZED_LEXICON = {}

# הוספת הגדרות לקובץ הביטויים (STT)
STT_PHRASES_FILE = "stt_phrases.txt"
STT_PHRASES = []

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
    logging.warning("⚠️ GOOGLE_APPLICATION_CREDENTIALS_B64 not found. TTS/STT will fail.")


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

# פונקציה חדשה לטעינת ביטויי התמלול
def load_stt_phrases():
    """טוען את רשימת הביטויים לשיפור התמלול (עבור STT)."""
    global STT_PHRASES
    try:
        with open(STT_PHRASES_FILE, "r", encoding="utf-8") as f:
            STT_PHRASES = [line.strip() for line in f if line.strip()]
        logging.info(f"✅ Loaded {len(STT_PHRASES)} phrases for STT adaptation.")
    except FileNotFoundError:
        logging.warning(f"⚠️ STT Phrases file {STT_PHRASES_FILE} not found. Running STT without custom phrases.")
    except Exception as e:
        logging.error(f"❌ Error loading STT phrases: {e}")

# טעינת הלקסיקונים בעת עליית השרת
load_vowelized_lexicon()
load_stt_phrases()


def add_silence(input_path: str) -> AudioSegment:
    audio = AudioSegment.from_file(input_path, format="wav")
    silence = AudioSegment.silent(duration=1000)
    return silence + audio + silence


# ----------------------------------------------------
# ⬇️ פונקציית התמלול המתוקנת ⬇️
# ----------------------------------------------------
def recognize_speech(audio_segment: AudioSegment) -> str:
    """
    ממיר אודיו לטקסט בעזרת Google Cloud Speech-to-Text API,
    תוך שימוש ב-Model Adaptation (PhraseSet) לחיזוק זיהוי מונחים תורניים.
    """
    
    # 1. יצירת לקוח
    try:
        client = speech.SpeechClient()
    except Exception as e:
        logging.error(f"❌ Failed to initialize SpeechClient: {e}")
        return ""

    # 2. הכנת נתוני האודיו
    try:
        # שמירה זמנית של האודיו כ-WAV כדי לקבל את ה-bytes בפורמט הנכון.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
            audio_segment.export(temp_wav.name, format="wav")
            # קריאת התוכן של הקובץ הזמני
            with open(temp_wav.name, "rb") as f:
                 audio_bytes = f.read()

        audio_content = speech.RecognitionAudio(content=audio_bytes)
    except Exception as e:
        logging.error(f"❌ Error preparing audio content: {e}")
        return ""

    # 3. הגדרת התאמת מודל (Model Adaptation)
    adaptation = None
    if STT_PHRASES:
        # ⬇️ --- התיקון --- ⬇️
        # המרת רשימת המחרוזות (str) לאובייקטים מסוג (Phrase)
        phrase_objects = [speech.PhraseSet.Phrase(value=phrase) for phrase in STT_PHRASES]
        
        # יצירת PhraseSet עם רשימת האובייקטים המתוקנת
        phrase_set = speech.PhraseSet(phrases=phrase_objects, boost=15.0) 
        # ⬆️ --- סוף התיקון --- ⬆️
        
        adaptation = speech.SpeechAdaptation(phrase_sets=[phrase_set])

    # 4. הגדרת התצורה לבקשה
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000, 
        language_code="he-IL",
        model="command_and_search", # מודל מותאם לדיבור ממוקד
        adaptation=adaptation if adaptation else None 
    )

    # 5. שליחת הבקשה
    try:
        response = client.recognize(config=config, audio=audio_content)
        
        # 6. עיבוד התוצאה
        if response.results and response.results[0].alternatives:
            text = response.results[0].alternatives[0].transcript
            logging.info(f"✅ Recognized text (STT): {text}")
            return text
        
        logging.info("⚠️ No clear speech recognized (STT).")
        return ""
    
    except Exception as e:
        # כאן השגיאה שראית בלוג תירשם
        logging.error(f"❌ Speech recognition API error: {e}")
        return ""
# ----------------------------------------------------
# ⬆️ סוף הפונקציה המתוקנת ⬆️
# ----------------------------------------------------


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
        if time.time() - history.get("last_updated", 0) > 1 * 3600: # נשאר עם שעה אחת, כפי שבקוד שלך
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
        "generationConfig": {"temperature": 0.6, "max_output_tokens": 1300}
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
playfile_control_play_goto=/8/6/1
playfile_end_goto=/8/6/11
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


# ------------------ Routes ------------------

@app.route("/health", methods=["GET"])
def health():
    return Response("OK", status=200, mimetype="text/plain")


def process_audio_request(request, remember_history: bool, instruction_file: str):
    file_url = request.args.get("file_url")
    call_id = request.args.get("ApiCallId", str(int(time.time())))
    phone_number = request.args.get("ApiPhone", "unknown")

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
            
            # כאן פועלת הפונקציה המתוקנת
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
        # החזרת השגיאה גם לימות המשיח כדי שתדע שיש בעיה
        return Response(f"שגיאה קריטית בעיבוד: {e}", mimetype="text/plain")


@app.route("/upload_audio_continue", methods=["GET"])
def upload_audio_continue():
    return process_audio_request(request, remember_history=True, instruction_file=INSTRUCTIONS_CONTINUE_FILE)


@app.route("/upload_audio_new", methods=["GET"])
def upload_audio_new():
    return process_audio_request(request, remember_history=False, instruction_file=INSTRUCTIONS_NEW_FILE)


# ------------------ Run ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
```eof
