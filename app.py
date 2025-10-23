import os
import tempfile
import base64
import json
import logging
import time
import requests
from flask import Flask, request, Response
from pydub import AudioSegment
import speech_recognition as sr
from google.cloud import texttospeech

# ------------------ Logging ------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

app = Flask(__name__)

# ------------------ Configuration ------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_CREDENTIALS_B64 = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_B64")

# הגדרת נתיב העלאת הקובץ ונקודת מעבר לאחר ההשמעה
YEMOT_UPLOAD_FOLDER = "ivr2:/85" # לדוגמה: מעלה לתיקייה 85 בראשי (ivr2)
YEMOT_FILE_NAME = "dvartorah.wav"
# לאן לעבור אחרי ההשמעה? (חזרה לתפריט הראשי '/')
POST_PLAYBACK_GOTO = "/000" 

# יצירת קובץ JSON זמני עם המפתח של Google Cloud
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

def add_silence(input_path: str) -> AudioSegment:
    """מוסיף שניית שקט לפני ואחרי קטע האודיו כדי לשפר את הדיוק בזיהוי דיבור."""
    audio = AudioSegment.from_file(input_path, format="wav")
    silence = AudioSegment.silent(duration=1000)
    return silence + audio + silence

def recognize_speech(audio_segment: AudioSegment) -> str:
    """מבצע זיהוי דיבור בעברית באמצעות Google Speech Recognition."""
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

def summarize_dvartorah_with_gemini(text_to_summarize: str) -> str:
    """ניסוח דבר תורה יפה ומסודר עם Gemini (כולל פיסוק)."""
    if not text_to_summarize or not GEMINI_API_KEY:
        logging.warning("Skipping Gemini summarization: Missing text or API key.")
        return "שגיאה: לא ניתן לנסח דבר תורה."

    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
    # הנחיה המדגישה פיסוק מלא לשיפור איכות ה-TTS
    prompt = (
        "אתה עורך תורני ומנסח דברי תורה. נסח מחדש את הטקסט המועתק ל'דבר תורה' קצר, ברור ומכובד. "
        "אין להשתמש בסימני * או אימוג'ים וכדומה. "
        "בתחילת הסיכום יהיה טקסט: אני מסכם את מה שאמרת. ואז תסכם את מה שנכתב בתמלול בקצרה. אין להוסיף שום דבר משלך. "
        "הפלט צריך להיות רק הכותרת והטקסט."
        "אם הטקסט אינו דבר תורה - תאמר שאינך יכול לענות על שום שאלה או לדבר על נושאים אחרים, אתה יכול רק לסכם את דברי התורה הנאמרים."
    )

    payload = {
        "contents": [{"parts": [{"text": text_to_summarize}]}],
        "systemInstruction": {"parts": [{"text": prompt}]},
        "generationConfig": {"temperature": 0.3}
    }
    
    last_error_message = "שגיאה בניסוח."
    MAX_RETRIES = 2

    for attempt in range(MAX_RETRIES):
        try:
            if attempt > 0: time.sleep(1)
            response = requests.post(f"{API_URL}?key={GEMINI_API_KEY}", json=payload, timeout=25)
            response.raise_for_status()
            data = response.json()
            result = data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()
            return result or text_to_summarize
        except Exception as e:
            logging.error(f"Gemini API error (Attempt {attempt+1}): {e}")
            last_error_message = f"שגיאת AI: {str(e)}"
            if attempt == MAX_RETRIES - 1:
                break
    
    return f"כשל בניסוח דבר התורה. {last_error_message}. נשמיע את התמלול המקורי: {text_to_summarize}"


def synthesize_with_google_tts(text: str) -> str:
    """ממיר טקסט לקובץ שמע WAV (LINEAR16) בעזרת Google Cloud TTS."""
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        raise EnvironmentError("Google Cloud credentials not configured for TTS.")

    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # שימוש בקול Wavenet איכותי בעברית
    voice = texttospeech.VoiceSelectionParams(
        language_code="he-IL",
        name="he-IL-Wavenet-B"  # קול גברי טבעי בעברית
    )

    # פורמט אודיו WAV (LINEAR16)
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000 # קצב דגימה סטנדרטי למערכות טלפוניות
        speaking_rate=1.2
    )
    
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    with open(output_path, "wb") as out:
        out.write(response.audio_content)
    logging.info(f"✅ Google TTS file created: {output_path}")
    return output_path

def upload_to_yemot(audio_path: str, yemot_full_path: str):
    """מעלה קובץ לימות המשיח לשלוחה שנבחרה."""
    system_token = "0733181406:80809090"
    url = "https://www.call2all.co.il/ym/api/UploadFile"

    path_no_file = os.path.dirname(yemot_full_path)
    file_name = os.path.basename(yemot_full_path)

    with open(audio_path, "rb") as f:
        files = {"file": (file_name, f, 'audio/wav')}
        params = {
            "token": system_token,
            "path": path_no_file,
            "file_name": file_name, # הגדרת שם הקובץ במפורש
            "convertAudio": 1 # ממליץ להשאיר כדי שימות יוודא פורמט תקין
        }
        response = requests.post(url, params=params, files=files)
        data = response.json()
        if data.get("responseStatus") == "OK":
            logging.info(f"✅ File uploaded successfully to {yemot_full_path}")
            return True
        else:
            logging.error(f"❌ Upload failed: {data}")
            return False

# ------------------ Routes ------------------

@app.route("/health", methods=["GET"])
def health():
    return Response("OK", status=200, mimetype="text/plain")

@app.route("/upload_audio", methods=["GET"])
def upload_audio():
    file_url = request.args.get("file_url")
    system_token = "0733181406:80809090"

    # --- יצירת URL לקובץ המקור ---
    if not file_url:
        stockname = request.args.get("stockname")
        if stockname:
            file_url = f"https://www.call2all.co.il/ym/api/DownloadFile?token={system_token}&path=ivr2:/{stockname}"
        else:
            return Response("שגיאה: חסר קובץ להורדה.", mimetype="text/plain")

    if not file_url.startswith("http"):
        file_url = f"https://www.call2all.co.il/ym/api/DownloadFile?token={system_token}&path=ivr2:/{file_url}"

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
                return Response("לא זוהה דיבור ברור. נסה שוב.", mimetype="text/plain")

            # 1. ניסוח דבר תורה
            final_dvartorah = summarize_dvartorah_with_gemini(recognized_text)

            # 2. הפקת אודיו עם Google Cloud TTS
            tts_path = synthesize_with_google_tts(final_dvartorah)

            # 3. העלאה לימות המשיח
            yemot_full_path = f"{YEMOT_UPLOAD_FOLDER}/{YEMOT_FILE_NAME}"
            upload_success = upload_to_yemot(tts_path, yemot_full_path)
            
            # 4. מחיקת קובץ ה-TTS המקומי (ניקוי)
            os.remove(tts_path)

            if upload_success:
                # 5. החזרת פקודת השמעה ל-IVR (הפקודה חייבת להיות בטקסט פשוט)
                # go_to_folder_and_play=שלוחה,קובץ,0.
                # אנחנו משתמשים בנתיב היחסי של הקובץ (שלוחה 85), ומורים למערכת לעבור אח"כ ל-POST_PLAYBACK_GOTO
                
                # הפקודה המלאה תהיה: go_to_folder_and_play=/85,dvartorah.wav,0.
                # הערה: מכיוון שימות דורש go_to_folder_and_play=שלוחה,קובץ...
                # והשלוחה לאן לעבור אחרי ההשמעה מוגדרת ב-api_end_goto
                # אנחנו מחזירים רק את הפעולה הראשית go_to_folder_and_play
                
                # אם נרצה לעבור לשלוחה POST_PLAYBACK_GOTO לאחר ההשמעה, נשתמש בשרשור פעולות (אפשרות ב'):
                # אפשרות א': go_to_folder_and_play=/85,dvartorah.wav,0
                # אפשרות ב' (מומלץ): go_to_folder_and_play=/85,dvartorah.wav,0.go_to_folder=/ (במקרה של תשובת שרת שמכילה הגדרות)
                
                # מכיוון שהקוד שלנו מחזיר פקודה אחת, הפקודה go_to_folder_and_play תשמיע ותחזור ל-api_end_goto המוגדר.
                
                # אנחנו משתמשים בנתיב המלא של הקובץ כפי שהועלה: 85/dvartorah.wav
                # כדי להבטיח שהשלוחה תמצא את הקובץ בלי קשר לשלוחה שבה היא נמצאת
                
                # הפקודה הבאה תורה ל-IVR להשמיע את הקובץ הספציפי שהועלה
                playback_command = f"go_to_folder_and_play={YEMOT_UPLOAD_FOLDER.replace('ivr2:', '')},{YEMOT_FILE_NAME},0"
                
                logging.info(f"Returning IVR command: {playback_command}")
                return Response(playback_command, status=200, mimetype='text/plain')

            else:
                logging.error("Final upload failed. Returning error to IVR.")
                return Response("שגיאה חמורה: נכשל בתהליך העלאת קובץ האודיו.", status=200, mimetype='text/plain')

    except Exception as e:
        logging.error(f"Critical error: {e}")
        return Response(f"שגיאה קריטית בעיבוד: {e}", status=200, mimetype='text/plain')

# ------------------ Run ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
