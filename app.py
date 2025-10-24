import os
import tempfile
import base64
import json
import logging
import time
import requests
import re # ייבוא לצורך ניקוי טקסט
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

SYSTEM_TOKEN = "0733181406:80809090"
BASE_YEMOT_FOLDER = "ivr2:/85"  # שלוחה ראשית לכל הקבצים

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

def add_silence(input_path: str) -> AudioSegment:
    """מוסיף שניית שקט לפני ואחרי קטע האודיו כדי לשפר את הדיוק בזיהוי דיבור."""
    audio = AudioSegment.from_file(input_path, format="wav")
    silence = AudioSegment.silent(duration=1000)
    return silence + audio + silence

def recognize_speech(audio_segment: AudioSegment) -> str:
    """ממיר אודיו לטקסט בעברית בעזרת Google Speech Recognition."""
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

def summarize_dvartorah_with_gemini(text_to_summarize: str, phone_number: str) -> str:
    """מסכם דבר תורה עם זיכרון עד 20 הודעות אחרונות לפי מספר הטלפון של המאזין."""
    if not text_to_summarize or not GEMINI_API_KEY:
        logging.warning("Skipping Gemini summarization: Missing text or API key.")
        return "שגיאה: לא ניתן לנסח דבר תורה."

    os.makedirs("/tmp/conversations", exist_ok=True)
    history_path = f"/tmp/conversations/{phone_number}.json"

    history = {"messages": [], "last_updated": time.time()}
    if os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            pass

    # ניקוי היסטוריה ישנה (מעל 24 שעות)
    if time.time() - history.get("last_updated", 0) > 48 * 3600:
        history = {"messages": [], "last_updated": time.time()}

    # הוספת ההודעה החדשה
    history["messages"].append(text_to_summarize)
    history["last_updated"] = time.time()

    # שמירה של עד 20 הודעות אחרונות בלבד
    history["messages"] = history["messages"][-20:]

    # בניית prompt חכם
    context_text = "\n---\n".join(history["messages"])
    prompt = (
        "אתה עורך תורני שמסכם דברי תורה שנאמרו בהמשכים על ידי אותו אדם. "
        "לפניך רשימה של עד 20 קטעים שנאמרו בעבר. "
        "אם ההודעה החדשה ממשיכה את אותו נושא – שלב אותה יחד עם הקודמות בסיכום אחד זורם. "
        "אם נראה שהיא נושא חדש לגמרי – התחל דבר תורה חדש נפרד, בלי קשר לטקסטים הישנים. "
        "נסח מחדש את הטקסט המועתק ל'דבר תורה' קצר, ברור ומכובד. "
        "אין להשתמש כלל בסימני * או אימוג'ים וכדומה, בפלט לא יהיה שום כוכביות. "
        "בתחילת הסיכום תגיד בנוסח שלך משהו כמו שהדברים שנאמרו נפלאים ואתה מסכם אותם, ואז תסכם בקצרה. "
        "אם זה לא דבר תורה, אמור רק שאתה לא יכול לסכם נושאים שאינם דברי תורה."
        f"דברי תורה שנאמרו עד כה:\n{context_text}"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3}
    }

    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"

    try:
        response = requests.post(f"{API_URL}?key={GEMINI_API_KEY}", json=payload, timeout=35)
        response.raise_for_status()
        data = response.json()
        result = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()

        if result:
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            return result
        else:
            return text_to_summarize
    except Exception as e:
        logging.error(f"Gemini API error: {e}")
        return text_to_summarize

def synthesize_with_google_tts(text: str) -> str:
    """
    ממיר טקסט לאודיו (WAV) בעזרת Google Cloud Text-to-Speech.
    כולל ניקוי של כוכביות ותווים לא רצויים לפני ההמרה.
    """
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        raise EnvironmentError("Google Cloud credentials not configured for TTS.")

    # --- הוספת ניקוי טקסט קריטי ---
    # 1. הסרת כוכביות (Asterisks)
    cleaned_text = text.replace('*', '') 
    # 2. הסרת ניקוד (כדי למנוע שגיאות הקראה)
    # הסיבה: מנועי TTS לפעמים מקריאים ניקוד כתווים מילוליים.
    cleaned_text = re.sub(r'[\u0591-\u05BD\u05BF-\u05C7]', '', cleaned_text)
    # 3. הסרת אימוג'ים ותווים שאינם אלפביתיים-נומריים 
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    cleaned_text = emoji_pattern.sub(r'', cleaned_text)
    # -------------------------------

    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=cleaned_text) # שימוש בטקסט הנקי
    
    voice = texttospeech.VoiceSelectionParams(
        language_code="he-IL",
        name="he-IL-Wavenet-B"
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        speaking_rate=1.15,
        pitch=2.0
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    with open(output_path, "wb") as out:
        out.write(response.audio_content)
    logging.info(f"✅ Google TTS file created: {output_path}")
    return output_path

def upload_to_yemot(audio_path: str, yemot_full_path: str):
    """מעלה קובץ לשלוחה בימות המשיח."""
    url = "https://www.call2all.co.il/ym/api/UploadFile"
    
    # 1. מנקים את קידומת ה-ivr2:
    path_with_file_name = yemot_full_path.replace('ivr2:', '') 
    
    # 2. מחלצים את נתיב התיקייה בלבד (לדוגמה: /85)
    path_only = os.path.dirname(path_with_file_name).strip('/')
    
    # 3. שם הקובץ המלא
    file_name = os.path.basename(yemot_full_path)

    with open(audio_path, "rb") as f:
        files = {"file": (file_name, f, "audio/wav")}
        params = {
            "token": SYSTEM_TOKEN,
            "path": path_only,  
            "file_name": file_name,
            "convertAudio": 1
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
    """
    Health Check / Keep-Alive Endpoint.
    מחזיר OK כדי למנוע את הכיבוי האוטומטי של Render (Idle Timeout).
    """
    return Response("OK", status=200, mimetype="text/plain")

@app.route("/upload_audio", methods=["GET"])
def upload_audio():
    file_url = request.args.get("file_url")
    call_id = request.args.get("ApiCallId", str(int(time.time())))
    phone_number = request.args.get("ApiPhone", "unknown")

    # בניית URL מלא לקובץ
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

            # ניסוח דבר תורה עם הקשר קיים
            final_dvartorah = summarize_dvartorah_with_gemini(recognized_text, phone_number)

            # המרת הטקסט לשמע
            tts_path = synthesize_with_google_tts(final_dvartorah)

            # --- התיקון הקריטי להעלאה לתיקייה אחת עם שם ייחודי ---
            
            # 1. יצירת שם קובץ ייחודי
            FILE_NAME_WITH_ID = f"dvartorah_{call_id}.wav"
            
            # 2. בניית הנתיב המלא: ivr2:/85/dvartorah_<ApiCallId>.wav
            yemot_full_path = f"{BASE_YEMOT_FOLDER}/{FILE_NAME_WITH_ID}" 
            
            upload_success = upload_to_yemot(tts_path, yemot_full_path)
            os.remove(tts_path)

            if upload_success:
                # 3. החזרת פקודת השמעת קובץ ספציפי
                
                # שם התיקייה בלבד: /85
                folder_to_play_from = BASE_YEMOT_FOLDER.replace('ivr2:', '')
                
                # הפקודה: go_to_folder_and_play=/85,dvartorah_<ApiCallId>.wav,0
                playback_command = f"go_to_folder_and_play={folder_to_play_from},{FILE_NAME_WITH_ID},0"
                
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
