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

# יצירת קובץ JSON זמני עם המפתח של Google Cloud
if GOOGLE_CREDENTIALS_B64:
    creds_json = base64.b64decode(GOOGLE_CREDENTIALS_B64).decode("utf-8")
    temp_cred_path = "/tmp/google_creds.json"
    with open(temp_cred_path, "w") as f:
        f.write(creds_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_cred_path
    logging.info("✅ Google Cloud credentials loaded successfully.")
else:
    logging.warning("⚠️ GOOGLE_APPLICATION_CREDENTIALS_B64 not found. TTS will fail if used.")

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
    """ניסוח דבר תורה יפה ומסודר עם Gemini."""
    if not text_to_summarize or not GEMINI_API_KEY:
        logging.warning("Skipping Gemini summarization: Missing text or API key.")
        return "שגיאה: לא ניתן לנסח דבר תורה."

    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
    prompt = (
        "אתה עורך תורני ומנסח דברי תורה. נסח מחדש את הטקסט המועתק ל'דבר תורה' קצר, ברור ומכובד. "
        "הוסף כותרת יפה בתחילת הסיכום, וסכם לשתי פסקאות קצרות בלבד. "
        "הפלט צריך להיות רק הכותרת והטקסט."
    )

    payload = {
        "contents": [{"parts": [{"text": text_to_summarize}]}],
        "systemInstruction": {"parts": [{"text": prompt}]},
        "generationConfig": {"temperature": 0.3}
    }

    try:
        response = requests.post(f"{API_URL}?key={GEMINI_API_KEY}", json=payload, timeout=25)
        response.raise_for_status()
        data = response.json()
        result = data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()
        return result or text_to_summarize
    except Exception as e:
        logging.error(f"Gemini API error: {e}")
        return text_to_summarize

def synthesize_with_google_tts(text: str) -> str:
    """ממיר טקסט לקובץ שמע בעזרת Google Cloud TTS."""
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="he-IL",
        name="he-IL-Wavenet-D"  # קול גברי טבעי בעברית
    )

    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    with open(output_path, "wb") as out:
        out.write(response.audio_content)
    logging.info(f"✅ Google TTS file created: {output_path}")
    return output_path

def upload_to_yemot(audio_path: str, yemot_path: str):
    """מעלה קובץ לימות המשיח לשלוחה שנבחרה."""
    system_token = "0733181406:80809090"  # החלף אם צריך
    url = "https://www.call2all.co.il/ym/api/UploadFile"

    with open(audio_path, "rb") as f:
        files = {"file": f}
        params = {
            "token": system_token,
            "path": yemot_path,
            "convertAudio": 1
        }
        response = requests.post(url, params=params, files=files)
        data = response.json()
        if data.get("responseStatus") == "OK":
            logging.info(f"✅ File uploaded to {yemot_path}")
        else:
            logging.error(f"❌ Upload failed: {data}")
    return yemot_path

# ------------------ Routes ------------------

@app.route("/health", methods=["GET"])
def health():
    return Response("OK", mimetype="text/plain")

@app.route("/upload_audio", methods=["GET"])
def upload_audio():
    file_url = request.args.get("file_url")
    system_token = "0733181406:80809090"

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

            # ניסוח דבר תורה
            final_dvartorah = summarize_dvartorah_with_gemini(recognized_text)

            # הפקת אודיו עם Google Cloud
            tts_path = synthesize_with_google_tts(final_dvartorah)

            # העלאה לימות המשיח לשלוחה 85
            yemot_path = upload_to_yemot(tts_path, "ivr2:85/dvartorah.wav")

            return Response(f"הודעה הוקלטה והועלתה בהצלחה לשלוחה 85.\n{yemot_path}", mimetype="text/plain")

    except Exception as e:
        logging.error(f"Critical error: {e}")
        return Response(f"שגיאה קריטית בעיבוד: {e}", mimetype="text/plain")

# ------------------ Run ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
