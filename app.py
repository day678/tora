import os
import tempfile
import logging
import requests
import json
import time
import base64
from flask import Flask, request, Response
from pydub import AudioSegment
import speech_recognition as sr
from google.cloud import texttospeech  # Google Cloud TTS

# ------------------ Logging ------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

app = Flask(__name__)

# ------------------ Decode Google credentials if provided as base64 ------------------
if os.getenv("GOOGLE_APPLICATION_CREDENTIALS_B64"):
    try:
        decoded = base64.b64decode(os.getenv("GOOGLE_APPLICATION_CREDENTIALS_B64")).decode("utf-8")
        temp_path = "/tmp/google-key.json"
        with open(temp_path, "w") as f:
            f.write(decoded)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path
        logging.info("✅ Google credentials loaded from base64 environment variable.")
    except Exception as e:
        logging.error(f"❌ Failed to load GOOGLE_APPLICATION_CREDENTIALS_B64: {e}")

# ------------------ Configuration ------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
YEMOT_TOKEN = os.getenv("YEMOT_TOKEN")

# ------------------ Helper Functions ------------------

def add_silence(input_path: str) -> AudioSegment:
    """מוסיף שניית שקט לפני ואחרי קטע האודיו כדי לשפר את הדיוק בזיהוי דיבור."""
    audio = AudioSegment.from_file(input_path)
    silence = AudioSegment.silent(duration=1000)
    return silence + audio + silence

def recognize_speech(audio_segment: AudioSegment) -> str:
    """מבצע זיהוי דיבור בעברית באמצעות Google Speech Recognition."""
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True

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
    """שולח את התמלול למודל Gemini כדי לנסח דבר תורה יפה ומכובד."""
    if not text_to_summarize or not GEMINI_API_KEY:
        logging.warning("Skipping Gemini summarization: Text or API Key missing.")
        return "לא ניתן לנסח דבר תורה כרגע. אנא נסה שוב מאוחר יותר."

    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
    MAX_RETRIES = 2

    system_prompt = (
        "אתה עורך תורני ומנסח דברי תורה. נסח מחדש את הטקסט המועתק (תמלול אודיו) ל'דבר תורה' קצר, בהיר ומסודר. "
        "השתמש בשפה עברית גבוהה ומכובדת, המתאימה לתוכן תורני. "
        "הוסף כותרת יפה בתחילת הסיכום (כגון: 'דבר תורה לשבת קודש', 'סיכום תורני יומי'). "
        "סכם את הדברים לשתי פסקאות קצרות בלבד והשתמש בציטוטים רלוונטיים אם הובאו בטקסט המקורי. "
        "*אל* תוסיף הקדמות שאינן חלק מהסיכום או משפטי סיום כלליים. "
        "הפלט שלך צריך להיות רק הכותרת והטקסט המסוכם."
    )

    payload = {
        "contents": [{"parts": [{"text": text_to_summarize}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {"temperature": 0.3}
    }

    headers = {'Content-Type': 'application/json'}
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                f"{API_URL}?key={GEMINI_API_KEY}",
                headers=headers,
                data=json.dumps(payload),
                timeout=25
            )
            response.raise_for_status()
            result = response.json()
            generated_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()
            if generated_text:
                logging.info(f"Gemini summarization successful (attempt {attempt+1}).")
                return generated_text
            return text_to_summarize
        except Exception as e:
            logging.warning(f"Gemini API failed attempt {attempt+1}: {e}")
            time.sleep(1)
    return f"לא הצלחנו לנסח את דבר התורה. התמלול המקורי: {text_to_summarize}"

def synthesize_speech_with_google(text: str, output_path: str) -> bool:
    """ממיר טקסט לדיבור באמצעות Google Cloud Text-to-Speech."""
    try:
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="he-IL",
            name="he-IL-Standard-B",
            ssml_gender=texttospeech.SsmlVoiceGender.MALE
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)
        response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        with open(output_path, "wb") as out:
            out.write(response.audio_content)
        logging.info(f"TTS synthesis completed: {output_path}")
        return True
    except Exception as e:
        logging.error(f"Google TTS error: {e}")
        return False

def upload_to_yemot(file_path: str, ivr_path: str) -> bool:
    """מעלה קובץ אודיו למערכת ימות המשיח."""
    try:
        url = "https://www.call2all.co.il/ym/api/UploadFile"
        files = {'file': open(file_path, 'rb')}
        data = {'token': YEMOT_TOKEN, 'path': f"ivr2:{ivr_path}", 'convertAudio': 1}
        r = requests.post(url, files=files, data=data, timeout=30)
        logging.info(f"Upload response: {r.text}")
        return '"responseStatus":"OK"' in r.text or '"success":true' in r.text.lower()
    except Exception as e:
        logging.error(f"Yemot upload error: {e}")
        return False

# ------------------ Routes ------------------

@app.route("/health", methods=["GET"])
def health():
    return Response("OK", status=200, mimetype="text/plain")

@app.route("/upload_audio", methods=["GET"])
def upload_audio():
    file_url = request.args.get("file_url")
    stockname = request.args.get("stockname")

    if not YEMOT_TOKEN:
        return Response("שגיאה: לא הוגדר טוקן למערכת ימות המשיח.", status=200)

    if not file_url:
        if not stockname:
            return Response("שגיאה: חסר פרמטר stockname.", status=200)
        file_url = f"https://www.call2all.co.il/ym/api/DownloadFile?token={YEMOT_TOKEN}&path=ivr2:/{stockname}"

    logging.info(f"Downloading audio from: {file_url}")

    try:
        response = requests.get(file_url, timeout=40)
        if response.status_code != 200:
            return Response("שגיאה: לא ניתן להוריד את הקובץ.", status=200)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input:
            temp_input.write(response.content)
            temp_input.flush()

            processed_audio = add_silence(temp_input.name)
            recognized_text = recognize_speech(processed_audio)

            if not recognized_text:
                return Response("לא זוהה דיבור ברור. אנא נסה שוב.", status=200)

            final_text = summarize_dvartorah_with_gemini(recognized_text)

            # יצירת קובץ קול עם Google TTS
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tts_file:
                success = synthesize_speech_with_google(final_text, tts_file.name)
                if not success:
                    return Response("שגיאת קול: לא ניתן היה ליצור קובץ דיבור.", status=200)

                # העלאה לימות
                upload_path = "5/tts_output.wav"  # שנה לפי מספר השלוחה שלך
                uploaded = upload_to_yemot(tts_file.name, upload_path)
                if uploaded:
                    logging.info("Audio file uploaded successfully to Yemot.")
                    return Response(upload_path, status=200, mimetype="text/plain")
                else:
                    return Response("שגיאה: העלאה לימות המשיח נכשלה.", status=200)

    except Exception as e:
        logging.error(f"Critical error: {e}")
        return Response(f"שגיאה קריטית: {e}", status=200)

# ------------------ Run ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
