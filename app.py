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
import re

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

INSTRUCTIONS_CONTINUE_FILE = "instructions_continue.txt"
INSTRUCTIONS_NEW_FILE = "instructions_new.txt"

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


def load_instructions(file_path: str) -> str:
    """קורא את קובץ ההנחיות של ג'מיני."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        logging.warning(f"⚠️ Instruction file {file_path} not found or unreadable.")
        return "אתה עורך תורני המסכם דברי תורה בקצרה ובבהירות."


def clean_text_for_tts(text: str) -> str:
    """מסיר אימוג'ים, סימונים ואותיות לועזיות כדי למנוע הקראת תווים מיותרים."""
    text = re.sub(r'[A-Za-z*#@^_^~\[\]{}()<>+=_|\\\/]', '', text)
    text = re.sub(r'[^\w\s,.!?אבגדהוזחטיכלמנסעפצקרשתםןףךץ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def summarize_with_gemini(text_to_summarize: str, phone_number: str, instruction_file: str, remember_history: bool) -> str:
    """מסכם דבר תורה עם או בלי זיכרון."""
    if not text_to_summarize or not GEMINI_API_KEY:
        logging.warning("Skipping Gemini summarization: Missing text or API key.")
        return "שגיאה: לא ניתן לנסח דבר תורה."

    instruction_text = load_instructions(instruction_file)

    os.makedirs("/tmp/conversations", exist_ok=True)
    history_path = f"/tmp/conversations/{phone_number}.json"
    history = {"messages": [], "last_updated": time.time()}

    # אם צריך לזכור את ההיסטוריה
    if remember_history and os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            pass

        # ניקוי היסטוריה ישנה
        if time.time() - history.get("last_updated", 0) > 48 * 3600:
            history = {"messages": [], "last_updated": time.time()}

        history["messages"].append(text_to_summarize)
        history["messages"] = history["messages"][-20:]
        history["last_updated"] = time.time()
        context_text = "\n---\n".join(history["messages"])
    else:
        # ללא זיכרון
        history = {"messages": [text_to_summarize], "last_updated": time.time()}
        context_text = text_to_summarize

    prompt = f"{instruction_text}\n\nדברי התורה שנאמרו:\n{context_text}"

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

        if remember_history:
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

        return result or text_to_summarize
    except Exception as e:
        logging.error(f"Gemini API error: {e}")
        return text_to_summarize


def synthesize_with_google_tts(text: str) -> str:
    """ממיר טקסט לאודיו (WAV) בעזרת Google Cloud Text-to-Speech."""
    text = clean_text_for_tts(text)
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        raise EnvironmentError("Google Cloud credentials not configured for TTS.")
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="he-IL", name="he-IL-Wavenet-B")
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16, sample_rate_hertz=16000, speaking_rate=1.15, pitch=2.0)
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    with open(output_path, "wb") as out:
        out.write(response.audio_content)
    logging.info(f"✅ Google TTS file created: {output_path}")
    return output_path


def upload_to_yemot(audio_path: str, yemot_full_path: str):
    """מעלה קובץ לשלוחה בימות המשיח."""
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


# ------------------ Routes ------------------

@app.route("/health", methods=["GET"])
def health():
    return Response("OK", status=200, mimetype="text/plain")


def process_audio_request(request, remember_history: bool, instruction_file: str):
    """פונקציה משותפת לעיבוד האודיו (עם או בלי זיכרון)."""
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
            recognized_text = recognize_speech(processed_audio)
            if not recognized_text:
                return Response("לא זוהה דיבור ברור. אנא נסה שוב.", mimetype="text/plain")

            final_dvartorah = summarize_with_gemini(recognized_text, phone_number, instruction_file, remember_history)
            tts_path = synthesize_with_google_tts(final_dvartorah)

            yemot_full_path = f"{BASE_YEMOT_FOLDER}/dvartorah_{call_id}.wav"
            upload_success = upload_to_yemot(tts_path, yemot_full_path)
            os.remove(tts_path)

            if upload_success:
                playback_command = f"go_to_folder_and_play=/85,dvartorah_{call_id}.wav,0.go_to_folder=/000"
                logging.info(f"Returning IVR command: {playback_command}")
                return Response(playback_command, mimetype="text/plain")
            else:
                return Response("שגיאה בהעלאת הקובץ לשרת.", mimetype="text/plain")
    except Exception as e:
        logging.error(f"Critical error: {e}")
        return Response(f"שגיאה קריטית בעיבוד: {e}", mimetype="text/plain")


@app.route("/upload_audio_continue", methods=["GET"])
def upload_audio_continue():
    """פונקציה שממשיכה שיחה קודמת (שומרת זיכרון)."""
    return process_audio_request(request, remember_history=True, instruction_file=INSTRUCTIONS_CONTINUE_FILE)


@app.route("/upload_audio_new", methods=["GET"])
def upload_audio_new():
    """פונקציה שמתחילה נושא חדש (ללא זיכרון)."""
    return process_audio_request(request, remember_history=False, instruction_file=INSTRUCTIONS_NEW_FILE)


# ------------------ Run ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
