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
from google.cloud import texttospeech
from openai import OpenAI

# ------------------ Configuration ------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_CREDENTIALS_B64 = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_B64")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SYSTEM_TOKEN = "0733181406:80809090"
BASE_YEMOT_FOLDER = "ivr2:/85"

INSTRUCTIONS_CONTINUE_FILE = "instructions_continue.txt"
INSTRUCTIONS_NEW_FILE = "instructions_new.txt"

VOWELIZED_LEXICON_FILE = "vowelized_lexicon.txt"
CUSTOM_WORDS_FILE = "custom_words.json"

VOWELIZED_LEXICON = {}
CUSTOM_WORDS = {}

# ------------------ Logging ------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

app = Flask(__name__)

# ------------------ Google Credentials ------------------
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
    global VOWELIZED_LEXICON
    try:
        with open(VOWELIZED_LEXICON_FILE, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) == 2:
                    VOWELIZED_LEXICON[parts[0].strip()] = parts[1].strip()
        logging.info(f"✅ Loaded {len(VOWELIZED_LEXICON)} words into vowelized lexicon.")
    except FileNotFoundError:
        logging.warning(f"⚠️ Lexicon file {VOWELIZED_LEXICON_FILE} not found.")

def load_custom_words():
    global CUSTOM_WORDS
    try:
        with open(CUSTOM_WORDS_FILE, "r", encoding="utf-8") as f:
            CUSTOM_WORDS = json.load(f)
        logging.info(f"✅ Loaded {len(CUSTOM_WORDS)} custom words for Whisper correction.")
    except FileNotFoundError:
        logging.warning("⚠️ custom_words.json not found, running without custom replacements.")
    except Exception as e:
        logging.error(f"❌ Error loading custom words: {e}")

def add_silence(input_path: str) -> AudioSegment:
    audio = AudioSegment.from_file(input_path, format="wav")
    silence = AudioSegment.silent(duration=1000)
    return silence + audio + silence

# ✅ Whisper transcription (replaces recognize_speech)
def recognize_speech(audio_segment: AudioSegment) -> str:
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            audio_segment.export(temp_wav.name, format="wav")
            client = OpenAI(api_key=OPENAI_API_KEY)
            with open(temp_wav.name, "rb") as f:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language="he"
                )
            text = transcript.text.strip()
            os.remove(temp_wav.name)
            # החל מילים לפי המילון
            for wrong, correct in CUSTOM_WORDS.items():
                text = re.sub(r"\b" + re.escape(wrong) + r"\b", correct, text)
            logging.info(f"🎧 Whisper recognized: {text}")
            return text
    except Exception as e:
        logging.error(f"Speech recognition error: {e}")
        return ""

def load_instructions(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return "אתה עורך תורני המסכם דברי תורה בקצרה ובבהירות."

def clean_text_for_tts(text: str) -> str:
    text = re.sub(r"[A-Za-z*#@^_^~\[\]{}()<>+=_|\\\/]", "", text)
    text = re.sub(r"[^\w\s,.!?אבגדהוזחטיכלמנסעפצקרשתםןףךץ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def apply_vowelized_lexicon(text: str) -> str:
    if not VOWELIZED_LEXICON:
        return f"<speak lang='he-IL'>{text}</speak>"
    for plain, vow in VOWELIZED_LEXICON.items():
        text = re.sub(r"\b" + re.escape(plain) + r"\b", vow, text)
    return f"<speak lang='he-IL'>{text}</speak>"

def summarize_with_gemini(text_to_summarize: str, phone_number: str, instruction_file: str, remember_history: bool) -> str:
    if not text_to_summarize or not GEMINI_API_KEY:
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
        if time.time() - history.get("last_updated", 0) > 3600:
            history = {"messages": [], "last_updated": time.time()}
        history["messages"].append(text_to_summarize)
        history["messages"] = history["messages"][-20:]
        context_text = "\n---\n".join(history["messages"])
    else:
        context_text = text_to_summarize
    prompt = f"{instruction_text}\n\nדברי התורה שנאמרו:\n{context_text}"
    payload = {"contents": [{"parts": [{"text": prompt}]}],
               "generationConfig": {"temperature": 0.6, "max_output_tokens": 1300}}
    try:
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={GEMINI_API_KEY}",
            json=payload, timeout=35)
        resp.raise_for_status()
        data = resp.json()
        result = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        if remember_history:
            history["last_updated"] = time.time()
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        return result.strip() or text_to_summarize
    except Exception as e:
        logging.error(f"Gemini API error: {e}")
        return text_to_summarize

def synthesize_with_google_tts(text: str) -> str:
    cleaned_text = clean_text_for_tts(text)
    ssml_text = apply_vowelized_lexicon(cleaned_text)
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)
    voice = texttospeech.VoiceSelectionParams(language_code="he-IL", name="he-IL-Wavenet-B")
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                                            sample_rate_hertz=16000, speaking_rate=1.15, pitch=2.0)
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    with open(output_path, "wb") as out:
        out.write(response.audio_content)
    return output_path

def upload_to_yemot(audio_path: str, yemot_full_path: str):
    url = "https://www.call2all.co.il/ym/api/UploadFile"
    path_no_file = os.path.dirname(yemot_full_path)
    file_name = os.path.basename(yemot_full_path)
    with open(audio_path, "rb") as f:
        files = {"file": (file_name, f, "audio/wav")}
        params = {"token": SYSTEM_TOKEN, "path": f"{path_no_file}/{file_name}", "convertAudio": 1}
        r = requests.post(url, params=params, files=files)
        d = r.json()
        return d.get("responseStatus") == "OK"

def ensure_personal_folder_exists(phone_number: str):
    folder_path = f"{BASE_YEMOT_FOLDER}/{phone_number}"
    url_check = "https://www.call2all.co.il/ym/api/GetFiles"
    try:
        r = requests.get(url_check, params={"token": SYSTEM_TOKEN, "path": folder_path})
        d = r.json()
        if d.get("responseStatus") == "OK":
            return
    except:
        pass
    ini_content = """type=playfile
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
    files = {"file": ("ext.ini", ini_content.encode("utf-8"), "text/plain")}
    params = {"token": SYSTEM_TOKEN, "path": f"{folder_path}/ext.ini"}
    requests.post("https://www.call2all.co.il/ym/api/UploadFile", params=params, files=files)

load_vowelized_lexicon()
load_custom_words()

# ------------------ Flask Routes ------------------

@app.route("/health")
def health():
    return Response("OK", mimetype="text/plain")

def process_audio_request(request, remember_history, instruction_file):
    file_url = request.args.get("file_url")
    call_id = request.args.get("ApiCallId", str(int(time.time())))
    phone_number = request.args.get("ApiPhone", "unknown")
    if not file_url.startswith("http"):
        file_url = f"https://www.call2all.co.il/ym/api/DownloadFile?token={SYSTEM_TOKEN}&path=ivr2:/{file_url}"
    try:
        r = requests.get(file_url, timeout=20)
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_input:
            temp_input.write(r.content)
            temp_input.flush()
            processed_audio = add_silence(temp_input.name)
            recognized_text = recognize_speech(processed_audio)
            if not recognized_text:
                return Response("לא זוהה דיבור ברור. אנא נסה שוב.", mimetype="text/plain")
            gemini_result = {}
            def run_gemini():
                gemini_result["text"] = summarize_with_gemini(recognized_text, phone_number, instruction_file, remember_history)
            t = threading.Thread(target=run_gemini)
            t.start()
            t.join()
            final_text = gemini_result.get("text", recognized_text)
            tts_path = synthesize_with_google_tts(final_text)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            ensure_personal_folder_exists(phone_number)
            yemot_path = f"{BASE_YEMOT_FOLDER}/{phone_number}/dvartorah_{timestamp}.wav"
            upload_to_yemot(tts_path, yemot_path)
            os.remove(tts_path)
            cmd = f"go_to_folder_and_play=/85/{phone_number},dvartorah_{timestamp}.wav,0.go_to_folder=/8/6"
            return Response(cmd, mimetype="text/plain")
    except Exception as e:
        logging.error(f"Critical error: {e}")
        return Response(f"שגיאה בעיבוד: {e}", mimetype="text/plain")

@app.route("/upload_audio_continue")
def upload_audio_continue():
    return process_audio_request(request, remember_history=True, instruction_file=INSTRUCTIONS_CONTINUE_FILE)

@app.route("/upload_audio_new")
def upload_audio_new():
    return process_audio_request(request, remember_history=False, instruction_file=INSTRUCTIONS_NEW_FILE)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
