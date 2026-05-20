import os
import io
import re
import json
import time
import struct
import threading
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from urllib.parse import quote
from urllib.request import urlopen
import numpy as np
import subprocess
import tempfile
import pyaudio
import pvporcupine
import soxr
from openai import OpenAI
from elevenlabs import ElevenLabs
from duckduckgo_search import DDGS
from dotenv import load_dotenv

_ENV_PATH = os.path.expanduser("~/.env")
load_dotenv(dotenv_path=_ENV_PATH, override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
PORCUPINE_KEY = os.getenv("PORCUPINE_KEY")

# Personal context — set these in ~/.env (see jarvis_personal.env.example)
JARVIS_LOCATION = os.getenv("JARVIS_LOCATION", "").strip()
JARVIS_LAT = os.getenv("JARVIS_LAT", "").strip()
JARVIS_LON = os.getenv("JARVIS_LON", "").strip()
JARVIS_TIMEZONE = os.getenv("JARVIS_TIMEZONE", "").strip()
CALENDAR_ICS_URL = os.getenv("CALENDAR_ICS_URL", "").strip()

def _require_keys():
    missing = [
        name
        for name, val in (
            ("OPENAI_API_KEY", OPENAI_API_KEY),
            ("ELEVENLABS_API_KEY", ELEVENLABS_API_KEY),
            ("ELEVENLABS_VOICE_ID", ELEVENLABS_VOICE_ID),
            ("PORCUPINE_KEY", PORCUPINE_KEY),
        )
        if not val
    ]
    if missing:
        raise SystemExit(
            f"Missing in {_ENV_PATH}: {', '.join(missing)}. "
            "Create ~/.env with your API keys."
        )


_require_keys()
openai_client = OpenAI(api_key=OPENAI_API_KEY)
eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

MIC_SAMPLE_RATE = 44100
PLAYBACK_VOLUME = 0.5  # 50% — USB speaker has no ALSA mixer
MIC_NAME_HINTS = ("USB PnP", "C-Media", "PnP Sound", "PCM2902", "Texas")
MIC_EXCLUDE_HINTS = ("UACDemo", "Jieli", "vc4hdmi", "bcm2835", "Headphones")
SPEAKER_PREFERRED_HINTS = ("UACDemo", "Jieli")
SKIP_PLAYBACK_CARDS = ("vc4hdmi0", "vc4hdmi1")


def find_alsa_card(hints):
    """Return ALSA card number whose name matches any hint (from /proc/asound/cards)."""
    try:
        with open("/proc/asound/cards", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or not line[0].isdigit():
                    continue
                card = line.split()[0]
                if any(h in line for h in hints):
                    return card
    except OSError:
        pass
    return None


def find_speaker_alsa_card():
    """Prefer the UACDemo USB speaker; fallback to any USB playback, then 3.5mm jack."""
    try:
        cards_text = open("/proc/asound/cards", encoding="utf-8").read()
        usb_cards = []
        other_cards = []
        for raw in cards_text.splitlines():
            line = raw.strip()
            if not line or not line[0].isdigit():
                continue
            card = line.split()[0]
            card_id = line.split("[", 1)[1].split("]")[0] if "[" in line else ""
            if card_id in SKIP_PLAYBACK_CARDS:
                continue
            if not os.path.exists(f"/proc/asound/card{card}/pcm0p"):
                continue
            if any(h in line for h in SPEAKER_PREFERRED_HINTS):
                return card
            if "USB" in line:
                usb_cards.append(card)
            else:
                other_cards.append(card)
        if usb_cards:
            return usb_cards[-1]  # highest card # (UACDemo is usually after the mic)
        if other_cards:
            return other_cards[0]
    except OSError:
        pass
    return "0"


def find_mic():
    """USB mic only — never use the USB speaker for capture."""
    pa = pyaudio.PyAudio()
    try:
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] < 1:
                continue
            name = info["name"]
            if any(x in name for x in MIC_EXCLUDE_HINTS):
                continue
            if any(hint in name for hint in MIC_NAME_HINTS):
                ch = 2 if info["maxInputChannels"] >= 2 else 1
                return i, ch, name
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            name = info["name"]
            if info["maxInputChannels"] >= 1 and not any(
                x in name for x in MIC_EXCLUDE_HINTS
            ):
                ch = 2 if info["maxInputChannels"] >= 2 else 1
                return i, ch, name
    finally:
        pa.terminate()
    return 1, 1, "default"


def get_playback_rate(card):
    """USB speaker (UACDemo) is 48000 Hz; Pi jack is usually 44100/48000."""
    try:
        with open(f"/proc/asound/card{card}/stream0", encoding="utf-8") as f:
            text = f.read()
        if "Rates:" in text:
            rates = text.split("Rates:")[1].split("\n")[0]
            if "48000" in rates:
                return 48000
    except OSError:
        pass
    return 44100


def init_audio_devices():
    """Re-detect devices each run (USB must be plugged in before starting)."""
    global MIC_ALSA_CARD, SPEAKER_ALSA_CARD, PLAYBACK_DEVICE, PLAYBACK_RATE
    global MIC_DEVICE_INDEX, MIC_CHANNELS, MIC_DEVICE_NAME
    MIC_ALSA_CARD = find_alsa_card(MIC_NAME_HINTS) or "3"
    SPEAKER_ALSA_CARD = find_speaker_alsa_card()
    PLAYBACK_DEVICE = f"plughw:{SPEAKER_ALSA_CARD},0"
    PLAYBACK_RATE = get_playback_rate(SPEAKER_ALSA_CARD)
    MIC_DEVICE_INDEX, MIC_CHANNELS, MIC_DEVICE_NAME = find_mic()


init_audio_devices()


def pcm_to_mono(pcm_bytes, channels):
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
    if channels == 2:
        pcm = pcm.reshape(-1, 2).mean(axis=1).astype(np.int16)
    return pcm

conversation_history = []

_weather_cache = {"at": 0.0, "text": None}
_calendar_cache = {"at": 0.0, "text": None}
_coords_cache = {"lat": None, "lon": None, "label": None}
_WEATHER_TTL = 600
_CALENDAR_TTL = 300

WMO_WEATHER = {
    0: "clear",
    1: "mostly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "foggy",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    80: "rain showers",
    81: "rain showers",
    82: "heavy rain showers",
    95: "thunderstorm",
}

SYSTEM_PROMPT = """You are Jarvis, a witty and intelligent AI assistant inspired by Iron Man.
You are helpful, concise, and occasionally humorous. Keep responses short and conversational.

IMPORTANT: Your built-in knowledge is outdated. Never state current events, prices, weather,
sports, politics, product releases, or "who is X now" from memory — you will be wrong.

Each request includes [LIVE CONTEXT]: clock, user location, current weather, and calendar.
Use LIVE CONTEXT for time, date, weather, temperature, and schedule — never guess these.

If [WEB RESULTS] are included, base your answer ONLY on those results plus LIVE CONTEXT.
If the user needs up-to-date facts and there are no WEB RESULTS, reply with [SEARCH] plus a
short search query (include the current year when relevant).

If the user wants to set a timer, start with [TIMER] followed by the number of seconds.
If the user wants to shutdown the Pi, reply with exactly [SHUTDOWN].
If the user wants to change volume, start with [VOLUME] followed by a number 0-100.
Otherwise reply normally (jokes, how things work, coding help — no search needed)."""

# Questions that need the web, not GPT memory
_WEB_HINTS = re.compile(
    r"\b(weather|forecast|temperature|rain|snow)\b|"
    r"\b(news|headline|latest|recent|current|today|tonight|yesterday|tomorrow|now)\b|"
    r"\b(who won|score|standings|playoff|champion)\b|"
    r"\b(president|election|prime minister|ceo|stock|price|market)\b|"
    r"\b(20\d{2})\b|"  # e.g. 2024, 2025, 2026
    r"\b(release|update|version|announced|launched)\b|"
    r"\b(search|look up|google|find out)\b|"
    r"\b(happening|going on)\b",
    re.I,
)


def _http_get_json(url, timeout=12):
    with urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _local_tz():
    if JARVIS_TIMEZONE:
        try:
            return ZoneInfo(JARVIS_TIMEZONE)
        except Exception:
            pass
    return datetime.now().astimezone().tzinfo


def _resolve_coords():
    """Lat/lon from env or geocode JARVIS_LOCATION (cached)."""
    global _coords_cache
    if _coords_cache["lat"] is not None:
        return _coords_cache["lat"], _coords_cache["lon"], _coords_cache["label"]

    label = JARVIS_LOCATION or "your area"
    if JARVIS_LAT and JARVIS_LON:
        lat, lon = float(JARVIS_LAT), float(JARVIS_LON)
        _coords_cache.update({"lat": lat, "lon": lon, "label": label})
        return lat, lon, label

    if not JARVIS_LOCATION:
        return None, None, None

    try:
        url = (
            "https://geocoding-api.open-meteo.com/v1/search?name="
            + quote(JARVIS_LOCATION)
            + "&count=1"
        )
        data = _http_get_json(url)
        results = data.get("results") or []
        if not results:
            return None, None, None
        hit = results[0]
        lat, lon = hit["latitude"], hit["longitude"]
        label = hit.get("name", JARVIS_LOCATION)
        if hit.get("admin1"):
            label = f"{label}, {hit['admin1']}"
        _coords_cache.update({"lat": lat, "lon": lon, "label": label})
        return lat, lon, label
    except Exception as e:
        print(f"Geocoding error: {e}")
        return None, None, None


def get_weather_context():
    """Current temperature and conditions via Open-Meteo (free, no API key)."""
    now = time.time()
    if _weather_cache["text"] and now - _weather_cache["at"] < _WEATHER_TTL:
        return _weather_cache["text"]

    lat, lon, label = _resolve_coords()
    if lat is None:
        return (
            "Weather: not configured. Set JARVIS_LOCATION (city name) "
            "or JARVIS_LAT and JARVIS_LON in ~/.env."
        )

    try:
        q = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,apparent_temperature,weather_code,wind_speed_10m"
            "&temperature_unit=fahrenheit&wind_speed_unit=mph"
        )
        data = _http_get_json(q)
        cur = data["current"]
        temp = cur["temperature_2m"]
        feels = cur["apparent_temperature"]
        desc = WMO_WEATHER.get(int(cur["weather_code"]), "unknown")
        wind = cur.get("wind_speed_10m")
        text = (
            f"Weather in {label}: {temp:.0f}°F (feels like {feels:.0f}°F), {desc}."
        )
        if wind is not None:
            text += f" Wind {wind:.0f} mph."
        _weather_cache.update({"at": now, "text": text})
        return text
    except Exception as e:
        print(f"Weather error: {e}")
        return "Weather: could not fetch current conditions."


def _parse_ics_datetime(raw, tz):
    raw = (raw or "").strip()
    if not raw:
        return None
    if len(raw) == 8:
        return datetime.strptime(raw, "%Y%m%d").replace(tzinfo=tz)
    clean = raw.replace("Z", "+00:00")
    if "T" not in clean:
        return None
    try:
        if "+" in clean[9:] or "-" in clean[9:]:
            return datetime.fromisoformat(clean).astimezone(tz)
        return datetime.strptime(clean[:15], "%Y%m%dT%H%M%S").replace(tzinfo=tz)
    except ValueError:
        return None


def get_calendar_context():
    """Upcoming events from a Google/Apple calendar ICS URL in ~/.env."""
    now = time.time()
    if _calendar_cache["text"] and now - _calendar_cache["at"] < _CALENDAR_TTL:
        return _calendar_cache["text"]

    if not CALENDAR_ICS_URL:
        return (
            "Calendar: not configured. Add CALENDAR_ICS_URL to ~/.env "
            "(Google Calendar → Settings → your calendar → Secret iCal URL)."
        )

    try:
        with urlopen(CALENDAR_ICS_URL, timeout=15) as resp:
            ics = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"Calendar fetch error: {e}")
        return "Calendar: could not fetch events."

    tz = _local_tz()
    now_dt = datetime.now(tz)
    end_dt = now_dt + timedelta(days=7)
    events = []

    for block in ics.split("BEGIN:VEVENT")[1:]:
        summary = ""
        dtstart = None
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("SUMMARY"):
                summary = line.split(":", 1)[-1].strip()
            elif line.startswith("DTSTART"):
                val = line.split(":", 1)[-1].strip()
                dtstart = _parse_ics_datetime(val, tz)

        if not dtstart or not summary:
            continue
        if dtstart < now_dt - timedelta(hours=1) or dtstart > end_dt:
            continue
        events.append((dtstart, summary))

    events.sort(key=lambda x: x[0])
    if not events:
        text = "Calendar: no upcoming events in the next 7 days."
    else:
        lines = ["Calendar (upcoming):"]
        for dt, title in events[:8]:
            lines.append(f"- {dt.strftime('%a %b %d %I:%M %p')}: {title}")
        text = "\n".join(lines)

    _calendar_cache.update({"at": now, "text": text})
    return text


def get_live_context():
    """Clock, location, weather, and calendar — injected into every GPT request."""
    tz = _local_tz()
    now = datetime.now(tz)
    tz_name = now.tzname() or "local time"
    parts = [
        f"Date: {now.strftime('%A, %B %d, %Y')}.",
        f"Time: {now.strftime('%I:%M %p')} ({now.strftime('%H:%M:%S')} 24h).",
        f"Timezone: {tz_name}.",
    ]
    if JARVIS_LOCATION:
        parts.append(f"User location: {JARVIS_LOCATION}.")
    elif _coords_cache.get("label"):
        parts.append(f"User location: {_coords_cache['label']}.")

    parts.append(get_weather_context())
    parts.append(get_calendar_context())
    return " ".join(parts)


def try_local_answer(user_input):
    """Fast answers from Pi clock, weather API, and calendar (no GPT)."""
    text = user_input.lower().strip()
    now = datetime.now(_local_tz())

    if re.search(r"\b(what('s| is) the )?time\b|\btime is it\b|\bcurrent time\b", text):
        return f"It's {now.strftime('%I:%M %p')}, sir."

    if re.search(
        r"\b(what('s| is) the )?date\b|\btoday('s)? date\b|\bwhat day\b|\bday is it\b",
        text,
    ):
        return f"Today is {now.strftime('%A, %B %d, %Y')}, sir."

    if re.search(
        r"\b(temperature|temp|weather|forecast|how hot|how cold|degrees)\b",
        text,
    ):
        wx = get_weather_context()
        if wx.startswith("Weather: not configured"):
            return (
                "I don't have your location yet, sir. "
                "Add JARVIS_LOCATION to your .env file, for example your city name."
            )
        if wx.startswith("Weather: could not"):
            return "I couldn't reach the weather service, sir. Try again in a moment."
        # e.g. "Weather in Los Angeles: 72°F (feels like 70°F), clear."
        m = re.search(
            r"Weather in ([^:]+): ([\d.]+)°F \(feels like ([\d.]+)°F\), ([^.]+)",
            wx,
        )
        if m:
            place, temp, feels, cond = m.groups()
            return (
                f"In {place}, it's {float(temp):.0f} degrees Fahrenheit, "
                f"feels like {float(feels):.0f}, and {cond}, sir."
            )
        return wx.replace("Weather in ", "In ", 1) + " Sir."

    if re.search(
        r"\b(calendar|schedule|appointments?|meetings?|events?)\b|"
        r"\bwhat('s| is) on (my |the )?(calendar|schedule)\b|"
        r"\banything (on|for) today\b|"
        r"\bwhat do i have (today|tomorrow)\b",
        text,
    ):
        cal = get_calendar_context()
        if cal.startswith("Calendar: not configured"):
            return (
                "Your calendar isn't connected yet, sir. "
                "Add CALENDAR_ICS_URL to your .env file with your Google Calendar secret iCal link."
            )
        if cal.startswith("Calendar: could not"):
            return "I couldn't load your calendar, sir."
        if cal.startswith("Calendar: no upcoming"):
            return "You have nothing on the calendar for the next week, sir."
        # Speak first few events
        lines = [ln for ln in cal.splitlines() if ln.startswith("- ")][:4]
        if not lines:
            return "Your calendar looks clear, sir."
        spoken = "Here's what's coming up, sir. "
        spoken += ". ".join(ln[2:] for ln in lines)
        return spoken

    return None


def needs_web_search(user_input):
    """True when the question likely needs current info from the internet."""
    if try_local_answer(user_input):
        return False
    text = user_input.lower()
    # Weather/calendar handled locally when configured
    if re.search(
        r"\b(weather|temperature|temp|forecast|calendar|schedule|appointment)\b", text
    ):
        if JARVIS_LOCATION or (JARVIS_LAT and JARVIS_LON) or CALENDAR_ICS_URL:
            return False
    # Skip obvious non-factual chat
    if len(text.split()) <= 3 and not _WEB_HINTS.search(text):
        if text in ("hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye"):
            return False
    return bool(_WEB_HINTS.search(text))


def search_web(query):
    """DuckDuckGo search — returns formatted snippets for GPT."""
    year = datetime.now().year
    if str(year) not in query:
        query = f"{query} {year}"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
    except Exception as e:
        print(f"Web search error: {e}")
        return None
    if not results:
        return None
    parts = []
    for r in results[:4]:
        title = r.get("title", "")
        body = r.get("body", "")
        if title or body:
            parts.append(f"- {title}: {body}".strip())
    return "\n".join(parts)[:1200]


def answer_with_web(user_input):
    """Search the web first, then have GPT answer from those results."""
    print(f"Web search: {user_input}")
    results = search_web(user_input)
    if not results:
        return ask_gpt(
            user_input,
            extra_system="Web search returned nothing. Say you could not find current info.",
        )

    # Drop a pending [SEARCH] tag from a prior ask_gpt() turn
    if conversation_history and conversation_history[-1]["content"].startswith(
        "[SEARCH]"
    ):
        conversation_history.pop()

    web_block = f"\n\n[WEB RESULTS — treat as current facts]\n{results}"
    history_for_api = list(conversation_history)

    if history_for_api and history_for_api[-1]["role"] == "user":
        history_for_api[-1] = {
            "role": "user",
            "content": history_for_api[-1]["content"] + web_block,
        }
        append_user = False
    else:
        history_for_api.append(
            {"role": "user", "content": f"{user_input}{web_block}"}
        )
        conversation_history.append({"role": "user", "content": user_input})
        append_user = True

    system = f"{SYSTEM_PROMPT}\n\n[LIVE CONTEXT]\n{get_live_context()}"
    messages = [{"role": "system", "content": system}, *history_for_api]
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    reply = response.choices[0].message.content.strip()
    if append_user:
        pass  # user already appended above
    conversation_history.append({"role": "assistant", "content": reply})
    return reply


def set_speaker_volume():
    """Unmute speaker; use PLAYBACK_VOLUME in ffmpeg if no ALSA mixer."""
    pct = int(PLAYBACK_VOLUME * 100)
    for control in ("PCM", "Speaker", "Master", "PCM Playback Volume"):
        for level in (f"{pct}%", "50%", "400"):
            rc = os.system(
                f"amixer -c {SPEAKER_ALSA_CARD} sset '{control}' {level} unmute 2>/dev/null"
            )
            if rc == 0:
                return


def play_mp3(audio_bytes):
    """Play MP3 on the USB speaker via ffmpeg + aplay."""
    mp3_path = wav_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_bytes)
            mp3_path = f.name
        wav_path = mp3_path.replace(".mp3", ".wav")
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", mp3_path,
                "-af", f"volume={PLAYBACK_VOLUME}",
                "-ar", str(PLAYBACK_RATE), "-ac", "2", wav_path,
            ],
            check=True,
        )
        print(f"Playing on {PLAYBACK_DEVICE} @ {PLAYBACK_RATE}Hz")
        subprocess.run(
            ["aplay", "-D", PLAYBACK_DEVICE, wav_path],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Audio playback failed: {e}")
    finally:
        for path in (mp3_path, wav_path):
            if path and os.path.exists(path):
                os.unlink(path)


def speak(text):
    print(f"Jarvis: {text}")
    audio = eleven_client.text_to_speech.convert(
        voice_id=ELEVENLABS_VOICE_ID,
        text=text,
        model_id="eleven_multilingual_v2",
    )
    play_mp3(b"".join(audio))


def safe_speak(text):
    """Speak without crashing if TTS/network fails at boot."""
    try:
        speak(text)
    except Exception as e:
        print(f"TTS failed: {e}")


def record_audio(stream, duration=5):
    """Record from the already-open mic stream (cannot open mic twice)."""
    print("Listening for command...")
    frames = []
    for _ in range(0, int(MIC_SAMPLE_RATE / 4096 * duration)):
        data = stream.read(4096, exception_on_overflow=False)
        frames.append(data)
    audio_bytes = b"".join(frames)
    # Resample from 44100 to 16000 for Whisper
    audio_np = pcm_to_mono(audio_bytes, MIC_CHANNELS).astype(np.float32)
    resampled = soxr.resample(audio_np, MIC_SAMPLE_RATE, 16000)
    resampled_int16 = resampled.astype(np.int16)
    return resampled_int16.tobytes(), 16000

def transcribe(audio_bytes, sample_rate):
    import wave
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)
    wav_buffer.seek(0)
    wav_buffer.name = "audio.wav"
    transcript = openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=wav_buffer
    )
    return transcript.text.strip()

def set_timer(seconds):
    def timer_thread():
        time.sleep(seconds)
        speak(f"Your {seconds} second timer is done, sir.")
    threading.Thread(target=timer_thread, daemon=True).start()
    speak(f"Timer set for {seconds} seconds, sir.")

def set_volume(level):
    global PLAYBACK_VOLUME
    PLAYBACK_VOLUME = max(0.1, min(1.0, level / 100.0))
    os.system(
        f"amixer -c {SPEAKER_ALSA_CARD} sset 'PCM' {level}% unmute 2>/dev/null"
    )
    speak(f"Volume set to {level} percent, sir.")

def shutdown():
    speak("Shutting down. Goodbye sir.")
    time.sleep(2)
    os.system("sudo shutdown now")

def ask_gpt(user_input, extra_system=None):
    conversation_history.append({"role": "user", "content": user_input})
    system = f"{SYSTEM_PROMPT}\n\n[LIVE CONTEXT]\n{get_live_context()}"
    if extra_system:
        system += f"\n\n{extra_system}"
    messages = [{"role": "system", "content": system}] + conversation_history
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    reply = response.choices[0].message.content.strip()
    conversation_history.append({"role": "assistant", "content": reply})
    return reply

def handle_response(reply):
    if reply.startswith("[SEARCH]"):
        query = reply.replace("[SEARCH]", "").strip()
        speak("One moment, sir.")
        final = answer_with_web(query)
        speak(final)
    elif reply.startswith("[TIMER]"):
        seconds = int(reply.replace("[TIMER]", "").strip())
        set_timer(seconds)
    elif reply.startswith("[VOLUME]"):
        level = int(reply.replace("[VOLUME]", "").strip())
        set_volume(level)
    elif reply == "[SHUTDOWN]":
        shutdown()
    else:
        speak(reply)

def main():
    init_audio_devices()
    print(f"Speaker: ALSA card {SPEAKER_ALSA_CARD} ({PLAYBACK_DEVICE}) @ {PLAYBACK_RATE}Hz")
    print(
        f"Mic: ALSA card {MIC_ALSA_CARD}, PyAudio index {MIC_DEVICE_INDEX} "
        f"({MIC_DEVICE_NAME}), channels {MIC_CHANNELS}"
    )
    if SPEAKER_ALSA_CARD == "0":
        print("WARNING: Using 3.5mm jack — plug in USB speaker before starting, then restart.")
    porcupine = pvporcupine.create(
        access_key=PORCUPINE_KEY,
        keywords=["jarvis"]
    )
    
    # Calculate how many 44100Hz frames equal one Porcupine frame at 16000Hz
    porcupine_frame_length = porcupine.frame_length  # at 16000Hz
    mic_frame_length = int(porcupine_frame_length * MIC_SAMPLE_RATE / 16000)

    pa = pyaudio.PyAudio()
    stream = pa.open(
        rate=MIC_SAMPLE_RATE,
        channels=MIC_CHANNELS,
        format=pyaudio.paInt16,
        input=True,
        input_device_index=MIC_DEVICE_INDEX,
        frames_per_buffer=mic_frame_length
    )

    set_speaker_volume()
    safe_speak("Jarvis online. How can I help you, sir?")
    print("Waiting for wake word 'Jarvis'...")

    try:
        while True:
            pcm_bytes = stream.read(mic_frame_length, exception_on_overflow=False)
            # Resample from 44100 to 16000 for Porcupine
            pcm_np = pcm_to_mono(pcm_bytes, MIC_CHANNELS).astype(np.float32)
            resampled = soxr.resample(pcm_np, MIC_SAMPLE_RATE, 16000)
            resampled_int16 = resampled.astype(np.int16)
            # Make sure we have exactly the right frame length
            if len(resampled_int16) < porcupine_frame_length:
                continue
            pcm = struct.unpack_from("h" * porcupine_frame_length, resampled_int16[:porcupine_frame_length].tobytes())
            result = porcupine.process(pcm)
            if result >= 0:
                speak("Yes sir?")
                audio_bytes, sample_rate = record_audio(stream, duration=5)
                user_input = transcribe(audio_bytes, sample_rate)
                if user_input:
                    print(f"You: {user_input}")
                    local = try_local_answer(user_input)
                    if local:
                        speak(local)
                    elif needs_web_search(user_input):
                        speak("One moment, sir.")
                        reply = answer_with_web(user_input)
                        speak(reply)
                    else:
                        reply = ask_gpt(user_input)
                        handle_response(reply)
    except KeyboardInterrupt:
        speak("Going offline. Goodbye sir.")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        porcupine.delete()

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
