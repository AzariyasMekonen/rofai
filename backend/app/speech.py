import io
import re
import wave
from xml.sax.saxutils import escape, quoteattr

import azure.cognitiveservices.speech as speechsdk

from .config import get_settings

SAMPLE_RATE = 16000
STT_LOCALES = {"en": "en-US", "am": "am-ET"}
_ETHIOPIC_RE = re.compile(r"[\u1200-\u137F]")


class SpeechNotConfiguredError(RuntimeError):
    pass


class SpeechServiceError(RuntimeError):
    pass


class InvalidAudioError(ValueError):
    pass


def _config() -> speechsdk.SpeechConfig:
    s = get_settings()
    if not s.speech_configured:
        raise SpeechNotConfiguredError("AZURE_SPEECH_KEY and AZURE_SPEECH_REGION must be set in .env")
    cfg = speechsdk.SpeechConfig(subscription=s.azure_speech_key, region=s.azure_speech_region)
    cfg.set_property(speechsdk.PropertyId.SpeechSynthesis_FrameTimeoutInterval, "15000")
    cfg.set_property(speechsdk.PropertyId.SpeechSynthesis_RtfTimeoutThreshold, "10")
    return cfg



def read_wav_pcm(data: bytes, max_seconds: int) -> bytes:
    """Validate a 16 kHz mono 16-bit WAV and return its raw PCM frames."""
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (SAMPLE_RATE, 1, 2):
                raise InvalidAudioError("audio must be 16 kHz, mono, 16-bit PCM WAV")
            if wav.getnframes() > SAMPLE_RATE * max_seconds:
                raise InvalidAudioError(f"audio longer than {max_seconds} seconds")
            return wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as exc:
        raise InvalidAudioError("not a valid WAV file") from exc


def recognize(pcm: bytes, language: str) -> tuple[str, str | None]:
    """Speech-to-text. language: 'en', 'am', or 'auto'. Returns (text, detected short code)."""
    cfg = _config()
    fmt = speechsdk.audio.AudioStreamFormat(samples_per_second=SAMPLE_RATE, bits_per_sample=16, channels=1)
    stream = speechsdk.audio.PushAudioInputStream(stream_format=fmt)
    stream.write(pcm)
    stream.close()
    audio_cfg = speechsdk.audio.AudioConfig(stream=stream)

    if language in STT_LOCALES:
        cfg.speech_recognition_language = STT_LOCALES[language]
        recognizer = speechsdk.SpeechRecognizer(speech_config=cfg, audio_config=audio_cfg)
    else:
        auto = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=list(STT_LOCALES.values()))
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=cfg, auto_detect_source_language_config=auto, audio_config=audio_cfg
        )

    result = recognizer.recognize_once_async().get()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        if language in STT_LOCALES:
            return result.text, language
        locale = speechsdk.AutoDetectSourceLanguageResult(result).language or ""
        return result.text, locale.split("-")[0] or None
    if result.reason == speechsdk.ResultReason.NoMatch:
        return "", None
    details = result.cancellation_details
    raise SpeechServiceError(f"recognition canceled: {details.reason} {details.error_details}")


def detect_text_language(text: str) -> str:
    return "am" if _ETHIOPIC_RE.search(text) else "en"


def build_ssml(text: str, voice: str, locale: str, rate: str, pitch: str) -> str:
    return (
        f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang={quoteattr(locale)}>'
        f"<voice name={quoteattr(voice)}>"
        f"<prosody rate={quoteattr(rate)} pitch={quoteattr(pitch)}>{escape(text)}</prosody>"
        "</voice></speak>"
    )


# Ge'ez fidel base consonant articulatory categories (Viseme ID for consonant onset)
_ETHIOPIC_CONSONANTS: dict[int, int] = {
    # Bilabial: lips pressed together (viseme 21)
    0x1210: 21,  # መ
    0x1260: 21,  # በ
    0x1338: 21,  # ጰ
    0x1358: 21,  # ፐ
    # Labiodental: lower lip to upper teeth (viseme 18)
    0x1268: 18,  # ቨ
    0x1348: 18,  # ፈ
    # Sibilants / dental: closed teeth, tongue blade (viseme 15)
    0x1230: 15,  # ሰ
    0x1228: 15,  # ሠ
    0x1238: 15,  # ሸ
    0x12D8: 15,  # ዘ
    0x12E0: 15,  # ዠ
    0x1330: 15,  # ጸ
    0x1340: 15,  # ፀ
    # Alveolar & others (viseme 19)
    0x1270: 19,  # ተ
    0x1278: 19,  # ቸ
    0x1290: 19,  # ነ
    0x1298: 19,  # ኘ
    0x12A8: 19,  # ከ
    0x12B8: 19,  # ኸ
    0x12C8: 7,   # ወ
    0x12E8: 6,   # የ
    0x12F0: 19,  # ደ
    0x12F8: 19,  # ጀ
    0x1300: 19,  # ገ
    0x1320: 19,  # ጠ
    0x1328: 19,  # ጨ
}

# Ge'ez 7 vowel orders (0 to 6) mapped onto Azure Viseme IDs
_ORDER_TO_VISEME: dict[int, int] = {
    0: 1,   # 1st order (ግዕዝ /ä/): open-mid vowel -> viseme 1 (/æ/, /ʌ/)
    1: 7,   # 2nd order (ካዕብ /u/): rounded vowel -> viseme 7 (/u/, /w/)
    2: 6,   # 3rd order (ሣልስ /i/): spread vowel -> viseme 6 (/i/, /y/)
    3: 2,   # 4th order (ራብዕ /a/): wide open jaw -> viseme 2 (/ɑ/)
    4: 4,   # 5th order (ኃምስ /e/): spread mid -> viseme 4 (/eɪ/, /ɛ/)
    5: 19,  # 6th order (ሳድስ /ə/ or coda consonant): closed/alveolar -> viseme 19
    6: 8,   # 7th order (ሳብዕ /o/): rounded open -> viseme 8 (/oʊ/, /ɔ/)
}


def generate_amharic_visemes(text: str, duration_ms: float) -> list[dict[str, float]]:
    """Generate high-precision phonetic viseme timeline for Ge'ez/Amharic text."""
    items: list[tuple[int, float]] = []
    for ch in text:
        code = ord(ch)
        if ch in " \t":
            items.append((0, 1.0))
        elif ch in "፣,;":
            items.append((0, 1.8))
        elif ch in "።.!?፦":
            items.append((0, 2.5))
        elif 0x1200 <= code <= 0x137F:
            order = (code - 0x1200) % 8
            base = code - order
            cons_viseme = _ETHIOPIC_CONSONANTS.get(base)
            if cons_viseme in (21, 18) and order != 5:
                # Bilabial or labiodental closure before vowel
                items.append((cons_viseme, 0.4))
                items.append((_ORDER_TO_VISEME.get(order, 1), 0.8))
            elif order < 7:
                vowel_viseme = _ORDER_TO_VISEME.get(order, 1)
                items.append((vowel_viseme, 1.0))
            else:
                items.append((1, 1.0))
        elif ch.isalnum():
            items.append((1, 1.0))

    if not items:
        return [{"t": 0.0, "id": 0}]

    total_weight = sum(w for _, w in items)
    lead_in = min(60.0, duration_ms * 0.05)
    lead_out = min(80.0, duration_ms * 0.06)
    active_ms = max(50.0, duration_ms - lead_in - lead_out)
    ms_per_weight = active_ms / max(1.0, total_weight)

    visemes: list[dict[str, float]] = [{"t": 0.0, "id": 0}]
    cur_t = lead_in
    for vid, w in items:
        visemes.append({"t": round(cur_t, 1), "id": vid})
        cur_t += w * ms_per_weight

    visemes.append({"t": round(duration_ms, 1), "id": 0})
    return visemes


def synthesize(text: str, language: str) -> tuple[bytes, list[dict[str, float]]]:
    """Text-to-speech to MP3. Returns (audio bytes, visemes as [{t: ms, id}])."""
    s = get_settings()
    if language == "am":
        voice, locale = s.tts_voice_am, "am-ET"
    else:
        voice, locale = s.tts_voice_en, "en-US"

    cfg = _config()
    cfg.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio24Khz48KBitRateMonoMp3)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=cfg, audio_config=None)

    visemes: list[dict[str, float]] = []
    synthesizer.viseme_received.connect(
        lambda evt: visemes.append({"t": evt.audio_offset / 10_000, "id": evt.viseme_id})
    )

    result = synthesizer.speak_ssml_async(build_ssml(text, voice, locale, s.tts_rate, s.tts_pitch)).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        duration_ms = result.audio_duration.total_seconds() * 1000
        if duration_ms <= 0:
            # Fallback estimation for 24kHz 48kbps MP3 (6000 bytes/sec)
            duration_ms = (len(result.audio_data) / 6000.0) * 1000.0

        # am-ET voices emit dummy (0.0, 0) visemes; generate real phonetic visemes for Amharic
        if language == "am" or not visemes or all(v.get("id", 0) == 0 for v in visemes):
            visemes = generate_amharic_visemes(text, duration_ms)
        elif visemes and duration_ms > 0:
            # Ensure English speech ends with resting mouth shape
            if visemes[-1]["t"] < duration_ms:
                visemes.append({"t": round(duration_ms, 1), "id": 0})

        return result.audio_data, visemes

    details = result.cancellation_details
    raise SpeechServiceError(f"synthesis canceled: {details.reason} {details.error_details}")

