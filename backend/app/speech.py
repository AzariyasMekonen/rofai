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
    return speechsdk.SpeechConfig(subscription=s.azure_speech_key, region=s.azure_speech_region)


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
    # audio_offset is in 100-ns ticks; am-ET voices emit no visemes, the client falls back to loudness
    synthesizer.viseme_received.connect(
        lambda evt: visemes.append({"t": evt.audio_offset / 10_000, "id": evt.viseme_id})
    )

    result = synthesizer.speak_ssml_async(build_ssml(text, voice, locale, s.tts_rate, s.tts_pitch)).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return result.audio_data, visemes
    details = result.cancellation_details
    raise SpeechServiceError(f"synthesis canceled: {details.reason} {details.error_details}")
