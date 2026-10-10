import io
import wave

import pytest

from app.speech import (
    InvalidAudioError,
    build_ssml,
    detect_text_language,
    generate_amharic_visemes,
    read_wav_pcm,
)



def _wav(rate=16000, channels=1, width=2, frames=1600) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(b"\x00" * frames * channels * width)
    return buf.getvalue()


def test_ssml_escapes_user_text():
    ssml = build_ssml('Hi <break time="9s"/> & "bye"', "en-US-AvaMultilingualNeural", "en-US", "-8%", "-4%")
    assert "<break" not in ssml
    assert "&lt;break" in ssml
    assert "&amp;" in ssml
    assert "name=\"en-US-AvaMultilingualNeural\"" in ssml


def test_detect_text_language():
    assert detect_text_language("ሰላም ልጆች") == "am"
    assert detect_text_language("Hello children") == "en"


def test_read_wav_pcm_accepts_16k_mono():
    assert len(read_wav_pcm(_wav(), max_seconds=30)) == 1600 * 2


@pytest.mark.parametrize("kwargs", [{"rate": 44100}, {"channels": 2}, {"width": 1}])
def test_read_wav_pcm_rejects_wrong_format(kwargs):
    with pytest.raises(InvalidAudioError):
        read_wav_pcm(_wav(**kwargs), max_seconds=30)


def test_read_wav_pcm_rejects_too_long():
    with pytest.raises(InvalidAudioError):
        read_wav_pcm(_wav(frames=16000 * 3), max_seconds=2)


def test_read_wav_pcm_rejects_garbage():
    with pytest.raises(InvalidAudioError):
        read_wav_pcm(b"not a wav", max_seconds=30)


def test_generate_amharic_visemes():
    visemes = generate_amharic_visemes("ሰላም እንደምን አላችሁ?", 2000.0)
    assert len(visemes) >= 10
    assert visemes[0]["t"] == 0.0 and visemes[0]["id"] == 0
    assert visemes[-1]["t"] == 2000.0 and visemes[-1]["id"] == 0
    # Checks that non-zero visemes exist
    non_zero = [v["id"] for v in visemes if v["id"] > 0]
    assert len(non_zero) > 0
    # In 'ሰላም', 'ላ' has 4th order (a) -> viseme 2
    assert 2 in non_zero

