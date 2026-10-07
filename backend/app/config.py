from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""

    embed_base_url: str = ""
    embed_api_key: str = ""
    embed_model: str = "text-embedding-3-small"

    azure_speech_key: str = ""
    azure_speech_region: str = ""
    tts_voice_en: str = "en-US-AvaMultilingualNeural"
    tts_voice_am: str = "am-ET-MekdesNeural"
    tts_rate: str = "-8%"
    tts_pitch: str = "-4%"

    school_name: str = "our school"
    persona_name: str = "Tsehay Roschli"

    data_dir: Path = ROOT_DIR / "data"
    top_k: int = 5

    allowed_origins: str = "http://localhost:5173"
    rate_limit_chat_per_minute: int = 15
    rate_limit_speech_per_minute: int = 60

    @property
    def origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def sources_dir(self) -> Path:
        return self.data_dir / "sources"

    @property
    def facts_file(self) -> Path:
        return self.data_dir / "facts.md"

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_base_url and self.llm_model)

    @property
    def speech_configured(self) -> bool:
        return bool(self.azure_speech_key and self.azure_speech_region)


@lru_cache
def get_settings() -> Settings:
    return Settings()
