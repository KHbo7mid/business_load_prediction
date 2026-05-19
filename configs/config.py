from pydantic_settings import BaseSettings,SettingsConfigDict

class Config(BaseSettings):
    MONGO_URL: str
    MONGO_DB: str
    EXPORT_DATE_MIN: str
    EXPORT_DATE_MAX: str
    MODEL_VERSION: str
    SERVICE_PORT: int
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
        
        
def load_config():
    return Config()