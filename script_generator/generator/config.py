import os
from dotenv import load_dotenv


load_dotenv()


class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    DB_NAME = os.getenv("DB_NAME", "BDDL-Script")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
    DB_PORT = os.getenv("DB_PORT", "5433")

    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")
    GROQ_MODEL_SUMMARY = os.getenv("GROQ_MODEL_SUMMARY", "llama-3.1-8b-instant")
    GROQ_MODEL_GENERATE = os.getenv("GROQ_MODEL_GENERATE", "llama-3.3-70b-versatile")

    @classmethod
    def validate(cls):
        missing = []

        if not cls.GROQ_API_KEY:
            missing.append("GROQ_API_KEY")

        if not cls.DB_PASSWORD:
            missing.append("DB_PASSWORD")

        if missing:
            raise RuntimeError(
                "Missing required environment variables: " + ", ".join(missing)
            )