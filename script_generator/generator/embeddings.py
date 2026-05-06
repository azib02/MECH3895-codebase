from functools import lru_cache

from sentence_transformers import SentenceTransformer

from generator.config import Config


@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(Config.EMBEDDING_MODEL)


def embed_text(text):
    model = get_embedding_model()
    return model.encode(text).tolist()