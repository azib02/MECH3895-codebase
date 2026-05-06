import psycopg2
from psycopg2.extras import RealDictCursor

from generator.config import Config


def get_connection(cursor_factory=None):
    Config.validate()

    return psycopg2.connect(
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        cursor_factory=cursor_factory,
    )


def get_dict_connection():
    return get_connection(cursor_factory=RealDictCursor)