import argparse
import json
import os

from groq import Groq
from psycopg2.extras import Json

from generator.config import Config
from generator.db import get_connection
from generator.embeddings import embed_text


def get_llama_summary(description):
    client = Groq(api_key=Config.GROQ_API_KEY)

    completion = client.chat.completions.create(
        model=Config.GROQ_MODEL_SUMMARY,
        messages=[
            {
                "role": "system",
                "content": "You are a robotics expert. Summarise the robot task briefly and clearly.",
            },
            {
                "role": "user",
                "content": f"Task: {description}",
            },
        ],
        temperature=0.0,
    )

    return completion.choices[0].message.content.strip()


def archive_policy(task_name, description, script_path, matrices_path=None):
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script file not found: {script_path}")

    enhanced_description = get_llama_summary(description)
    task_vector = embed_text(enhanced_description)

    with open(script_path, "r", encoding="utf-8") as file:
        script_code = file.read()

    matrices = []
    if matrices_path:
        if not os.path.exists(matrices_path):
            raise FileNotFoundError(f"Matrices JSON file not found: {matrices_path}")

        with open(matrices_path, "r", encoding="utf-8") as file:
            matrices = json.load(file)

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO tasks (task_name, description, embedding)
            VALUES (%s, %s, %s)
            ON CONFLICT (task_name) DO UPDATE SET
                description = EXCLUDED.description,
                embedding = EXCLUDED.embedding
            RETURNING id
            """,
            (task_name, enhanced_description, task_vector),
        )

        task_id = cursor.fetchone()[0]

        cursor.execute("DELETE FROM matrices WHERE task_id = %s", (task_id,))
        cursor.execute("DELETE FROM scripts WHERE task_id = %s", (task_id,))

        for matrix in matrices:
            cursor.execute(
                """
                INSERT INTO matrices (task_id, label, data)
                VALUES (%s, %s, %s)
                """,
                (task_id, matrix["label"], Json(matrix["data"])),
            )

        cursor.execute(
            """
            INSERT INTO scripts (task_id, code_content, success)
            VALUES (%s, %s, %s)
            """,
            (task_id, script_code, True),
        )

        conn.commit()

    finally:
        conn.close()

    print(f"Saved task: {task_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Archive a successful LIBERO policy into the script database."
    )

    parser.add_argument("--task-name", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--script", required=True)
    parser.add_argument("--matrices", default=None)

    args = parser.parse_args()

    archive_policy(
        task_name=args.task_name,
        description=args.description,
        script_path=args.script,
        matrices_path=args.matrices,
    )


if __name__ == "__main__":
    main()