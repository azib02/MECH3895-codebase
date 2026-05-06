import argparse
import os

from groq import Groq

from generator.config import Config
from generator.db import get_dict_connection
from generator.embeddings import embed_text
from generator.prompts import build_generation_prompt


def strip_code_fence(text):
    cleaned = text.strip()
    cleaned = cleaned.replace("```python", "")
    cleaned = cleaned.replace("```", "")
    return cleaned.strip()


def read_text_file(path):
    if not path:
        return ""

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def find_similar_task_and_matrices(bddl_content):
    query_vector = embed_text(bddl_content)

    conn = get_dict_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT t.task_name, s.code_content
            FROM tasks t
            JOIN scripts s ON t.id = s.task_id
            ORDER BY t.embedding <=> %s::vector
            LIMIT 1
            """,
            (query_vector,),
        )

        db_match = cursor.fetchone()

        cursor.execute(
            """
            SELECT label, data
            FROM matrices
            ORDER BY label
            """
        )

        matrix_rows = cursor.fetchall()

    finally:
        conn.close()

    if matrix_rows:
        database_matrices = "\n".join(
            f"{row['label']} = np.array({row['data']})"
            for row in matrix_rows
        )
    else:
        database_matrices = ""

    return db_match, database_matrices


def generate_policy(
    bddl_path,
    output_path,
    manual_matrices="",
    reference_script_path=None,
    notes="",
):
    bddl_content = read_text_file(bddl_path)
    manual_script_content = read_text_file(reference_script_path) if reference_script_path else ""

    db_match, database_matrices = find_similar_task_and_matrices(bddl_content)

    similar_task_name = db_match["task_name"] if db_match else None
    similar_script = db_match["code_content"] if db_match else None

    prompt = build_generation_prompt(
        bddl_content=bddl_content,
        manual_matrices=manual_matrices,
        manual_script_content=manual_script_content,
        additional_notes=notes,
        similar_task_name=similar_task_name,
        similar_script=similar_script,
        database_matrices=database_matrices,
    )

    client = Groq(api_key=Config.GROQ_API_KEY)

    completion = client.chat.completions.create(
        model=Config.GROQ_MODEL_GENERATE,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise robotics compiler. "
                    "Return only valid Python code for a LIBERO scripted policy."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.0,
    )

    generated_code = strip_code_fence(completion.choices[0].message.content)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(generated_code)

    print(f"Generated policy saved to: {os.path.abspath(output_path)}")

    if similar_task_name:
        print(f"Nearest database match: {similar_task_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a new LIBERO policy using BDDL + archived successful scripts."
    )

    parser.add_argument("--bddl-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-script", default=None)
    parser.add_argument("--manual-matrices", default="")
    parser.add_argument("--notes", default="")

    args = parser.parse_args()

    generate_policy(
        bddl_path=args.bddl_file,
        output_path=args.output,
        manual_matrices=args.manual_matrices,
        reference_script_path=args.reference_script,
        notes=args.notes,
    )


if __name__ == "__main__":
    main()