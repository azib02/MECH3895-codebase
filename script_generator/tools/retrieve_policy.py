import argparse
import os

from generator.db import get_connection


def retrieve_policy(task_name, output_path):
    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT s.code_content
            FROM tasks t
            JOIN scripts s ON t.id = s.task_id
            WHERE t.task_name = %s
            ORDER BY s.created_at DESC
            LIMIT 1
            """,
            (task_name,),
        )

        result = cursor.fetchone()

        if not result:
            print(f"Task not found: {task_name}")
            return False

        code_content = result[0]

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as file:
            file.write(code_content)

    finally:
        conn.close()

    print(f"Recovered script saved to: {os.path.abspath(output_path)}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve a saved LIBERO policy from the script database."
    )

    parser.add_argument("--task-name", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    retrieve_policy(
        task_name=args.task_name,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()