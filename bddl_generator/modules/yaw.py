import os
import re
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from modules.grouping import analyze_relational_bddl
from modules.proximity_validator import ProximityValidator
from modules.validator import BDDLValidator


load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "llama-3.3-70b-versatile"
MAX_ATTEMPTS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_groq_client() -> Groq:
    """
    Create a Groq client from the GROQ_API_KEY environment variable.

    Your local .env file should contain:
        GROQ_API_KEY=your_key_here
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "Missing GROQ_API_KEY. Add it to your .env file before running the pipeline."
        )

    return Groq(api_key=api_key)


def _build_prompt(
    current_bddl: str,
    grouping_data: dict,
    errors: list[str],
    attempt: int,
) -> str:
    """Build the LLM prompt for yaw randomisation."""
    shielded = grouping_data.get("shielded", [])
    standalone = grouping_data.get("standalone", [])

    shielded_text = ", ".join(shielded) if shielded else "None"
    standalone_text = ", ".join(standalone) if standalone else "None"

    feedback = ""

    if errors and attempt > 1:
        formatted_errors = "\n".join(f"  - {error}" for error in errors)
        feedback = (
            "\nPrevious attempt failed validation with these errors:\n"
            f"{formatted_errors}\n"
            "Fix only the issue described. Do not rename objects, fixtures, or regions.\n"
        )

    return (
        "You are modifying a BDDL robot scene file.\n\n"
        "Task:\n"
        "Randomise the orientation yaw of movable standalone regions.\n\n"
        f"Standalone target regions: {standalone_text}\n"
        f"Shielded fixed regions: {shielded_text}\n\n"
        "Rules:\n"
        "1. For each standalone region, set :yaw_rotation to a random float between 0.0 and 6.28.\n"
        "2. If a standalone region is missing :yaw_rotation, add it.\n"
        "3. Use this syntax when adding yaw:\n"
        "   (:yaw_rotation (\n"
        "     (<angle> <angle>)\n"
        "   ))\n"
        "4. Do not change :ranges coordinates.\n"
        "5. Do not change shielded regions.\n"
        "6. Do not add, remove, or rename objects, fixtures, or regions.\n"
        "7. Do not change the :language field.\n"
        "8. Return only the raw BDDL file starting with (define. No markdown.\n"
        f"{feedback}\n"
        f"Current BDDL file:\n{current_bddl}"
    )


def _clean_llm_output(text: str) -> str:
    """Remove markdown fences and keep only the raw BDDL from '(define' onwards."""
    text = text.strip()
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    text = text.replace("```", "")

    define_index = text.find("(define")

    if define_index >= 0:
        text = text[define_index:]

    return text.strip()


def _call_llm(client: Groq, model: str, prompt: str) -> str | None:
    """Call the LLM and return cleaned BDDL text."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        result = response.choices[0].message.content
        return _clean_llm_output(result)

    except Exception as exc:
        print(f"  [Yaw] LLM error: {exc}")
        return None


def _save_text(text: str, save_path: str | Path) -> None:
    """Save text to disk, creating parent folders if needed."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as file:
        file.write(text)


def _load_text(path: str | Path) -> str:
    """Load text from disk."""
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    input_bddl: str,
    original_bddl: str,
    bddl_path: str | Path,
    save_path: str | Path,
) -> tuple[bool, str, list[str]]:
    """
    Randomise yaw rotation for standalone BDDL regions.

    Args:
        input_bddl:
            BDDL text entering this stage.

        original_bddl:
            Original BDDL text before augmentation.

        bddl_path:
            Path to the original BDDL file. Used for grouping analysis.

        save_path:
            Where the yaw-modified BDDL file should be saved.

    Returns:
        success:
            True if a valid yaw-modified BDDL file was produced.

        result_bddl:
            Yaw-modified BDDL text if successful, otherwise the unchanged input.

        errors:
            Validation errors from the final failed attempt.
    """
    try:
        client = _get_groq_client()
    except ValueError as exc:
        return False, input_bddl, [str(exc)]

    print("  [Yaw] Running grouping analysis...")

    grouping_data = analyze_relational_bddl(bddl_path)

    syntax_validator = BDDLValidator(original_bddl)
    proximity_validator = ProximityValidator(min_clearance=0.005)

    current_working_bddl = input_bddl
    last_errors = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"  [Yaw] Attempt {attempt}/{MAX_ATTEMPTS}...")

        prompt = _build_prompt(
            current_bddl=current_working_bddl,
            grouping_data=grouping_data,
            errors=last_errors,
            attempt=attempt,
        )

        candidate_bddl = _call_llm(
            client=client,
            model=MODEL_NAME,
            prompt=prompt,
        )

        if not candidate_bddl:
            last_errors = ["Yaw Error: LLM returned an empty response."]
            continue

        _save_text(candidate_bddl, save_path)
        saved_text = _load_text(save_path)

        syntax_passed, syntax_errors = syntax_validator.validate(
            saved_text,
            require_language_change=False,
        )

        proximity_passed, proximity_errors = proximity_validator.validate_proximity(
            original_content=original_bddl,
            generated_content=saved_text,
            grouping_data=grouping_data,
        )

        last_errors = syntax_errors + proximity_errors

        if syntax_passed and proximity_passed:
            print(f"  [Yaw] Passed on attempt {attempt}.")
            return True, saved_text, []

        current_working_bddl = saved_text
        print("  [Yaw] Validation failed. Retrying with feedback...")

    print(f"  [Yaw] Failed after {MAX_ATTEMPTS} attempts.")
    return False, input_bddl, last_errors


# ---------------------------------------------------------------------------
# Optional manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    original_file = Path("../input_bddl/example.bddl")
    shifted_file = Path("../attempts/manual_shift_test.bddl")
    save_file = Path("../attempts/manual_yaw_test.bddl")

    if not original_file.exists():
        print(f"Original file not found: {original_file}")
    elif not shifted_file.exists():
        print(f"Shifted file not found: {shifted_file}")
    else:
        original_text = _load_text(original_file)
        shifted_text = _load_text(shifted_file)

        success, result_text, errors = run(
            input_bddl=shifted_text,
            original_bddl=original_text,
            bddl_path=original_file,
            save_path=save_file,
        )

        print("\n" + "=" * 60)

        if success:
            print("Yaw randomisation complete")
            print(f"Saved to: {save_file}")
        else:
            print("Yaw failed")
            for error in errors:
                print(f"  - {error}")

        print("=" * 60)