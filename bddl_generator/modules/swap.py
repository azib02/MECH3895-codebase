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
    """Build the LLM prompt for swapping two standalone region locations."""
    standalone = grouping_data.get("standalone", [])

    target_regions = ", ".join(standalone) if len(standalone) >= 2 else "None"

    feedback = ""

    if errors and attempt > 1:
        formatted_errors = "\n".join(f"  - {error}" for error in errors)
        feedback = (
            "\nPrevious attempt failed validation with these errors:\n"
            f"{formatted_errors}\n"
            "Fix only the swap issue. Do not rename objects, fixtures, or regions.\n"
        )

    return (
        "You are modifying a BDDL robot scene file.\n\n"
        "Task:\n"
        "Swap the physical locations (:ranges) of two movable standalone regions.\n\n"
        f"Standalone regions available for swapping: {target_regions}\n\n"
        "Rules:\n"
        "1. Choose any two regions from the standalone list and swap only their :ranges coordinates.\n"
        "2. Do not swap region names.\n"
        "3. Do not swap :target fixtures.\n"
        "4. Do not swap or change :yaw_rotation.\n"
        "5. Do not change :language.\n"
        "6. Do not add, remove, or rename objects, fixtures, or regions.\n"
        "7. Do not modify any region outside the standalone list.\n"
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
            temperature=0.2,
        )

        result = response.choices[0].message.content
        return _clean_llm_output(result)

    except Exception as exc:
        print(f"  [Swap] LLM error: {exc}")
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
    Swap the :ranges coordinates of two standalone BDDL regions.

    Args:
        input_bddl:
            BDDL text entering this stage.

        original_bddl:
            Original BDDL text before augmentation.

        bddl_path:
            Path to the original BDDL file. Used for grouping analysis.

        save_path:
            Where the swapped BDDL file should be saved.

    Returns:
        success:
            True if a valid swapped BDDL file was produced.

        result_bddl:
            Swapped BDDL text if successful, otherwise the unchanged input.

        errors:
            Validation errors from the final failed attempt.
    """
    print("  [Swap] Running grouping analysis...")

    grouping_data = analyze_relational_bddl(bddl_path)
    standalone_regions = grouping_data.get("standalone", [])

    if len(standalone_regions) < 2:
        print("  [Swap] Not enough standalone regions to swap. Skipping.")
        return True, input_bddl, []

    try:
        client = _get_groq_client()
    except ValueError as exc:
        return False, input_bddl, [str(exc)]

    syntax_validator = BDDLValidator(original_bddl)
    proximity_validator = ProximityValidator(min_clearance=0.01)

    current_working_bddl = input_bddl
    last_errors = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"  [Swap] Attempt {attempt}/{MAX_ATTEMPTS}...")

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
            last_errors = ["Swap Error: LLM returned an empty response."]
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
            print(f"  [Swap] Passed on attempt {attempt}.")
            return True, saved_text, []

        current_working_bddl = saved_text
        print("  [Swap] Validation failed. Retrying with feedback...")

    print(f"  [Swap] Failed after {MAX_ATTEMPTS} attempts.")
    return False, input_bddl, last_errors


# ---------------------------------------------------------------------------
# Optional manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    original_file = Path("../input_bddl/example.bddl")
    yaw_file = Path("../attempts/manual_yaw_test.bddl")
    save_file = Path("../attempts/manual_swap_test.bddl")

    if not original_file.exists():
        print(f"Original file not found: {original_file}")
    elif not yaw_file.exists():
        print(f"Yaw file not found: {yaw_file}")
    else:
        original_text = _load_text(original_file)
        yaw_text = _load_text(yaw_file)

        success, result_text, errors = run(
            input_bddl=yaw_text,
            original_bddl=original_text,
            bddl_path=original_file,
            save_path=save_file,
        )

        print("\n" + "=" * 60)

        if success:
            print("Swap complete")
            print(f"Saved to: {save_file}")
        else:
            print("Swap failed")
            for error in errors:
                print(f"  - {error}")

        print("=" * 60)