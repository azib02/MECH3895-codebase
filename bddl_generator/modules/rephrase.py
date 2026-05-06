import os
import random
import re
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from modules.parser import BDDLParser
from modules.validator import BDDLValidator


load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "llama-3.3-70b-versatile"
MAX_ATTEMPTS = 5

PERSONAS = {
    "Helpful_Friend": "a helpful friend who is multitasking and speaking naturally",
    "Technical_Writer": "a technical manual writer providing precise operation steps",
    "Hurried_Person": "a person in a hurry who is direct but still descriptive",
    "Polite_Assistant": "a polite assistant using slightly more formal vocabulary",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_groq_client() -> Groq:
    """
    Create a Groq client from the GROQ_API_KEY environment variable.

    Your real key should be stored in a local .env file:
        GROQ_API_KEY=your_key_here
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "Missing GROQ_API_KEY. Add it to your .env file before running the pipeline."
        )

    return Groq(api_key=api_key)


def _replace_language(bddl_text: str, new_language: str) -> str:
    """
    Replace only the :language field in a BDDL file.

    Everything outside the :language field is left unchanged.
    """
    return re.sub(
        r"(\(:language\s+).*?(\))",
        lambda match: f"{match.group(1)}{new_language}{match.group(2)}",
        bddl_text,
        count=1,
        flags=re.DOTALL,
    )


def _clean_llm_output(text: str) -> str:
    """Remove quotes, markdown fences, and extra whitespace from LLM output."""
    text = text.strip()
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    text = text.replace("```", "")
    return text.strip().strip("\"'")


def _build_prompt(
    original_language: str,
    persona_description: str,
    errors: list[str],
) -> str:
    """Build the prompt used for language rephrasing."""
    feedback = ""

    if errors:
        formatted_errors = "\n".join(f"  - {error}" for error in errors)
        feedback = (
            "\nYour previous attempt failed validation with these errors:\n"
            f"{formatted_errors}\n"
            "Fix only the language field. Do not change object names or meaning.\n"
        )

    return (
        f"Rewrite the following robot task instruction in the style of "
        f"{persona_description}.\n\n"
        "Rules:\n"
        "- Keep the exact same meaning.\n"
        "- Keep all object names unchanged.\n"
        "- The new sentence must be different from the original.\n"
        "- Output only the new sentence. No quotes, no explanation, no markdown.\n"
        f"{feedback}\n"
        f"Original instruction: {original_language}"
    )


def _call_llm(
    client: Groq,
    model: str,
    original_language: str,
    persona_description: str,
    errors: list[str],
) -> str | None:
    """Ask the LLM to rephrase the original language instruction."""
    prompt = _build_prompt(
        original_language=original_language,
        persona_description=persona_description,
        errors=errors,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        result = response.choices[0].message.content
        return _clean_llm_output(result)

    except Exception as exc:
        print(f"  [Rephrase] LLM error: {exc}")
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

def run(original_bddl: str, save_path: str | Path) -> tuple[bool, str, list[str]]:
    """
    Rephrase the :language field of a BDDL file.

    Args:
        original_bddl:
            Original BDDL text.

        save_path:
            Path where the rephrased candidate should be saved.

    Returns:
        success:
            True if a valid rephrased BDDL file was produced.

        result_bddl:
            Rephrased BDDL text if successful, otherwise the original BDDL text.

        errors:
            Validation errors from the last failed attempt.
    """
    try:
        client = _get_groq_client()
    except ValueError as exc:
        return False, original_bddl, [str(exc)]

    original_language = BDDLParser.extract_language(BDDLParser.get_tree(original_bddl))

    if not original_language:
        return False, original_bddl, [
            "Rephrase Error: could not find a :language block in the input BDDL."
        ]

    print(f"  [Rephrase] Original language: '{original_language}'")

    validator = BDDLValidator(original_bddl)

    persona_name = random.choice(list(PERSONAS.keys()))
    persona_description = PERSONAS[persona_name]

    print(f"  [Rephrase] Persona: {persona_name}")

    last_errors = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"  [Rephrase] Attempt {attempt}/{MAX_ATTEMPTS}...")

        new_language = _call_llm(
            client=client,
            model=MODEL_NAME,
            original_language=original_language,
            persona_description=persona_description,
            errors=last_errors,
        )

        if not new_language:
            last_errors = ["Rephrase Error: LLM returned an empty response."]
            continue

        print(f"  [Rephrase] Candidate language: '{new_language}'")

        candidate_bddl = _replace_language(original_bddl, new_language)
        _save_text(candidate_bddl, save_path)

        saved_text = _load_text(save_path)
        is_valid, last_errors = validator.validate(
            saved_text,
            require_language_change=True,
        )

        if is_valid:
            print(f"  [Rephrase] Passed on attempt {attempt}.")
            return True, saved_text, []

        print("  [Rephrase] Validation failed:")
        for error in last_errors:
            print(f"    - {error}")

        if any("Language" in error or "Instruction" in error for error in last_errors):
            persona_name = random.choice(list(PERSONAS.keys()))
            persona_description = PERSONAS[persona_name]
            print(f"  [Rephrase] Switching persona to: {persona_name}")

    print(f"  [Rephrase] Failed after {MAX_ATTEMPTS} attempts.")
    return False, original_bddl, last_errors


# ---------------------------------------------------------------------------
# Optional manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_file = Path("../input_bddl/example.bddl")
    save_file = Path("../attempts/manual_rephrase_test.bddl")

    if not test_file.exists():
        print(f"Test file not found: {test_file}")
    else:
        original_text = _load_text(test_file)
        success, result_text, errors = run(original_text, save_file)

        print("\n" + "=" * 60)

        if success:
            new_language = BDDLParser.extract_language(BDDLParser.get_tree(result_text))
            print("Rephrase complete")
            print(f"New :language → '{new_language}'")
            print(f"Saved to: {save_file}")
        else:
            print("Rephrase failed")
            for error in errors:
                print(f"  - {error}")

        print("=" * 60)