import random
import re
from pathlib import Path

from modules.grouping import analyze_relational_bddl
from modules.parser import BDDLParser
from modules.proximity_validator import ProximityValidator
from modules.validator import BDDLValidator


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_ATTEMPTS = 100
MIN_SHIFT = 0.02
MAX_SHIFT = 0.04


# ---------------------------------------------------------------------------
# Region shifting
# ---------------------------------------------------------------------------

def surgical_shift(
    content: str,
    region_name: str,
    dx: float,
    dy: float,
) -> tuple[str, bool]:
    """
    Shift a single BDDL region by dx and dy.

    Only the coordinates inside that region's :ranges block are changed.

    Args:
        content:
            BDDL text to modify.

        region_name:
            Name of the region to shift.

        dx:
            X-axis offset.

        dy:
            Y-axis offset.

    Returns:
        updated_content:
            Modified BDDL text.

        changed:
            True if the target region was found and modified.
    """
    tree = BDDLParser.get_tree(content)
    regions_section = BDDLParser.find_section(tree, ":regions")

    current_coords = None

    if regions_section:
        for region in regions_section:
            if not isinstance(region, list) or not region:
                continue

            if str(region[0]) != region_name:
                continue

            for element in region:
                if isinstance(element, list) and element and element[0] == ":ranges":
                    current_coords = element[1][0]
                    break

    if not current_coords:
        return content, False

    new_coords = [
        float(current_coords[0]) + dx,
        float(current_coords[1]) + dy,
        float(current_coords[2]) + dx,
        float(current_coords[3]) + dy,
    ]

    new_range_string = " ".join(f"{coord:.4f}" for coord in new_coords)

    pattern = (
        rf"(\(\s*{re.escape(region_name)}\b.*?"
        rf":\s*ranges\s*\(\s*\(\s*)"
        rf"([^)]+)"
        rf"(\s*\)\s*\))"
    )

    def replacement(match: re.Match) -> str:
        prefix, _, suffix = match.groups()
        return f"{prefix}{new_range_string}{suffix}"

    updated_content = re.sub(
        pattern,
        replacement,
        content,
        count=1,
        flags=re.DOTALL,
    )

    return updated_content, updated_content != content


def _random_offset() -> tuple[float, float]:
    """Generate a random x/y shift."""
    dx = random.uniform(MIN_SHIFT, MAX_SHIFT) * random.choice([-1, 1])
    dy = random.uniform(MIN_SHIFT, MAX_SHIFT) * random.choice([-1, 1])
    return dx, dy


def _save_text(text: str, save_path: str | Path) -> None:
    """Save text to disk, creating parent folders if needed."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as file:
        file.write(text)


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
    Shift movable BDDL regions and validate the result.

    Families are shifted together using the same offset.
    Standalone regions are shifted independently.

    Args:
        input_bddl:
            BDDL text entering this stage.

        original_bddl:
            Original BDDL text before augmentation.

        bddl_path:
            Path to the original BDDL file. Used for grouping analysis.

        save_path:
            Where the successful shifted BDDL file should be saved.

    Returns:
        success:
            True if a valid shifted BDDL file was produced.

        result_bddl:
            Shifted BDDL text if successful, otherwise the unchanged input.

        errors:
            Validation errors from the final failed attempt.
    """
    print("  [Shift] Running grouping analysis...")

    grouping_data = analyze_relational_bddl(bddl_path)

    syntax_validator = BDDLValidator(original_bddl)
    proximity_validator = ProximityValidator(min_clearance=0.01)

    last_errors = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"  [Shift] Attempt {attempt}/{MAX_ATTEMPTS}...")

        candidate_bddl = input_bddl

        # Shift synced families together.
        for _, members in grouping_data.get("families", {}).items():
            dx, dy = _random_offset()

            for region_name in members:
                candidate_bddl, _ = surgical_shift(
                    content=candidate_bddl,
                    region_name=region_name,
                    dx=dx,
                    dy=dy,
                )

        # Shift standalone regions independently.
        for region_name in grouping_data.get("standalone", []):
            dx, dy = _random_offset()

            candidate_bddl, _ = surgical_shift(
                content=candidate_bddl,
                region_name=region_name,
                dx=dx,
                dy=dy,
            )

        syntax_passed, syntax_errors = syntax_validator.validate(
            candidate_bddl,
            require_language_change=False,
        )

        proximity_passed, proximity_errors = proximity_validator.validate_proximity(
            original_content=original_bddl,
            generated_content=candidate_bddl,
            grouping_data=grouping_data,
        )

        last_errors = syntax_errors + proximity_errors

        if syntax_passed and proximity_passed:
            _save_text(candidate_bddl, save_path)
            print(f"  [Shift] Passed on attempt {attempt}.")
            return True, candidate_bddl, []

        print("  [Shift] Failed validation. Retrying with new offsets...")

    print(f"  [Shift] Failed after {MAX_ATTEMPTS} attempts.")
    return False, input_bddl, last_errors


# ---------------------------------------------------------------------------
# Optional manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_file = Path("../input_bddl/example.bddl")
    save_file = Path("../attempts/manual_shift_test.bddl")

    if not test_file.exists():
        print(f"Test file not found: {test_file}")
    else:
        with open(test_file, "r", encoding="utf-8") as file:
            original_text = file.read()

        success, result_text, errors = run(
            input_bddl=original_text,
            original_bddl=original_text,
            bddl_path=test_file,
            save_path=save_file,
        )

        print("\n" + "=" * 60)

        if success:
            print("Shift complete")
            print(f"Saved to: {save_file}")
        else:
            print("Shift failed")
            for error in errors:
                print(f"  - {error}")

        print("=" * 60)