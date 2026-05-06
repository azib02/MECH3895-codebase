import os
import time
from pathlib import Path

from modules import rephrase
from modules import shift
from modules import yaw
from modules import swap


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_FOLDER = BASE_DIR / "input_bddl"
OUTPUT_FOLDER = BASE_DIR / "output_bddl"
ATTEMPTS_FOLDER = BASE_DIR / "attempts"

NUM_VARIATIONS = 5


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_evaluation_summary(
    total_files: int,
    successful_variations: int,
    total_attempts: int,
    start_time: float,
    expected_total: int,
) -> None:
    """Print a simple summary of the batch generation results."""
    runtime = time.time() - start_time
    success_rate = (successful_variations / expected_total * 100) if expected_total > 0 else 0
    average_attempts = (total_attempts / successful_variations) if successful_variations > 0 else 0

    print("\n" + "=" * 55)
    print(f"{'BDDL AUGMENTATION PIPELINE SUMMARY':^55}")
    print("=" * 55)

    stats = [
        ("Input files processed", total_files),
        ("Target variations", expected_total),
        ("Successful variations", successful_variations),
        ("Total pipeline attempts", total_attempts),
        ("Success rate", f"{success_rate:.2f}%"),
        ("Average attempts per success", f"{average_attempts:.2f}"),
        ("Total runtime", f"{runtime:.2f}s"),
    ]

    for label, value in stats:
        print(f"{label:<35} {value:>15}")

    print("=" * 55)


# ---------------------------------------------------------------------------
# Core Pipeline
# ---------------------------------------------------------------------------

def process_single_variation(
    original_path: Path,
    original_content: str,
    work_dir: Path,
) -> tuple[str | None, int]:
    """
    Generate one augmented variation of a BDDL file.

    Pipeline stages:
        1. Rephrase the language instruction.
        2. Shift object regions.
        3. Randomise yaw rotation.
        4. Swap two standalone object locations.

    Returns:
        final_text:
            The final augmented BDDL text if successful, otherwise None.

        step_count:
            Number of pipeline stages attempted.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    rephrased_path = work_dir / "1_rephrased.bddl"
    shifted_path = work_dir / "2_shifted.bddl"
    yawed_path = work_dir / "3_yawed.bddl"
    swapped_path = work_dir / "4_swapped.bddl"

    step_count = 0

    # Stage 1: Rephrase task instruction
    success, current_bddl, errors = rephrase.run(original_content, str(rephrased_path))
    step_count += 1

    if not success:
        print(f"      [Rephrase failed] {errors}")
        return None, step_count

    # Stage 2: Shift object regions
    success, current_bddl, errors = shift.run(
        input_bddl=current_bddl,
        original_bddl=original_content,
        bddl_path=str(original_path),
        save_path=str(shifted_path),
    )
    step_count += 1

    if not success:
        print(f"      [Shift failed] {errors}")
        return None, step_count

    # Stage 3: Randomise yaw rotation
    success, current_bddl, errors = yaw.run(
        input_bddl=current_bddl,
        original_bddl=original_content,
        bddl_path=str(original_path),
        save_path=str(yawed_path),
    )
    step_count += 1

    if not success:
        print(f"      [Yaw failed] {errors}")
        return None, step_count

    # Stage 4: Swap two standalone object locations
    success, current_bddl, errors = swap.run(
        input_bddl=current_bddl,
        original_bddl=original_content,
        bddl_path=str(original_path),
        save_path=str(swapped_path),
    )
    step_count += 1

    if not success:
        print(f"      [Swap failed] {errors}")
        return None, step_count

    return current_bddl, step_count


def run_batch_process() -> None:
    """Run the BDDL augmentation pipeline over every .bddl file in input_bddl/."""
    start_time = time.time()

    INPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    ATTEMPTS_FOLDER.mkdir(parents=True, exist_ok=True)

    bddl_files = sorted(INPUT_FOLDER.glob("*.bddl"))

    if not bddl_files:
        print(f"No .bddl files found in: {INPUT_FOLDER}")
        print("Place your original BDDL files inside the input_bddl folder and run again.")
        return

    total_files = len(bddl_files)
    expected_total = total_files * NUM_VARIATIONS
    successful_variations = 0
    total_attempts = 0

    print(f"Starting BDDL augmentation for {total_files} file(s)...")

    for original_path in bddl_files:
        with open(original_path, "r", encoding="utf-8") as file:
            original_content = file.read()

        print(f"\n[FILE] {original_path.name}")

        for variation_index in range(1, NUM_VARIATIONS + 1):
            file_slug = original_path.stem
            work_dir = ATTEMPTS_FOLDER / file_slug / f"variation_{variation_index}"

            final_text, attempts_used = process_single_variation(
                original_path=original_path,
                original_content=original_content,
                work_dir=work_dir,
            )

            total_attempts += attempts_used

            if final_text:
                output_name = f"{original_path.stem}_var{variation_index}.bddl"
                output_path = OUTPUT_FOLDER / output_name

                with open(output_path, "w", encoding="utf-8") as output_file:
                    output_file.write(final_text)

                successful_variations += 1
                print(f"Variation {variation_index}: saved to {output_path.name}")
            else:
                print(f"Variation {variation_index}: failed")

    print_evaluation_summary(
        total_files=total_files,
        successful_variations=successful_variations,
        total_attempts=total_attempts,
        start_time=start_time,
        expected_total=expected_total,
    )


if __name__ == "__main__":
    run_batch_process()