import os
import re
from pathlib import Path

from modules.parser import BDDLParser


class BDDLValidator:
    def __init__(self, base_bddl_text: str) -> None:
        self.base_tree = BDDLParser.get_tree(base_bddl_text)

        print("\n--- INITIALIZING ORIGINAL FILE ---")

        self.original_objects = self._extract_entities(self.base_tree)
        self.original_language = self._extract_language(self.base_tree)

    def _extract_entities(self, tree) -> set[str]:
        """
        Extract all object and fixture names from the BDDL tree.

        Expected format:
            name - type
        """
        entities = set()

        for section in [":fixtures", ":objects"]:
            content = BDDLParser.find_section(tree, section)

            if content and isinstance(content, list):
                index = 0

                while index < len(content):
                    item = str(content[index])

                    if index + 1 < len(content) and str(content[index + 1]) == "-":
                        entities.add(item)
                        index += 3
                    else:
                        entities.add(item)
                        index += 1

        return entities

    def _check_identifiers(self, generated_text: str) -> list[str]:
        """
        Check if identifiers with underscores were split into spaces.

        Example:
            plate_1 should not become plate 1
        """
        errors = []

        parts = generated_text.split(":language")
        non_language_text = parts[-1] if len(parts) > 1 else generated_text

        for obj in self.original_objects:
            if "_" not in obj:
                continue

            name_part, number_part = obj.rsplit("_", 1)
            pattern = rf"\b{re.escape(name_part)}\s+{re.escape(number_part)}(\.0)?\b"

            if re.search(pattern, non_language_text):
                errors.append(
                    f"Identifier Error: '{obj}' was split into "
                    f"'{name_part} {number_part}'."
                )

        return errors

    def _extract_language(self, tree) -> str:
        """Extract the :language string from the BDDL tree."""
        language_section = BDDLParser.find_section(tree, ":language")

        if language_section:
            return " ".join(str(token) for token in language_section).strip()

        return ""

    def _check_region_structure(self, tree) -> list[str]:
        """
        Check that every existing :ranges block has exactly four coordinates.

        Important:
            Some semantic/helper regions may not have :ranges at all.
            Those are allowed and are skipped.
        """
        errors = []
        regions_content = BDDLParser.find_section(tree, ":regions")

        if regions_content:
            for region in regions_content:
                if not isinstance(region, list):
                    continue

                region_name = region[0]

                ranges_block = next(
                    (
                        item
                        for item in region
                        if isinstance(item, list) and item and item[0] == ":ranges"
                    ),
                    None,
                )

                # Keep your original logic:
                # if there is no :ranges block, do not complain.
                if ranges_block and len(ranges_block) > 1:
                    coord_lists = ranges_block[1]

                    for coords in coord_lists:
                        if len(coords) != 4:
                            errors.append(
                                f"Range Count Error in {region_name}: "
                                f"Expected 4 values, got {len(coords)}."
                            )

        return errors

    def _check_math_expressions(self, tree) -> list[str]:
        """Catch coordinates that contain unsolved maths expressions."""
        errors = []
        regions_content = BDDLParser.find_section(tree, ":regions")

        if regions_content:
            for region in regions_content:
                if not isinstance(region, list):
                    continue

                region_name = region[0]

                ranges_block = next(
                    (
                        item
                        for item in region
                        if isinstance(item, list) and item and item[0] == ":ranges"
                    ),
                    None,
                )

                if ranges_block and len(ranges_block) > 1:
                    for coord_list in ranges_block[1]:
                        for value in coord_list:
                            value_str = str(value)

                            if re.search(r"\d\s*[\+\*/]\s*\d|\d\s*-\s*\d", value_str):
                                errors.append(
                                    f"Math Error in {region_name}: "
                                    f"Unsolved expression '{value_str}'."
                                )

                            if value_str in ["+", "*", "/"]:
                                errors.append(
                                    f"Math Error in {region_name}: "
                                    f"Raw symbol '{value_str}' found in coordinates."
                                )

        return errors

    def _check_coordinate_ordering(self, tree) -> list[str]:
        """Check that x_min < x_max and y_min < y_max."""
        errors = []
        regions_content = BDDLParser.find_section(tree, ":regions")

        if regions_content:
            for region in regions_content:
                if not isinstance(region, list):
                    continue

                region_name = region[0]

                ranges_block = next(
                    (
                        item
                        for item in region
                        if isinstance(item, list) and item and item[0] == ":ranges"
                    ),
                    None,
                )

                if ranges_block and len(ranges_block) > 1:
                    for coords in ranges_block[1]:
                        if len(coords) == 4:
                            try:
                                x_min, y_min, x_max, y_max = [float(value) for value in coords]

                                if x_min >= x_max:
                                    errors.append(
                                        f"Ordering Error in {region_name}: "
                                        f"x_min ({x_min}) must be less than x_max ({x_max})."
                                    )

                                if y_min >= y_max:
                                    errors.append(
                                        f"Ordering Error in {region_name}: "
                                        f"y_min ({y_min}) must be less than y_max ({y_max})."
                                    )

                            except (ValueError, TypeError):
                                continue

        return errors

    def _check_parentheses(self, text: str) -> list[str]:
        """Check for balanced parentheses in the raw text."""
        open_count = text.count("(")
        close_count = text.count(")")

        if open_count != close_count:
            difference = abs(open_count - close_count)
            side = "closing ')'" if open_count > close_count else "opening '('"
            return [f"Syntax Error: Missing {difference} {side} parenthesis."]

        return []

    def validate(
        self,
        generated_text: str,
        require_language_change: bool = True,
    ) -> tuple[bool, list[str]]:
        errors = []

        print("\n--- VALIDATING GENERATED FILE ---")

        try:
            generated_tree = BDDLParser.get_tree(generated_text)

            if not generated_tree:
                return False, ["Could not parse generated BDDL. Tree is empty."]

        except Exception as exc:
            return False, [f"Syntax Error during parse: {exc}"]

        # Check 1: entity integrity
        generated_objects = self._extract_entities(generated_tree)
        missing_objects = self.original_objects - generated_objects

        if missing_objects:
            errors.append(f"Missing objects/fixtures: {missing_objects}")

        # Check 2: identifier syntax
        errors.extend(self._check_identifiers(generated_text))

        # Check 3: instruction check
        if require_language_change:
            new_language = self._extract_language(generated_tree)

            if new_language.lower() == self.original_language.lower():
                errors.append(
                    f"Instruction Error: Language was not rephrased "
                    f"('{new_language}')."
                )

        # Check 4: range values count
        errors.extend(self._check_region_structure(generated_tree))

        # Check 5: maths expressions
        errors.extend(self._check_math_expressions(generated_tree))

        # Check 6: coordinate ordering
        errors.extend(self._check_coordinate_ordering(generated_tree))

        # Check 7: parentheses
        errors.extend(self._check_parentheses(generated_text))

        return len(errors) == 0, errors


def run_validation_test(original_path: str | Path, generated_path: str | Path) -> None:
    """Simple runner to compare two specific files."""
    original_path = Path(original_path)
    generated_path = Path(generated_path)

    if not original_path.exists():
        print(f"Original file not found: {original_path}")
        return

    if not generated_path.exists():
        print(f"Generated file not found: {generated_path}")
        return

    with open(original_path, "r", encoding="utf-8") as file:
        original_text = file.read()

    with open(generated_path, "r", encoding="utf-8") as file:
        generated_text = file.read()

    validator = BDDLValidator(original_text)
    is_valid, errors = validator.validate(generated_text)

    print("\n" + "=" * 60)
    print("TESTING PAIR:")
    print(f"  Original:  {original_path.name}")
    print(f"  Generated: {generated_path.name}")
    print("-" * 60)

    if is_valid:
        print("TEST PASSED: no issues found.")
    else:
        print("TEST FAILED: issues found.")
        for error in errors:
            print(f"  - {error}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_validation_test(
        original_path=Path("../input_bddl/example_original.bddl"),
        generated_path=Path("../output_bddl/example_generated.bddl"),
    )