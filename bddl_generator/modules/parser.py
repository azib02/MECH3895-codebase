import json
import re
from pathlib import Path
from typing import Any


class BDDLParser:
    """
    Utility class for parsing BDDL text into a nested Python list structure.

    This parser is used by the validation and augmentation modules to inspect
    sections such as :language, :objects, :fixtures, and :regions.
    """

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """
        Remove BDDL comments and split the text into tokens.

        Args:
            text: Raw BDDL text.

        Returns:
            A list of tokens.
        """
        text = re.sub(r";.*", "", text)
        text = text.replace("(", " ( ").replace(")", " ) ")
        return text.split()

    @staticmethod
    def parse(tokens: list[str]) -> Any:
        """
        Recursively parse a token list into a nested list tree.

        Args:
            tokens: Flat list of BDDL tokens.

        Returns:
            Nested list representation of the BDDL file.
        """
        if not tokens:
            return None

        token = tokens.pop(0)

        if token == "(":
            parsed_list = []

            while tokens and tokens[0] != ")":
                parsed_list.append(BDDLParser.parse(tokens))

            if tokens:
                tokens.pop(0)

            return parsed_list

        try:
            return float(token)
        except ValueError:
            return token

    @staticmethod
    def get_tree(bddl_text: str) -> Any:
        """
        Convert raw BDDL text into a nested list tree.

        Args:
            bddl_text: Raw BDDL text.

        Returns:
            Parsed BDDL tree.
        """
        tokens = BDDLParser.tokenize(bddl_text)
        return BDDLParser.parse(tokens)

    @staticmethod
    def find_section(tree: Any, section_name: str) -> list[Any] | None:
        """
        Search the parsed BDDL tree for a named section.

        Example section names:
            :language
            :fixtures
            :objects
            :regions
            :init
            :goal

        Args:
            tree: Parsed BDDL tree.
            section_name: Name of the section to find.

        Returns:
            Section contents if found, otherwise None.
        """
        if not isinstance(tree, list):
            return None

        if tree and tree[0] == section_name:
            return tree[1:]

        for item in tree:
            if isinstance(item, list):
                result = BDDLParser.find_section(item, section_name)
                if result is not None:
                    return result

        return None

    @staticmethod
    def extract_entities(tree: Any) -> set[str]:
        """
        Extract all object and fixture names from a parsed BDDL tree.

        Expected BDDL pattern:
            object_name - object_type
            fixture_name - fixture_type

        Args:
            tree: Parsed BDDL tree.

        Returns:
            Set of object and fixture names.
        """
        entities = set()

        for section_name in [":fixtures", ":objects"]:
            section = BDDLParser.find_section(tree, section_name)

            if not section or not isinstance(section, list):
                continue

            index = 0

            while index < len(section):
                item = str(section[index])

                if index + 1 < len(section) and str(section[index + 1]) == "-":
                    entities.add(item)
                    index += 3
                else:
                    entities.add(item)
                    index += 1

        return entities

    @staticmethod
    def extract_language(tree: Any) -> str:
        """
        Extract the :language instruction from a parsed BDDL tree.

        Args:
            tree: Parsed BDDL tree.

        Returns:
            Language instruction as a string, or an empty string if missing.
        """
        section = BDDLParser.find_section(tree, ":language")

        if not section:
            return ""

        return " ".join(str(token) for token in section).strip()

    @staticmethod
    def extract_regions(tree: Any) -> dict[str, list[Any]]:
        """
        Extract all region blocks from the :regions section.

        Args:
            tree: Parsed BDDL tree.

        Returns:
            Dictionary mapping region names to their parsed region blocks.
        """
        regions = {}
        section = BDDLParser.find_section(tree, ":regions")

        if not section:
            return regions

        for branch in section:
            if isinstance(branch, list) and branch:
                regions[str(branch[0])] = branch

        return regions


# ---------------------------------------------------------------------------
# Optional debug runner
# ---------------------------------------------------------------------------

def debug_file(bddl_path: Path) -> None:
    """
    Print key parsed sections from a single BDDL file.

    This is only for manual debugging and is not used by the main pipeline.
    """
    sections_to_debug = [
        ":language",
        ":fixtures",
        ":objects",
        ":obj_of_interest",
        ":init",
        ":goal",
        ":regions",
    ]

    if not bddl_path.exists():
        print(f"File not found: {bddl_path}")
        return

    with open(bddl_path, "r", encoding="utf-8") as file:
        raw_text = file.read()

    tree = BDDLParser.get_tree(raw_text)

    if not tree:
        print("Parser returned an empty tree.")
        return

    print("\n" + "=" * 80)
    print(f"DEBUGGING FILE: {bddl_path.name}")
    print("=" * 80)

    for section_name in sections_to_debug:
        print(f"\n--- Section: {section_name} ---")
        section_data = BDDLParser.find_section(tree, section_name)

        if section_data:
            print("Status: found")
            print(json.dumps(section_data, indent=2))
        else:
            print("Status: not found")

    print(f"\nEntities: {BDDLParser.extract_entities(tree)}")
    print(f"Language: '{BDDLParser.extract_language(tree)}'")
    print(f"Regions: {list(BDDLParser.extract_regions(tree).keys())}")


if __name__ == "__main__":
    example_path = Path("../input_bddl/example.bddl")
    debug_file(example_path)