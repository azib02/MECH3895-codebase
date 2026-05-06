import re
from pathlib import Path
from typing import Any

from modules.parser import BDDLParser


SURFACE_KEYWORDS = [
    "table",
    "floor",
    "desk",
    "countertop",
]

FIXED_LANDMARK_KEYWORDS = [
    "center",
    "front",
    "back",
    "top",
    "bottom",
    "cabinet",
    "stove",
    "bin",
    "rack",
    "sink",
    "microwave",
    "basket",
    "desk",
    "contain",
]


def _get_region_target(region_branch: list[Any]) -> str:
    """
    Extract the :target value from a region branch.

    Returns an empty string if the region has no target.
    """
    target_section = BDDLParser.find_section(region_branch, ":target")

    if not target_section:
        return ""

    return str(target_section[0]).lower()


def _is_internal_region(target: str) -> bool:
    """
    Decide whether a region target points to an internal/fixed object.

    Surface targets such as tables and countertops are allowed to move objects
    around on top of them, so they are not treated as internal.
    """
    if not target:
        return False

    return not any(surface in target for surface in SURFACE_KEYWORDS)


def _is_landmark_region(region_name: str) -> bool:
    """Check whether a region name refers to a fixed scene landmark."""
    region_name = region_name.lower()
    return any(keyword in region_name for keyword in FIXED_LANDMARK_KEYWORDS)


def _find_between_family(region_name: str, all_region_names: list[str]) -> list[str]:
    """
    Build a synced family for a 'between_X_Y_region' style region.

    The family contains:
        - the between region itself
        - the base regions for both parent objects, if present
        - related next_to_ or front_of_ regions for either parent
    """
    parents = re.findall(
        r"between_([a-zA-Z0-9_]+?)_([a-zA-Z0-9_]+?)_region",
        region_name,
    )

    if not parents:
        return []

    parent_a, parent_b = parents[0]

    family_members = [region_name]

    parent_a_base = f"{parent_a}_region"
    parent_b_base = f"{parent_b}_region"

    if parent_a_base in all_region_names:
        family_members.append(parent_a_base)

    if parent_b_base in all_region_names:
        family_members.append(parent_b_base)

    for possible_child in all_region_names:
        is_related_child = any(
            prefix + parent_a in possible_child or prefix + parent_b in possible_child
            for prefix in ["next_to_", "front_of_"]
        )

        if is_related_child:
            family_members.append(possible_child)

    return sorted(set(family_members))


def analyze_relational_bddl(bddl_path: str | Path) -> dict[str, Any]:
    """
    Analyse BDDL regions and group them into movement categories.

    Returns:
        shielded:
            Fixed or internal regions that should not be moved.

        families:
            Linked regions that should move together.

        standalone:
            Independent movable regions.
    """
    bddl_path = Path(bddl_path)

    with open(bddl_path, "r", encoding="utf-8") as file:
        content = file.read()

    print(f"\n{'=' * 20} REGION GROUPING: {bddl_path.name} {'=' * 20}")

    tree = BDDLParser.get_tree(content)
    regions_section = BDDLParser.find_section(tree, ":regions")

    if not regions_section:
        print("No :regions section found.")
        return {
            "shielded": [],
            "families": {},
            "standalone": [],
        }

    all_region_names = [
        str(region[0])
        for region in regions_section
        if isinstance(region, list) and region
    ]

    shielded = []
    standalone = []
    families = {}
    assigned = set()

    # Step 1: identify regions that should not be moved.
    for region_branch in regions_section:
        if not isinstance(region_branch, list) or not region_branch:
            continue

        region_name = str(region_branch[0])
        target = _get_region_target(region_branch)

        if _is_internal_region(target) or _is_landmark_region(region_name):
            shielded.append(region_name)
            assigned.add(region_name)

    # Step 2: create synced families for between regions.
    for region_name in all_region_names:
        if "between" not in region_name.lower() or region_name in assigned:
            continue

        family_members = _find_between_family(region_name, all_region_names)

        if family_members:
            families[region_name] = family_members
            assigned.update(family_members)

    # Step 3: create synced families for next_to/front_of regions.
    for region_name in all_region_names:
        if region_name in assigned:
            continue

        spatial_match = re.search(
            r"(?:next_to_|front_of_)([a-zA-Z0-9_]+?)(?:_region)",
            region_name,
        )

        if not spatial_match:
            continue

        core_name = spatial_match.group(1)

        base_region = next(
            (
                candidate
                for candidate in all_region_names
                if core_name in candidate
                and "next_to" not in candidate
                and "between" not in candidate
            ),
            None,
        )

        if base_region and base_region not in shielded:
            families.setdefault(base_region, [base_region])

            if region_name not in families[base_region]:
                families[base_region].append(region_name)

            assigned.add(region_name)
            assigned.add(base_region)

    # Step 4: anything not assigned or shielded is standalone.
    for region_name in all_region_names:
        if region_name not in assigned and region_name not in shielded:
            standalone.append(region_name)

    print(f"\n[SHIELDED - FIXED]\n  {shielded}")

    print("\n[FAMILIES - SYNCED]")
    for family_root, members in families.items():
        print(f"  {family_root} -> {members}")

    print(f"\n[STANDALONE - INDEPENDENT]\n  {standalone}")

    return {
        "shielded": shielded,
        "families": families,
        "standalone": standalone,
    }


# Backwards-compatible name used by the older scripts.
def analyze_relational_bddl_v4(bddl_path: str | Path) -> dict[str, Any]:
    return analyze_relational_bddl(bddl_path)


# ---------------------------------------------------------------------------
# Optional debug runner
# ---------------------------------------------------------------------------

def main() -> None:
    """Run grouping analysis on every .bddl file in input_bddl/."""
    input_folder = Path("../input_bddl")
    files = sorted(input_folder.glob("*.bddl"))

    if not files:
        print(f"No .bddl files found in: {input_folder}")
        return

    stats = []

    print(f"Found {len(files)} file(s). Starting grouping analysis...")

    for index, bddl_path in enumerate(files, start=1):
        print(f"\n[{index}/{len(files)}] Processing: {bddl_path.name}")

        try:
            result = analyze_relational_bddl(bddl_path)

            stats.append(
                {
                    "file": bddl_path.name,
                    "families": len(result.get("families", {})),
                    "standalone": len(result.get("standalone", [])),
                    "shielded": len(result.get("shielded", [])),
                }
            )

        except Exception as exc:
            print(f"Failed to analyse {bddl_path.name}: {exc}")

    print("\n" + "=" * 80)
    print(f"{'BATCH GROUPING SUMMARY':^80}")
    print("=" * 80)
    print(f"{'File Name':<50} | {'Fam':<5} | {'Std':<5} | {'Shd':<5}")
    print("-" * 80)

    for item in stats:
        print(
            f"{item['file'][:48]:<50} | "
            f"{item['families']:<5} | "
            f"{item['standalone']:<5} | "
            f"{item['shielded']:<5}"
        )

    print("=" * 80)
    print(f"Done. Total files: {len(files)}")


if __name__ == "__main__":
    main()