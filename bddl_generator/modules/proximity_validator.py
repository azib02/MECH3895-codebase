from typing import Any

from modules.parser import BDDLParser


class ProximityValidator:
    """
    Checks whether generated BDDL regions overlap.

    This uses simple axis-aligned bounding box collision detection on the
    :ranges values inside the :regions section.

    It does not currently check how far objects moved from their original
    positions. It only checks whether generated regions collide.
    """

    def __init__(self, min_clearance: float = 0.03) -> None:
        """
        Args:
            min_clearance:
                Minimum spacing required between two regions.
        """
        self.min_clearance = min_clearance

    def _extract_region_coords(self, region_branch: list[Any]) -> dict[str, float] | None:
        """
        Extract normalised region coordinates from a parsed region branch.

        Expected BDDL pattern:
            (:ranges ((x_min y_min x_max y_max)))

        Args:
            region_branch:
                Parsed BDDL branch for one region.

        Returns:
            Dictionary containing x1, y1, x2, y2, or None if unavailable.
        """
        ranges_block = next(
            (
                item
                for item in region_branch
                if isinstance(item, list) and item and item[0] == ":ranges"
            ),
            None,
        )

        if not ranges_block or len(ranges_block) < 2:
            return None

        try:
            coords = ranges_block[1][0]

            if len(coords) != 4:
                return None

            x1, y1, x2, y2 = [float(value) for value in coords]

            return {
                "x1": min(x1, x2),
                "y1": min(y1, y2),
                "x2": max(x1, x2),
                "y2": max(y1, y2),
            }

        except (IndexError, TypeError, ValueError):
            return None

    def _boxes_overlap(self, box_a: dict[str, float], box_b: dict[str, float]) -> bool:
        """
        Check whether two bounding boxes overlap with the clearance buffer.

        Args:
            box_a: First bounding box.
            box_b: Second bounding box.

        Returns:
            True if the boxes overlap, otherwise False.
        """
        separated_x = (
            box_a["x2"] + self.min_clearance < box_b["x1"]
            or box_b["x2"] + self.min_clearance < box_a["x1"]
        )

        separated_y = (
            box_a["y2"] + self.min_clearance < box_b["y1"]
            or box_b["y2"] + self.min_clearance < box_a["y1"]
        )

        return not separated_x and not separated_y

    def validate_proximity(
        self,
        original_content: str,
        generated_content: str,
        grouping_data: dict | None = None,
    ) -> tuple[bool, list[str]]:
        """
        Validate that generated BDDL regions do not overlap.

        Args:
            original_content:
                Original BDDL text. Currently unused, but kept for compatibility
                with the rest of the pipeline.

            generated_content:
                Generated BDDL text to validate.

            grouping_data:
                Optional grouping information. Currently unused, but kept for
                compatibility with the rest of the pipeline.

        Returns:
            Tuple of:
                is_valid: True if no collisions were found.
                errors: List of collision error messages.
        """
        errors = []

        generated_tree = BDDLParser.get_tree(generated_content)
        regions_section = BDDLParser.find_section(generated_tree, ":regions")

        if not regions_section:
            return True, []

        parsed_regions = []

        for region_branch in regions_section:
            if not isinstance(region_branch, list) or not region_branch:
                continue

            coords = self._extract_region_coords(region_branch)

            if coords:
                parsed_regions.append(
                    {
                        "name": str(region_branch[0]),
                        **coords,
                    }
                )

        for i in range(len(parsed_regions)):
            for j in range(i + 1, len(parsed_regions)):
                region_a = parsed_regions[i]
                region_b = parsed_regions[j]

                if self._boxes_overlap(region_a, region_b):
                    errors.append(
                        f"Collision Error: '{region_a['name']}' overlaps with "
                        f"'{region_b['name']}'. Move these regions apart."
                    )

        return len(errors) == 0, errors