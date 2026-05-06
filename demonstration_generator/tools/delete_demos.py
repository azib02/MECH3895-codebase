import argparse
from pathlib import Path

import h5py


def list_demos(file_path: Path) -> None:
    """List all demos inside an HDF5 file."""
    with h5py.File(file_path, "r") as hdf5_file:
        demos = list(hdf5_file["data"].keys())

    print(f"\nFound {len(demos)} demos:")
    for demo in demos:
        print(f"  - {demo}")


def delete_demos(file_path: Path, demo_names: list[str], dry_run: bool) -> None:
    """
    Delete selected demos from an HDF5 file.

    Note:
        Deleting HDF5 groups unlinks them, but the file size may not shrink
        until the file is repacked.
    """
    with h5py.File(file_path, "a") as hdf5_file:
        data_group = hdf5_file["data"]

        for demo_name in demo_names:
            if demo_name not in data_group:
                print(f"{demo_name} not found.")
                continue

            if dry_run:
                print(f"[DRY RUN] Would delete: {demo_name}")
            else:
                print(f"Deleting: {demo_name}")
                del data_group[demo_name]

    if dry_run:
        print("\nDry run complete. No demos were deleted.")
    else:
        print("\nDeletion complete.")
        print("Note: HDF5 file size may not shrink until repacked.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete selected demos from an HDF5 file."
    )

    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the HDF5 file.",
    )

    parser.add_argument(
        "--demos",
        nargs="+",
        required=True,
        help="Demo names to delete, e.g. demo_0 demo_5.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without modifying the file.",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List demos before deleting.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    file_path = Path(args.file)

    if args.list:
        list_demos(file_path)

    delete_demos(
        file_path=file_path,
        demo_names=args.demos,
        dry_run=args.dry_run,
    )