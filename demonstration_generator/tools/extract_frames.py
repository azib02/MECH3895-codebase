import argparse
from pathlib import Path

import h5py
from PIL import Image


def print_hdf5_structure(file_path: Path) -> None:
    """Print all keys inside an HDF5 file for debugging."""
    with h5py.File(file_path, "r") as hdf5_file:
        print("\nHDF5 structure:")

        def print_key(name):
            print(name)

        hdf5_file.visit(print_key)


def resolve_image_path(layout: str, demo_id: str, image_key: str) -> str:
    """
    Build the image dataset path depending on HDF5 layout.

    Raw layout:
        data/demo_x/image

    Processed/restructured layout:
        data/demo_x/obs/agentview_rgb
    """
    if layout == "raw":
        return f"data/{demo_id}/{image_key}"

    if layout == "processed":
        return f"data/{demo_id}/obs/{image_key}"

    raise ValueError("layout must be either 'raw' or 'processed'")


def extract_frames(
    hdf5_path: Path,
    output_dir: Path,
    layout: str,
    frame_index: int,
    image_key: str,
) -> None:
    """
    Extract one frame from every demo in an HDF5 file.

    frame_index examples:
        -1  = last frame
         0  = first frame
         10 = frame 10
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Inspection started: {hdf5_path}")
    print(f"Layout: {layout}")
    print(f"Frame index: {frame_index}")

    try:
        with h5py.File(hdf5_path, "r") as hdf5_file:
            demos = list(hdf5_file["data"].keys())

            for demo_id in demos:
                image_path = resolve_image_path(layout, demo_id, image_key)
                images = hdf5_file[image_path]

                selected_frame = images[frame_index]

                image = Image.fromarray(selected_frame)

                output_name = f"{hdf5_path.stem}_{demo_id}_frame_{frame_index}.png"
                output_path = output_dir / output_name

                image.save(output_path)

            print(f"Extracted {len(demos)} frames to: {output_dir}")

    except Exception as exc:
        print(f"Error: {exc}")
        print_hdf5_structure(hdf5_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract selected frames from each demo in an HDF5 file."
    )

    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the HDF5 file.",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="videos/frames",
        help="Folder where extracted PNG frames are saved.",
    )

    parser.add_argument(
        "--layout",
        type=str,
        choices=["raw", "processed"],
        default="raw",
        help="Use 'raw' for data/demo_x/image, or 'processed' for data/demo_x/obs/agentview_rgb.",
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=-1,
        help="Frame index to extract. Use -1 for last frame, 0 for first frame.",
    )

    parser.add_argument(
        "--image-key",
        type=str,
        default=None,
        help="Image dataset key. Defaults to 'image' for raw and 'agentview_rgb' for processed.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.image_key is None:
        image_key = "image" if args.layout == "raw" else "agentview_rgb"
    else:
        image_key = args.image_key

    extract_frames(
        hdf5_path=Path(args.file),
        output_dir=Path(args.out),
        layout=args.layout,
        frame_index=args.frame,
        image_key=image_key,
    )