import argparse
from pathlib import Path

import cv2
import h5py
from tqdm import tqdm


def print_hdf5_structure(file_path: Path) -> None:
    """Print all keys inside an HDF5 file for debugging."""
    with h5py.File(file_path, "r") as hdf5_file:
        print("\nHDF5 structure:")

        def print_key(name):
            print(name)

        hdf5_file.visit(print_key)


def export_raw_hdf5_videos(
    file_path: Path,
    output_dir: Path,
    fps: int,
    image_key: str,
) -> None:
    """
    Export videos from raw demonstration HDF5.

    Raw files produced by run_collection.py usually store images at:
        data/demo_x/image
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with h5py.File(file_path, "r") as hdf5_file:
            demos = list(hdf5_file["data"].keys())
            print(f"Found {len(demos)} demonstrations in HDF5.")

            for demo_key in demos:
                images = hdf5_file[f"data/{demo_key}/{image_key}"][:]
                num_frames, height, width, channels = images.shape

                video_name = f"{file_path.stem}_{demo_key}_raw_replay.mp4"
                video_output_path = output_dir / video_name

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    str(video_output_path),
                    fourcc,
                    fps,
                    (width, height),
                )

                print(f"Processing {demo_key} ({num_frames} frames)...")

                for frame in tqdm(images, desc=f"Writing {demo_key}"):
                    # HDF5 images are RGB; OpenCV expects BGR.
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_bgr)

                video_writer.release()
                print(f"Saved: {video_output_path}")

    except Exception as exc:
        print(f"Error: {exc}")
        print_hdf5_structure(file_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create replay videos from raw demonstration HDF5 files."
    )

    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the raw HDF5 file.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="videos",
        help="Folder where replay videos will be saved.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Video FPS. Match this to the collection control frequency.",
    )

    parser.add_argument(
        "--image-key",
        type=str,
        default="image",
        help="Image dataset key inside each demo. Raw files usually use 'image'.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    export_raw_hdf5_videos(
        file_path=Path(args.file),
        output_dir=Path(args.output_dir),
        fps=args.fps,
        image_key=args.image_key,
    )