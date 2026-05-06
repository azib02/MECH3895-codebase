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


def export_regenerated_hdf5_video(
    file_path: Path,
    output_path: Path,
    demo_key: str,
    fps: int,
    image_key: str,
) -> None:
    """
    Export a replay video from a restructured HDF5 file.

    Restructured files usually store images at:
        data/demo_x/obs/agentview_rgb
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with h5py.File(file_path, "r") as hdf5_file:
            image_path = f"data/{demo_key}/obs/{image_key}"
            images = hdf5_file[image_path][:]

            num_frames, height, width, channels = images.shape

            print(f"Found {num_frames} frames.")
            print(f"Resolution: {width}x{height}")
            print(f"Image path: {image_path}")

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (width, height),
            )

            print(f"Creating video at: {output_path}")

            for frame in tqdm(images, desc=f"Writing {demo_key}"):
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)

            video_writer.release()

            print("Success.")
            print(f"Video saved to: {output_path}")
            print(f"Duration: {num_frames / fps:.2f} seconds")

    except Exception as exc:
        print(f"Error: {exc}")
        print_hdf5_structure(file_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create replay video from a restructured HDF5 file."
    )

    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the restructured HDF5 file.",
    )

    parser.add_argument(
        "--demo",
        type=str,
        default="demo_0",
        help="Demo key to export, e.g. demo_0, demo_1, demo_9.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="videos/regenerated_replay.mp4",
        help="Output MP4 path.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video FPS.",
    )

    parser.add_argument(
        "--image-key",
        type=str,
        default="agentview_rgb",
        help="Image key inside obs. Usually agentview_rgb.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    export_regenerated_hdf5_video(
        file_path=Path(args.file),
        output_path=Path(args.output),
        demo_key=args.demo,
        fps=args.fps,
        image_key=args.image_key,
    )