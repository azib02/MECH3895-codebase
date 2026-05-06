import argparse
import glob
import os
from pathlib import Path

import h5py
import numpy as np
import tqdm


def restructure_file(input_path: Path, output_path: Path) -> None:
    """Convert one raw HDF5 demo file into trainer-ready HDF5 format."""
    print(f"Restructuring: {input_path.name}")

    with h5py.File(input_path, "r") as old_file, h5py.File(output_path, "w") as new_file:
        # Copy top-level attributes
        for key, value in old_file.attrs.items():
            new_file.attrs[key] = value

        old_data = old_file["data"]
        new_data = new_file.create_group("data")

        for demo_id in tqdm.tqdm(old_data.keys(), desc="Processing demos"):
            old_demo = old_data[demo_id]
            new_demo = new_data.create_group(demo_id)

            # Copy demo-level attributes, e.g. language instruction
            for key, value in old_demo.attrs.items():
                new_demo.attrs[key] = value

            # Trainer expects observations inside an obs group
            obs_group = new_demo.create_group("obs")

            obs_group.create_dataset("agentview_rgb", data=old_demo["image"][()])
            obs_group.create_dataset("eye_in_hand_rgb", data=old_demo["wrist_image"][()])
            obs_group.create_dataset("joint_states", data=old_demo["joint_states"][()])
            obs_group.create_dataset("gripper_states", data=old_demo["gripper_state"][()])

            # EEF state = position + orientation
            eef_data = old_demo["EEF_state"][()]
            obs_group.create_dataset("ee_states", data=eef_data)
            obs_group.create_dataset("ee_pos", data=eef_data[:, :3])
            obs_group.create_dataset("ee_ori", data=eef_data[:, 3:])

            # Root-level datasets
            new_demo.create_dataset("actions", data=old_demo["actions"][()])
            new_demo.create_dataset("states", data=old_demo["states"][()])
            new_demo.create_dataset("robot_states", data=old_demo["robot_states"][()])

            # Success signals
            num_steps = old_demo["actions"].shape[0]

            dones = np.zeros(num_steps, dtype=np.uint8)
            dones[-1] = 1

            rewards = dones.copy()

            new_demo.create_dataset("dones", data=dones)
            new_demo.create_dataset("rewards", data=rewards)


def restructure_dataset(raw_dir: Path, target_dir: Path) -> None:
    """Restructure all HDF5 files inside raw_dir."""
    target_dir.mkdir(parents=True, exist_ok=True)

    hdf5_files = sorted(raw_dir.glob("*.hdf5"))

    if not hdf5_files:
        print(f"No HDF5 files found in: {raw_dir}")
        return

    for input_path in hdf5_files:
        output_path = target_dir / input_path.name
        restructure_file(input_path, output_path)

    print(f"\nRestructuring complete. Saved to: {target_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw demonstration HDF5 files into trainer-ready format."
    )

    parser.add_argument(
        "--raw-dir",
        type=str,
        default="raw_demos",
        help="Folder containing raw HDF5 files from run_collection.py.",
    )

    parser.add_argument(
        "--target-dir",
        type=str,
        default="processed_demos",
        help="Folder where trainer-ready HDF5 files will be saved.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    restructure_dataset(
        raw_dir=Path(args.raw_dir),
        target_dir=Path(args.target_dir),
    )