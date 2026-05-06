import argparse
import datetime
import importlib
import json
import os
import shutil
import time
from glob import glob
from pathlib import Path

import h5py
import numpy as np
import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import DataCollectionWrapper

from libero.libero.envs import TASK_MAPPING
import libero.libero.envs.bddl_utils as BDDLUtils


# ---------------------------------------------------------------------------
# Headless rendering setup for RunPod / server environments
# ---------------------------------------------------------------------------

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------

def load_policy(policy_name: str):
    """
    Load a policy module from the policies folder.

    Example:
        --policy turn_on_stove_put_moka_pot

    This loads:
        policies/turn_on_stove_put_moka_pot.py

    The policy file must contain:
        def run_solver(env, bddl_file): ...
    """
    module_path = f"policies.{policy_name}"

    try:
        policy_module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Could not import policy '{module_path}'. "
            f"Check that policies/{policy_name}.py exists."
        ) from exc

    if not hasattr(policy_module, "run_solver"):
        raise AttributeError(
            f"Policy '{module_path}' does not contain a run_solver(env, bddl_file) function."
        )

    return policy_module


# ---------------------------------------------------------------------------
# HDF5 compilation
# ---------------------------------------------------------------------------

def gather_demonstrations_as_hdf5(
    directory: str | Path,
    final_path: str | Path,
    args: argparse.Namespace,
    problem_info: dict,
    remove_directory: list[str] | None = None,
) -> None:
    """
    Replay recorded actions in a high-resolution offscreen environment.

    This converts the temporary raw DataCollectionWrapper output into a single
    HDF5 file containing states, robot states, actions, images, EEF state,
    joint states, and gripper state.
    """
    directory = Path(directory)
    final_path = Path(final_path)
    remove_directory = remove_directory or []

    problem_name = problem_info["problem_name"]
    language_instruction = problem_info["language_instruction"]

    print(f"\n[Post-Process] Physically replaying {problem_name} at 224x224...")

    hdf5_file = h5py.File(final_path, "w")
    data_group = hdf5_file.create_group("data")
    data_group.attrs["info"] = "RLDS-ready LIBERO dataset with 224x224 images"

    render_env = TASK_MAPPING[problem_name](
        bddl_file_name=args.bddl_file,
        robots=["Panda"],
        controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=224,
        camera_widths=224,
        control_freq=args.control_freq,
    )

    episode_dirs = sorted(
        [
            item
            for item in os.listdir(directory)
            if item.startswith("ep_") and item not in remove_directory
        ]
    )

    demo_count = 0

    for episode_dir in episode_dirs:
        state_paths = directory / episode_dir / "state_*.npz"

        all_states = []
        all_actions = []

        for state_file in sorted(glob(str(state_paths))):
            data = np.load(state_file, allow_pickle=True)

            all_states.extend(data["states"])

            for action_info in data["action_infos"]:
                all_actions.append(action_info["actions"])

        if len(all_actions) == 0:
            continue

        print(f" -> Replaying demo {demo_count} ({len(all_actions)} frames)")

        render_env.reset()
        render_env.sim.set_state_from_flattened(all_states[0])
        render_env.sim.forward()

        storage = {
            "states": [],
            "robot_states": [],
            "actions": [],
            "image": [],
            "wrist_image": [],
            "EEF_state": [],
            "joint_states": [],
            "gripper_state": [],
        }

        for action in all_actions:
            storage["states"].append(render_env.sim.get_state().flatten())

            obs, reward, done, info = render_env.step(action)

            storage["actions"].append(np.array(action).astype(np.float32))
            storage["image"].append(obs["agentview_image"][::-1])
            storage["wrist_image"].append(obs["robot0_eye_in_hand_image"][::-1])

            robot_state = np.concatenate(
                [
                    obs["robot0_gripper_qpos"],
                    obs["robot0_eef_pos"],
                    obs["robot0_eef_quat"],
                ]
            )
            storage["robot_states"].append(robot_state)

            eef_state = np.concatenate(
                [
                    obs["robot0_eef_pos"],
                    T.quat2axisangle(obs["robot0_eef_quat"]),
                ]
            ).astype(np.float32)
            storage["EEF_state"].append(eef_state)

            storage["joint_states"].append(obs["robot0_joint_pos"].astype(np.float32))

            gripper_state = np.array(
                [np.mean(obs["robot0_gripper_qpos"])]
            ).astype(np.float32)
            storage["gripper_state"].append(gripper_state)

        episode_group = data_group.create_group(f"demo_{demo_count}")
        episode_group.attrs["language_instruction"] = language_instruction

        for dataset_name, dataset_data in storage.items():
            compression = "gzip" if "image" in dataset_name else None

            episode_group.create_dataset(
                dataset_name,
                data=np.array(dataset_data),
                compression=compression,
            )

        demo_count += 1

    hdf5_file.attrs["date"] = datetime.datetime.now().strftime("%m-%d-%Y")
    hdf5_file.attrs["problem_info"] = json.dumps(problem_info)

    render_env.close()
    hdf5_file.close()

    print(f"\n[Done] Successfully compiled {demo_count} demos into:")
    print(f"       {final_path}")


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect_demonstrations(args: argparse.Namespace) -> None:
    """Collect successful demonstrations for one BDDL task using one policy."""
    policy_module = load_policy(args.policy)

    bddl_path = Path(args.bddl_file)
    bddl_base = bddl_path.stem

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    final_hdf5_path = output_folder / f"{bddl_base}_demo.hdf5"

    problem_info = BDDLUtils.get_problem_info(str(bddl_path))
    problem_name = problem_info["problem_name"]

    env = TASK_MAPPING[problem_name](
        bddl_file_name=str(bddl_path),
        robots=["Panda"],
        controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=args.control_freq,
    )

    timestamp = int(time.time())
    tmp_dir = Path(args.tmp_folder) / f"tmp_{bddl_base}_{timestamp}"
    tmp_dir.parent.mkdir(parents=True, exist_ok=True)

    env = DataCollectionWrapper(env, str(tmp_dir))

    success_count = 0
    remove_directory = []

    print("\nStarting collection")
    print(f"Task:       {bddl_base}")
    print(f"Policy:     {args.policy}")
    print(f"Requested:  {args.num_demos} demos")
    print(f"Output:     {final_hdf5_path}")

    while success_count < args.num_demos:
        env.reset()

        try:
            policy_module.run_solver(env, str(bddl_path))

            if env._check_success():
                success_count += 1
                print(f" TRUE SUCCESS: {success_count}/{args.num_demos}")
            else:
                print(" DISCARDED: environment success check failed")

                if hasattr(env, "ep_directory"):
                    remove_directory.append(Path(env.ep_directory).name)

        except Exception as exc:
            print(f" ERROR: {exc}")

            if hasattr(env, "ep_directory"):
                remove_directory.append(Path(env.ep_directory).name)

    if success_count > 0:
        gather_demonstrations_as_hdf5(
            directory=tmp_dir,
            final_path=final_hdf5_path,
            args=args,
            problem_info=problem_info,
            remove_directory=remove_directory,
        )

    env.close()

    if not args.keep_tmp and tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    print("\nCollection finished.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect LIBERO demonstrations using a scripted policy."
    )

    parser.add_argument(
        "--bddl-file",
        type=str,
        required=True,
        help="Path to the BDDL task file.",
    )

    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        help="Policy module name inside policies/, without .py extension.",
    )

    parser.add_argument(
        "--num-demos",
        type=int,
        default=1,
        help="Number of successful demonstrations to collect.",
    )

    parser.add_argument(
        "--output-folder",
        type=str,
        default="raw_demos",
        help="Folder where the final HDF5 file will be saved.",
    )

    parser.add_argument(
        "--tmp-folder",
        type=str,
        default="tmp",
        help="Folder for temporary DataCollectionWrapper files.",
    )

    parser.add_argument(
        "--control-freq",
        type=int,
        default=5,
        help="Control frequency for collection and replay environments.",
    )

    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep temporary raw npz files.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    collect_demonstrations(parse_args())