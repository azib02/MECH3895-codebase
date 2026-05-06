import argparse
import importlib
import os
from pathlib import Path

import cv2
import robosuite as suite
from robosuite.controllers import load_controller_config

from libero.libero.envs import TASK_MAPPING
import libero.libero.envs.bddl_utils as BDDLUtils


# Headless rendering settings, useful for RunPod / servers.
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"


def load_policy(policy_name: str):
    """
    Load a policy module from the policies folder.

    Example:
        --policy libero_10.turn_on_stove_put_moka_pot

    loads:
        policies/libero_10/turn_on_stove_put_moka_pot.py
    """
    module_path = f"policies.{policy_name}"
    policy_module = importlib.import_module(module_path)

    if not hasattr(policy_module, "run_solver"):
        raise AttributeError(
            f"Policy '{module_path}' does not define a run_solver(env, bddl_file) function."
        )

    return policy_module


def record_policy_test(
    bddl_path: Path,
    policy_name: str,
    output_video: Path,
    fps: int,
    camera_name: str,
    image_size: int,
    control_freq: int,
) -> None:
    """
    Test a scripted policy and record the rollout as an MP4 video.

    This does not create HDF5 data. It is only for quick visual debugging.
    """
    policy_module = load_policy(policy_name)

    problem_info = BDDLUtils.get_problem_info(str(bddl_path))
    controller_config = load_controller_config(default_controller="OSC_POSE")

    print(f"Initialising task: {problem_info['problem_name']}")
    print(f"BDDL file: {bddl_path}")
    print(f"Policy: policies.{policy_name}")

    env = TASK_MAPPING[problem_info["problem_name"]](
        bddl_file_name=str(bddl_path),
        robots=["Panda"],
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_names=[camera_name],
        camera_heights=image_size,
        camera_widths=image_size,
        controller_configs=controller_config,
        control_freq=control_freq,
    )

    frames = []

    original_step = env.step

    def stepping_with_render(action):
        obs, reward, done, info = original_step(action)

        image_obs_key = f"{camera_name}_image"
        frame = obs[image_obs_key][::-1]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frames.append(frame_bgr)

        return obs, reward, done, info

    env.step = stepping_with_render

    try:
        env.reset()

        print("Running policy...")
        success = policy_module.run_solver(env, str(bddl_path))

        if frames:
            output_video.parent.mkdir(parents=True, exist_ok=True)

            height, width, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                str(output_video),
                fourcc,
                fps,
                (width, height),
            )

            for frame in frames:
                video_writer.write(frame)

            video_writer.release()

            print(f"Video saved to: {output_video}")
            print(f"Frames recorded: {len(frames)}")
        else:
            print("No frames were recorded.")

        env_success = env._check_success()
        print(f"Policy returned: {success}")
        print(f"Environment success check: {env_success}")

    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test a scripted policy and record the rollout as a video."
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
        help="Policy module inside policies, e.g. libero_10.turn_on_stove_put_moka_pot.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="videos/policy_test.mp4",
        help="Output video path.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Video FPS.",
    )

    parser.add_argument(
        "--camera",
        type=str,
        default="agentview",
        help="Camera name to record.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Rendered image height and width.",
    )

    parser.add_argument(
        "--control-freq",
        type=int,
        default=5,
        help="Environment control frequency.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    record_policy_test(
        bddl_path=Path(args.bddl_file),
        policy_name=args.policy,
        output_video=Path(args.output),
        fps=args.fps,
        camera_name=args.camera,
        image_size=args.image_size,
        control_freq=args.control_freq,
    )