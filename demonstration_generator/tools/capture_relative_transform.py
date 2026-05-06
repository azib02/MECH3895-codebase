import argparse
import select
import sys

import cv2
import numpy as np

from robosuite import load_controller_config
from robosuite.devices import Keyboard
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import VisualizationWrapper

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import TASK_MAPPING


# ==========================================================
# UNIVERSAL MATRIX UTILITY
# ==========================================================

def get_4x4_matrix(env, body_name):
    print(f"[DEBUG] Attempting to fetch matrix for: {body_name}")

    try:
        pos = env.sim.data.get_body_xpos(body_name).copy()
        mat = env.sim.data.get_body_xmat(body_name).reshape(3, 3).copy()

        transform = np.eye(4)
        transform[:3, :3] = mat
        transform[:3, 3] = pos

        return transform

    except Exception as error:
        print(f"[ERROR] Could not find body '{body_name}': {error}")
        return None


def capture_relative_transform(env):
    print("\n" + "!" * 30)
    print("PAUSED FOR MATRIX CAPTURE")
    print("!" * 30)

    all_bodies = sorted(list(env.sim.model.body_names))

    print("\n[AVAILABLE BODIES IN SCENE]:")

    for index in range(0, len(all_bodies), 3):
        row = all_bodies[index:index + 3]
        print("  " + "".join(f"{name:<35}" for name in row))

    print("-" * 60)

    child = input("CHILD body name, e.g. robot0_right_hand: ").strip()
    parent = input("PARENT body name, e.g. moka_pot_1_main: ").strip()

    child_transform = get_4x4_matrix(env, child)
    parent_transform = get_4x4_matrix(env, parent)

    if child_transform is None or parent_transform is None:
        print("Capture failed. Check body names.")
        return

    relative_transform = np.linalg.inv(parent_transform) @ child_transform

    projected_child_transform = parent_transform @ relative_transform

    actual_xyz = child_transform[:3, 3]
    projected_xyz = projected_child_transform[:3, 3]

    error_mm = np.linalg.norm(actual_xyz - projected_xyz) * 1000

    print("\n" + "=" * 60)
    print("MATRIX VALIDATION REPORT")
    print("=" * 60)
    print(f"Child body:  {child}")
    print(f"Parent body: {parent}")
    print(f"Error: {error_mm:.6f} mm")

    if error_mm < 1e-5:
        print("STATUS: VERIFIED. The relative transform maps child to parent correctly.")
    else:
        print("STATUS: FAILED. The transform check is inconsistent.")

    print("\n--- COPY-PASTE THIS MATRIX ---")
    print(np.array2string(relative_transform, separator=", "))
    print("=" * 60)


# ==========================================================
# MAIN EXECUTION
# ==========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Manually capture relative 4x4 transforms between LIBERO scene bodies."
    )

    parser.add_argument(
        "--bddl-file",
        type=str,
        required=True,
        help="Path to the BDDL file used to load the LIBERO task.",
    )

    args = parser.parse_args()

    print("[DEBUG] Loading environment...")

    problem_info = BDDLUtils.get_problem_info(args.bddl_file)

    env = TASK_MAPPING[problem_info["problem_name"]](
        bddl_file_name=args.bddl_file,
        robots=["Panda"],
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )

    env = VisualizationWrapper(env)

    device = Keyboard(pos_sensitivity=1.5, rot_sensitivity=1.0)
    device.start_control()

    env.reset()

    print("\n" + "=" * 60)
    print("UNIVERSAL RELATIVE TRANSFORM CAPTURE TOOL")
    print("=" * 60)
    print("Use the keyboard controller to move the robot.")
    print("When the robot is at the desired grasp/place pose,")
    print("click the terminal and press ENTER to capture a matrix.")
    print("=" * 60 + "\n")

    try:
        while True:
            action, _ = input2action(
                device=device,
                robot=env.robots[0],
                active_arm="right",
                env_configuration="single-arm-opposed",
            )

            if action is None:
                break

            env.step(action)
            env.render()

            if select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                print("\n[!] TERMINAL TRIGGER DETECTED")
                capture_relative_transform(env)
                print("\n>>> Resuming robot control. Press ENTER in terminal for next capture.")

            cv2.waitKey(1)

    finally:
        env.close()


if __name__ == "__main__":
    main()