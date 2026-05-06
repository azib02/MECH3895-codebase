import numpy as np
from robosuite.utils.transform_utils import (
    mat2quat,
    quat_conjugate,
    quat_multiply,
    quat2axisangle,
)


# ---------------------------------------------------------------------------
# Smooth movement helpers
# ---------------------------------------------------------------------------

def nlerp(q1, q2, alpha):
    """Normalised linear interpolation for smooth rotations."""
    if np.dot(q1, q2) < 0:
        q2 = -q2

    q_interp = (1 - alpha) * q1 + alpha * q2
    return q_interp / np.linalg.norm(q_interp)


def get_matrix(env, name):
    """Get the world-frame pose matrix of an object/body."""
    pos = env.sim.data.get_body_xpos(name).copy()
    mat = env.sim.data.get_body_xmat(name).reshape(3, 3).copy()

    transform = np.eye(4)
    transform[:3, :3] = mat
    transform[:3, 3] = pos

    return transform


def move_to_smooth(
    env,
    target_matrix,
    offset=None,
    steps=100,
    grip=-1.0,
    pos_gain=4.0,
    rot_gain=2.0,
):
    """Smoothly move the end-effector to a target pose."""
    if offset is None:
        offset = [0, 0, 0]

    target_pos_world = target_matrix[:3, 3] + np.array(offset)
    target_quat = mat2quat(target_matrix[:3, :3])

    robot_base_pos = np.array(env.robots[0].base_pos)

    start_pos_world = env.robots[0]._hand_pos + robot_base_pos
    start_quat = env.robots[0]._hand_quat.copy()

    for step in range(1, steps + 1):
        alpha = step / steps

        interp_pos_world = start_pos_world + alpha * (
            target_pos_world - start_pos_world
        )
        interp_quat = nlerp(start_quat, target_quat, alpha)

        curr_pos_world = env.robots[0]._hand_pos + robot_base_pos
        curr_quat = env.robots[0]._hand_quat

        pos_error = interp_pos_world - curr_pos_world

        if np.dot(curr_quat, interp_quat) < 0:
            interp_quat = -interp_quat

        rel_quat = quat_multiply(interp_quat, quat_conjugate(curr_quat))
        rot_error = quat2axisangle(rel_quat)

        action = np.zeros(7)
        action[:3] = pos_error * pos_gain
        action[3:6] = rot_error * rot_gain
        action[6] = grip

        env.step(action)

        if getattr(env, "has_renderer", False):
            env.render()


def gripper_action(env, cmd, steps=40):
    """Open or close the gripper for a fixed number of steps."""
    action = np.zeros(7)
    action[6] = cmd

    for _ in range(steps):
        env.step(action)

        if getattr(env, "has_renderer", False):
            env.render()


# ---------------------------------------------------------------------------
# Captured body-to-body transforms
# ---------------------------------------------------------------------------

# Parent = white_yellow_mug_1_main, Child = robot0_right_hand
T_HAND_ON_MUG = np.array(
    [
        [-0.99065337, -0.0169295, -0.13534876, 0.07444197],
        [-0.01528217, 0.99979608, -0.01320081, 0.00105774],
        [0.13554465, -0.011009, -0.99071007, 0.16968589],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Parent = white_yellow_mug_1_main, Child = robot0_right_hand
T_PUSH_CONTACT = np.array(
    [
        [0.06618593, -0.38912362, -0.91880478, 0.15999232],
        [-0.96976129, 0.19169998, -0.15104356, 0.02805307],
        [0.23490948, 0.90101827, -0.36466918, 0.13676244],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Parent = microwave_1_main, Child = white_yellow_mug_1_main
T_MUG_IN_MICROWAVE = np.array(
    [
        [-0.02255241, 0.99947498, 0.02326273, -0.04120388],
        [-0.99939255, -0.02315673, 0.02604422, -0.04841469],
        [0.02656924, -0.02266124, 0.99939009, 0.02661956],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


# ---------------------------------------------------------------------------
# Main policy
# ---------------------------------------------------------------------------

def run_solver(env, bddl_file=None):
    """
    Scripted policy for:
    put_the_yellow_and_white_mug_in_the_microwave_and_close_it

    The environment is created by run_collection.py.
    This function only controls the robot.
    """
    print("Phase 1: Picking up the mug...")

    pick_pose = get_matrix(env, "white_yellow_mug_1_main") @ T_HAND_ON_MUG

    move_to_smooth(env, pick_pose, offset=[0, 0, 0.15], steps=100, grip=-1.0)
    move_to_smooth(env, pick_pose, offset=[0, 0, 0.00], steps=100, grip=-1.0)

    gripper_action(env, cmd=1.0, steps=40)

    move_to_smooth(env, pick_pose, offset=[0, 0, 0.20], steps=100, grip=1.0)

    print("Phase 2: Placing mug in microwave...")

    place_pose = get_matrix(env, "microwave_1_main") @ T_MUG_IN_MICROWAVE @ T_HAND_ON_MUG

    move_to_smooth(env, place_pose, offset=[0.20, 0, 0.05], steps=100, grip=1.0)
    move_to_smooth(env, place_pose, offset=[0.00, 0, 0.00], steps=100, grip=1.0)

    gripper_action(env, cmd=-1.0, steps=40)

    move_to_smooth(env, place_pose, offset=[0.15, 0, 0.05], steps=100, grip=-1.0)

    print("Phase 3: Pushing mug deeper...")

    push_pose = get_matrix(env, "white_yellow_mug_1_main") @ T_PUSH_CONTACT

    move_to_smooth(env, push_pose, offset=[0.10, 0, 0.00], steps=100, grip=-1.0)

    move_to_smooth(
        env,
        push_pose,
        offset=[-0.05, 0, 0.00],
        steps=100,
        grip=-1.0,
        pos_gain=2.0,
        rot_gain=2.0,
    )

    move_to_smooth(env, push_pose, offset=[0.20, 0, 0.15], steps=100, grip=-1.0)

    print("Mission complete.")

    return True