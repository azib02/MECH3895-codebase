## Main Pipeline
The main scripts are:
# Demonstration Generator

This folder contains the scripts used to test scripted LIBERO policies, collect robot demonstrations, restructure the collected HDF5 files, inspect the generated data, and remove bad demonstrations.

It is part of the project:

**Beyond LIBERO: LLM-Augmented Data for VLAs**

The purpose of this folder is to generate demonstration datasets for custom or augmented LIBERO BDDL tasks. These demonstrations can then be used for training, evaluation, or later conversion into another dataset format such as RLDS.

---

## Folder Structure

```text
demonstration_generator/
├── README.md
├── requirements.txt
├── .gitignore
│
├── run_collection.py
├── restructure_dataset.py
│
├── policies/
│   ├── __init__.py
│   └── libero_10/
│       ├── __init__.py
│       └── turn_on_stove_put_moka_pot.py
│
├── tools/
│   ├── __init__.py
│   ├── generate_pruned_init.py
│   ├── check_raw.py
│   ├── check_regenerated.py
│   ├── extract_frames.py
│   ├── delete_demos.py
│   └── test_and_record_policy.py
│
├── input_bddl/
├── raw_demos/
├── processed_demos/
├── pruned_init/
├── videos/
└── tmp/
```

---

## Main Pipeline

The overall demonstration generation process is:

```text
BDDL task file
   ↓
scripted policy
   ↓
raw HDF5 demonstrations
   ↓
restructured HDF5 demonstrations
   ↓
inspection and filtering
   ↓
final dataset for training or conversion
```

The main scripts are:

```text
run_collection.py          Collects successful robot demonstrations
restructure_dataset.py     Converts raw HDF5 files into trainer-ready HDF5 format
tools/                     Contains testing, checking, frame extraction, and cleanup tools
policies/                  Contains scripted robot policies
```

---

## Requirements

This repository assumes that the following are already installed and working:

```text
LIBERO
robosuite
MuJoCo
h5py
NumPy
OpenCV
Pillow
tqdm
PyTorch
```

Install the local Python requirements with:

```bash
pip install -r requirements.txt
```

Run all scripts from the root of this folder:

```bash
cd demonstration_generator
```

---

## 1. Add BDDL Files

Place the BDDL task files inside:

```text
input_bddl/
```

Example:

```text
input_bddl/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl
```

These BDDL files define the LIBERO tasks that the scripted policies will solve.

---

## 2. Add Scripted Policies

Each scripted policy should go inside the `policies/` folder.

For example:

```text
policies/libero_10/turn_on_stove_put_moka_pot.py
```

Each policy file must contain a function called:

```python
def run_solver(env, bddl_file=None):
    ...
```

The environment is created by `run_collection.py` or `tools/test_and_record_policy.py`.

The policy file should only control the robot. It should not create its own environment, reset the environment, or close the environment.

A policy should follow this basic structure:

```python
def run_solver(env, bddl_file=None):
    # Move robot
    # Grasp objects
    # Place objects
    # Complete task

    return True
```

The policy can use helper functions inside the same file, such as:

```python
move_to_smooth(...)
gripper_action(...)
get_matrix(...)
```

---

## 3. Test a Policy and Record a Video

Before collecting demonstrations, test the policy visually.

Use:

```bash
python tools/test_and_record_policy.py \
  --bddl-file input_bddl/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl \
  --policy libero_10.turn_on_stove_put_moka_pot \
  --output videos/test_result.mp4
```

This creates a video of the policy rollout.

It does **not** create an HDF5 dataset.

This is useful for checking whether the robot is actually completing the task before collecting demonstrations.

---

## 4. Collect Raw Demonstrations

Use `run_collection.py` to collect successful demonstrations.

Example:

```bash
python run_collection.py \
  --bddl-file input_bddl/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl \
  --policy libero_10.turn_on_stove_put_moka_pot \
  --num-demos 1
```

The policy name refers to the Python file inside the `policies/` folder.

This command:

```bash
--policy libero_10.turn_on_stove_put_moka_pot
```

loads:

```text
policies/libero_10/turn_on_stove_put_moka_pot.py
```

Raw demonstrations are saved to:

```text
raw_demos/
```

Temporary data is saved to:

```text
tmp/
```

Failed demonstrations are ignored. Only successful demonstrations are written into the final HDF5 file.

---

## 5. Raw HDF5 Layout

The raw demonstration files produced by `run_collection.py` usually have this structure:

```text
data/
└── demo_0/
    ├── image
    ├── wrist_image
    ├── actions
    ├── states
    ├── robot_states
    ├── EEF_state
    ├── joint_states
    └── gripper_state
```

The important datasets are:

```text
image           Agentview camera frames
wrist_image     Eye-in-hand camera frames
actions         Robot action commands
states          Full simulator states
robot_states    Robot state information
EEF_state       End-effector position and orientation
joint_states    Robot joint positions
gripper_state   Gripper state
```

---

## 6. Restructure Raw Demonstrations

After collecting raw demonstrations, restructure them into a trainer-ready format.

Run:

```bash
python restructure_dataset.py \
  --raw-dir raw_demos \
  --target-dir processed_demos
```

This reads from:

```text
raw_demos/
```

and saves to:

```text
processed_demos/
```

---

## 7. Processed HDF5 Layout

The processed HDF5 files are structured like this:

```text
data/
└── demo_0/
    ├── obs/
    │   ├── agentview_rgb
    │   ├── eye_in_hand_rgb
    │   ├── joint_states
    │   ├── gripper_states
    │   ├── ee_states
    │   ├── ee_pos
    │   └── ee_ori
    │
    ├── actions
    ├── states
    ├── robot_states
    ├── dones
    └── rewards
```

The processed format is easier to use for training or later conversion.

This script does **not** directly convert the data to RLDS. It creates an intermediate structured HDF5 format.

---

## 8. Check Raw Demonstrations

To create replay videos from a raw HDF5 file, use:

```bash
python tools/check_raw.py \
  --file raw_demos/your_demo_file.hdf5
```

This expects the raw image path:

```text
data/demo_x/image
```

Videos are saved to:

```text
videos/
```

This is useful for quickly checking whether the collected raw demos look correct.

---

## 9. Check Processed Demonstrations

To create a replay video from a processed HDF5 file, use:

```bash
python tools/check_regenerated.py \
  --file processed_demos/your_demo_file.hdf5 \
  --demo demo_0
```

This expects the processed image path:

```text
data/demo_x/obs/agentview_rgb
```

You can choose another demo by changing the `--demo` value:

```bash
python tools/check_regenerated.py \
  --file processed_demos/your_demo_file.hdf5 \
  --demo demo_9
```

The video is saved to:

```text
videos/regenerated_replay.mp4
```

---

## 10. Extract Frames for Inspection

To inspect demonstrations quickly, extract frames from each demo.

For the last frame of each raw demo:

```bash
python tools/extract_frames.py \
  --file raw_demos/your_demo_file.hdf5 \
  --layout raw \
  --frame -1
```

For the first frame of each raw demo:

```bash
python tools/extract_frames.py \
  --file raw_demos/your_demo_file.hdf5 \
  --layout raw \
  --frame 0
```

For the last frame of each processed demo:

```bash
python tools/extract_frames.py \
  --file processed_demos/your_demo_file.hdf5 \
  --layout processed \
  --frame -1
```

Frames are saved to:

```text
videos/frames/
```

This is useful for checking whether each demo ends in the correct final state.

---

## 11. Delete Bad Demonstrations

If some demonstrations are bad, they can be removed from the HDF5 file.

Always run a dry run first:

```bash
python tools/delete_demos.py \
  --file processed_demos/your_demo_file.hdf5 \
  --demos demo_0 demo_5 \
  --dry-run
```

Then delete them for real:

```bash
python tools/delete_demos.py \
  --file processed_demos/your_demo_file.hdf5 \
  --demos demo_0 demo_5
```

You can also list demos before deleting:

```bash
python tools/delete_demos.py \
  --file processed_demos/your_demo_file.hdf5 \
  --list \
  --demos demo_0 demo_5
```

Note: deleting demos from an HDF5 file unlinks them from the file structure, but the file size may not shrink until the file is repacked.

---

## 12. Generate Pruned Initial States

For custom LIBERO benchmark tasks, pruned initial states can be generated with:

```bash
python tools/generate_pruned_init.py \
  --benchmark libero_custom \
  --task-id 0 \
  --num-states 50
```

This saves:

```text
pruned_init/<task_name>.pruned_init
```

The generated file can then be copied into the LIBERO init files folder:

```text
LIBERO/libero/libero/init_files/libero_custom/
```

These initial states help LIBERO reset tasks from valid starting states.

---

## Example Full Workflow

A typical workflow is:

```bash
# 1. Test the policy visually
python tools/test_and_record_policy.py \
  --bddl-file input_bddl/task.bddl \
  --policy libero_10.turn_on_stove_put_moka_pot \
  --output videos/test_result.mp4

# 2. Collect raw demonstrations
python run_collection.py \
  --bddl-file input_bddl/task.bddl \
  --policy libero_10.turn_on_stove_put_moka_pot \
  --num-demos 10

# 3. Restructure raw demonstrations
python restructure_dataset.py \
  --raw-dir raw_demos \
  --target-dir processed_demos

# 4. Check raw demonstration videos
python tools/check_raw.py \
  --file raw_demos/your_demo_file.hdf5

# 5. Check processed demonstration video
python tools/check_regenerated.py \
  --file processed_demos/your_demo_file.hdf5 \
  --demo demo_0

# 6. Extract last frames for quick inspection
python tools/extract_frames.py \
  --file processed_demos/your_demo_file.hdf5 \
  --layout processed \
  --frame -1

# 7. Delete bad demos if needed
python tools/delete_demos.py \
  --file processed_demos/your_demo_file.hdf5 \
  --demos demo_0 demo_5 \
  --dry-run
```

---

## Notes

- Use `.py` files, not `.pyw`.
- Run scripts from the root of `demonstration_generator/`.
- Keep large generated files out of GitHub.
- Store BDDL files in `input_bddl/`.
- Store raw demonstrations in `raw_demos/`.
- Store processed demonstrations in `processed_demos/`.
- Store generated videos in `videos/`.
- Store extracted frames in `videos/frames/`.
- Store pruned initial states in `pruned_init/`.
- Store temporary demonstration data in `tmp/`.

---

## GitHub Notes

The following folders are useful locally, but the large generated files inside them should not be committed:

```text
raw_demos/
processed_demos/
videos/
tmp/
pruned_init/
```

These folders can be kept in GitHub using empty `.gitkeep` files.

Example:

```text
raw_demos/.gitkeep
processed_demos/.gitkeep
videos/.gitkeep
tmp/.gitkeep
pruned_init/.gitkeep
```

---

## Purpose in the Project

This demonstration generator supports the wider project by producing demonstration data for augmented LIBERO tasks.

The BDDL generator creates new or modified task files.

The demonstration generator then uses scripted policies to solve those tasks and collect successful robot demonstrations.

The collected data can then be inspected, cleaned, restructured, and used for training or evaluation.