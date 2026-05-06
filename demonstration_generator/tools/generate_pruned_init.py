import argparse
from pathlib import Path

import numpy as np
import torch

from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv


def generate_pruned_init_states(
    benchmark_name: str,
    task_id: int,
    num_states: int,
    settle_steps: int,
    save_dir: Path,
    render_gpu_device_id: int,
) -> None:
    """
    Generate LIBERO pruned initial states for a benchmark task.

    These states are saved as:
        <task_name>.pruned_init

    They can then be copied into:
        LIBERO/libero/libero/init_files/<benchmark_name>/
    """
    benchmark_instance = get_benchmark(benchmark_name)()
    task = benchmark_instance.get_task(task_id)
    bddl_file = benchmark_instance.get_task_bddl_file_path(task_id)

    print(f"Starting generation for task: {task.name}")
    print(f"Benchmark: {benchmark_name}")
    print(f"BDDL file: {bddl_file}")

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        render_gpu_device_id=render_gpu_device_id,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        horizon=500,
        control_freq=20,
    )

    init_states = []
    dummy_action = np.zeros(7)

    for index in range(num_states):
        env.reset()

        # Let physics settle before saving the state.
        for _ in range(settle_steps):
            env.step(dummy_action)

        state = env.sim.get_state().flatten()
        init_states.append(state)

        if (index + 1) % 5 == 0 or index == 0:
            print(f"Generated state {index + 1}/{num_states}")

    env.close()

    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{task.name}.pruned_init"

    torch.save(init_states, save_path)

    print(f"\n[SUCCESS] Saved {len(init_states)} states to:")
    print(f"          {save_path}")
    print("\nCopy it to:")
    print(f"LIBERO/libero/libero/init_files/{benchmark_name}/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pruned initial states for a LIBERO benchmark task."
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        default="libero_custom",
        help="LIBERO benchmark name.",
    )

    parser.add_argument(
        "--task-id",
        type=int,
        default=0,
        help="Task index inside the benchmark.",
    )

    parser.add_argument(
        "--num-states",
        type=int,
        default=50,
        help="Number of initial states to generate.",
    )

    parser.add_argument(
        "--settle-steps",
        type=int,
        default=50,
        help="Number of no-op steps after reset before saving each state.",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="pruned_init",
        help="Folder where the .pruned_init file will be saved.",
    )

    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU device ID for offscreen rendering.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    generate_pruned_init_states(
        benchmark_name=args.benchmark,
        task_id=args.task_id,
        num_states=args.num_states,
        settle_steps=args.settle_steps,
        save_dir=Path(args.save_dir),
        render_gpu_device_id=args.gpu_id,
    )