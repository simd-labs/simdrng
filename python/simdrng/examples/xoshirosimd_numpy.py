"""XoshiroSIMD with node/gpu/thread stream levels for parallel simulations."""

import numpy as np
import simdrng


def main() -> None:
    # Thread-offset seeding: same seed, different streams
    streams = [simdrng.XoshiroSIMD(seed=42, thread=i) for i in range(4)]

    for i, rng in enumerate(streams):
        vals = rng.random(5)
        print(f"thread {i}: {vals}")

    # Gpu-offset for distributed computing (this backend has no node level)
    rng = simdrng.XoshiroSIMD(seed=42, thread=0, gpu=1)
    print(f"gpu 1: {rng.random(5)}")


if __name__ == "__main__":
    main()
