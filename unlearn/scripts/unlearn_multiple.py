"""

unlearn_multiple.py

Unlearns multiple questions from the dataset in varying group sizes (1, 5, 10, 20).

"""

import random
import time
import subprocess
from dataclasses import dataclass

@dataclass(frozen=True)
class Consts:
    rank: int = 8
    epochs: int = 600
    sleep_time: int = 5
    total_questions: int = 3600
    chunk_size: int = 20
    num_runs: int = 12


def get_chunk_indices(index, chunk_size):
    """Returns list of all indices in the chunk containing the given index."""
    start = (index // chunk_size) * chunk_size
    return list(range(start, start + chunk_size))


def main():
    consts = Consts()
    random_indices = random.sample(range(consts.total_questions), consts.num_runs)  

    for idx in random_indices:
        chunk_indices = get_chunk_indices(idx, consts.chunk_size)
        others = [i for i in chunk_indices if i != idx]

        group_1 = sorted([idx])
        group_5 = sorted(random.sample(others, 4) + [idx])
        group_10 = sorted(random.sample(others, 9) + [idx])
        group_20 = sorted(chunk_indices)

        for group in [group_5, group_10, group_20]:

            indices_str = "-".join(map(str, group))
            print(f"Running with indices={indices_str} and rank={RANK} for {EPOCHS} epochs...")
            subprocess.run([
                "python", "main.py",
                "--indices", indices_str,
                "--epochs", str(EPOCHS),
                "--dataset", "tofu",
                "--rank", str(RANK)
            ])
            print(f"Finished run with indices={indices_str}. Waiting {SLEEP_TIME} seconds...")
            time.sleep(SLEEP_TIME)

    print("All runs complete.")


if __name__ == "__main__":
    main()
