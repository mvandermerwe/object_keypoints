import os
import argparse
from typing import List

import numpy as np


def save_split(filename: str, examples: List[str]):
    with open(filename, 'w') as f:
        f.write("\n".join(examples))


def split_dataset(dataset_dir: str, train_split: float, val_split: float, test_split: float):
    assert (train_split + val_split + test_split) == 1.0, "Splits should add to 1.0!"

    examples = np.array([f for f in os.listdir(dataset_dir) if '.npy' in f])

    num_examples = len(examples)
    num_train = int(num_examples * train_split)
    num_val = int(num_examples * val_split)
    num_test = num_examples - num_train - num_val

    example_ids = np.arange(num_examples, dtype=int)
    np.random.shuffle(example_ids)

    if not os.path.exists(os.path.join(dataset_dir, 'splits')):
        os.makedirs(os.path.join(dataset_dir, 'splits'))

    save_split(os.path.join(dataset_dir, 'splits', 'train.txt'), examples[example_ids[:num_train]])
    save_split(os.path.join(dataset_dir, 'splits', 'val.txt'), examples[example_ids[num_train:num_train + num_val]])
    save_split(os.path.join(dataset_dir, 'splits', 'test.txt'), examples[example_ids[num_train + num_val:]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split provided dataset.")
    parser.add_argument('dataset_dir', type=str, help="Dataset directory.")
    parser.add_argument('--train', type=float, default=0.8, help="Train split percentage.")
    parser.add_argument('--val', type=float, default=0.1, help="Val split percentage.")
    parser.add_argument('--test', type=float, default=0.1, help="Test split percentage.")
    args = parser.parse_args()

    split_dataset(args.dataset_dir, args.train, args.val, args.test)
