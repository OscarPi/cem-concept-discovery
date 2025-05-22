import argparse
import random
from pathlib import Path
import pickle

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-train", 
        type=int,
        required=True,
        help="Number of training examples.")
    parser.add_argument(
        "--n-test", 
        type=int,
        required=True,
        help="Number of test examples.")
    parser.add_argument(
        "--n-val", 
        type=int,
        required=True,
        help="Number of validation examples.")
    parser.add_argument(
        "--output", 
        type=Path,
        required=True,
        help="File to store dataset info.")
    return parser.parse_args()

def generate_examples(n):
    examples = []
    for _ in range(n):
        [shape_colour, background_colour] = random.sample(["red", "green", "blue", "purple"], k=2)
        n_obstructions = range(random.randint(500, 1000))
        examples.append({
            "shape": random.choice(["square", "circle", "triangle", "hexagon"]),
            "shape_colour": shape_colour,
            "background_colour": background_colour,
            "obstructions_n_sides": random.randint(3, 7),
            "obstructions_radius": random.randint(5, 10),
            "obstructions_centres": [(random.randint(0, 255), random.randint(0, 255)) for _ in n_obstructions],
            "obstructions_rotations": [random.randint(0, 360) for _ in n_obstructions]

        })
    return examples

if __name__ == "__main__":
    args = parse_arguments()

    train_examples = generate_examples(args.n_train)
    test_examples = generate_examples(args.n_test)
    val_examples = generate_examples(args.n_val)
    dataset = {
        "train": train_examples,
        "test": test_examples,
        "val": val_examples
    }

    with args.output.open("wb") as f:
        pickle.dump(dataset, f)
