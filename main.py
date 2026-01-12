# main.py
# Entry point to run all GPredict examples

import numpy as np

# Set global random seed for reproducibility
np.random.seed(42)

from examples import run_sinusoidal_example, run_linear_example, run_quadratic_example

def main():
    # List of examples: (display name, function)
    examples = [
        ("Sinusoidal", run_sinusoidal_example),
        ("Linear", run_linear_example),
        ("Quadratic", run_quadratic_example)
    ]

    # Run each example sequentially
    for name, func in examples:
        print(f"\n==== Running {name} example ====")
        func()  # execute the GP example
        print(f"==== {name} example completed ====\n")

if __name__ == "__main__":
    main()
