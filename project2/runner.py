import importlib
import os
import sys
import time
# TODO: remove tqdm
from tqdm import tqdm


def run_fold(fold_data, code_path):
    os.chdir(fold_data)
    start_time = time.time()

    try:
        module = importlib.import_module(code_path)
        module.main()
        # os.system(f"python3 '{code_path}'")

    except Exception as e:
        print(f"Error running {code_path} in {fold_data}: {e}")

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time for {fold_data}: {execution_time:.2f} seconds")


def main(code_script="mymain"):
    folds = [f"fold_{i}" for i in range(1, 11)]

    cwd = os.getcwd()

    project_data_path = os.path.join(cwd, "Proj2_Data")

    for fold in tqdm(folds, desc="Running Project 2"):
        fold_data = os.path.join(project_data_path, fold)

        run_fold(fold_data, code_script)

        print()

    print("All folds processed!")


if __name__ == "__main__":
    code_script = "mymain.py" if len(sys.argv) < 1 else sys.argv[1]
    main(code_script=code_script)
