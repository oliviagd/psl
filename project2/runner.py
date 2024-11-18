import os
import time
from tqdm import tqdm

def run_fold(fold_data, code_path):
    os.chdir(fold_data)
    start_time = time.time()

    try:
        os.system(f"python3 {code_path}")
        
    except Exception as e:
        print(f"Error running {code_path} in {fold_data}: {e}")

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time for {fold_data}: {execution_time:.2f} seconds")


if __name__ == "__main__":
    folds = [f"fold_{i}" for i in range(1, 11)]

    cwd = os.getcwd()

    project_data_path = os.path.join(cwd, 'Proj2_Data')
    code_path = os.path.join(cwd, "mymain.py")

    for fold in tqdm(folds, desc='Running Project 2'):
        fold_data = os.path.join(project_data_path, fold)

        run_fold(fold_data, code_path)

        print()

    print("All folds processed!")

