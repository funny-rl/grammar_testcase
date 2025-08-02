import os
import json
import traceback
import jsonlines
import subprocess
import py_compile
from tqdm import tqdm
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Tuple

from utils.get_testcases import get_testcases
from pathos.multiprocessing import ProcessingPool as Pool

def precompile_solution(solution_path: Path) -> bool:
    try:
        py_compile.compile(str(solution_path), doraise=True)
        return True
    except py_compile.PyCompileError as e:
        return False

def get_correct_solutions(
    name: str,
    correct_sol_list: List[str] = None,
) -> List[Path]:
    if correct_sol_list is not None:
        correct_solutions: List[Path] = []
        solutions_dir = Path(CORRECT_SOLUTIONS_DIR) / name
        for file in correct_sol_list:
            file_path: Path = solutions_dir / file # full path of file
            correct_solutions.append(file_path)
        
    elif correct_sol_list is None:
        solutions_dir: Path = Path(CORRECT_SOLUTIONS_DIR) / name
        correct_solutions: List[Path] = list(solutions_dir.glob("*.py"))
    
    correct_sol_paths: List[Path] = []

    with Pool(processes=cpu_count) as pool:
        compiled_flags: List[bool] = list(pool.map(precompile_solution, correct_solutions))
        for idx, sol_path in enumerate(correct_solutions):
            if compiled_flags[idx]:
                correct_sol_paths.append(sol_path)

    if len(correct_sol_paths) < min_correct_solutions:
        raise ValueError(f"Not enough correct solutions found for {name}. \
            Expected at least {min_correct_solutions}, found {len(correct_sol_paths)}.")
    return correct_sol_paths

def get_incorrect_solutions(
    name: str,
    incorrect_sol_list: List[str] = None,
) -> List[Path]:
    if incorrect_sol_list is not None:
        incorrect_solutions: List[Path] = []
        solutions_dir = Path(INCORRECT_SOLUTIONS_DIR) / name
        for file in incorrect_sol_list:
            file_path: Path = solutions_dir / file # full path of file
            incorrect_solutions.append(file_path)
    elif incorrect_sol_list is None:
        solutions_dir: Path = Path(INCORRECT_SOLUTIONS_DIR) / name
        incorrect_solutions: List[Path] = list(solutions_dir.glob("*.py"))

    incorrect_sol_paths: List[Path] = []

    with Pool(processes=cpu_count) as pool:
        compiled_flags: List[bool] = list(pool.map(precompile_solution, incorrect_solutions))
        for idx, sol_path in enumerate(incorrect_solutions):
            if compiled_flags[idx]:
                incorrect_sol_paths.append(sol_path)

    if len(incorrect_sol_paths) < 1:
        raise ValueError(f"Not enough incorrect solutions found for {name}. \
            Expected at least 1, found {len(incorrect_sol_paths)}.")
    return incorrect_sol_paths

def exe_code(args: Tuple[str, int, Path]) -> Any:
    testcase, code_timeout, sol_path = args
    python_file: str = sol_path.name
    try:
        process = subprocess.run(
            ["python", sol_path],
            capture_output=True,
            input=testcase,
            timeout=code_timeout,
            text=True,
            check=True,
        )
        if process.returncode != 0:
            return (python_file, None)
        else:
            return (python_file, " ".join(process.stdout.split()).lower())
        
    except Exception as e:
        return (python_file, None)
    
def filtering(data: Dict[str, Any]) -> None:
    name: str = data["name"]
    grammar: Dict[str, List[str]] = data["grammar"]
    code_timeout: int = 2 * timeout_dict[name] # Adjusted timeout for code generation

    testcases = get_testcases(
        data=grammar,
        num_testcase=num_testcase,
        timeout=timeout_sample_tc
    )
    
    correct_sol_paths = get_correct_solutions(
        name=name,
        correct_sol_list=data.get("correct_solutions", None),
    )
    # incorrect_sol_paths = get_incorrect_solutions(
    #     name=name,
    #     incorrect_sol_list=data.get("incorrect_solutions", None),
    # )
    
    total_correct_solution_set = set()
    error_correct_solution_set = set()
    #filtered_incorrect_solution_set = set()
    
    for testcase in testcases:
        correct_all_args: List[Tuple[str, int, Path]] = [
            (testcase, code_timeout, sol_path) for sol_path in correct_sol_paths
        ]

        # incorrect_all_args: List[Tuple[str, int, Path]] = [
        #     (testcase, code_timeout, sol_path) for sol_path in incorrect_sol_paths
        # ]
        
        with Pool(processes=cpu_count) as pool:
            correct_outputs: List[Any] = list(
                pool.map(exe_code, correct_all_args)
            )
            filtered_correct_outputs = [pair for pair in correct_outputs if pair[1] is not None]
            total_correct_solution_set.update([pair[0] for pair in filtered_correct_outputs])
            
        # with Pool(processes=cpu_count) as pool:
        #     incorrect_outputs: List[Any] = list(
        #         pool.map(exe_code, incorrect_all_args)
        #     )
            
        outputs = [out[1] for out in filtered_correct_outputs]
        # most common output: str | common_count: int
        common_output, common_count = Counter(outputs).most_common(1)[0] 
            
        error_correct_solution_set.update(
            [pair[0] for pair in filtered_correct_outputs if pair[1] != common_output]
        )
        
        if only_correct_filtering:
            pass
        # else:
            # for file_name ,output in incorrect_outputs:
            #     if output != common_output:
            #         data["testcase"] = testcase
            #         data["incorrect_file_name"] = file_name
            #         

    data["correct_solutions"] = list(total_correct_solution_set - error_correct_solution_set)
    data["correct_common_ratio"] = round(len(data["correct_solutions"]) / len(total_correct_solution_set), 3)
    data["need_check"] = data["correct_common_ratio"] < need_check_ratio

    if len(data["correct_solutions"]) >= min_correct_solutions:
        with open(output_data, 'a', encoding='utf-8') as outfile:
            json.dump(data, outfile)
            outfile.write('\n')

def main():
    data_list: List[Dict[str, Any]] = []
    
    with open(input_data, "r", encoding="utf-8") as input_file:
        for line in input_file:
            data: Dict[str, Any] = json.loads(line.strip())
            data_list.append(data)
    
    for data in tqdm(data_list, desc="Processing data samples"):
        try:
            filtering(data)
        except Exception as e:
            print(f"Error processing {data['name']}: {e}")
            traceback.print_exc()
            pass

if __name__ == "__main__":
    cpu_count = os.cpu_count()
    need_check_ratio: float = 0.8
    num_testcase: int = 10
    timeout_sample_tc = 10
    min_correct_solutions: int = 10
    timeout_dict: dict[str, int] = {}
    
    SOLUTIONS_DIR = "./reward_model/data/solutions"
    CORRECT_SOLUTIONS_DIR = f"{SOLUTIONS_DIR}/solutions"
    INCORRECT_SOLUTIONS_DIR = f"{SOLUTIONS_DIR}/incorrect_solutions"
    PUBLIC_TESTCASE_DIR = "./reward_model/data/testcase/code-contest/public/test.jsonl"
    
    only_correct_filtering: bool = True
    
    public_dataset = jsonlines.open(PUBLIC_TESTCASE_DIR, 'r') 
    with jsonlines.open(PUBLIC_TESTCASE_DIR, 'r') as public_dataset:
        for data in public_dataset:
            timeout_dict[data["name"]] = max(
                1,
                int(
                    data["time_limit"]["seconds"] + data["time_limit"]["nanos"] / 1e9
                )
            )
    input_data: str = "./data/data_preprocess/filter4/filter1.jsonl"
    output_data: str = "./data/data_preprocess/filter5/filter1.jsonl"

    with open(output_data, 'w', encoding='utf-8') as f: 
        pass  
    
    main()