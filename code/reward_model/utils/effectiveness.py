import re
import os
import time
from collections import Counter
import random 
import tempfile
import jsonlines
import itertools
import py_compile
import subprocess
from pathlib import Path
from itertools import chain
from tqdm import tqdm
from typing import IO, Optional, TypedDict, List, Tuple, Dict
from pathos.multiprocessing import ProcessingPool as Pool

SOLUTIONS_DIR = "reward_model/data/solutions"
CORRECT_SOLUTIONS_DIR = f"{SOLUTIONS_DIR}/solutions"
INCORRECT_SOLUTIONS_DIR = f"{SOLUTIONS_DIR}/incorrect_solutions"
PUBLIC_TESTCASE_DIR = "reward_model/data/testcase/code-contest/public/test.jsonl"

n_samples = 10
timeout_dict: Dict[str, int] = {}

public_dataset = jsonlines.open(PUBLIC_TESTCASE_DIR, 'r')
with jsonlines.open(PUBLIC_TESTCASE_DIR, 'r') as public_dataset:
    for data in public_dataset:
        timeout_dict[data["name"]] = max(
            1,
            int(
                data["time_limit"]["seconds"] + data["time_limit"]["nanos"] / 1e9
            )
        )
        
def precompile_solution(solution_path: Path) -> bool:
    try:
        py_compile.compile(str(solution_path), doraise=True)
        return True
    except py_compile.PyCompileError as e:
        return False
        

def sample_solutions(
    name: str,
    num_sample: int,
    correctness: str = "correct",
    correct_solution_set: List[str] = None,
) -> Tuple[List[Path], int]:
    """
    return only the Compilable solutions.
    if num_sample is larger than the number of compilable solutions, 
    it will return only compilable solutions.
    """
    if correctness == "correct":
        solutions_dir = Path(CORRECT_SOLUTIONS_DIR) / name
        solutions: List[Path] = [Path(solutions_dir / p) for p in correct_solution_set]
    elif correctness == "incorrect":
        solutions_dir = Path(INCORRECT_SOLUTIONS_DIR) / name
        solutions: List[Path] = list(solutions_dir.glob("*.py"))
    else:
        raise ValueError("Correctness must be either 'correct' or 'incorrect'")
    
    if len(solutions) < 1:
        raise ValueError(f"No solutions found for {name} in {correctness} directory")
    total_sample = min(num_sample, len(solutions))
    
    tried_solutions = set()
    n_sample:int = total_sample
    complied_solutions = []
    
    sol_set = set(solutions)
    with Pool() as compile_pool:
        while len(complied_solutions) < total_sample:
            untried_solutions = list(sol_set - tried_solutions)
            if len(untried_solutions) < 1:
                break
            if len(untried_solutions) < n_sample:
                n_solution = len(untried_solutions)
            else:
                n_solution = n_sample
            sampled_solution = random.sample(untried_solutions, n_solution)    
            compiled_flags = list(compile_pool.map(precompile_solution, sampled_solution))
            tried_solutions.update(sampled_solution)
            for idx, sol in enumerate(sampled_solution):
                if compiled_flags[idx]:
                    complied_solutions.append(sol)
                    n_sample -= 1

    if len(complied_solutions) < 1:
        raise ValueError(f"No compilable solutions found for {name} in {correctness} directory")
    
    return complied_solutions

def get_stdout(
    python_file: Path, 
    testcase_str: str, 
    timeout: int
) -> Optional[str]:
    try:
        process = subprocess.run(
            ["python", str(python_file)],
            capture_output=True,
            input=testcase_str,
            timeout=timeout,
            text=True,
            check=True,
        )
        if process.returncode != 0:
            return None
    except Exception as e:
        return None

    return " ".join(process.stdout.split()).lower()

def test_pairs(
    args: Tuple[str, Path, int]
):
    testcase, solution, timeout = args 
    return get_stdout(solution, testcase, timeout)


class ExecutionResultPerTestcase(TypedDict):
    """ExecutionResultPerTestcase is a dictionary that contains the results of
    execution for a single testcase.
    """

    correct_outputs: list[Optional[str]]
    incorrect_outputs: list[Optional[str]]
    correct_solutions: list[str]
    incorrect_solutions: list[str]
    
def get_mode(xs: list[str]) -> tuple[str, int]:
    groupby_iterable = itertools.groupby(sorted(xs))
    groups = [(k, len(list(v))) for k, v in groupby_iterable]
    groups = sorted(groups, key=lambda e: e[1], reverse=True)
    mode, num_of_mode = groups[0]
    return mode, num_of_mode
    
def summarize_and_score(results: list[ExecutionResultPerTestcase]) -> float:
    if not results:
        return 0.0

    total_incorrect_flags = [True] * len(results[0]["incorrect_outputs"])

    for r in results:
        correct_outputs = [o for o in r["correct_outputs"] if o is not None]
        if not correct_outputs:
            return 0.0
        answer, _ = get_mode(correct_outputs)
        if any(o != answer for o in r["correct_outputs"]):
            return 0.0
        for i, o in enumerate(r["incorrect_outputs"]):
            total_incorrect_flags[i] &= (o == answer)

    return 1.0 - sum(total_incorrect_flags) / len(total_incorrect_flags)
    
def get_effectiveness(
    data: Dict[str, str],
    testcases: List[Tuple[str, int]],
) -> Tuple[float, float]:
    """
    Compute the effectiveness of the test cases against the provided data.
    
    Args:
        data: Dictionary containing the ground truth data.
        testcases: List of tuples containing test case strings and their degrees.
    
    Returns:
        Tuple containing validity and effectiveness scores.
    """
    name = data["name"]
    
    timeout = timeout_dict[name]
    
    correct_solutions = sample_solutions(
        name=name,
        num_sample=n_samples,
        correctness="correct",
        correct_solution_set = data["correct_solutions"],
    )
    incorrect_solutions = sample_solutions(
        name=name,
        num_sample=n_samples,
        correctness="incorrect"
    )

    n_correct_solution = len(correct_solutions)
    n_incorrect_solution = len(incorrect_solutions)

    num_testcase = len(testcases)
    all_solutions = list(chain(correct_solutions, incorrect_solutions))
    tc_sol_pairs = list(itertools.product(testcases, all_solutions))
    all_args = [(*pair, timeout) for pair in tc_sol_pairs]
    
    with Pool() as test_tc:
        test_result = test_tc.map(
            test_pairs,
            all_args,
        )
        test_result = list(test_result)
        
    assert len(test_result) == num_testcase * len(all_solutions), \
        f"Expected {num_testcase * len(all_solutions)} results, got {len(test_result)}"
        
    n_solution = n_correct_solution + n_incorrect_solution
    testcase_outputs = [test_result[i:i + n_solution] for i in range(0, len(test_result), n_solution)]
    
    assert len(testcase_outputs) == num_testcase, \
        f"Length of correct outputs and incorrect outputs must match: {len(testcase_outputs)} != {num_testcase}"

    results = []
    for testcase_output in testcase_outputs:
        tc_correct_output = testcase_output[:n_correct_solution]
        tc_incorrect_output = testcase_output[n_correct_solution:]
        
        assert len(tc_correct_output) == n_correct_solution, \
            f"Length of correct outputs must match: {len(tc_correct_output)} != {n_correct_solution}"
        assert len(tc_incorrect_output) == n_incorrect_solution, \
            f"Length of incorrect outputs must match: {len(tc_incorrect_output)} != {n_incorrect_solution}"
            
        filtered_correct_outputs = [o for o in tc_correct_output if o is not None]
        print(filtered_correct_outputs)
        if not filtered_correct_outputs:
            print("[Empty Error] No valid correct outputs")
            return 0, 0, n_correct_solution, n_incorrect_solution

        threshold = len(filtered_correct_outputs) * 0.90
        counter_fco = Counter(filtered_correct_outputs)
        
        valid = any(count >= threshold for count in counter_fco.values())

        if not valid:
            print("[Mismatch Error] Multiple correct outputs found")
            print("Not Valid output: ", counter_fco)
            return 0, 0, n_correct_solution, n_incorrect_solution

        result_per_testcase = ExecutionResultPerTestcase(
            correct_outputs=tc_correct_output,
            incorrect_outputs=tc_incorrect_output,
            correct_solutions=[str(e.name) for e in correct_solutions],
            incorrect_solutions=[str(e.name) for e in incorrect_solutions],
        )
        results.append(result_per_testcase)

    effectiveness = summarize_and_score(results)
    return 1.0, effectiveness, n_correct_solution, n_incorrect_solution