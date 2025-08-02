import ast
import time

from reward_model.utils.utils import extract_solution
from reward_model.utils.testcase import get_tc_from_assignemt
from reward_model.utils.effectiveness import get_effectiveness
def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
)-> float:
    print("=" * 100)
    
    solution = extract_solution(solution_str=solution_str)
    if solution is None:
        return -0.5
    grammar = ground_truth["grammar"]
    start_time = time.time()
    try:
        testcases = get_tc_from_assignemt(
            grammar=grammar,
            assignments=ast.literal_eval(solution),
        )
        (
            validity,
            effectiveness,
            n_correct_solution,
            n_incorrect_solution,
        ) = get_effectiveness(
            data = ground_truth,
            testcases = testcases,
        )
        rewards = validity * effectiveness
        end_time = time.time()
        print(f"Name {ground_truth['name']} | Total Reward {rewards} | n_correct_solution: {n_correct_solution} | n_incorrect_solution: {n_incorrect_solution} | Time taken: {end_time - start_time:.3f} seconds")
        return rewards 
    
    except Exception as e:
        print(f"Error parsing solution string | Error: {e}")
        print(f"[Reward Error] {e}")
        return 0.0
    
    
    