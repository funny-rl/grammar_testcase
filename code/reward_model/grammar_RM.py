
import ast
import time

from reward_model.utils.utils import extract_solution
from reward_model.utils.testcase import get_testcases
from reward_model.utils.effectiveness import get_effectiveness

def compute_score(
  data_source,
  solution_str,
  ground_truth,
  extra_info=None,
):
    solution = extract_solution(solution_str=solution_str)
    
    if solution is None:
      return -0.5
    
    total_reward = 0.0
    num_testcase = 5
    tc_timeout = 10
    
    start_time = time.time()
    try:
      grammar = ast.literal_eval(solution)
      print("[Grammar] ", grammar)
      
      testcases = get_testcases(
          data=grammar,
          num_testcase=num_testcase,
          timeout=tc_timeout,
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
      total_reward = validity * effectiveness
    except Exception as e:
      print(f"Error parsing solution string: {solution}, Error: {e}")
      return total_reward

    end_time = time.time()
    print(f"Name {ground_truth['name']} | Total Reward {total_reward} | n_correct_solution: {n_correct_solution} | n_incorrect_solution: {n_incorrect_solution} | Time taken: {end_time - start_time:.3f} seconds")
    print("=" * 100)
    return total_reward