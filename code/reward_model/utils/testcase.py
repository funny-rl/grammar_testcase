import re
import os 
from typing import List, Tuple, Dict
import timeout_decorator
from pathos.multiprocessing import ProcessingPool as Pool

from reward_model.utils.grammar.counting_context_free_grammar import CountingContextFreeGrammar as Ccfg

import warnings
warnings.filterwarnings("ignore")

# PUBLIC_TESTCASE_DIR = os.getenv("PUBLIC_TESTCASE_DIR")
# if PUBLIC_TESTCASE_DIR is None:
#     raise ValueError("Environment variable PUBLIC_TESTCASE_DIR is not set")

def is_invalid_terminal(s: str) -> bool:
    return bool(
        any(
            re.search(pattern, s)
            for pattern in [
                r"\[[^\]]+\]\+",
                r"\\d\+",
                r"\\w\+",
                r"\\s\+",
                r"\\b",
                r"\\S",
                r"\\W",
            ]
        )
    )

def get_testcase(
    ccfg: Ccfg,
    timeout: int,
    min_degree: int,
    num_testcase: int,
) -> list[tuple[str, int]]:
    @timeout_decorator.timeout(timeout) 
    def _generate(degree: int) -> str:
        return ccfg.generate(degree=degree)  
    if min_degree == -1:
        assert num_testcase == 1
        val = _generate(-1)
        if is_invalid_terminal(val):
            raise ValueError(f"Invalid testcase generated: {val}")
        return [(val, -1)]
    
    degree = min_degree
    degrees = [degree] * num_testcase
    def _generate_parallel(degree: int) -> Tuple[str, int]:
        while True:
            try:
                val = _generate(degree)
                if is_invalid_terminal(val):
                    raise ValueError(f"Invalid testcase generated: {val}")
                
                return (val, degree)
            except TimeoutError as e:
                if degree >= 2:
                    raise e
                degree += 1
    
    with Pool() as pool:
        testcases = pool.uimap(_generate_parallel, degrees)
    return testcases


def get_testcases(
        data: dict[str, str],
        num_testcase: int,
        timeout: int, 
    ) ->  List[str]:
    
    productions = data["productions"]
    constraints = data["constraints"]
    
    # Raise error if any regex production contains a '+' quantifier
    for prod in productions:
        if re.search(r"\[[^\]]+\]\+", prod) or re.search(r"\\[dws]\+", prod): # 
            raise ValueError(f"Invalid regex pattern with '+' found in production: {prod}")
        if re.search(r"\[[^\]]*\]\*", prod) or re.search(r"\\[dws]\*", prod):
            raise ValueError(f"Invalid regex pattern with '*' found in production: {prod}")
    
    ccfg = Ccfg(productions, constraints)
    testcases = []
    try:
        tuples = get_testcase(ccfg, timeout, -1, 1) # deterministic 
    except TimeoutError:
        tuples = []
    tuples += get_testcase(ccfg, timeout, 2, num_testcase-1)
    tuples += get_testcase(ccfg, timeout, 1, num_testcase)
    tuples += get_testcase(ccfg, timeout, 0, num_testcase)
    testcases: List[str] = [t[0] for t in tuples]
    return testcases

def get_tc_from_assignemt(
    grammar: Dict[str, List[str]],
    assignments: List[Dict[str, int]],
)-> List[str]:
    productions = grammar["productions"]
    constraints = grammar["constraints"]
    
    ccfg = Ccfg(productions, constraints)
    
    testcases = []
    
    for assignment in assignments:
        generated = ccfg.generate_with_assignment(assignment)
        testcases.append(generated.strip())
        
    return testcases