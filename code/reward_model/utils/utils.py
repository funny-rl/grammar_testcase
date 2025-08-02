import re

def extract_solution(solution_str: str) -> str | None:
    think_token_occurrences = re.findall(r'</think>', solution_str)
    if len(think_token_occurrences) != 1:
        return None
    match = re.search(r'</think>(.+)', solution_str)
    if match and match.group(1).strip():
        return match.group(1)
    return None