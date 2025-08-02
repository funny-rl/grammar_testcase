import jsonlines
from pathlib import Path
from typing import Any
from tqdm import tqdm
import signal

from reward_model.utils.grammar.counting_context_free_grammar import CountingContextFreeGrammar as Ccfg  

class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Testcase generation timeout")


def generate_fixed_testcase(data: dict[str, Any], timeout_sec: int = 5) -> dict[str, Any]:
    grammar_data = data.get("grammar", {})
    productions = grammar_data.get("productions", [])
    constraints = grammar_data.get("constraints", [])

    raw_assignment_list = data.get("assignment", [])
    ccfg = Ccfg(productions, constraints)

    testcases = []
    try:
        signal.signal(signal.SIGALRM, timeout_handler)  
        signal.alarm(timeout_sec)  

        if isinstance(raw_assignment_list, list):
            for assignment in raw_assignment_list:
                generated = ccfg.generate_with_assignment(assignment)
                testcases.append(generated.strip())
        else:
            generated = ccfg.generate_with_assignment(raw_assignment_list)
            testcases.append(generated.strip())

        signal.alarm(0)  # Cancel alarm
        data["testcase"] = testcases

    except TimeoutException as te:
        data["testcase"] = []
        data["error"] = f"Generation timeout: {str(te)}"
    except Exception as e:
        data["testcase"] = []
        data["error"] = f"Generation failed: {str(e)}"
    finally:
        signal.alarm(0)  # Ensure alarm is off

    return data


def main(input_path: Path, output_path: Path) -> None:
    with jsonlines.open(input_path, mode="r") as reader, jsonlines.open(output_path, mode="w") as writer:
        for i, data in enumerate(tqdm(reader, desc="Generating testcases")):
            try:
                result = generate_fixed_testcase(data)
                writer.write(result)
            except Exception as e:
                data["testcase"] = []
                data["error"] = f"Processing failed: {str(e)}"
                writer.write(data)

            if i % 10 == 0:
                # Log progress every 10 testcases
                print(f"Processed {i} testcases...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Path to input .jsonl file")
    parser.add_argument("--output", type=Path, required=True, help="Path to output .jsonl file")
    args = parser.parse_args()

    main(args.input, args.output)
