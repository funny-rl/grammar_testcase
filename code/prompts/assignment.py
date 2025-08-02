import json
from transformers import AutoTokenizer

from typing import Dict, List, Any
from prompts._base_instruction import (
    get_prompt_func,
    get_incorrect_solution,    
)

def preprocess_data(file_path: str) -> List[Dict[str, Any]]:
    processed_data: List[Dict[str, Any]] = [] 
    with open(file_path, 'r', encoding='utf-8') as infile:
        for idx, line in enumerate(infile):

            data = json.loads(line.strip())
            if 'assignment' in data:
                data['assignment'] = json.dumps(data['assignment'], ensure_ascii=False)
            if 'testcase' in data:
                del data['testcase']
            if 'methods' in data:
                del data['methods']
            processed_data.append(data)
    return processed_data

 
def generate_parquet_dataset(
        model_name: str,
        data_path: str,
        tokenizer: AutoTokenizer,
        prompt_type: str
    )-> None:
    """Generate a parquet dataset from the given model and jsonl data path.
    Args:
        model_name (str): LLM model name
        data_path (str): jsonl data path
        tokenizer (AutoTokenizer): tokenize the string
    """
    def make_map_fn(split: str):
        def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
            
            grammar: str = json.dumps(example["grammar"], ensure_ascii=False)
            
            try:
                code_string: str = get_incorrect_solution(example["name"], tokenizer, grammar)
            except Exception as e:
                print("[Incorrect Code Error] ", e)
                return None
            
            chat_prompt: List[Dict[str, str]] = get_prompt_func(
                model_name = model_name,
                grammar = grammar, 
                code_string = code_string,
            )
            
            if prompt_type == "sft":
                assignment: str = example["assignment"]
                answer: str = f"The assignment for the testcases is as follows.</think>{assignment}"
                data = {
                    "prompt": chat_prompt,
                    "answer": answer,
                    "extra_info": {
                        'split': split,
                        'index': idx,
                    }
                }
            if prompt_type == "rl":
                data = {
                    "data_source": jsonl_path,
                    "prompt": chat_prompt,
                    "reward_model" : {
                        "style": "rule",
                        "ground_truth": example
                    },
                    "extra_info": {
                        'split': split,
                        'index': idx,
                    }
                }
            elif prompt_type == "inference":
                data = {
                    "prompt": chat_prompt,
                    "extra_info": {
                        'split': split,
                        'index': idx
                    }
                }
            return data
        return process_fn

    import os
    import json
    from datasets import load_dataset, DatasetDict, Dataset

    jsonl_path: str = os.path.join(data_path, 'jsonl')
    parquet_path: str = os.path.join(data_path, 'parquet')
    os.makedirs(parquet_path, exist_ok=True)
    
    if prompt_type == "sft" or prompt_type == "rl":
        train_file_name: str = os.environ.get('train_file_name', None)
        val_file_name: str = os.environ.get('val_file_name', None)
        
        assert train_file_name is not None and val_file_name is not None, \
            "Environment variables 'train_file_name' and 'val_file_name' must be set."
        
        train_file_path: str = f"{jsonl_path}/{train_file_name}.jsonl"
        val_file_path: str = f"{jsonl_path}/{val_file_name}.jsonl"
        
        processed_train_data = preprocess_data(train_file_path)
        processed_val_data = preprocess_data(val_file_path)
        
        print("\nData preprocessing completed...\n")

        dataset: DatasetDict = DatasetDict({
            train_file_name: Dataset.from_list(processed_train_data),
            val_file_name: Dataset.from_list(processed_val_data),
        })
        
        
        train_dataset = dataset[train_file_name]
        valid_dataset = dataset[val_file_name]
        
        print("\n", "="*20, "Before filtering", "="*20)
        print("Number of train data: ", len(train_dataset))
        print("Number of valid data: ", len(valid_dataset))
        print("="*50, "\n")

        train_dataset = train_dataset.map(
            function=make_map_fn(train_file_name), 
            with_indices=True
        )
        valid_dataset = valid_dataset.map(
            function=make_map_fn(val_file_name), 
            with_indices=True
        )
        
        filtered_train_dataset = train_dataset.filter(lambda x: x is not None)
        filtered_valid_dataset = valid_dataset.filter(lambda x: x is not None)

        print("\n", "="*20, "After filtering", "="*20)
        print("Number of filtered train data: ", len(filtered_train_dataset))
        print("Number of filtered valid data: ", len(filtered_valid_dataset))
        print("="*50, "\n")

        filtered_train_dataset.to_parquet(os.path.join(parquet_path, f'{train_file_name}.parquet'))
        filtered_valid_dataset.to_parquet(os.path.join(parquet_path, f'{val_file_name}.parquet'))
        
    elif prompt_type == "inference":
        test_file_name: str = os.environ.get('test_file_name', None)

        assert test_file_name is not None, \
            "Environment variables 'test_file_name' must be set."

        dataset: DatasetDict = load_dataset(
            "json",
            data_files={
                test_file_name: f"{jsonl_path}/{test_file_name}.jsonl",
            },
        )
        test_dataset = dataset[test_file_name]
        
        test_dataset = test_dataset.map(function=make_map_fn(test_file_name), with_indices=True)
        
        print("\n", "="*20, "Before filtering", "="*20)
        print("Number of test data: ", len(test_dataset))
        print("="*50, "\n")
        
        filtered_test_dataset = test_dataset.filter(lambda x: x is not None)

        print("\n", "="*20, "After filtering", "="*20)
        print("Number of filtered test data: ", len(filtered_test_dataset))
        print("="*50, "\n")

        filtered_test_dataset.to_parquet(os.path.join(parquet_path, f'{test_file_name}.parquet'))