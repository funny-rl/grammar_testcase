import json

from transformers import AutoTokenizer
from typing import Dict, List, Any
from tqdm import tqdm
from prompts._base_instruction import (
    get_prompt_func, 
)

def preprocess_data(file_path: str) -> List[Dict[str, Any]]:
    processed_data: List[Dict[str, Any]] = [] 
    with open(file_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line.strip())
            if 'correct_solutions' in data:
                del data['correct_solutions']
            if "correct_common_ratio" in data:
                del data["correct_common_ratio"]
            if "need_check" in data:
                del data["need_check"]
            processed_data.append(data)
    return processed_data


def generate_parquet_dataset(
        model_name: str,
        data_path: str,
        tokenizer: AutoTokenizer,
        prompt_type: str
) -> None:
    """Generate a parquet dataset from the given model and jsonl data path.
    Args:
        model_name (str): LLM model name
        data_path (str): jsonl data path
    """
    import os
    import json
    from datasets import load_dataset, DatasetDict, Dataset
    
    jsonl_path: str = os.path.join(data_path, 'jsonl')
    parquet_path: str = os.path.join(data_path, 'parquet')
    os.makedirs(parquet_path, exist_ok=True)
    
    def make_map_fn(split: str):
        def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:

                description: str = json.dumps(example["description"], ensure_ascii=False)
                chat_prompt: List[Dict[str, str]] = get_prompt_func(
                    model_name = model_name,
                    description = description, 
                )
                    
                if prompt_type == "sft":
                    answer = f"The <Grammar> for the <Description> is as follouw.</think>{json.dumps(example['grammar'], ensure_ascii=False)}"
                    data = {
                        "prompt": chat_prompt,
                        "answer": answer,
                        "extra_info": {
                            'split': split,
                            'index': idx,
                        }
                    }
                
                elif prompt_type == "rl":
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
                else:
                    raise ValueError(f"Unknown prompt type: {prompt_type}")
                return data
        return process_fn
    
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
        
        train_dataset = train_dataset.map(
            function=make_map_fn(train_file_name), 
            with_indices=True
        )
        valid_dataset = valid_dataset.map(
            function=make_map_fn(val_file_name), 
            with_indices=True
        )
        
        print("\nDataset mapping completed...\n")
        
        train_dataset.to_parquet(os.path.join(parquet_path, f'{train_file_name}.parquet'))
        valid_dataset.to_parquet(os.path.join(parquet_path, f'{val_file_name}.parquet'))
        
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
        test_dataset.to_parquet(os.path.join(parquet_path, f'{test_file_name}.parquet'))