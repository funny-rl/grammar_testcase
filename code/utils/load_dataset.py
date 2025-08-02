from transformers import AutoTokenizer

def load_dataset(
    task: str,
    model_name: str = None,
    data_path: str = None,
    tokenizer: AutoTokenizer = None,
    prompt_type: str = "sft"
    ) -> None:
    
    if task == 'grammar':
        from prompts.grammar import generate_parquet_dataset
    elif task == 'assignment':
        from prompts.assignment import generate_parquet_dataset
    else:
        raise ValueError(f"Unknown task: {task}")
    
    generate_parquet_dataset(
        model_name=model_name,
        data_path=data_path,
        tokenizer=tokenizer,
        prompt_type=prompt_type
    )