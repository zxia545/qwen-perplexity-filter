#!/usr/bin/env python
import argparse
import json
import time
import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_utils import prepare_encodings_tmp

def read_jsonl(file_path):
    """Read a JSONL file and return a list of JSON objects."""
    items = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line.strip()))
    return items

def write_jsonl(file_path, items):
    """Write a list of JSON objects to a JSONL file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")

def detect_question_type(data_item):
    """Detect if the question is multiple choice or open-ended."""
    prompt = data_item.get("prompt", "")
    groundtruth = data_item.get("groundtruth", "")
    answer_format = data_item.get("answer_format", "")
    
    # Check if it's multiple choice based on prompt containing "Choices:" or groundtruth starting with a letter
    if answer_format == "MC":
        return "multiple_choice"
    elif answer_format == "Open-ended":
        return "open_ended"
    else:
        raise ValueError(f"Unknown answer format: {answer_format}")

# System prompt for multiple choice questions
MC_SYSTEM_PROMPT = """You are a helpful AI assistant specialized in evaluating multiple choice question answers.

You are given:
1. A multiple choice question with options (A, B, C, D, E, etc.)
2. The correct answer (groundtruth)
3. A model's response to evaluate

Please determine if the model's response is correct by:
1. Extracting the final answer choice from the model's response (look for patterns like "The answer is A", "Thus, the correct answer is: B", etc.)
2. Comparing the extracted answer with the correct answer
3. Responding with 'True' if the model's extracted answer matches the groundtruth, or 'False' if it doesn't match or cannot be clearly extracted.

Focus on identifying the model's final answer choice and comparing it with the correct answer."""

# System prompt for open-ended questions  
OPEN_SYSTEM_PROMPT = """You are a helpful AI assistant specialized in evaluating open-ended question answers.

You are given:
1. An open-ended question
2. The correct/reference answer (groundtruth)
3. A model's response to evaluate

Please determine if the model's response is correct by:
1. Analyzing the model's response for factual accuracy and completeness
2. Comparing the key information with the reference answer
3. Considering that there may be multiple valid ways to express the same concepts
4. Responding with 'True' if the response is accurate and contains the key information, or 'False' if it contains significant factual errors or missing key information.

Focus on the factual accuracy and key concepts rather than exact wording."""

def construct_eval_prompt(question_type):
    """Get system prompt based on question type."""
    if question_type == "multiple_choice":
        return MC_SYSTEM_PROMPT
    else:
        return OPEN_SYSTEM_PROMPT

def construct_eval_user_prompt(data_entry, question_type):
    """Construct user prompt for evaluation based on question type."""
    question = data_entry.get("question", data_entry.get("prompt", ""))
    model_answer = data_entry.get("llm_answer", data_entry.get("output", ""))
    correct_answer = data_entry.get("groundtruth", data_entry.get("reference_answer", ""))
    
    if question_type == "multiple_choice":
        prompt = (
            f"Question: {question}\n\n"
            f"Correct Answer: {correct_answer}\n\n"
            f"Model's Response to Evaluate:\n{model_answer}\n\n"
            "Is the model's answer correct? Respond with True if correct, False if incorrect."
        )
    else:
        prompt = (
            f"Question: {question}\n\n"
            f"Reference Answer: {correct_answer}\n\n"
            f"Model's Response to Evaluate:\n{model_answer}\n\n"
            "Is the model's answer correct? Respond with True if correct, False if incorrect."
        )
    
    return prompt

def evaluate_entry(data_entry, tokenizer, model, device, max_length=4096, mini_batch_size=12):
    """
    Evaluate one data entry using True/False candidates with confidence scores.
    """
    # Detect question type
    question_type = detect_question_type(data_entry)
    
    # Get appropriate prompts based on question type
    system_prompt = construct_eval_prompt(question_type)
    user_prompt = construct_eval_user_prompt(data_entry, question_type)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    # Use the tokenizer's chat template to get the full prompt text
    prompt_template = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Candidate responses for True/False evaluation
    candidates = ["True", "False"]
    
    # Get loss scores for each candidate
    try:
        scores = prepare_encodings_tmp(
            messages_template=prompt_template,
            categories=candidates,
            tokenizer=tokenizer,
            device=device,
            mini_batch_size=mini_batch_size,
            max_length=max_length,
            add_start_token=False,
            model=model
        )
    except Exception as e:
        raise ValueError(f"Error processing item: {e}")
    
    true_loss = scores["True"]
    false_loss = scores["False"]
    
    if scores["True"] >= scores["False"]:
        is_correct = True

    else:
        is_correct = False
    
    return is_correct, true_loss, false_loss, question_type

def process_item(data_item, tokenizer, model, device, max_length, mini_batch_size, max_retries=3):
    """
    Process one JSON data entry with retries.
    """

    
    is_correct, true_loss, false_loss, question_type = evaluate_entry(
        data_item, tokenizer, model, device, max_length, mini_batch_size
    )
    result = {
        "is_correct": is_correct,
        "true_loss": true_loss,
        "false_loss": false_loss,
        "question_type": question_type
    }
    
    # Add evaluation results to the data item
    data_item.update(result)
    return data_item

def main():
    parser = argparse.ArgumentParser(description="Evaluate model responses using True/False evaluation with confidence scores for both MC and open-ended questions.")
    parser.add_argument("--input_file", type=str, help="Path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, help="Path to the output JSONL file.")
    parser.add_argument("--input_folder", type=str, help="Path to the input folder.")
    parser.add_argument("--output_folder", type=str, help="Path to the output folder.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the evaluation model")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens for model inputs (default 4096).")
    parser.add_argument("--mini_batch_size", type=int, default=12, help="Mini-batch size for candidate evaluation (default 12).")
    parser.add_argument("--num-gpu-host-model", type=int, default=1, help="Number of GPUs to host the model (default 1).")
    args = parser.parse_args()
    
    # Validate arguments
    if args.input_folder and args.output_folder:
        # Folder mode - process all JSONL files in input folder
        if not os.path.exists(args.input_folder):
            raise ValueError(f"Input folder does not exist: {args.input_folder}")
        
        # Create output folder if it doesn't exist
        os.makedirs(args.output_folder, exist_ok=True)
        
        # Get all JSONL files in input folder
        jsonl_files = glob.glob(os.path.join(args.input_folder, "*.jsonl"))
        if not jsonl_files:
            raise ValueError(f"No JSONL files found in input folder: {args.input_folder}")
            
        print(f"Found {len(jsonl_files)} JSONL files in input folder")
        
    elif args.input_file and args.output_file:
        # Single file mode
        if not os.path.exists(args.input_file):
            raise ValueError(f"Input file does not exist: {args.input_file}")
        jsonl_files = [args.input_file]
    else:
        raise ValueError("Either provide --input_file and --output_file, or --input_folder and --output_folder")

    # Load tokenizer and model instances
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    device_list = []
    model_list = []
    num_gpu_host_model = args.num_gpu_host_model
    
    for i in range(num_gpu_host_model):
        device = f"cuda:{i}" if torch.cuda.is_available() and torch.cuda.device_count() > i else "cpu"
        device_list.append(device)
        model_instance = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype="auto").to(device)
        model_instance.config.use_cache = True
        model_list.append(model_instance)
    
    # Process each JSONL file
    total_files = len(jsonl_files)
    
    for file_idx, input_file in enumerate(jsonl_files):
        print(f"\n=== Processing file {file_idx + 1}/{total_files}: {os.path.basename(input_file)} ===")
        
        # Determine output file path
        if args.input_folder and args.output_folder:
            # Folder mode - construct output file path
            input_filename = os.path.basename(input_file)
            output_file = os.path.join(args.output_folder, input_filename)
        else:
            # Single file mode
            output_file = args.output_file
        
        # Read input JSONL file
        data_items = read_jsonl(input_file)
        processed_items = []

        # Process items with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_gpu_host_model) as executor:
            futures = []
            for idx, item in enumerate(data_items):
                assigned_idx = idx % num_gpu_host_model
                device = device_list[assigned_idx]
                model = model_list[assigned_idx]
                futures.append(
                    executor.submit(
                        process_item,
                        item,
                        tokenizer,
                        model,
                        device,
                        args.max_tokens,
                        args.mini_batch_size
                    )
                )
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {os.path.basename(input_file)}"):
                processed_items.append(future.result())
        
        # Write results for this file
        write_jsonl(output_file, processed_items)
        print(f"{len(processed_items)} items written to {output_file}")
        

if __name__ == "__main__":
    main()
