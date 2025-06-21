#!/usr/bin/env python
import argparse
import json
import time
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
    attempts = 0
    result = None
    
    while attempts < max_retries and result is None:
        # try:
        #     is_correct, true_loss, false_loss, question_type = evaluate_entry(
        #         data_item, tokenizer, model, device, max_length, mini_batch_size
        #     )
        #     result = {
        #         "is_correct": is_correct,
        #         "true_loss": true_loss,
        #         "false_loss": false_loss,
        #         "question_type": question_type
        #     }
        # except Exception as e:
        #     raise ValueError(f"Error processing item (attempt {attempts}): {e}")
        #     attempts += 1
        #     print(f"Error processing item (attempt {attempts}): {e}")
        #     time.sleep(1)
        is_correct, true_loss, false_loss, question_type = evaluate_entry(
            data_item, tokenizer, model, device, max_length, mini_batch_size
        )
        result = {
            "is_correct": is_correct,
            "true_loss": true_loss,
            "false_loss": false_loss,
            "question_type": question_type
        }
    
    if result is None:
        # Fallback values if all attempts failed
        result = {
            "is_correct": False,
            "true_loss": 999999999.0,
            "false_loss": 999999999.0,
            "question_type": "unknown"
        }
    
    # Add evaluation results to the data item
    data_item.update(result)
    data_item["eval_attempts"] = attempts + 1
    return data_item

def main():
    parser = argparse.ArgumentParser(description="Evaluate model responses using True/False evaluation with confidence scores for both MC and open-ended questions.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the evaluation model")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens for model inputs (default 4096).")
    parser.add_argument("--mini_batch_size", type=int, default=12, help="Mini-batch size for candidate evaluation (default 12).")
    parser.add_argument("--num-gpu-host-model", type=int, default=1, help="Number of GPUs to host the model (default 1).")
    args = parser.parse_args()

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
    
    # Read input JSONL file
    data_items = read_jsonl(args.input_file)
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
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing items"):
            processed_items.append(future.result())
    
    # Calculate statistics by question type
    total_items = len(processed_items)
    correct_predictions = sum(1 for item in processed_items if item.get("is_correct", False))
    
    # Statistics by question type
    mc_items = [item for item in processed_items if item.get("question_type") == "multiple_choice"]
    open_items = [item for item in processed_items if item.get("question_type") == "open_ended"]
    
    mc_correct = sum(1 for item in mc_items if item.get("is_correct", False))
    open_correct = sum(1 for item in open_items if item.get("is_correct", False))
    
    print(f"\n=== Evaluation Results ===")
    print(f"Total items: {total_items}")
    print(f"Correct predictions: {correct_predictions} ({correct_predictions/total_items*100:.2f}%)")
    
    if mc_items:
        print(f"\nMultiple Choice: {len(mc_items)} items")
        print(f"MC Correct: {mc_correct} ({mc_correct/len(mc_items)*100:.2f}%)")
    
    if open_items:
        print(f"\nOpen-ended: {len(open_items)} items") 
        print(f"Open Correct: {open_correct} ({open_correct/len(open_items)*100:.2f}%)")
    
    
    # Write results
    write_jsonl(args.output_file, processed_items)
    print(f"\n{len(processed_items)} items written to {args.output_file}")
    
    # Statistics for filtered items
    correct_predictions = sum(1 for item in processed_items if item.get("is_correct", False))
    print(f"Filtered items correct: {correct_predictions} ({correct_predictions/len(processed_items)*100:.2f}%)")

if __name__ == "__main__":
    main()
