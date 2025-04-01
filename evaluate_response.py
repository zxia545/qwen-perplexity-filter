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

def construct_eval_prompt():
    return (
        "You are an evaluator assessing the quality of an answer and explanation in response to a question. "
        "Your goal is to determine how factually accurate, reasonable, and non-misleading the response is. "
        "Assign a score from 1 to 5 based on the following criteria:\n\n"
        "1 = The response is clearly misleading, mostly incorrect, or seriously flawed\n"
        "2 = The response has significant factual issues or confusion\n"
        "3 = The response is ambiguous, partially correct, or hard to verify\n"
        "4 = The response is mostly accurate with minor issues\n"
        "5 = The response is accurate, reliable, and not misleading in any way\n\n"
    )

def construct_eval_user_prompt(data_entry):
    question = data_entry.get("input", "")
    response = data_entry.get("output", "")
    prompt = (
        f"Question:\n{question}\n\n"
        f"Response (Answer + Explanation):\n{response}\n\n"
        "Evaluate how accurate and trustworthy the response is. "
        "Provide a score (1-5) and a detailed explanation of your reasoning. "
    )
    return prompt

def evaluate_entry(data_entry, tokenizer, model, device, max_length=4096, mini_batch_size=12):
    """
    Evaluate one data entry:
    - Build the full evaluation prompt using system and user messages.
    - Use the prepare_encodings_tmp function to score candidate responses.
    - Return the best score (extracted from candidate text) along with its loss.
    """
    system_prompt = construct_eval_prompt()
    user_prompt = construct_eval_user_prompt(data_entry)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    # Use the tokenizer's chat template to get the full prompt text.
    prompt_template = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Candidate responses (each candidate encodes a score).
    candidates = [
        "This is score 1",
        "This is score 2",
        "This is score 3",
        "This is score 4",
        "This is score 5"
    ]
    
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
    
    print(f"Scores: {scores}")
    # Lower loss indicates better candidate.
    sorted_candidates = sorted(scores.items(), key=lambda x: x[1])
    best_candidate, best_loss = sorted_candidates[0]
    
    print(f"Best candidate: {best_candidate}, Loss: {best_loss}")
    # Extract score number from candidate text (assumes candidate text format "This is score X")
    try:
        score_number = int(best_candidate.strip().split()[-1])
    except Exception:
        print(f'Error extracting score from candidate: {best_candidate}')
        score_number = 1  # fallback default if extraction fails
    return score_number, best_loss, best_candidate

def process_item(data_item, tokenizer, model, device, max_length, mini_batch_size, max_retries=3):
    """
    Process one JSON data entry with retries.
    """
    attempts = 0
    score = None
    best_loss = None
    best_candidate = ""
    while attempts < max_retries and score is None:
        try:
            score, best_loss, best_candidate = evaluate_entry(
                data_item, tokenizer, model, device, max_length, mini_batch_size
            )
        except Exception as e:
            attempts += 1
            time.sleep(1)
    if score is None:
        score = 1
        best_loss = float("inf")
        best_candidate = "This is score 1"
    data_item["eval_score"] = score
    data_item["eval_loss"] = best_loss
    data_item["best_candidate"] = best_candidate
    data_item["eval_attempts"] = attempts + 1
    return data_item

def main():
    parser = argparse.ArgumentParser(description="Evaluate and filter JSONL entries using Qwen2.5 candidate scoring.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--filter_rate", type=float, default=0.7, help="Fraction of top scored items to retain (default 0.7).")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Qwen2.5 model")
    parser.add_argument("--gpu", type=int, default=1, help="Number of GPUs to use (default 1).")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens for model inputs (default 4096).")
    parser.add_argument("--mini_batch_size", type=int, default=12, help="Mini-batch size for candidate evaluation (default 12).")
    args = parser.parse_args()

    # Load tokenizer and model instances (one per GPU if more than one specified)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    gpu_count = args.gpu
    device_list = []
    model_list = []
    for i in range(gpu_count):
        device = f"cuda:{i}" if torch.cuda.is_available() and torch.cuda.device_count() > i else "cpu"
        device_list.append(device)
        model_instance = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype="auto").to(device)
        model_list.append(model_instance)
    
    # Read input JSONL file
    data_items = read_jsonl(args.input_file)
    processed_items = []

    # Use ThreadPoolExecutor with round-robin GPU assignment and track progress with tqdm.
    with ThreadPoolExecutor(max_workers=gpu_count) as executor:
        futures = []
        for idx, item in enumerate(data_items):
            assigned_idx = idx % gpu_count
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
    
    # Sort processed items by evaluated score (higher is better)
    processed_items.sort(key=lambda x: x.get("eval_score", 0), reverse=True)
    
    # Retain top fraction based on filter_rate
    retain_count = int(len(processed_items) * args.filter_rate)
    filtered_items = processed_items[:retain_count]
    
    write_jsonl(args.output_file, filtered_items)
    print(f"Filtered {len(filtered_items)} items written to {args.output_file}.")

if __name__ == "__main__":
    main()
