import copy
import torch
from torch.nn import CrossEntropyLoss

@torch.no_grad()
def prepare_encodings_tmp(messages_template, categories, tokenizer, device, mini_batch_size=12, max_length=None,
                          add_start_token=False, model=None):
    """
    Computes loss values for a list of candidate responses (categories) given a prompt template.

    Parameters:
    - messages_template (str): The prompt template (Q&A paired string) with its last character removed.
    - categories (list of str): List of candidate response strings.
    - tokenizer: The tokenizer associated with the language model.
    - device (str): The computation device ('cuda' or 'cpu').
    - mini_batch_size (int, optional): Batch size for processing candidates.
    - max_length (int, optional): Maximum token length for candidate inputs.
    - add_start_token (bool, optional): Flag to indicate if a start token is added.
    - model: The Qwen2.5 language model.

    Returns:
    - dict: A dictionary mapping each original candidate string to its computed loss value.
    """
    # Make a copy of the original candidate list for final mapping.
    categories_tmp = copy.deepcopy(categories)
    
    # Process the prompt template (remove the last character as in the original function)
    inputs = tokenizer([messages_template[:-1]], return_tensors="pt").to(device)
    outputs = model(**inputs)
    past_key_values = outputs.past_key_values

    result_list = []
    result_dict = {}

    # Prepend a newline to each candidate string and tokenize
    categories = ["\n" + candidate.strip() for candidate in categories]
    inputs_cal = tokenizer(categories, return_tensors="pt", padding=True).to(device)
    
    # For each prompt input (should be one in our case)
    for idx in range(inputs['input_ids'].shape[0]):
        example_result = []
        # Process candidates in mini-batches
        for i in range(0, len(categories), mini_batch_size):
            mini_cal_ids = inputs_cal['input_ids'][i:i + mini_batch_size]
            mini_cal_attention_mask = inputs_cal['attention_mask'][i:i + mini_batch_size]
            current_mini_bs = mini_cal_ids.shape[0]
            
            # Expand past key values for the mini-batch
            expand_current_mini = tuple(
                [tuple([d[[idx]].expand(current_mini_bs, -1, -1, -1) for d in dd]) for dd in past_key_values]
            )
            expand_current_mini_attention_mask = inputs['attention_mask'][[idx]].expand(current_mini_bs, -1)
            # Concatenate attention masks
            cal_attention_mask = torch.cat([expand_current_mini_attention_mask,
                                            mini_cal_attention_mask], dim=1)

            # Forward pass with past key values
            cal_outputs = model(input_ids=mini_cal_ids,
                                attention_mask=cal_attention_mask,
                                past_key_values=expand_current_mini)
            mini_logits = cal_outputs.logits
            mini_labels = torch.where(mini_cal_attention_mask == 0, -100, mini_cal_ids)
            shift_logits = mini_logits[..., :-1, :].contiguous()
            shift_labels = mini_labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(-1, model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            
            # Compute token-level loss and aggregate per sample
            loss = loss_fct(shift_logits, shift_labels)
            loss = loss.view(cal_attention_mask.shape[0], -1).sum(1) / (mini_cal_attention_mask.sum(1) - 1)
            example_result.append(loss.cpu())
        result_list.append(torch.cat(example_result, dim=0))

    # Extract loss values from result list and map back to original candidate strings.
    result_list_tmp = copy.deepcopy(result_list)[0].tolist()
    for cat_idx, cat in enumerate(categories_tmp):
        result_dict[cat] = result_list_tmp[cat_idx]
    return result_dict
