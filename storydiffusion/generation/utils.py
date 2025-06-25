"""Generation utility functions."""

import torch
import random
import numpy as np
from typing import List, Tuple
from utils.style_template import styles
from ..config import DEFAULT_STYLE_NAME


def setup_seed(seed: int, device: str = "cpu") -> None:
    """
    Set up random seeds for reproducible generation across all random number generators.

    Args:
        seed (int): The seed value to use for all random number generators
        device (str): The device being used ("cuda", "mps", or "cpu")
    """
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def apply_style_positive(style_name: str, positive: str) -> str:
    """
    Apply style template to a single positive prompt.

    Args:
        style_name (str): Name of the style template
        positive (str): Positive prompt to style

    Returns:
        str: Styled positive prompt
    """
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive)


def apply_style(
    style_name: str, 
    positives: List[str], 
    negative: str = ""
) -> Tuple[List[str], str]:
    """
    Apply style template to multiple prompts.

    Args:
        style_name (str): Name of the style template
        positives (list): List of positive prompts to style
        negative (str): Additional negative prompt text

    Returns:
        tuple: (styled_positives, styled_negative)
    """
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    styled_positives = [p.replace("{prompt}", positive) for positive in positives]
    styled_negative = n + " " + negative if negative else n
    return styled_positives, styled_negative


def process_prompts_for_generation(
    prompt_array: str,
    id_length: int
) -> Tuple[List[str], List[int], List[str]]:
    """
    Process the prompt array to extract prompts, NC indices, and captions.
    
    Args:
        prompt_array: Raw prompt array from user input
        id_length: Number of ID prompts
        
    Returns:
        Tuple of (processed_prompts, nc_indices, captions)
    """
    prompts = prompt_array.strip().splitlines()
    nc_indices = []
    captions = []
    processed_prompts = []
    
    for ind, prompt in enumerate(prompts):
        # Check for [NC] flag
        if "[NC]" in prompt:
            nc_indices.append(ind)
            if ind < id_length:
                raise ValueError(
                    f"The first {id_length} rows are id prompts, cannot use [NC]!"
                )
            prompt = prompt.replace("[NC]", "").strip()
        
        # Extract caption if present (after #)
        if "#" in prompt:
            prompt_part, caption_part = prompt.split("#", 1)
            processed_prompts.append(prompt_part.strip())
            captions.append(caption_part.strip())
        else:
            processed_prompts.append(prompt.strip())
            captions.append(prompt.strip())
    
    return processed_prompts, nc_indices, captions