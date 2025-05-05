"""
Model loading module for pretrained protein language models from Hugging Face
"""
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForSeq2SeqLM
from transformers import T5Tokenizer, T5Model, T5EncoderModel
import torch
import re

# Dictionary mapping model prefixes to their appropriate model types and loader functions
MODEL_TYPE_MAPPING = {
    "Rostlab/prot_t5": {
        "type": "encoder",
        "tokenizer_class": T5Tokenizer,
        "model_class": T5EncoderModel  # or T5Model depending on your needs
    },
    "facebook/esm2": {
        "type": "masked_lm",
        "tokenizer_class": AutoTokenizer,
        "model_class": AutoModelForMaskedLM
    },
    "ElnaggarLab/ankh": {
        "type": "seq2seq",
        "tokenizer_class": AutoTokenizer,
        "model_class": AutoModelForSeq2SeqLM
    }
}

def get_model_config(model_name):
    """
    Determine the type and classes to use for a model based on its name
    
    Args:
        model_name (str): Name of the model on Hugging Face
        
    Returns:
        tuple: (model_category, tokenizer_class, model_class)
    """
    for prefix, config in MODEL_TYPE_MAPPING.items():
        if model_name.startswith(prefix):
            return config["type"], config["tokenizer_class"], config["model_class"]
    
    # Default to encoder model if no match is found
    print(f"Warning: Unknown model type for {model_name}. Defaulting to AutoModel.")
    return "encoder", AutoTokenizer, AutoModel

def load_model(model_name="Rostlab/prot_t5_xl_half_uniref50-enc", device=None):
    """
    Load a pretrained protein language model
    
    Args:
        model_name (str): Name of the model on Hugging Face
        device (str): Device to run on ('cuda' or 'cpu'), auto-detects if None
        
    Returns:
        tuple: (tokenizer, model) The loaded model and tokenizer
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        print(f"Loading model: {model_name}")
        
        # Get the appropriate model type and classes
        model_category, tokenizer_class, model_class = get_model_config(model_name)
        
        # Special handling for ProtT5 models
        if model_name.startswith("Rostlab/prot_t5"):
            # Load tokenizer with appropriate settings for ProtT5
            tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=False)
            
            # Load model
            model = model_class.from_pretrained(model_name)
        else:
            # Standard loading for other models
            tokenizer = tokenizer_class.from_pretrained(model_name)
            model = model_class.from_pretrained(model_name)
        
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        
        print(f"Successfully loaded {model_name} as {model_category} model")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model {model_name}: {str(e)}")
        raise e

def list_available_models():
    """
    List all available pretrained protein language models supported by this loader
    
    Returns:
        dict: Dictionary with model categories and available models
    """
    available_models = {
        "Encoder Models (T5EncoderModel)": [
            "Rostlab/prot_t5_xl_half_uniref50-enc",
            "Rostlab/prot_t5_xl_uniref50",
            "Rostlab/prot_t5_base_mt_uniref50"
        ],
        "Masked Language Models (AutoModelForMaskedLM)": [
            "facebook/esm2_t6_8M_UR50D",
            "facebook/esm2_t12_35M_UR50D", 
            "facebook/esm2_t30_150M_UR50D", 
            "facebook/esm2_t33_650M_UR50D", 
            "facebook/esm2_t36_3B_UR50D"
        ],
        "Sequence-to-Sequence Models (AutoModelForSeq2SeqLM)": [
            "ElnaggarLab/ankh-base",
            "ElnaggarLab/ankh-large"
        ]
    }
    
    return available_models