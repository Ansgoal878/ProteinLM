"""
Embedding generation module for protein sequences
"""
import torch
import numpy as np
from Bio import SeqIO
import pandas as pd
import re


class ProteinEmbeddingGenerator:
    def __init__(self, model, tokenizer, device=None):
        """
        Initialize with a pretrained model and tokenizer
        
        Args:
            model: Pretrained language model
            tokenizer: Corresponding tokenizer
            device: Device to run inference on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = self._determine_model_type()
        print(f"Model type detected: {self.model_type}")
        
    def _determine_model_type(self):
        """
        Determine the type of model we're working with
        
        Returns:
            str: Model type identifier ('esm', 'prot_t5', 'ankh', or 'generic')
        """
        model_class_str = str(self.model.__class__)
        model_name = getattr(self.model, "name_or_path", "") if hasattr(self.model, "name_or_path") else ""
        
        if "facebook/esm" in model_class_str or "facebook/esm" in model_name or "MaskedLM" in model_class_str:
            return "esm"
        elif "ankh" in model_name.lower() or "ElnaggarLab/ankh" in model_name:
            return "ankh"
        elif "prot_t5" in model_name or "t5" in model_class_str.lower() or "Rostlab/prot" in model_name:
            return "prot_t5"
        else:
            return "generic"
            
    def _is_encoder_only_model(self):
        """Check if the model is encoder-only (like T5EncoderModel)"""
        return "T5EncoderModel" in str(self.model.__class__)
    
    def _prepare_input_for_t5(self, sequence):
        """
        Special processing for ProtT5 models
        
        Args:
            sequence (str): Protein sequence
            
        Returns:
            str: Processed sequence suitable for T5 models
        """
        # T5 models expect space-separated sequences
        # Also replace rare amino acids with X
        sequence = " ".join(sequence)
        sequence = re.sub(r"[UZOB]", "X", sequence)
        return sequence
    
    def _prepare_input_for_ankh(self, sequence):
        """
        Special processing for ANKH models
        
        Args:
            sequence (str): Protein sequence
            
        Returns:
            str: Processed sequence suitable for ANKH models
        """
        # ANKH models expect space-separated sequences like ProtT5
        return " ".join(sequence)
    
    def _prepare_input_for_esm(self, sequence):
        """
        Special processing for ESM models
        
        Args:
            sequence (str): Protein sequence
            
        Returns:
            str: Processed sequence suitable for ESM models
        """
        # ESM models expect the raw sequence
        return sequence
        
    def generate_embedding(self, sequence, pooling_strategy="mean"):
        """
        Generate embedding for a single protein sequence
        
        Args:
            sequence (str): Protein sequence
            pooling_strategy (str): Strategy for pooling embeddings ('mean', 'cls', etc.)
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        # Prepare sequence based on model type
        if self.model_type == "prot_t5":
            prep_sequence = self._prepare_input_for_t5(sequence)
        elif self.model_type == "ankh":
            prep_sequence = self._prepare_input_for_ankh(sequence)
        else:  # ESM or generic
            prep_sequence = self._prepare_input_for_esm(sequence)
        
        # Tokenize sequence
        inputs = self.tokenizer(prep_sequence, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings based on model type
        with torch.no_grad():
            if self.model_type == "esm":
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
            elif self.model_type == "prot_t5":
                # Check if it's an encoder-only model (T5EncoderModel)
                if self._is_encoder_only_model():
                    # For encoder-only models
                    outputs = self.model(**inputs)
                    hidden_states = outputs.last_hidden_state
                else:
                    # For full T5 models
                    outputs = self.model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        decoder_input_ids=None
                    )
                    # Get the encoder embedding from index 2 as per the reference example
                    hidden_states = outputs[2] if len(outputs) > 2 else outputs.last_hidden_state
            elif self.model_type == "ankh":
                # For ANKH models, only use the encoder's output
                encoder = self.model.get_encoder()
                outputs = encoder(**inputs)
                hidden_states = outputs.last_hidden_state
            else:
                # Generic fallback
                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs
            
        # Apply pooling strategy
        if pooling_strategy == "mean":
            # Mean pooling of token embeddings
            if 'attention_mask' in inputs:
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                embeddings = torch.sum(hidden_states * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
            else:
                embeddings = hidden_states.mean(dim=1)
        elif pooling_strategy == "cls":
            # Use CLS token embedding or first token
            embeddings = hidden_states[:, 0]
        else:
            raise ValueError(f"Unsupported pooling strategy: {pooling_strategy}")
            
        return embeddings.cpu().numpy()[0]
    
    def generate_embeddings_batch(self, sequences, pooling_strategy="mean", batch_size=8):
        """
        Generate embeddings for multiple protein sequences
        
        Args:
            sequences (list): List of protein sequences
            pooling_strategy (str): Strategy for pooling embeddings
            batch_size (int): Batch size for processing
            
        Returns:
            numpy.ndarray: Matrix of embeddings
        """
        all_embeddings = []
        
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            
            # Prepare sequences based on model type
            if self.model_type == "prot_t5":
                prep_batch = [self._prepare_input_for_t5(seq) for seq in batch]
            elif self.model_type == "ankh":
                prep_batch = [self._prepare_input_for_ankh(seq) for seq in batch]
            else:  # ESM or generic
                prep_batch = [self._prepare_input_for_esm(seq) for seq in batch]
            
            # Tokenize batch
            inputs = self.tokenizer(prep_batch, padding=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings based on model type
            with torch.no_grad():
                if self.model_type == "esm":
                    outputs = self.model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
                elif self.model_type == "prot_t5":
                    # Check if it's an encoder-only model (T5EncoderModel)
                    if self._is_encoder_only_model():
                        # For encoder-only models
                        outputs = self.model(**inputs)
                        hidden_states = outputs.last_hidden_state
                    else:
                        # For full T5 models
                        outputs = self.model(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            decoder_input_ids=None
                        )
                        # Get the encoder embedding from index 2 as per the reference example
                        hidden_states = outputs[2] if len(outputs) > 2 else outputs.last_hidden_state
                elif self.model_type == "ankh":
                    # For ANKH models, only use the encoder part
                    encoder = self.model.get_encoder()
                    outputs = encoder(**inputs)
                    hidden_states = outputs.last_hidden_state
                else:
                    # Generic fallback
                    outputs = self.model(**inputs)
                    hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs
                
            # Apply pooling strategy
            if pooling_strategy == "mean":
                # Mean pooling (over non-padding tokens)
                if 'attention_mask' in inputs:
                    attention_mask = inputs['attention_mask'].unsqueeze(-1)
                    embeddings = torch.sum(hidden_states * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
                else:
                    embeddings = hidden_states.mean(dim=1)
            elif pooling_strategy == "cls":
                # Use first token embedding
                embeddings = hidden_states[:, 0]
            else:
                raise ValueError(f"Unsupported pooling strategy: {pooling_strategy}")
                
            all_embeddings.append(embeddings.cpu().numpy())
                
        return np.vstack(all_embeddings)