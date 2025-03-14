#!/usr/bin/env python

"""\
Tutorial for obtaining a hands-on understanding of embeddings, the BERT model, transformers and self-attention.

# Intro

This is a standalone app that implements a simple BERT encoder using self-attention. The emphasis
is on simplicity, visualization, and being able to train and run the model on a CPU, while still
remaining as close as possible to the "real thing".

# Design

- Embed two-word sentences usig BERT that each contain a noun and an attribute (in random order),
like "tree green" "green frog", "car fast", etc. This would allow the model to work with 2-long
contexts and a minimal vocabulary (say, 64 nouns and 64 attributes).
- No tokenization, each word fed into the self-attention layer via a trainable linear
  transformationas of a trivial one-hot encoding.
- Trained using masked word prediction.
- Programmed from scratch, with no external dependencies other than numpy.

## TSV format for training:

word1  word2  type1  type2
frog   green  noun   attribute
car    fast   noun   attribute
...

# Usage

## Create an example TSV file

python transformer.py --mode create_tsv --tsv_path my_data.tsv

## Train the model using the TSV data

python transformer.py --mode train --tsv_path my_data.tsv --steps 2000

## Run inference with specific input

python transformer.py --mode inference --input "frog [MASK]"

## Or run inference with default examples

python transformer.py --mode

"""

import numpy as np
import pickle
import os
import csv
from typing import List, Tuple, Dict, Optional, Set

# Random seed for reproducibility
np.random.seed(42)

class SimpleTransformer:
    def __init__(
            self,
            vocab_size: int,
            d_model: int = 64,
            d_ff: int = 128,
            dropout_rate: float = 0.1,
    ):
        """
        Initialize a simple transformer for 2-word sentences
        
        Args:
            vocab_size: Size of the vocabulary
            d_model: Dimension of the embeddings
            d_ff: Dimension of the feed-forward network
            dropout_rate: Dropout probability
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize all model weights"""
        # Word embedding matrix: maps one-hot word indices to dense vectors
        self.W_emb = np.random.randn(self.vocab_size, self.d_model) * 0.1
        
        # Positional encoding for position 0 and 1
        self.pos_encoding = np.random.randn(2, self.d_model) * 0.1
        
        # Self-attention weights
        self.W_query = np.random.randn(self.d_model, self.d_model) * 0.1
        self.W_key = np.random.randn(self.d_model, self.d_model) * 0.1
        self.W_value = np.random.randn(self.d_model, self.d_model) * 0.1
        self.W_output = np.random.randn(self.d_model, self.d_model) * 0.1
        
        # Feed-forward network weights
        self.W_ff1 = np.random.randn(self.d_model, self.d_ff) * 0.1
        self.b_ff1 = np.zeros(self.d_ff)
        self.W_ff2 = np.random.randn(self.d_ff, self.d_model) * 0.1
        self.b_ff2 = np.zeros(self.d_model)
        
        # Output projection for word prediction
        self.W_out = np.random.randn(self.d_model, self.vocab_size) * 0.1
        self.b_out = np.zeros(self.vocab_size)
        
    def embed_inputs(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Convert input word indices to embeddings
        
        Args:
            input_ids: Input word indices [seq_len=2]
            
        Returns:
            Word embeddings with positional encoding [seq_len=2, d_model]
        """
        # Get word embeddings
        seq_len = input_ids.shape[0]  # Always 2 in our case
        embeddings = np.zeros((seq_len, self.d_model))
        
        for s in range(seq_len):
            # Get word embedding
            word_idx = input_ids[s]
            embeddings[s] = self.W_emb[word_idx]
            
            # Add positional encoding
            embeddings[s] += self.pos_encoding[s]
            
        return embeddings
    
    def self_attention(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute self-attention mechanism for 2-word context
        
        Args:
            embeddings: Word embeddings [seq_len=2, d_model]
            
        Returns:
            attention_output: Contextualized embeddings [seq_len=2, d_model]
            attention_weights: Attention weights for visualization [seq_len=2, seq_len=2]
        """
        seq_len = embeddings.shape[0]  # Always 2 in our case
        
        # Create query, key and value projections
        queries = np.matmul(embeddings, self.W_query)  # [seq_len, d_model]
        keys = np.matmul(embeddings, self.W_key)      # [seq_len, d_model]
        values = np.matmul(embeddings, self.W_value)  # [seq_len, d_model]
        
        # Compute attention scores (scaled dot-product attention)
        # For 2-word sentences, this gives us a 2x2 attention matrix
        attention_scores = np.zeros((seq_len, seq_len))
        for q in range(seq_len):
            for k in range(seq_len):
                # Dot product between query and key
                attention_scores[q, k] = np.dot(queries[q], keys[k])
                
        # Scale attention scores
        attention_scores = attention_scores / np.sqrt(self.d_model)
        
        # Apply softmax to get attention weights
        attention_weights = np.zeros_like(attention_scores)
        for q in range(seq_len):
            # Softmax along the key dimension
            exp_scores = np.exp(attention_scores[q])
            attention_weights[q] = exp_scores / np.sum(exp_scores)
            
        # Apply attention weights to values
        context_vectors = np.zeros((seq_len, self.d_model))
        for q in range(seq_len):
            # Weighted sum of values
            for k in range(seq_len):
                context_vectors[q] += attention_weights[q, k] * values[k]
                
        # Project back to model dimension
        attention_output = np.matmul(context_vectors, self.W_output)
        
        return attention_output, attention_weights
    
    def feed_forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply feed-forward network to each position
        
        Args:
            x: Input features [seq_len=2, d_model]
            
        Returns:
            Output features [seq_len=2, d_model]
        """
        # First dense layer with ReLU activation
        hidden = np.maximum(0, np.matmul(x, self.W_ff1) + self.b_ff1)
        
        # Second dense layer
        output = np.matmul(hidden, self.W_ff2) + self.b_ff2
        
        return output
    
    def forward(self, input_ids: np.ndarray, mask_position=None) -> Tuple[np.ndarray, Dict]:
        """
        Forward pass of the transformer
        
        Args:
            input_ids: Input word indices [seq_len=2]
            mask_position: Position to predict (for MLM) - 0 or 1, or None for all positions
            
        Returns:
            logits: Prediction logits [vocab_size] if mask_position is specified, else [seq_len=2, vocab_size]
            attention_weights: Attention weights for visualization
        """
        seq_len = input_ids.shape[0]  # Always 2 in our case
        
        # Embedding layer
        embeddings = self.embed_inputs(input_ids)
        
        # Self-attention block
        attention_output, attention_weights = self.self_attention(embeddings)
        
        # Add & Norm (simplified, no layer norm for now)
        attention_output = embeddings + attention_output
        
        # Feed-forward block
        ff_output = self.feed_forward(attention_output)
        
        # Add & Norm (simplified)
        output = attention_output + ff_output
        
        # Final prediction layer
        if mask_position is not None:
            # Only predict at masked position
            hidden = output[mask_position]
            logits = np.matmul(hidden, self.W_out) + self.b_out
        else:
            # Predict at all positions
            logits = np.zeros((seq_len, self.vocab_size))
            for s in range(seq_len):
                logits[s] = np.matmul(output[s], self.W_out) + self.b_out
                
        return logits, {"attention_weights": attention_weights}
    
    def compute_loss(self, logits: np.ndarray, target_id: int) -> float:
        """
        Compute cross-entropy loss
        
        Args:
            logits: Prediction logits [vocab_size]
            target_id: Target word index
            
        Returns:
            loss: Cross-entropy loss
        """
        # Softmax over vocabulary
        exp_logits = np.exp(logits - np.max(logits))  # numerical stability
        probs = exp_logits / np.sum(exp_logits)
        
        # Cross-entropy loss
        target_prob = probs[target_id]
        loss = -np.log(target_prob + 1e-10)  # add small epsilon to avoid log(0)
        
        return loss
    
    def compute_gradients(self, input_ids, mask_pos, target_id, learning_rate=0.01) -> float:
        """
        Compute gradients using backpropagation and update weights
        
        Args:
            input_ids: Input word indices [seq_len=2]
            mask_pos: Position that was masked (0 or 1)
            target_id: Original word ID at masked position
            learning_rate: Learning rate for SGD updates
        
        Returns:
            loss: Loss value after the forward pass
        """
        # Forward pass with tracking intermediate values
        # 1. Word embeddings + positional encoding
        embeddings = self.embed_inputs(input_ids)
        
        # 2. Self-attention
        queries = np.matmul(embeddings, self.W_query)
        keys = np.matmul(embeddings, self.W_key)
        values = np.matmul(embeddings, self.W_value)
        
        # Compute attention scores
        seq_len = embeddings.shape[0]  # Always 2 in our case
        attention_scores = np.zeros((seq_len, seq_len))
        for q in range(seq_len):
            for k in range(seq_len):
                attention_scores[q, k] = np.dot(queries[q], keys[k])
                
        # Scale attention scores
        attention_scores = attention_scores / np.sqrt(self.d_model)
    
        # Apply softmax to get attention weights
        attention_weights = np.zeros_like(attention_scores)
        for q in range(seq_len):
            exp_scores = np.exp(attention_scores[q])
            attention_weights[q] = exp_scores / np.sum(exp_scores)
        
        # Apply attention weights to values
        context_vectors = np.zeros((seq_len, self.d_model))
        for q in range(seq_len):
            for k in range(seq_len):
                context_vectors[q] += attention_weights[q, k] * values[k]
            
        # Project back to model dimension
        attention_output = np.matmul(context_vectors, self.W_output)
    
        # 3. Add & Norm (simplified, just add for now)
        post_attention = embeddings + attention_output
    
        # 4. Feed-forward network
        # First dense layer with ReLU activation
        ff_hidden = np.matmul(post_attention, self.W_ff1) + self.b_ff1
        ff_hidden_activated = np.maximum(0, ff_hidden)  # ReLU
    
        # Second dense layer
        ff_output = np.matmul(ff_hidden_activated, self.W_ff2) + self.b_ff2
    
        # 5. Final Add & Norm
        final_output = post_attention + ff_output
    
        # 6. Output projection for prediction
        masked_hidden = final_output[mask_pos]
        logits = np.matmul(masked_hidden, self.W_out) + self.b_out
    
        # Compute softmax and loss
        exp_logits = np.exp(logits - np.max(logits))  # for numerical stability
        probs = exp_logits / np.sum(exp_logits)
        loss = -np.log(probs[target_id] + 1e-10)  # add small epsilon to avoid log(0)
    
        # ==== BACKPROPAGATION ====
    
        # 1. Output layer gradients
        dlogits = probs.copy()
        dlogits[target_id] -= 1.0  # -1 for the true class
    
        # 2. Gradients for output projection
        dW_out = np.outer(masked_hidden, dlogits)
        db_out = dlogits
        dhidden = np.matmul(self.W_out, dlogits)  # gradient to previous layer
    
        # 3. Initialize gradient for final layer outputs
        dfinal_output = np.zeros_like(final_output)
        dfinal_output[mask_pos] = dhidden
    
        # 4. Gradient flowing to feed-forward output and post-attention (residual connection)
        dff_output = dfinal_output
        dpost_attention = dfinal_output.copy()  # Copy for the residual connection
    
        # 5. Feed-forward second layer gradient
        dff_hidden_activated = np.zeros((seq_len, self.d_ff))
        for i in range(seq_len):
            dff_hidden_activated[i] = np.matmul(self.W_ff2, dff_output[i])
        
        dW_ff2 = np.zeros_like(self.W_ff2)
        db_ff2 = np.zeros_like(self.b_ff2)
        for i in range(seq_len):
            dW_ff2 += np.outer(ff_hidden_activated[i], dff_output[i])
            db_ff2 += dff_output[i]
        
        # 6. Gradient through ReLU
        dff_hidden = dff_hidden_activated.copy()
        dff_hidden[ff_hidden <= 0] = 0  # ReLU gradient: 0 where input was <= 0
    
        # 7. Feed-forward first layer gradient
        dpost_attention_ff = np.zeros_like(post_attention)
        for i in range(seq_len):
            dpost_attention_ff[i] = np.matmul(self.W_ff1, dff_hidden[i])
        
        dW_ff1 = np.zeros_like(self.W_ff1)
        db_ff1 = np.zeros_like(self.b_ff1)
        for i in range(seq_len):
            dW_ff1 += np.outer(post_attention[i], dff_hidden[i])
            db_ff1 += dff_hidden[i]
        
        # 8. Add gradients from feed-forward to post-attention
        dpost_attention += dpost_attention_ff
    
        # 9. Gradient flows to both attention output and embeddings (residual)
        dattention_output = dpost_attention
        dembeddings_residual = dpost_attention.copy()
    
        # 10. Gradient for attention output projection
        dcontext_vectors = np.zeros_like(context_vectors)
        for i in range(seq_len):
            dcontext_vectors[i] = np.matmul(self.W_output, dattention_output[i])
        
        dW_output = np.zeros_like(self.W_output)
        for i in range(seq_len):
            dW_output += np.outer(context_vectors[i], dattention_output[i])
        
        # 11. Gradient for attention mechanism
        dvalues = np.zeros_like(values)
        dattention_weights = np.zeros_like(attention_weights)
    
        for q in range(seq_len):
            for k in range(seq_len):
                dvalues[k] += attention_weights[q, k] * dcontext_vectors[q]
                dattention_weights[q, k] = np.dot(dcontext_vectors[q], values[k])
            
        # 12. Gradient for softmax (attention weights)
        dattention_scores = np.zeros_like(attention_scores)
        for q in range(seq_len):
            for i in range(seq_len):
                for j in range(seq_len):
                    if i == j:
                        dattention_scores[q, i] += dattention_weights[q, i] * attention_weights[q, i] * (1 - attention_weights[q, i])
                    else:
                        dattention_scores[q, i] -= dattention_weights[q, j] * attention_weights[q, i] * attention_weights[q, j]
                    
        # 13. Gradient for scaled dot-product
        scale_factor = 1.0 / np.sqrt(self.d_model)
        dattention_scores *= scale_factor
    
        # 14. Gradients for queries and keys
        dqueries = np.zeros_like(queries)
        dkeys = np.zeros_like(keys)
    
        for q in range(seq_len):
            for k in range(seq_len):
                dqueries[q] += dattention_scores[q, k] * keys[k]
                dkeys[k] += dattention_scores[q, k] * queries[q]
            
        # 15. Gradients for attention weight matrices
        dW_query = np.zeros_like(self.W_query)
        dW_key = np.zeros_like(self.W_key)
        dW_value = np.zeros_like(self.W_value)
    
        for i in range(seq_len):
            dW_query += np.outer(embeddings[i], dqueries[i])
            dW_key += np.outer(embeddings[i], dkeys[i])
            dW_value += np.outer(embeddings[i], dvalues[i])
        
        # 16. Gradient for embeddings from attention mechanism
        dembeddings_q = np.zeros_like(embeddings)
        dembeddings_k = np.zeros_like(embeddings)
        dembeddings_v = np.zeros_like(embeddings)
    
        for i in range(seq_len):
            dembeddings_q[i] = np.matmul(self.W_query, dqueries[i])
            dembeddings_k[i] = np.matmul(self.W_key, dkeys[i])
            dembeddings_v[i] = np.matmul(self.W_value, dvalues[i])
        
        # 17. Combine all gradients flowing to embeddings
        dembeddings = dembeddings_residual + dembeddings_q + dembeddings_k + dembeddings_v
    
        # 18. Gradient for word embeddings and positional encoding
        dW_emb = np.zeros_like(self.W_emb)
        dpos_encoding = np.zeros_like(self.pos_encoding)
    
        for i in range(seq_len):
            word_idx = input_ids[i]
            dW_emb[word_idx] += dembeddings[i]
            dpos_encoding[i] += dembeddings[i]
        
        # Update weights using SGD
        self.W_emb -= learning_rate * dW_emb
        self.pos_encoding -= learning_rate * dpos_encoding
        self.W_query -= learning_rate * dW_query
        self.W_key -= learning_rate * dW_key
        self.W_value -= learning_rate * dW_value
        self.W_output -= learning_rate * dW_output
        self.W_ff1 -= learning_rate * dW_ff1
        self.b_ff1 -= learning_rate * db_ff1
        self.W_ff2 -= learning_rate * dW_ff2
        self.b_ff2 -= learning_rate * db_ff2
        self.W_out -= learning_rate * dW_out
        self.b_out -= learning_rate * db_out
    
        return loss

    def save(self, filepath: str) -> None:
        """
        Save model weights to a file
        
        Args:
            filepath: Path to save the model weights
        """
        # Create a dictionary of all weights
        weights = {
            'W_emb': self.W_emb,
            'pos_encoding': self.pos_encoding,
            'W_query': self.W_query,
            'W_key': self.W_key,
            'W_value': self.W_value,
            'W_output': self.W_output,
            'W_ff1': self.W_ff1,
            'b_ff1': self.b_ff1,
            'W_ff2': self.W_ff2,
            'b_ff2': self.b_ff2,
            'W_out': self.W_out,
            'b_out': self.b_out,
            'model_config': {
                'vocab_size': self.vocab_size,
                'd_model': self.d_model,
                'd_ff': self.d_ff,
                'dropout_rate': self.dropout_rate
            }
        }
        
        # Save to file
        with open(filepath, 'wb') as f:
            pickle.dump(weights, f)
            
        print(f"Model weights saved to {filepath}")
        
    @classmethod
    def load(cls, filepath: str) -> 'SimpleTransformer':
        """
        Load model weights from a file
        
        Args:
            filepath: Path to the saved model weights
            
        Returns:
            Loaded model instance
        """
        # Load weights from file
        with open(filepath, 'rb') as f:
            weights = pickle.load(f)
            
        # Create model with the same configuration
        config = weights['model_config']
        model = cls(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            d_ff=config['d_ff'],
            dropout_rate=config['dropout_rate']
        )
        
        # Load weights
        model.W_emb = weights['W_emb']
        model.pos_encoding = weights['pos_encoding']
        model.W_query = weights['W_query']
        model.W_key = weights['W_key']
        model.W_value = weights['W_value']
        model.W_output = weights['W_output']
        model.W_ff1 = weights['W_ff1']
        model.b_ff1 = weights['b_ff1']
        model.W_ff2 = weights['W_ff2']
        model.b_ff2 = weights['b_ff2']
        model.W_out = weights['W_out']
        model.b_out = weights['b_out']
        
        print(f"Model loaded from {filepath}")
        return model


# Data generation
class DataGenerator:
    def __init__(
            self,
            vocab: List[str],
            mask_token: str = "[MASK]",
            noun_types: Optional[Set[str]] = None,
            attribute_types: Optional[Set[str]] = None
    ):
        """
        Initialize data generator for word pairs
        
        Args:
            vocab: List of words in the vocabulary
            mask_token: The mask token string
            noun_types: Set of words that are considered nouns (optional)
            attribute_types: Set of words that are considered attributes (optional)
        """
        # Set mask token and add it to vocab if not present
        self.mask_token = mask_token
        if mask_token not in vocab:
            vocab = [mask_token] + vocab
            
        # Create vocabulary mappings
        self.vocab = vocab
        self.word2id = {word: idx for idx, word in enumerate(self.vocab)}
        self.id2word = {idx: word for idx, word in enumerate(self.vocab)}
        self.mask_token_id = self.word2id[mask_token]
        
        # Store noun and attribute types if provided
        self.noun_types = noun_types if noun_types is not None else set()
        self.attribute_types = attribute_types if attribute_types is not None else set()
        
        # Create ID lists for nouns and attributes if available
        if noun_types:
            self.noun_ids = [self.word2id[word] for word in noun_types if word in self.word2id]
        else:
            self.noun_ids = []
            
        if attribute_types:
            self.attribute_ids = [self.word2id[word] for word in attribute_types if word in self.word2id]
        else:
            self.attribute_ids = []
            
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
    
    def generate_example(self, pairs: List[Tuple[str, str]], masking_prob: float = 0.5) -> Tuple[np.ndarray, int, int]:
        """
        Generate a single example for masked language modeling from a list of valid pairs
        
        Args:
            pairs: List of valid word pairs (word1, word2)
            masking_prob: Probability of masking the first vs second word
            
        Returns:
            input_ids: Input word IDs with masking [2]
            mask_position: Position that was masked
            target_id: Original word ID at masked position
        """
        # Randomly select a pair
        word1, word2 = pairs[np.random.randint(0, len(pairs))]
        
        # Convert to IDs
        word1_id = self.word2id[word1]
        word2_id = self.word2id[word2]
        
        # Apply masking to first or second position randomly
        original_ids = np.array([word1_id, word2_id])
        mask_pos = 0 if np.random.random() < masking_prob else 1
        target_id = original_ids[mask_pos]  # Save original word
        
        # Create input with mask
        input_ids = original_ids.copy()
        input_ids[mask_pos] = self.mask_token_id  # Replace with mask
        
        return input_ids, mask_pos, target_id
    
    def convert_ids_to_text(self, ids: np.ndarray) -> List[str]:
        """Convert ID sequence to text"""
        return [self.id2word[idx] for idx in ids]
    
    def save(self, filepath: str) -> None:
        """
        Save data generator configuration to a file
        
        Args:
            filepath: Path to save the configuration
        """
        config = {
            'vocab': self.vocab,
            'mask_token': self.mask_token,
            'word2id': self.word2id,
            'id2word': self.id2word,
            'mask_token_id': self.mask_token_id,
            'noun_types': self.noun_types,
            'attribute_types': self.attribute_types,
            'noun_ids': self.noun_ids,
            'attribute_ids': self.attribute_ids
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(config, f)
            
        print(f"Data generator configuration saved to {filepath}")
        
    @classmethod
    def load(cls, filepath: str) -> 'DataGenerator':
        """
        Load data generator configuration from a file
        
        Args:
            filepath: Path to the saved configuration
            
        Returns:
            Loaded data generator instance
        """
        with open(filepath, 'rb') as f:
            config = pickle.load(f)
            
        # Create an empty instance and load the saved attributes
        instance = cls.__new__(cls)
        for key, value in config.items():
            setattr(instance, key, value)
            
        print(f"Data generator loaded from {filepath}")
        return instance
    
    @classmethod
    def from_tsv(cls, tsv_path: str, mask_token: str = "[MASK]") -> Tuple['DataGenerator', List[Tuple[str, str]]]:
        """
        Create a data generator from a TSV file containing word pairs
        
        Args:
            tsv_path: Path to TSV file with word pairs
            mask_token: The mask token string
            
        Returns:
            data_generator: Initialized data generator
            pairs: List of valid word pairs
        """
        # Read word pairs from TSV file
        pairs = []
        all_words = set()
        noun_types = set()
        attribute_types = set()
        
        with open(tsv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            # Skip header if present
            try:
                header = next(reader)
                has_types = len(header) > 2
            except StopIteration:
                print("Warning: TSV file is empty")
                has_types = False
                
            for row in reader:
                if len(row) >= 2:
                    word1, word2 = row[0].strip(), row[1].strip()
                    pairs.append((word1, word2))
                    all_words.add(word1)
                    all_words.add(word2)
                    
                    # If the TSV has type information (noun/attribute)
                    if has_types and len(row) >= 4:
                        if row[2].strip().lower() == "noun":
                            noun_types.add(word1)
                        elif row[2].strip().lower() == "attribute":
                            attribute_types.add(word1)
                            
                        if row[3].strip().lower() == "noun":
                            noun_types.add(word2)
                        elif row[3].strip().lower() == "attribute":
                            attribute_types.add(word2)
                            
        # Create vocabulary (unique words + mask token)
        vocab = sorted(list(all_words))
        
        # Initialize data generator
        data_gen = cls(vocab, mask_token, noun_types, attribute_types)
        
        print(f"Loaded {len(pairs)} word pairs from {tsv_path}")
        print(f"Vocabulary size: {len(vocab) + 1} (including mask token)")
        if noun_types:
            print(f"Identified {len(noun_types)} nouns and {len(attribute_types)} attributes")
            
        return data_gen, pairs


# Training function with gradient updates
def train_model(
        num_steps: int = 1000,
        learning_rate: float = 0.01,
        model_path: str = "model.pkl",
        data_path: str = "data_generator.pkl",
        tsv_path: Optional[str] = None
):
    """
    Train the transformer model and save the weights
    
    Args:
        num_steps: Number of training steps
        learning_rate: Learning rate for gradient updates
        model_path: Path to save the trained model
        data_path: Path to save the data generator
        tsv_path: Path to TSV file with training data (if None, use default data)
    """
    if tsv_path:
        # Load data from TSV file
        data_gen, pairs = DataGenerator.from_tsv(tsv_path)
    else:
        # Use default data
        nouns = ["tree", "car", "house", "dog", "cat", "bird", "fish", "book"]
        attributes = ["green", "red", "tall", "small", "fast", "slow", "old", "new"]
        
        # Create data generator with default data
        vocab = nouns + attributes
        data_gen = DataGenerator(vocab, "[MASK]", set(nouns), set(attributes))
        
        # Create all possible pairs for training
        pairs = [(noun, attr) for noun in nouns for attr in attributes]
        
    # Create model
    model = SimpleTransformer(
        vocab_size=data_gen.vocab_size,
        d_model=32,
        d_ff=64
    )
    
    # Training loop
    losses = []
    print("Starting training...")
    for step in range(num_steps):
        # Generate single example
        input_ids, mask_pos, target_id = data_gen.generate_example(pairs)
        
        # Forward pass and gradient updates
        loss = model.compute_gradients(input_ids, mask_pos, target_id, learning_rate)
        losses.append(loss)
        
        # Print progress
        if (step + 1) % 100 == 0 or step == 0:
            print(f"Step {step+1}/{num_steps}, Loss: {loss:.4f}, Avg Loss (last 100): {np.mean(losses[-100:]):.4f}")
            
            # Show example predictions every 500 steps
            if (step + 1) % 500 == 0:
                print("\nExample predictions:")
                example_inputs = [
                    ["[MASK]", "red"],
                    ["car", "[MASK]"],
                    ["[MASK]", "tall"],
                    ["dog", "[MASK]"]
                ]
                
                for example in example_inputs:
                    try:
                        example_ids = np.array([data_gen.word2id.get(word, data_gen.mask_token_id) for word in example])
                        mask_position = example.index("[MASK]")
                        
                        # Get predictions
                        logits, _ = model.forward(example_ids, mask_position)
                        top_idx = np.argmax(logits)
                        top_word = data_gen.id2word[top_idx]
                        
                        # Get probability
                        exp_logits = np.exp(logits - np.max(logits))
                        probs = exp_logits / np.sum(exp_logits)
                        top_prob = probs[top_idx]
                        
                        print(f"  Input: {' '.join(example)}, Prediction: {top_word} (prob: {top_prob:.2f})")
                    except (ValueError, KeyError) as e:
                        print(f"  Error with example {' '.join(example)}: {e}")
                        print("")
                        
    # Save the model and data generator
    model.save(model_path)
    data_gen.save(data_path)
    
    return model, data_gen

# Inference function
def run_inference(
        model_path: str,
        data_path: str,
        examples: Optional[List[List[str]]] = None,
        input_text: Optional[str] = None
):
    """
    Load a trained model and run inference on examples
    
    Args:
        model_path: Path to the saved model
        data_path: Path to the saved data generator
        examples: List of example pairs to run inference on (each is [word1, word2])
        input_text: Raw input text with [MASK] token to predict (e.g., "frog [MASK]")
    """
    # Load model and data generator
    model = SimpleTransformer.load(model_path)
    data_gen = DataGenerator.load(data_path)
    
    # Process command-line input if provided
    if input_text:
        words = input_text.strip().split()
        if len(words) != 2:
            print("Error: Input must be exactly 2 words (e.g., 'frog [MASK]' or '[MASK] green')")
            return
        
        examples = [words]
        
    # Use default examples if none provided
    if not examples:
        examples = [
            ["[MASK]", "green"],
            ["tree", "[MASK]"],
            ["[MASK]", "fast"],
            ["dog", "[MASK]"]
        ]
        
    print("\nRunning inference on examples:")
    
    for example in examples:
        # Convert words to IDs
        try:
            input_ids = np.array([data_gen.word2id.get(word, data_gen.mask_token_id) for word in example])
            mask_pos = None
            
            # Find if there's a mask and where it is
            for i, word in enumerate(example):
                if word == data_gen.mask_token:
                    mask_pos = i
                    break
                
            if mask_pos is None:
                print(f"No {data_gen.mask_token} token found in example: {example}")
                continue
            
            # Run forward pass
            logits, outputs = model.forward(input_ids, mask_pos)
            
            # Get top 5 predictions
            top_k = 5
            top_indices = np.argsort(logits)[-top_k:][::-1]
            top_words = [data_gen.id2word[idx] for idx in top_indices]
            
            # Calculate probabilities
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            top_probs = probs[top_indices]
            
            # Print results
            print(f"\nInput: {' '.join(example)}")
            print(f"Top {top_k} predictions for {data_gen.mask_token}:")
            for word, prob in zip(top_words, top_probs):
                print(f"  {word}: {prob:.4f}")
                
            # Display attention weights
            attn_weights = outputs["attention_weights"]
            print("Attention weights:")
            print(attn_weights)
            print("Word 0 attends to Word 0: {:.4f}, Word 1: {:.4f}".format(
                attn_weights[0, 0], attn_weights[0, 1]))
            print("Word 1 attends to Word 0: {:.4f}, Word 1: {:.4f}".format(
                attn_weights[1, 0], attn_weights[1, 1]))
            
        except KeyError as e:
            print(f"Error: Word not in vocabulary: {e}")


# Example TSV creation function - for demonstration purposes
def create_example_tsv(filepath: str):
    """Create an example TSV file with word pairs"""
    pairs = [
        ["frog", "green", "noun", "attribute"],
        ["green", "frog", "attribute", "noun"],
        ["car", "fast", "noun", "attribute"],
        ["fast", "car", "attribute", "noun"],
        ["dog", "friendly", "noun", "attribute"],
        ["cat", "quiet", "noun", "attribute"],
        ["bird", "small", "noun", "attribute"],
        ["tree", "tall", "noun", "attribute"],
        ["rose", "red", "noun", "attribute"],
        ["sky", "blue", "noun", "attribute"],
        ["grass", "green", "noun", "attribute"],
        ["snow", "white", "noun", "attribute"],
        ["tiger", "striped", "noun", "attribute"],
        ["elephant", "large", "noun", "attribute"],
        ["mouse", "tiny", "noun", "attribute"],
    ]
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["word1", "word2", "type1", "type2"])  # Header
        writer.writerows(pairs)
        
    print(f"Example TSV file created at {filepath}")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Transformer for educational purposes")
    parser.add_argument("--mode", type=str, choices=["train", "inference", "create_tsv"], 
                        help="Whether to train the model, run inference, or create example TSV")
    parser.add_argument("--model_path", type=str, default="model.pkl",
                        help="Path to save/load the model")
    parser.add_argument("--data_path", type=str, default="data_generator.pkl",
                        help="Path to save/load the data generator")
    parser.add_argument("--tsv_path", type=str, default=None,
                        help="Path to TSV file with training data")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Number of training steps (for train mode)")
    parser.add_argument("--input", type=str, default=None,
                        help="Input text for inference (e.g., 'frog [MASK]')")
    
    args = parser.parse_args()
    
    if args.mode == "create_tsv":
        tsv_path = args.tsv_path or "example_pairs.tsv"
        create_example_tsv(tsv_path)
        
    elif args.mode == "train":
        train_model(
            num_steps=args.steps,
            model_path=args.model_path,
            data_path=args.data_path,
            tsv_path=args.tsv_path
        )
        print(f"\nTraining completed. Model saved to {args.model_path}")
        print(f"Data generator saved to {args.data_path}")
        
    elif args.mode == "inference":
        if not os.path.exists(args.model_path) or not os.path.exists(args.data_path):
            print(f"Error: Model file {args.model_path} or data file {args.data_path} not found.")
            print("Please train the model first or specify the correct paths.")
            exit(1)
            
        run_inference(
            model_path=args.model_path,
            data_path=args.data_path,
            input_text=args.input
        )
        
    else:
        parser.print_help()
