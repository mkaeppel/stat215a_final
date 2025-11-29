import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm


# --- MLM MASKING ---
def mask_tokens(input_ids, vocab_size, mask_token_id, pad_token_id, mlm_prob=0.15):
    '''
    Implements Masked Language Modeling (MLM) masking strategy.
    
    For each token, with probability mlm_prob:
    - 80% of the time: replace with [MASK] token
    - 10% of the time: replace with random token
    - 10% of the time: keep original token
    
    Args:
        input_ids: torch.Tensor, shape (batch_size, seq_length)
        vocab_size: int, size of vocabulary
        mask_token_id: int, ID of [MASK] token
        pad_token_id: int, ID of [PAD] token (we don't mask padding)
        mlm_prob: float, probability of masking a token (default 0.15)
    
    Returns:
        inputs: torch.Tensor, masked input_ids
        labels: torch.Tensor, original tokens for loss calculation (-100 for non-masked)
    '''
    device = input_ids.device
    # Clone input_ids to create labels (targets for prediction)
    labels = input_ids.clone()
    
    # Create probability matrix for masking decisions
    # Shape: (batch_size, seq_length)
    probability_matrix = torch.full(labels.shape, mlm_prob, device=device)
    
    # Don't mask special tokens (padding tokens)
    # Set their probability to 0
    special_tokens_mask = input_ids == pad_token_id
    special_tokens_mask = special_tokens_mask.to(device)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    
    # Randomly select tokens to mask based on probability_matrix
    masked_indices = torch.bernoulli(probability_matrix).bool()
    
    # Set labels to -100 (ignored by CrossEntropyLoss) for tokens we're NOT predicting
    labels[~masked_indices] = -100
    
    # for 80% of masked tokens, replace with [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=device)).bool() & masked_indices
    input_ids[indices_replaced] = mask_token_id
    
    # for 10% of masked tokens, replace with random token
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=device)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=device)
    input_ids[indices_random] = random_words[indices_random]
    
    # for the remaining 10%, keep original token (i.e. do nothing)
    
    return input_ids, labels

def train_bert(model, dataloader, tokenizer, epochs=3, lr=5e-5, device='cuda'):
    '''
    Training loop for BERT with Masked Language Modeling.
    
    Args:
        model: BERT model (BertForMaskedLM from transformers)
        dataloader: DataLoader with tokenized text batches
        tokenizer: BERT tokenizer
        epochs: int, number of training epochs
        lr: float, learning rate (5e-5 is standard for BERT)
        device: str, 'cuda' or 'cpu'
    '''
    # Move model to device 
    model = model.to(device)
    model.train() 
    
    # Initialize optimizer 
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # Get special token IDs from tokenizer
    vocab_size = tokenizer.vocab_size
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get input_ids and attention_mask from batch
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Apply MLM masking
            masked_input_ids, labels = mask_tokens(
                input_ids.clone(),
                vocab_size=vocab_size,
                mask_token_id=mask_token_id,
                pad_token_id=pad_token_id,
                mlm_prob=0.15
            )
            
            # Move labels to device
            labels = labels.to(device)
            
            # Forward pass
            # BertForMaskedLM returns loss when labels are provided
            outputs = model(
                input_ids=masked_input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()  
            loss.backward()        
            optimizer.step()   
            
            # Track loss
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Print epoch summary
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
    
    print("Training complete!")
    return model


class TextDataset(torch.utils.data.Dataset):
    '''
    Simple dataset for text data.
    Tokenizes texts and returns batches for training.
    '''
    def __init__(self, texts, tokenizer, max_length=128):
        '''
        Args:
            texts: list of strings (your story texts)
            tokenizer: BERT tokenizer
            max_length: maximum sequence length
        '''
        print(f"Tokenizing {len(texts)} texts...")
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        print(f"Tokenization complete. Shape: {self.encodings['input_ids'].shape}")
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx]
        }

