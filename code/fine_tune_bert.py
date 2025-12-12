from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MLM MASKING ---

def mask_tokens(input_ids: torch.Tensor, tokenizer, mlm_prob: float = 0.15):
    """
    Apply BERT-style MLM masking.
    - 15% of non-special tokens selected
      * 80% -> [MASK]
      * 10% -> random token
      * 10% -> keep original
    """
    labels = input_ids.clone()

    # Do not mask special or padding tokens
    special_tokens_mask = (
        input_ids.eq(tokenizer.pad_token_id) |
        input_ids.eq(tokenizer.cls_token_id) |
        input_ids.eq(tokenizer.sep_token_id)
    )

    probability_matrix = torch.full(labels.shape, mlm_prob, device=input_ids.device)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # Only masked positions contribute to loss
    labels[~masked_indices] = -100

    # 80% -> [MASK]
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8, device=input_ids.device)).bool() &
        masked_indices
    )
    input_ids[indices_replaced] = tokenizer.mask_token_id

    # 10% -> random token
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5, device=input_ids.device)).bool() &
        masked_indices & ~indices_replaced
    )
    random_words = torch.randint(
        low=0,
        high=len(tokenizer),
        size=labels.shape,
        device=input_ids.device,
        dtype=torch.long
    )
    input_ids[indices_random] = random_words[indices_random]

    # Remaining 10%: keep original token
    return input_ids, labels


# --- LoRA wrapper ---

def apply_lora_to_bert(model, r: int = 8, alpha: int = 16, dropout: float = 0.05):
    """
    Wrap a BertForMaskedLM with LoRA adapters using PEFT.
    """
    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,  # close enough for MLM token-level work
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=["query", "key", "value", "dense"],  # typical for BERT
    )
    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()
    return lora_model


# --- Evaluation helper ---

@torch.no_grad()
def evaluate_mlm(model, dataloader, tokenizer, device=DEVICE):
    model.eval()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        masked_input_ids, labels = mask_tokens(input_ids.clone(), tokenizer)
        masked_input_ids = masked_input_ids.to(device)
        labels = labels.to(device)

        outputs = model(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    model.train()
    return avg_loss


# --- Training loop ---

def train_bert(
    model,
    train_loader,
    tokenizer,
    val_loader=None,
    epochs: int = 3,
    lr: float = 5e-5,
    device: str = "cuda",
):
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)

    num_training_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    for epoch in range(epochs):
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # MLM masking
            masked_input_ids, labels = mask_tokens(input_ids.clone(), tokenizer)
            masked_input_ids = masked_input_ids.to(device)
            labels = labels.to(device)

            outputs = model(
                input_ids=masked_input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} train loss: {avg_loss:.4f}")

        if val_loader is not None:
            val_loss = evaluate_mlm(model, val_loader, tokenizer, device)
            print(f"Epoch {epoch+1} val loss: {val_loss:.4f}")

    print("Training complete.")
    return model
