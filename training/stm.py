import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# Define constants
MODEL_DIR = "E:\\YuyangGPT\\models\\minilm-custom-eos"
CHUNK_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer + embedding model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModel.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()


class ChunkTransformer(nn.Module):
    def __init__(self, hidden_size, num_layers=2, num_heads=4):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Learnable summary token
        self.summary_token = nn.Parameter(
            torch.randn(1, 1, hidden_size)
        )

    def forward(self, hidden_states, attention_mask):
        """
        hidden_states: [B, T, d]
        attention_mask: [B, T]
        """

        B, T, d = hidden_states.shape

        # Expand summary token for batch
        summary = self.summary_token.expand(B, 1, d)

        # Prepend summary token
        x = torch.cat([summary, hidden_states], dim=1)  # [B, T+1, d]

        # Build attention mask (1 = keep, 0 = mask)
        summary_mask = torch.ones(B, 1, device=attention_mask.device)
        attn_mask = torch.cat([summary_mask, attention_mask], dim=1)

        # TransformerEncoder uses True = masked
        key_padding_mask = attn_mask == 0

        out = self.encoder(
            x,
            src_key_padding_mask=key_padding_mask
        )

        # Return summary token output
        return out[:, 0]  # [B, d]

stm = ChunkTransformer(
    hidden_size=model.config.hidden_size,
    num_layers=2,
    num_heads=4
).to(DEVICE)

def chunk_tokens(input_ids, attention_mask, chunk_size=128):
    chunks = []

    seq_len = input_ids.size(1)

    for start in range(0, seq_len, chunk_size):
        end = start + chunk_size

        chunk_ids = input_ids[:, start:end]
        chunk_mask = attention_mask[:, start:end]

        if chunk_ids.size(1) == 0:
            continue

        chunks.append({
            "input_ids": chunk_ids,
            "attention_mask": chunk_mask
        })

    return chunks

def encode_chunks(chunks, model):
    summaries = []

    with torch.no_grad():
        for chunk in chunks:
            outputs = model(
                input_ids=chunk["input_ids"],
                attention_mask=chunk["attention_mask"]
            )

            # Token-level hidden states
            h = outputs.last_hidden_state  # [1, T, d]

            summary = stm(
                hidden_states=h,
                attention_mask=chunk["attention_mask"]
            )

            summaries.append(summary.squeeze(0))  # [d]

    return torch.stack(summaries)  # [num_chunks, d]

class STMTrainer(nn.Module):
    def __init__(self, stm, hidden_size, vocab_size):
        super().__init__()
        self.stm = stm
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states, attention_mask):
        """
        hidden_states: [B, T, d]  (from MiniLM)
        attention_mask: [B, T]
        """
        summary = self.stm(hidden_states, attention_mask)  # [B, d]
        logits = self.lm_head(summary)                      # [B, vocab]
        return logits

