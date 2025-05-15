from transformers import AutoTokenizer, AutoModel
from PM4DA2E import PM4DA2EModel
import torch

# Load mBERT or XLM-R (supports multiple languages)
MODEL_NAME = "PM4DA2E"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)


#MUX-
def multiplex(sent_dict):
    embeddings = {}
    for lang, sent in sent_dict.items():
        inputs = tokenizer(sent, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean pooling across tokens (excluding padding)
        last_hidden = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        masked_hidden = last_hidden * attention_mask.unsqueeze(-1)
        sum_hidden = masked_hidden.sum(1)
        count_non_pad_tokens = attention_mask.sum(1).unsqueeze(-1)
        sentence_embedding = sum_hidden / count_non_pad_tokens
        embeddings[lang] = sentence_embedding
    return embeddings

# Get multiplexed embeddings
lang_embeddings = multiplex(sentence)

# Combine into a single tensor (multiplexed)
multiplexed = torch.stack([lang_embeddings[lang] for lang in ['en', 'hi', 'de', 'es', 'fr']], dim=0)

def demultiplex(multiplexed_tensor, langs=['en', 'hi', 'de', 'es', 'fr']):
    return {lang: multiplexed_tensor[i] for i, lang in enumerate(langs)}

