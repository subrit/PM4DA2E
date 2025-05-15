import torch
from transformers import BertTokenizer
from PM4DA2E import PM4DA2EModel

# Load pre-trained BERT
model_name = "bert-base-multilingual-cased"
model = PM4DA2EModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Set model to eval mode
model.eval()

# Apply dynamic quantization to Linear layers
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save quantized model if needed
torch.save(quantized_model.state_dict(), "PM4DA2E_quantized.pth")
