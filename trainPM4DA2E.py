import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from PM4DA2E import PM4DA2EModel
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Load XLNI dataset
dataset = load_dataset('xlni')

# Load the pre-trained Multilingual BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

# Function to tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

# Apply tokenization to train and test data
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split into training and validation datasets
train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

# Load pre-trained Multilingual PM4DA2EModel model for sequence classification
model = PM4DA2EModel()

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch",     # evaluate after each epoch
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,           # evaluation dataset
    tokenizer=tokenizer                  # tokenizer for encoding inputs
)

# Train the model
trainer.train()

# Save the model and tokenizer after training
trainer.save_model("./PM4DA2EModel_model")
tokenizer.save_pretrained("./PM4DA2EModel_tokenizer")
