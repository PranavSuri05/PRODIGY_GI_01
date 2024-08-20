from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import torch

# Initialize the GPT-2 model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to end-of-sequence token

# Load your dataset
dataset = load_dataset('text', data_files={'train': 'c:/Users/Bhavya/OneDrive/Desktop/python/dataset.txt'})

# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

# Tokenize dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_dataset['train']

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=4,   # batch size per device during training
    save_steps=10_000,               # number of updates steps before saving model
    save_total_limit=2,              # limit the total amount of checkpoints
)

# Trainer
trainer = Trainer(
    model=model,                     # the instantiated 🤗 Transformers model to be trained
    args=training_args,              # training arguments, defined above
    train_dataset=train_dataset      # training dataset
)

# Define the compute_loss function
def compute_loss(model, inputs):
    labels = inputs["input_ids"]
    outputs = model(**inputs)
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    # Flatten the logits and labels
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss

trainer.compute_loss = compute_loss  # Set the compute_loss function for the trainer

# Train the model
trainer.train()
