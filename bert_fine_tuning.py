import torch
import torch.nn as nn
import os
from transformers import BertTokenizer, BertModel, AdamW

# Define the fine-tuned model for sentiment analysis
class SentimentClassifier(nn.Module):
    def __init__(self, bert_model):
        super(SentimentClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.bert.config.hidden_size, 2)  # Binary classification (positive/negative sentiment)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs['pooler_output']
        logits = self.dropout(pooled_output)
        logits = self.linear(logits)
        return logits

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Load and preprocess the IMDb dataset (assuming you have it in a CSV file)
import csv

def read_csv_file(file_path):
    data = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Read the first row as column headings
        for row in reader:
            data.append(dict(zip(headers, row)))
    return data

# Replace 'file_path.csv' with the actual path to your CSV file
file_path = './IMDB_Dataset.csv'
csv_data = read_csv_file(file_path)

# Now you can access the data using column headings
input_texts = []
labels = []
for row in csv_data:
    input_texts.append(row['review'])
    if row['sentiment'] == 'positive':
        labels.append(1)
    else:
        labels.append(0)

# ... (code to load and preprocess data)

# Convert text data to BERT input format
# input_texts = ['Sample text 1', 'Sample text 2', ...]  # List of input texts
# labels = [0, 1, ...]  # List of corresponding sentiment labels (0: negative, 1: positive)

input_ids = []
attention_masks = []
for text in input_texts:
    encoded_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,  # Adjust as needed
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids.append(encoded_text['input_ids'])
    attention_masks.append(encoded_text['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Create dataloader
from torch.utils.data import TensorDataset, DataLoader, random_split

dataset = TensorDataset(input_ids, attention_masks, labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Fine-tune the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentimentClassifier(bert_model).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)  # Adjust learning rate as needed
loss_fn = nn.CrossEntropyLoss()

num_epochs = 3  # Adjust as needed
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        input_ids, attention_masks, labels = batch
        input_ids, attention_masks, labels = input_ids.to(device), attention_masks.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids, attention_masks)
        loss = loss_fn(logits, labels)
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping to avoid exploding gradients
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss}')

    # Validation
    model.eval()
    val_loss = 0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_masks, labels = batch
            input_ids, attention_masks, labels = input_ids.to(device), attention_masks.to(device), labels.to(device)
            
            logits = model(input_ids, attention_masks)
            loss = loss_fn(logits, labels)
            val_loss += loss.item()

            _, predicted_labels = torch.max(logits, 1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += len(labels)

    avg_val_loss = val_loss / len(val_dataloader)
    accuracy = correct_predictions / total_samples
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss}, Validation Accuracy: {accuracy}')

model_path =  './final'
os.makedirs(model_path)
path_checkpoint = model_path + "/net_checkpoint.pkl"
net_checkpoint =    {"model": model.state_dict(),
                    "optimizer": optimizer.state_dict()}
                    #   "step": curr_steps,
                    #   "episode": curr_episodes,
                    #   "tasks":curr_tasks}
                    
torch.save(net_checkpoint, path_checkpoint)