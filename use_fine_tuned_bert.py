import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
# Load the fine-tuned model
bert_model = BertModel.from_pretrained('bert-base-uncased')
path_checkpoint = './final/net_checkpoint.pkl'
sentiment_model = SentimentClassifier(bert_model).to(device)
sentiment_model.load_state_dict(torch.load(path_checkpoint)['model'])
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Use the same tokenizer used during training

file  = pd.read_csv('final_df.csv')
numpy_arrays = file.to_numpy()

review_rating_values = []

for id, input_text, summary, summ_rating in numpy_arrays:
    # input_text = "This is a sample sentence for inference."
    print(input_text)
    inputs = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=99999,  # Adjust as needed to match the input length during training
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    # print(model)

    # Move the inputs to the appropriate device (CPU/GPU)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Forward pass through the model
    with torch.no_grad():
        outputs = sentiment_model(input_ids, attention_mask=attention_mask)

    # Extract the outputs you need (e.g., logits for classification)
    logits = outputs['pooler_output']  # For classification tasks, use 'pooler_output'

    # Make predictions (e.g., get the predicted class for sentiment analysis)
    _, predicted_class = torch.max(logits, 1)

    # Map the predicted class index to the corresponding label (if needed)
    label_map = {0: 'Negative', 1: 'Positive'}  # Replace with your label mapping
    predicted_label = label_map[predicted_class.item()]
    review_rating_values.append(predicted_label)

dic = {'new': review_rating_values}
df = pd.DataFrame(dic)
# df_csv['Names'] = df.Name
file['review_rating'] = dic.new
file.to_csv('try.csv', index=False, mode= 'w')
print(f"Input Text: {input_text}")
print(f"Predicted Label: {predicted_label}")
