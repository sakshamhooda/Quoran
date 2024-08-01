from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import load_metric
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_csv('cleaned_data.csv')

# Split the data into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# Prepare the data for T5 model
def convert_to_t5_format(df):
    df['input_text'] = "question: " + df['question'] + " </s>"
    df['target_text'] = df['answer'] + " </s>"
    return df[['input_text', 'target_text']]

train_df = convert_to_t5_format(train_df)
val_df = convert_to_t5_format(val_df)

# Path to the saved model
saved_model_path = 'results/checkpoint-18500'

# Load the saved model and tokenizer
print(f"Loading model from {saved_model_path}...")
model = T5ForConditionalGeneration.from_pretrained(saved_model_path)
model.to(device)
tokenizer = T5Tokenizer.from_pretrained('t5-small')
print("Model loaded.")
# Custom Dataset class
class QADataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        question = str(self.data.iloc[index]['input_text'])
        answer = str(self.data.iloc[index]['target_text'])

        # Encode the inputs and targets
        input_encodings = self.tokenizer(question, truncation=True, padding='max_length', max_length=self.max_len, return_tensors="pt")
        target_encodings = self.tokenizer(answer, truncation=True, padding='max_length', max_length=self.max_len, return_tensors="pt")

        input_ids = input_encodings['input_ids'].squeeze()
        attention_mask = input_encodings['attention_mask'].squeeze()
        labels = target_encodings['input_ids'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# Create datasets
train_dataset = QADataset(train_df, tokenizer)
val_dataset = QADataset(val_df, tokenizer)
# Function to generate predictions
def generate_predictions(dataset):
    model.eval()
    predictions = []
    for i in range(len(dataset)):
        data = dataset[i]
        input_ids = data['input_ids'].unsqueeze(0).to(model.device)
        attention_mask = data['attention_mask'].unsqueeze(0).to(model.device)
        output = model.generate(input_ids, attention_mask=attention_mask, max_length=50)
        prediction = tokenizer.decode(output[0], skip_special_tokens=True)
        predictions.append(prediction)
        if i < 5:  # Debug: Print first 5 predictions
            print(f"Prediction {i+1}: {prediction}")
    return predictions

# Generate predictions on the validation set
print("Generating predictions on the validation set...")
val_predictions = generate_predictions(val_dataset)
print("Predictions generated.")

# Extract references from the validation DataFrame
print("Extracting references from the validation set...")
references = val_df['target_text'].tolist()
print(f"References extracted: {references[:5]}")  # Debug: Print first 5 references
#additional debugging step
# Ensure all predictions and references are strings
val_predictions = [str(pred) for pred in val_predictions]
references = [str(ref) for ref in references]

# Check for any unexpected data types
for i, (pred, ref) in enumerate(zip(val_predictions, references)):
    if not isinstance(pred, str) or not isinstance(ref, str):
        print(f"Unexpected type at index {i}: pred type = {type(pred)}, ref type = {type(ref)}")
        
import evaluate
print("Calculating BLEU score...")
# BLEU Score
bleu_metric = load_metric('bleu')
bleu_score = bleu_metric.compute(predictions=[pred.split() for pred in val_predictions], references=[[ref.split()] for ref in references])
print(f"BLEU Score: {bleu_score}")

print("Calculating ROUGE score...")
# ROUGE Score
rouge_metric = load_metric('rouge')
rouge_score = rouge_metric.compute(predictions=val_predictions, references=references)
print(f"ROUGE Score: {rouge_score}")

def compute_f1(predictions, references):
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    num_samples = len(predictions)
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        common_tokens = set(pred_tokens) & set(ref_tokens)
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            precision = 0
            recall = 0
            f1 = 0
        else:
            precision = len(common_tokens) / len(pred_tokens)
            recall = len(common_tokens) / len(ref_tokens)
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        
        # Debug: Print precision, recall, and F1 score for the first 5 samples
        if num_samples <= 5:
            print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    
    avg_precision = total_precision / num_samples
    avg_recall = total_recall / num_samples
    avg_f1 = total_f1 / num_samples
    
    return avg_precision, avg_recall, avg_f1

avg_precision, avg_recall, avg_f1 = compute_f1(val_predictions, references)
print(f"Average Precision: {avg_precision}")
print(f"Average Recall: {avg_recall}")
print(f"Average F1 Score: {avg_f1}")
