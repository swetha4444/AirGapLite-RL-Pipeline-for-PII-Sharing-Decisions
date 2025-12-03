# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from sklearn.model_selection import train_test_split
# from pathlib import Path

# # Import your model
# from context_agent_classifier import ContextAgentClassifier

# # Configuration
# CSV_FILE = "/Users/aringarg/Downloads/690F_FP/final_project/690-Project-Dataset-balanced.csv"
# MODEL_SAVE_PATH = "context_agent_mlp.pth"
# BATCH_SIZE = 16
# EPOCHS = 10
# LEARNING_RATE = 1e-3

# class ContextDataset(Dataset):
#     def __init__(self, texts, labels):
#         self.texts = texts
#         self.labels = labels

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         return self.texts[idx], self.labels[idx]

# def load_data(csv_path):
#     """
#     Load data and create labels based on allowed_* columns.
#     Restaurant = 0, Bank = 1
#     """
#     # if not Path(csv_path).exists():
#     #     # Fallback for common locations
#     #     if Path(f"Downloads/690F_FP/final_project/690-Project-Dataset-final.csv").exists():
#     #         csv_path = "Downloads/690F_FP/final_project/690-Project-Dataset-final.csv"
#     #     else:
#     #         raise FileNotFoundError(f"Could not find {csv_path}")

#     df = pd.read_csv(csv_path)
    
#     texts = []
#     labels = []
    
#     print(f"Loading data from {csv_path}...")
    
#     # Logic: If 'allowed_restaurant' has data, it's Restaurant (0).
#     # If 'allowed_bank' has data, it's Bank (1).
#     for idx, row in df.iterrows():
#         # Clean up column content
#         rest_val = str(row.get('allowed_restaurant', ''))
#         bank_val = str(row.get('allowed_bank', ''))
        
#         # Check if they are valid lists/strings (not 'nan' or empty)
#         is_rest = len(rest_val) > 5 and rest_val.lower() != 'nan'
#         is_bank = len(bank_val) > 5 and bank_val.lower() != 'nan'
        
#         # Input text usually comes from 'task' or 'conversation' column
#         # Using 'conversation' as it likely contains the user query history
#         text = str(row.get('conversation', row.get('task', '')))
        
#         if is_rest:
#             texts.append(text)
#             labels.append(0) # Restaurant
#         elif is_bank:
#             texts.append(text)
#             labels.append(1) # Bank
            
#     print(f"Found {len(texts)} samples. (Restaurant: {labels.count(0)}, Bank: {labels.count(1)})")
#     return texts, labels

# def train():
#     # 1. Prepare Data
#     texts, labels = load_data(CSV_FILE)
    
#     # Split into Train/Val
#     train_texts, val_texts, train_labels, val_labels = train_test_split(
#         texts, labels, test_size=0.2, random_state=42
#     )
    
#     train_dataset = ContextDataset(train_texts, train_labels)
#     val_dataset = ContextDataset(val_texts, val_labels)
    
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
#     # 2. Initialize Model
#     model = ContextAgentClassifier()
#     optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
#     criterion = nn.CrossEntropyLoss()
    
#     print("\nStarting Training...")
    
#     # 3. Training Loop
#     for epoch in range(EPOCHS):
#         model.train()
#         total_loss = 0
        
#         for batch_texts, batch_labels in train_loader:
#             optimizer.zero_grad()
            
#             # Forward pass
#             logits = model(list(batch_texts)) # Convert tuple to list for encoder
            
#             # Calculate loss
#             loss = criterion(logits, batch_labels)
            
#             # Backward pass
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
            
#         # Validation
#         model.eval()
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for batch_texts, batch_labels in val_loader:
#                 logits = model(list(batch_texts))
#                 predictions = torch.argmax(logits, dim=1)
#                 correct += (predictions == batch_labels).sum().item()
#                 total += len(batch_labels)
        
#         val_acc = correct / total
#         print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2%}")

#     # 4. Save Model
#     model.save_model(MODEL_SAVE_PATH)

# if __name__ == "__main__":
#     train()



import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from pathlib import Path

# Import your model
from context_agent_classifier import ContextAgentClassifier

# Configuration
CSV_FILE = "/Users/aringarg/Downloads/690F_FP/final_project/restbankbig.csv"
MODEL_SAVE_PATH = "context_agent_mlp.pth"
BATCH_SIZE = 8  # Reduced batch size for small dataset
EPOCHS = 15     # Increased epochs slightly since data is small
LEARNING_RATE = 1e-3

class ContextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def load_data(csv_path):
    """
    Load data from restaurant_bank_prompts.csv
    Expects columns: 'prompt', 'domain'
    """
    # if not Path(csv_path).exists():
    #     # Check standard upload folder if file not found locally
    #     if Path(f"final_project/{csv_path}").exists():
    #         csv_path = f"final_project/{csv_path}"
    #     # If user just uploaded it, it might be in current dir, checked first
        
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    texts = []
    labels = []
    
    # Map string labels to integers
    label_map = {"restaurant": 0, "bank": 1}
    
    for idx, row in df.iterrows():
        text = str(row['prompt'])
        domain = str(row['domain']).lower().strip()
        
        if domain in label_map:
            texts.append(text)
            labels.append(label_map[domain])
            
    print(f"Found {len(texts)} samples.")
    print(f"Counts: Restaurant (0): {labels.count(0)}, Bank (1): {labels.count(1)}")
    
    return texts, labels

def train():
    # 1. Prepare Data
    try:
        texts, labels = load_data(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: {CSV_FILE} not found. Make sure you uploaded it!")
        return

    # Split into Train/Val
    # specific random_state ensures we get a good mix even with small data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_dataset = ContextDataset(train_texts, train_labels)
    val_dataset = ContextDataset(val_texts, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # 2. Initialize Model
    # Explicitly creating model on CPU to avoid M1/M2 crash
    model = ContextAgentClassifier()
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print("\nStarting Training...")
    print("-" * 40)
    
    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_texts, batch_labels in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(list(batch_texts)) 
            
            # Calculate loss
            loss = criterion(logits, batch_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_texts, batch_labels in val_loader:
                logits = model(list(batch_texts))
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == batch_labels).sum().item()
                total += len(batch_labels)
        
        val_acc = correct / total
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2%}")

    # 4. Save Model
    model.save_model(MODEL_SAVE_PATH)

if __name__ == "__main__":
    train()