# import torch
# import torch.nn as nn
# from sentence_transformers import SentenceTransformer
# import numpy as np

# class ContextAgentClassifier(nn.Module):
#     def __init__(self, model_name='all-MiniLM-L6-v2', num_labels=2, hidden_dim=64):
#         """
#         Initializes the Context Agent with a lightweight encoder and MLP classifier.
#         """
#         super(ContextAgentClassifier, self).__init__()
        
#         # FIX 1: Explicitly force CPU to avoid macOS MPS/Metal errors
#         self.device = torch.device("cpu")
        
#         # 1. The Encoder
#         # We pass device='cpu' to ensure embeddings stay on CPU
#         self.encoder = SentenceTransformer(model_name, device='cpu')
        
#         # Get embedding dimension (384 for MiniLM-L6-v2)
#         self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        
#         # 2. The MLP Classifier
#         self.classifier = nn.Sequential(
#             nn.Linear(self.embedding_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, num_labels)
#         )
        
#         # Ensure the whole module is on the CPU
#         self.to(self.device)

#     def forward(self, text_input):
#         """
#         Forward pass: Encode text -> Classify context.
#         """
#         # Step 1: Encode text to embeddings
#         with torch.no_grad():
#             embeddings = self.encoder.encode(text_input, convert_to_tensor=True)
            
#         # Step 2: Pass through MLP
#         logits = self.classifier(embeddings)
#         return logits

#     def predict(self, text_input, id2label):
#         """
#         Utility function to get readable labels.
#         """
#         self.eval()
#         with torch.no_grad():
#             # FIX 2: Ensure input is a list to get a 2D batch tensor [1, Dim]
#             # otherwise softmax(dim=1) fails on a 1D tensor.
#             if isinstance(text_input, str):
#                 input_batch = [text_input]
#             else:
#                 input_batch = text_input

#             logits = self(input_batch)
#             probs = torch.softmax(logits, dim=1)
            
#             # Get the result for the first item in the batch
#             predicted_idx = torch.argmax(probs, dim=1)[0].item()
#             confidence = probs[0][predicted_idx].item()
            
#         return {
#             "label": id2label[predicted_idx],
#             "confidence": confidence,
#             "probabilities": probs.tolist()[0]
#         }

#     def save_model(self, path):
#         """Save just the MLP state dict (encoder is frozen/standard)"""
#         torch.save(self.classifier.state_dict(), path)
#         print(f"Model saved to {path}")

#     def load_model(self, path):
#         """Load the MLP state dict"""
#         self.classifier.load_state_dict(torch.load(path, map_location=self.device))
#         self.eval()
#         print(f"Model loaded from {path}")

# # --- Example Usage ---
# if __name__ == "__main__":
#     # Define labels 
#     LABELS = {0: "restaurant", 1: "bank"}
    
#     # Initialize model
#     model = ContextAgentClassifier()
    
#     # Sample inputs
#     test_queries = [
#         "I need to reserve a table for two tonight.",
#         "What is my current checking account balance?",
#     ]
    
#     print(f"--- Context Agent Classification (MiniLM + MLP) ---")
#     for query in test_queries:
#         result = model.predict(query, LABELS)
#         print(f"Query: '{query}'")
#         print(f"Prediction: {result['label'].upper()} (Conf: {result['confidence']:.4f})")



import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import numpy as np

class ContextAgentClassifier(nn.Module):
    def __init__(self, model_name='all-MiniLM-L6-v2', num_labels=2, hidden_dim=64):
        """
        Initializes the Context Agent with a lightweight encoder and MLP classifier.
        """
        super(ContextAgentClassifier, self).__init__()
        
        # FIX 1: Explicitly force CPU to avoid macOS MPS/Metal errors
        self.device = torch.device("cpu")
        
        # 1. The Encoder
        # We pass device='cpu' to ensure embeddings stay on CPU
        self.encoder = SentenceTransformer(model_name, device='cpu')
        
        # Get embedding dimension (384 for MiniLM-L6-v2)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        
        # 2. The MLP Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_labels)
        )
        
        # Ensure the whole module is on the CPU
        self.to(self.device)

    def forward(self, text_input):
        """
        Forward pass: Encode text -> Classify context.
        """
        # Step 1: Encode text to embeddings
        with torch.no_grad():
            embeddings = self.encoder.encode(text_input, convert_to_tensor=True)
        
        # FIX 2: Clone the tensor to detach it from "inference mode".
        # This allows the MLP to use it for training (backward pass) without errors.
        embeddings = embeddings.clone()
            
        # Step 2: Pass through MLP
        logits = self.classifier(embeddings)
        return logits

    def predict(self, text_input, id2label):
        """
        Utility function to get readable labels.
        """
        self.eval()
        with torch.no_grad():
            # FIX 3: Ensure input is a list to get a 2D batch tensor [1, Dim]
            # otherwise softmax(dim=1) fails on a 1D tensor.
            if isinstance(text_input, str):
                input_batch = [text_input]
            else:
                input_batch = text_input

            logits = self(input_batch)
            probs = torch.softmax(logits, dim=1)
            
            # Get the result for the first item in the batch
            predicted_idx = torch.argmax(probs, dim=1)[0].item()
            confidence = probs[0][predicted_idx].item()
            
        return {
            "label": id2label[predicted_idx],
            "confidence": confidence,
            "probabilities": probs.tolist()[0]
        }

    def save_model(self, path):
        """Save just the MLP state dict (encoder is frozen/standard)"""
        torch.save(self.classifier.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load the MLP state dict"""
        self.classifier.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()
        print(f"Model loaded from {path}")

# --- Example Usage ---
if __name__ == "__main__":
    # Define labels 
    LABELS = {0: "restaurant", 1: "bank"}
    
    # Initialize model
    model = ContextAgentClassifier()
    
    # Sample inputs
    test_queries = [
        "I need to reserve a table for two tonight.",
        "What is my current checking account balance?",
    ]
    
    print(f"--- Context Agent Classification (MiniLM + MLP) ---")
    for query in test_queries:
        result = model.predict(query, LABELS)
        print(f"Query: '{query}'")
        print(f"Prediction: {result['label'].upper()} (Conf: {result['confidence']:.4f})")