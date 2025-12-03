import torch
import os
from MLP.context_agent_classifier import ContextAgentClassifier

# Singleton pattern: Load the model once, keep it in memory
_MODEL = None
_LABELS = {0: "restaurant", 1: "bank"}
_DEVICE = torch.device("cpu") # Force CPU for safety

def load_model(model_path="context_agent_mlp.pth"):
    """
    Loads the model into the global _MODEL variable.
    """
    global _MODEL
    if _MODEL is None:
        # Check if file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        print(f"Loading Context Agent from {model_path}...")
        model = ContextAgentClassifier()
        model.load_model(model_path)
        model.to(_DEVICE)
        model.eval() # Set to inference mode
        _MODEL = model
    return _MODEL

def predict_context(prompt: str, threshold: float = 0.0) -> dict:
    """
    The Main Interface Function.
    
    Args:
        prompt (str): The user input text.
        threshold (float): Optional tunable parameter. If confidence is below this,
                           we might return "uncertain" or flag it.
                           
    Returns:
        dict: {
            "label": "bank" or "restaurant",
            "confidence": float (0.0 to 1.0),
            "probabilities": [prob_restaurant, prob_bank],
            "is_confident": bool
        }
    """
    # 1. Ensure model is loaded
    model = load_model()
    
    # 2. Run Prediction
    # The model.predict method you wrote already handles tokenization
    result = model.predict(prompt, _LABELS)
    
    # 3. Add "Tunable" Logic (Thresholding)
    # The proposal mentions "Final inclusion confidence combines rule-based scores and classifier probabilities"[cite: 33].
    # So returning the raw probabilities is crucial for Phase 4.
    
    is_confident = result['confidence'] >= threshold
    
    return {
        "label": result['label'],
        "confidence": result['confidence'],
        "probabilities": result['probabilities'], # Critical for Fusion Layer 
        "is_confident": is_confident
    }

# Quick Test if run directly
if __name__ == "__main__":
    # Example of how the Integration person will use it:
    print(predict_context("I need to transfer money"))