import torch
from context_agent_classifier import ContextAgentClassifier

# 1. Config
MODEL_PATH = "context_agent_mlp.pth"
LABELS = {0: "restaurant", 1: "bank"}

def load_prediction_model():
    model = ContextAgentClassifier()
    model.load_model(MODEL_PATH)
    return model

def main():
    # Load the trained brain
    print(f"Loading model from {MODEL_PATH}...")
    model = load_prediction_model()
    
    print("\n" + "="*50)
    print(" 🧠 CONTEXT AGENT LIVE TEST")
    print(" Type a sentence to see if it predicts BANK or RESTAURANT.")
    print(" (Type 'quit' to exit)")
    print("="*50 + "\n")
    
    while True:
        user_input = input(">> Enter prompt: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        # Predict
        result = model.predict(user_input, LABELS)
        
        # Display Result
        label = result['label'].upper()
        conf = result['confidence'] * 100
        
        print(f"   [Prediction]: {label} (Confidence: {conf:.1f}%)")
        print("-" * 30)

if __name__ == "__main__":
    main()