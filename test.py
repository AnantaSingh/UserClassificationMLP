import torch
import numpy as np
from data_preprocessing import prepare_dataset, load_and_preprocess_file
from model import UserClassifierMLP

def test_model(model_path="final_weights.pth"):
    # Load the trained model
    model = UserClassifierMLP()
    model.load_weights(model_path)
    model.eval()  # Set to evaluation mode
    
    # Load and preprocess training data to get the scaler
    _, _, scaler = prepare_dataset("DataSamples")
    
    # Test files
    test_files = ["TestSamples/Test1.csv", "TestSamples/Test2.csv", "TestSamples/Test3.csv"]
    expected_labels = ["A", "A", "B"]  # As per the assignment
    
    print("\nTesting Results:")
    print("-" * 50)
    
    for test_file, expected in zip(test_files, expected_labels):
        # Load and preprocess test file
        data = load_and_preprocess_file(test_file)
        data = scaler.transform(data)  # Scale using the same scaler
        
        # Convert to PyTorch tensor
        data = torch.FloatTensor(data)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(data)
            predictions = outputs.mean(dim=0)  # Average predictions across all samples
            
        # Determine the predicted user
        predicted_user = "A" if predictions[1] > predictions[0] else "B"
        
        # Print results
        print(f"File: {test_file}")
        print(f"Expected User: {expected}")
        print(f"Predicted User: {predicted_user}")
        print(f"Confidence: {max(predictions[0], predictions[1]):.4f}")
        print("-" * 50)

if __name__ == "__main__":
    test_model() 