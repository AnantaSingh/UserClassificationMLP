import torch
from model import UserClassifierMLP

def print_weights(weight_file):
    print(f"\nLoading weights from: {weight_file}")
    print("-" * 50)
    
    # Load the model and its weights
    model = UserClassifierMLP()
    model.load_weights(weight_file)
    
    # Print each layer's weights and biases
    for name, param in model.named_parameters():
        print(f"\n{name}:")
        print("Shape:", param.shape)
        print("Values:")
        print(param.data.numpy())
        print("\nStatistics:")
        print(f"Mean: {param.data.mean().item():.6f}")
        print(f"Std: {param.data.std().item():.6f}")
        print(f"Min: {param.data.min().item():.6f}")
        print(f"Max: {param.data.max().item():.6f}")
        print("-" * 50)

if __name__ == "__main__":
    print("\nInitial Weights:")
    print("=" * 50)
    print_weights("initial_weights.pth")
    
    print("\nFinal Weights:")
    print("=" * 50)
    print_weights("final_weights.pth") 