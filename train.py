import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import prepare_dataset
from model import UserClassifierMLP

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Hyperparameters
learning_rate = 0.001
num_epochs = 1000

def train_model():
    # Load and preprocess training data
    X_train, y_train, scaler = prepare_dataset("DataSamples")
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    
    # Initialize model
    model = UserClassifierMLP()
    print("\nModel structure:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params}")
    
    # Save initial weights
    model.save_weights("initial_weights.pth")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    losses = []
    accuracies = []
    print("\nStarting training...")
    
    try:
        for epoch in range(num_epochs):
            # Forward pass
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch+1}")
                print("Model weights:")
                for name, param in model.named_parameters():
                    print(f"{name}: {param.data}")
                break
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total = y_train.size(0)
            correct = (predicted == y_train).sum().item()
            accuracy = 100 * correct / total
            
            # Record metrics
            losses.append(loss.item())
            accuracies.append(accuracy)
            
            # Print progress
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}]')
                print(f'Loss: {loss.item():.4f}')
                print(f'Accuracy: {accuracy:.2f}%')
                print("Sample predictions:", predicted[:10].tolist())
                print("Sample targets:", y_train[:10].tolist())
    
    except Exception as e:
        print(f"Training interrupted due to error: {str(e)}")
    
    # Save final weights
    model.save_weights("final_weights.pth")
    
    # Plot training metrics
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Training Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()
    
    print("\nTraining completed!")
    return model, scaler

if __name__ == "__main__":
    train_model() 