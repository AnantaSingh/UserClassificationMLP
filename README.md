# User Classification with MLP

This project implements a Multi-Layer Perceptron (MLP) for classifying users based on accelerometer data. The implementation follows specific requirements:
- Single hidden layer with less than 3 nodes
- Two output nodes representing users A and B
- Training for 1000 iterations
- Saving initial and final weights

## Project Structure

- `data_preprocessing.py`: Handles data loading and preprocessing
- `model.py`: Contains the MLP model implementation
- `train.py`: Training script
- `test.py`: Testing script for evaluation
- `requirements.txt`: Required Python packages

## Features

- Input features:
  - X, Y, Z accelerometer values
  - Magnitude of acceleration
  - Rolling mean and standard deviation
- Hidden layer with 2 nodes
- ReLU activation for hidden layer
- Softmax activation for output layer

## Setup and Usage

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train.py
```
This will:
- Save initial weights to `initial_weights.pth`
- Train for 1000 epochs
- Save final weights to `final_weights.pth`
- Generate a training loss plot (`training_loss.png`)

3. Test the model:
```bash
python test.py
```
This will evaluate the model on the test samples and print the results.

## Model Architecture

- Input Layer: 6 nodes (features)
- Hidden Layer: 2 nodes (ReLU activation)
- Output Layer: 2 nodes (Softmax activation)

## Data Processing

- Removes EID and timestamp columns
- Calculates magnitude of acceleration
- Computes rolling statistics
- Standardizes features using StandardScaler 