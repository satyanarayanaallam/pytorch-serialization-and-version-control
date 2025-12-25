# pytorch-serialization-and-version-control
# PyTorch Model Training: From Creation to Inference - Complete Learning Guide

**Date:** December 24, 2025  
**Topic:** PyTorch Model Creation, Training, Checkpointing, and Inference  
**Difficulty Level:** Beginner to Intermediate

---

## Table of Contents

1. [Introduction](#introduction)
2. [Understanding Tensor Dimensions](#understanding-tensor-dimensions)
3. [Convolution and Pooling Formulas](#convolution-and-pooling-formulas)
4. [Building a CNN Model](#building-a-cnn-model)
5. [Training Loop with Validation](#training-loop-with-validation)
6. [Model Checkpointing](#model-checkpointing)
7. [Inference and Predictions](#inference-and-predictions)
8. [Resuming Training from Checkpoint](#resuming-training-from-checkpoint)
9. [Transfer Learning](#transfer-learning)
10. [Complete Code Example](#complete-code-example)
11. [Key Takeaways](#key-takeaways)

---

## Introduction

Today, we covered the complete workflow of building, training, and deploying PyTorch neural networks. This guide walks through each step with practical examples, formulas, and best practices.

### What We'll Learn
- How to create neural network architectures (specifically CNNs)
- How to implement a proper training loop with validation
- How to save and load model checkpoints
- How to perform inference on new data
- How to resume training from saved checkpoints
- How to apply transfer learning techniques

---

## Understanding Tensor Dimensions

Before building models, it's crucial to understand tensor shapes and when to use different dimensions.

### 1D Tensors: `(batch_size,)`
Used for single features or labels.

```python
y = torch.randint(0, 10, (1000,))  # 1000 class labels
```

**Key Point:** The shape must be a **tuple**, not an integer. `(1000,)` is correct, `1000` will cause an error.

### 2D Tensors: `(batch_size, features)`
The most common format for tabular data and fully connected networks.

```python
X = torch.randn(32, 784)  # 32 samples, 784 features each
```

### 3D Tensors: `(batch_size, sequence_length, features)`
Used for sequential data like text or time-series.

```python
X = torch.randn(32, 50, 100)  # 32 sequences, 50 timesteps, 100 features per step
```

### 4D Tensors: `(batch_size, channels, height, width)`
The standard format for image data in CNNs.

```python
X = torch.randn(32, 3, 224, 224)  # 32 RGB images, 224×224 pixels
```

### 5D+ Tensors: `(batch, channels, depth, height, width)`
Used for 3D data like medical imaging or videos.

```python
X = torch.randn(8, 1, 64, 64, 64)  # 8 3D medical scans, grayscale, 64×64×64
```

### Quick Reference Table

| Tensor Dim | Shape Example | Use Case |
|-----------|--------------|----------|
| 1D | `(1000,)` | Labels, 1D signals |
| 2D | `(32, 784)` | Tabular data, FC networks |
| 3D | `(32, 50, 100)` | Text, sequences, time-series |
| 4D | `(32, 3, 224, 224)` | Images, CNNs |
| 5D+ | `(8, 1, 64, 64, 64)` | 3D volumes, videos |

---

## Convolution and Pooling Formulas

Understanding how dimensions change through layers is critical for building correct architectures.

### Convolution Formula

$$\text{Output Size} = \left\lfloor \frac{\text{Input Size} - \text{Kernel Size} + 2 \times \text{Padding}}{\text{Stride}} \right\rfloor + 1$$

**Variables:**
- **Input Size (I):** Spatial dimension of input (e.g., 32 for 32×32 image)
- **Kernel Size (K):** Size of convolutional filter (e.g., 3 for 3×3)
- **Stride (S):** Step size of kernel movement (default: 1)
- **Padding (P):** Zero-padding added around input (default: 0)

#### Example 1: Basic Convolution
```python
nn.Conv2d(3, 16, kernel_size=3)
# Input: (batch, 3, 32, 32)
# Kernel: 3×3, Stride: 1 (default), Padding: 0 (default)

Output = floor((32 - 3 + 0) / 1) + 1 = 30
# Result: (batch, 16, 30, 30)
```

#### Example 2: Convolution with Padding (Same Padding)
```python
nn.Conv2d(3, 16, kernel_size=3, padding=1)
# Input: (batch, 3, 32, 32)

Output = floor((32 - 3 + 2×1) / 1) + 1 = 32
# Result: (batch, 16, 32, 32)  ← Output size same as input!
```

#### Example 3: Convolution with Stride
```python
nn.Conv2d(3, 16, kernel_size=3, stride=2)
# Input: (batch, 3, 32, 32)

Output = floor((32 - 3 + 0) / 2) + 1 = 15
# Result: (batch, 16, 15, 15)
```

### Pooling Formula

$$\text{Output Size} = \left\lfloor \frac{\text{Input Size} - \text{Pool Size}}{\text{Stride}} \right\rfloor + 1$$

**Note:** Pooling typically uses **no padding** (default: 0).

#### Example 1: Standard MaxPool
```python
nn.MaxPool2d(kernel_size=2, stride=2)
# Input: (batch, channels, 32, 32)

Output = floor((32 - 2) / 2) + 1 = 16
# Result: (batch, channels, 16, 16)
```

#### Example 2: MaxPool with Different Parameters
```python
nn.MaxPool2d(kernel_size=3, stride=1)
# Input: (batch, channels, 32, 32)

Output = floor((32 - 3) / 1) + 1 = 30
# Result: (batch, channels, 30, 30)
```

#### Example 3: MaxPool with Default Stride
```python
nn.MaxPool2d(kernel_size=2)  # stride defaults to kernel_size
# Input: (batch, channels, 32, 32)

Output = floor((32 - 2) / 2) + 1 = 16
# Result: (batch, channels, 16, 16)
```

### Quick Reference

| Operation | Formula | Default Stride | Default Padding |
|-----------|---------|-----------------|-----------------|
| Convolution | $\left\lfloor \frac{I - K + 2P}{S} \right\rfloor + 1$ | 1 | 0 |
| Pooling | $\left\lfloor \frac{I - K}{S} \right\rfloor + 1$ | K (kernel size) | 0 |

---

## Building a CNN Model

Now that we understand dimensions, let's build a Convolutional Neural Network.

### Model Architecture Design

For CIFAR-10 style images (32×32 RGB):
- **Input:** `(batch_size, 3, 32, 32)`
- **Conv Layer 1:** 3 channels → 16 channels, kernel=3
- **MaxPool 1:** kernel=2, stride=2
- **Conv Layer 2:** 16 channels → 32 channels, kernel=3
- **MaxPool 2:** kernel=2, stride=2
- **Fully Connected:** FC1(hidden=128) → FC2(output=10)

### Dimension Tracing

```
Input:                    (batch, 3, 32, 32)
↓ Conv2d(3→16, k=3)      Output = (32 - 3)/1 + 1 = 30
                          (batch, 16, 30, 30)
↓ MaxPool2d(k=2, s=2)    Output = (30 - 2)/2 + 1 = 15
                          (batch, 16, 15, 15)
↓ Conv2d(16→32, k=3)     Output = (15 - 3)/1 + 1 = 13
                          (batch, 32, 13, 13)
↓ MaxPool2d(k=2, s=2)    Output = (13 - 2)/2 + 1 = 6
                          (batch, 32, 6, 6)
↓ Flatten                 (batch, 32 × 6 × 6) = (batch, 1152)
↓ fc1(1152 → 128)        (batch, 128)
↓ fc2(128 → 10)          (batch, 10)
```

### Complete Model Implementation

```python
import torch
import torch.nn as nn

class ConvolutionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Layer 1: Conv + ReLU + Pool
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 2: Conv + ReLU + Pool
        self.layer2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1152, 128)  # 32 × 6 × 6 = 1152
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # First convolutional block
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.pooling1(x)
        
        # Second convolutional block
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.pooling2(x)
        
        # Fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        
        return x

# Create model instance
model = ConvolutionModel()
print(model)
```

### Key Components Explained

- **`nn.Conv2d(in_channels, out_channels, kernel_size)`:** Applies 2D convolution. Computes $y = W \star x + b$ where $\star$ is the convolution operation.

- **`nn.ReLU()`:** Rectified Linear Unit activation function. Applies $f(x) = \max(0, x)$.

- **`nn.MaxPool2d(kernel_size, stride)`:** Downsampling layer that takes the maximum value in each window.

- **`nn.Linear(in_features, out_features)`:** Fully connected layer. Applies $y = Wx + b$.

- **`nn.Flatten()`:** Reshapes multi-dimensional tensor into 2D (batch_size, features).

- **`forward(x)`:** Defines the computation flow through the network.

### Common Mistakes to Avoid

❌ **Wrong:** `y = torch.randint(0, 10, 1000)` - shape is not a tuple  
✅ **Correct:** `y = torch.randint(0, 10, (1000,))` - shape is a tuple

❌ **Wrong:** `self.fc1 = nn.Linear(32, 128)` - incorrect input size  
✅ **Correct:** `self.fc1 = nn.Linear(1152, 128)` - traced dimensions correctly

---

## Training Loop with Validation

A proper training loop is essential for building robust models. It includes:
- **Training phase:** Update model weights
- **Validation phase:** Evaluate on unseen data
- **Tracking metrics:** Monitor loss over time

### Complete Training Loop Implementation

```python
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Create dummy data
X_train = torch.rand(500, 3, 32, 32)
y_train = torch.randint(0, 10, (500,))
X_val = torch.rand(100, 3, 32, 32)
y_val = torch.randint(0, 10, (100,))

# Step 2: Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32)

# Step 3: Setup training infrastructure
criterion = nn.CrossEntropyLoss()  # Loss function for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer
num_epochs = 5

# Step 4: Training loop
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # ===== TRAINING PHASE =====
    model.train()  # Enable training mode (dropout, batch norm, etc.)
    train_loss = 0
    
    for batch_data, batch_target in train_dataloader:
        # Forward pass
        output = model(batch_data)
        loss = criterion(output, batch_target)
        
        # Backward pass
        optimizer.zero_grad()      # Clear old gradients
        loss.backward()            # Compute gradients via backpropagation
        optimizer.step()           # Update weights using gradients
        
        train_loss += loss.item()
    
    train_loss /= len(train_dataloader)
    
    # ===== VALIDATION PHASE =====
    model.eval()  # Disable training-specific behaviors
    val_loss = 0
    
    with torch.no_grad():  # Disable gradient computation for efficiency
        for batch_data, batch_target in val_dataloader:
            output = model(batch_data)
            loss = criterion(output, batch_target)
            val_loss += loss.item()
    
    val_loss /= len(val_dataloader)
    
    # ===== LOGGING =====
    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
```

### Understanding Each Component

**`model.train()` vs `model.eval()`:**
- `model.train()`: Enables dropout (random neuron deactivation), batch norm updates statistics
- `model.eval()`: Disables dropout, uses learned batch norm parameters

**`optimizer.zero_grad()`:**
- Clears gradients from previous iteration
- **Important:** Forgetting this causes gradient accumulation (incorrect updates)

**`loss.backward()`:**
- Computes gradients using backpropagation
- Stores gradients in `param.grad` for each parameter

**`optimizer.step()`:**
- Updates weights using computed gradients
- Formula: `param = param - lr * param.grad`

**`torch.no_grad()`:**
- Disables gradient computation during validation/inference
- Saves memory and increases speed
- Prevents accidental weight updates during validation

### Loss Functions Explained

**`nn.CrossEntropyLoss()`:**
- Combines softmax activation with negative log likelihood loss
- **Used for:** Multi-class classification
- **Input:** Raw logits (not probabilities)
- **Formula:** $L = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$

**`nn.MSELoss()`:**
- Mean squared error loss
- **Used for:** Regression tasks
- **Formula:** $L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$

### Optimizer Types

**`optim.Adam()`:**
- Adaptive moment estimation
- **Pros:** Fast convergence, works well for most tasks
- **Default learning rate:** 0.001
- **Use when:** You want reliable, fast training

**`optim.SGD()`:**
- Stochastic gradient descent
- **Pros:** Simple, often produces better generalization
- **Cons:** Slower convergence, requires learning rate tuning
- **Use when:** You have time for hyperparameter tuning

**`optim.RMSprop()`:**
- Root mean square propagation
- **Pros:** Good for RNNs, adaptive learning rates
- **Use when:** Training recurrent networks

---

## Model Checkpointing

Saving models during training is crucial for:
1. **Recovering from crashes**
2. **Loading the best model** (based on validation performance)
3. **Resuming training** without losing progress
4. **Sharing trained models** with others

### What to Save in a Checkpoint

```python
checkpoint = {
    'epoch': epoch,                          # Current epoch
    'model_state_dict': model.state_dict(),  # Model weights
    'optimizer_state_dict': optimizer.state_dict(),  # Optimizer state
    'loss': val_loss,                        # Best loss or current loss
    'hyperparameters': {                     # Optional: save settings
        'learning_rate': 0.001,
        'batch_size': 32,
    }
}
```

**Why save optimizer state?**
- Optimizer maintains internal state (e.g., momentum buffers, adaptive learning rates)
- Without it, resuming training loses optimization history
- Results in slower convergence when resuming

### Saving Checkpoints

```python
def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"✓ Checkpoint saved to {filepath}")

# Usage during training
if val_loss < best_val_loss:
    best_val_loss = val_loss
    save_checkpoint(model, optimizer, epoch, val_loss, 'checkpoints/best_model.pt')
```

### Loading Checkpoints

```python
def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return model, optimizer, epoch, loss

# Usage
model, optimizer, start_epoch, last_loss = load_checkpoint(
    model, optimizer, 'checkpoints/best_model.pt'
)
```

### Checkpoint Strategy: Best Model Only

```python
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # ... training and validation code ...
    
    # Save checkpoint if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }
        torch.save(checkpoint, 'checkpoints/best_model.pt')
        print("✓ Best model saved!")
```

**Advantages:**
- ✅ Saves only the best performing model
- ✅ Prevents overfitting by selecting best validation performance
- ✅ Minimal disk space usage

---

## Inference and Predictions

Inference is the process of using a trained model to make predictions on new data.

### Key Differences: Training vs Inference

| Aspect | Training | Inference |
|--------|----------|-----------|
| `model.train()` or `model.eval()` | `.train()` | `.eval()` |
| Compute gradients | Yes (requires memory) | No |
| Update dropout/batchnorm | Yes (training) | No (use learned) |
| `torch.no_grad()` | No | Yes |
| Speed | Slower | Faster |

### Inference Implementation

```python
# Step 1: Load best checkpoint
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Step 2: Prepare for inference
model.eval()  # Set to evaluation mode
test_data = torch.randn(10, 3, 32, 32)  # 10 new samples

# Step 3: Perform inference
with torch.no_grad():  # Disable gradient computation
    output = model(test_data)  # Forward pass
    probabilities = torch.softmax(output, dim=1)  # Convert to probabilities
    
    # Get predictions and confidence scores
    confidence, predictions = torch.max(probabilities, dim=1)

# Step 4: Display results
print("Predictions (class):", predictions)
print("Confidence (probability):", confidence)

# Step 5: Detailed per-sample results
print("\n--- Detailed Results ---")
for i in range(len(predictions)):
    print(f"Sample {i+1}: Predicted Class={predictions[i].item()}, Confidence={confidence[i].item():.4f}")
```

### Understanding Inference Output

**Raw Output (Logits):**
```python
output = model(test_data)  # Shape: (10, 10) for 10 classes
# Values: [-2.3, 0.5, 1.2, -0.8, ...]  # Not normalized
```

**Probabilities:**
```python
probabilities = torch.softmax(output, dim=1)  # Shape: (10, 10)
# Values: [0.05, 0.25, 0.45, 0.1, ...]  # Sum to 1 per sample
```

**Class Predictions:**
```python
confidence, predictions = torch.max(probabilities, dim=1)
# predictions: [2, 1, 4, 0, ...]  # Predicted class (0-9)
# confidence: [0.45, 0.35, 0.62, ...]  # Probability of predicted class
```

### Common Inference Operations

**Get top-K predictions:**
```python
# Get top 3 predicted classes
top_k_probs, top_k_classes = torch.topk(probabilities, k=3, dim=1)
```

**Batch inference:**
```python
# Process data in batches for large datasets
test_dataloader = DataLoader(test_dataset, batch_size=32)

all_predictions = []
all_confidences = []

model.eval()
with torch.no_grad():
    for batch_data in test_dataloader:
        output = model(batch_data)
        probs = torch.softmax(output, dim=1)
        conf, preds = torch.max(probs, dim=1)
        
        all_predictions.append(preds)
        all_confidences.append(conf)

predictions = torch.cat(all_predictions)
confidences = torch.cat(all_confidences)
```

---

## Resuming Training from Checkpoint

One of the key advantages of checkpointing is the ability to resume training without losing progress.

### Why Resume Training?

1. **Unexpected interruptions:** Power loss, network failure
2. **Limited compute resources:** Train in multiple sessions
3. **Hyperparameter adjustment:** Continue training with different settings
4. **Data availability:** Add more training data and continue

### Implementation

```python
# Step 1: Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1

print(f"Resuming from epoch {start_epoch}...")

# Step 2: Continue training
total_epochs = num_epochs + 2  # Train for 2 more epochs

for epoch in range(start_epoch, total_epochs):
    # Training Phase
    model.train()
    train_loss = 0
    
    for batch_data, batch_target in train_dataloader:
        output = model(batch_data)
        loss = criterion(output, batch_target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_dataloader)
    
    # Validation Phase
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for batch_data, batch_target in val_dataloader:
            output = model(batch_data)
            loss = criterion(output, batch_target)
            val_loss += loss.item()
    
    val_loss /= len(val_dataloader)
    
    # Logging and Checkpointing
    print(f"Epoch {epoch+1}/{total_epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}", end="")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }
        torch.save(checkpoint, 'checkpoints/best_model.pt')
        print(" ✓ Best model saved!")
    else:
        print()
```

### Critical Points

✅ **Always load optimizer state** for proper gradient-based optimization  
✅ **Verify shapes match** before loading (device compatibility important)  
✅ **Check if file exists** before loading to avoid errors  
✅ **Continue from correct epoch** to avoid overlap  

---

## Transfer Learning

Transfer learning leverages pre-trained models to solve new tasks efficiently.

### Core Idea

1. **Start with pre-trained weights** (trained on large dataset like ImageNet)
2. **Freeze early layers** (they learn general features)
3. **Train only last layers** (learn task-specific features)
4. **Benefit:** Faster convergence, better generalization, less data needed

### Why Transfer Learning Works

**Early layers** learn generic features:
- Edge detection
- Texture recognition
- Shape recognition

**Later layers** learn task-specific features:
- Object parts
- Category-specific patterns

By keeping early layer features and only retraining late layers, we:
- ✅ Save computation time
- ✅ Require less training data
- ✅ Achieve better performance

### Implementation

```python
# Step 1: Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Step 2: Freeze all layers
for param in model.parameters():
    param.requires_grad = False

print("Freezing all layers...")

# Step 3: Replace last layer for new task
# Original: 10 classes → New: 5 classes
model.fc2 = nn.Linear(128, 5)

# Step 4: Unfreeze last layer
for param in model.fc2.parameters():
    param.requires_grad = True

print("Unfroze last layer (fc2)")

# Step 5: Setup optimizer for only trainable parameters
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 6: Verify which parameters are trainable
print("\n--- Trainable Parameters ---")
trainable_count = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  ✓ {name}")
        trainable_count += param.numel()

print(f"Total trainable parameters: {trainable_count}")

# Step 7: Train on new dataset
new_X_train = torch.rand(200, 3, 32, 32)
new_y_train = torch.randint(0, 5, (200,))  # 5 new classes

train_dataset = TensorDataset(new_X_train, new_y_train)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

criterion = nn.CrossEntropyLoss()

for epoch in range(3):  # Train for 3 epochs
    model.train()
    train_loss = 0
    
    for batch_data, batch_target in train_dataloader:
        output = model(batch_data)
        loss = criterion(output, batch_target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_dataloader)
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}")
```

### Advanced: Fine-tuning Strategy

```python
# Strategy 1: Freeze all except last 2 layers
for name, param in model.named_parameters():
    if 'fc2' not in name and 'fc1' not in name:
        param.requires_grad = False

# Strategy 2: Different learning rates for different layers
# (requires custom optimizer setup)
layer_groups = [
    {'params': model.layer1.parameters(), 'lr': 0.00001},  # Very low LR
    {'params': model.layer2.parameters(), 'lr': 0.0001},   # Low LR
    {'params': model.fc1.parameters(), 'lr': 0.001},       # Medium LR
    {'params': model.fc2.parameters(), 'lr': 0.01},        # High LR
]
optimizer = optim.Adam(layer_groups)
```

### When to Use Transfer Learning

✅ **Use transfer learning when:**
- You have limited training data
- Your task is similar to the pre-training task
- You want faster training
- You need better generalization

❌ **Don't use transfer learning when:**
- Your task is very different from pre-training task
- You have large amounts of task-specific data
- You need to optimize for inference speed over accuracy

---

## Complete Code Example

Here's the complete, production-ready code for the entire workflow:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# ===== CONFIGURATION =====
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ===== STEP 1: CREATE DATA =====
print("Creating synthetic CIFAR-10 style data...")
X_train = torch.rand(500, 3, 32, 32)
y_train = torch.randint(0, 10, (500,))
X_val = torch.rand(100, 3, 32, 32)
y_val = torch.randint(0, 10, (100,))

# ===== STEP 2: CREATE DATALOADERS =====
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ===== STEP 3: DEFINE MODEL =====
class ConvolutionModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Layer 1
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 2
        self.layer2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # FC Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1152, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.pooling1(x)
        
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.pooling2(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        
        return x

model = ConvolutionModel(num_classes=10)

# ===== STEP 4: SETUP TRAINING =====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ===== STEP 5: TRAINING LOOP =====
print("\nStarting training...")
best_val_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    train_loss = 0
    
    for batch_data, batch_target in train_dataloader:
        output = model(batch_data)
        loss = criterion(output, batch_target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_dataloader)
    
    # Validation
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for batch_data, batch_target in val_dataloader:
            output = model(batch_data)
            loss = criterion(output, batch_target)
            val_loss += loss.item()
    
    val_loss /= len(val_dataloader)
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}", end="")
    
    # Checkpointing
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }
        torch.save(checkpoint, f'{CHECKPOINT_DIR}/best_model.pt')
        print(" ✓ Best model saved!")
    else:
        print()

# ===== STEP 6: INFERENCE =====
print("\nPerforming inference...")
checkpoint = torch.load(f'{CHECKPOINT_DIR}/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
test_data = torch.randn(5, 3, 32, 32)

with torch.no_grad():
    output = model(test_data)
    probabilities = torch.softmax(output, dim=1)
    confidence, predictions = torch.max(probabilities, dim=1)

print("\nPredictions (class):", predictions)
print("Confidence scores:", confidence)

# ===== STEP 7: RESUME TRAINING =====
print("\nResuming training from checkpoint...")
checkpoint = torch.load(f'{CHECKPOINT_DIR}/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1

for epoch in range(start_epoch, NUM_EPOCHS + 2):
    model.train()
    train_loss = 0
    
    for batch_data, batch_target in train_dataloader:
        output = model(batch_data)
        loss = criterion(output, batch_target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_dataloader)
    
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for batch_data, batch_target in val_dataloader:
            output = model(batch_data)
            loss = criterion(output, batch_target)
            val_loss += loss.item()
    
    val_loss /= len(val_dataloader)
    
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

print("\nTraining complete!")
```

---

## Key Takeaways

### 1. **Tensor Dimensions**
- Always use tuples for shapes: `(1000,)` not `1000`
- Choose dimensions based on data type (1D for labels, 4D for images, etc.)

### 2. **Convolution & Pooling**
- Use formulas to calculate dimension changes
- Track dimensions through entire network
- Verify FC input size before building model

### 3. **Model Architecture**
- Build modular components (Conv blocks, FC layers)
- Use `forward()` method to define computation
- Initialize model before training

### 4. **Training Loop**
- Use `model.train()` and `model.eval()` appropriately
- Always call `optimizer.zero_grad()` before backprop
- Don't forget `torch.no_grad()` during validation

### 5. **Checkpointing**
- Save both model AND optimizer state
- Save checkpoint when validation improves
- Load checkpoint correctly before inference/resuming

### 6. **Inference**
- Always set `model.eval()` before inference
- Use `torch.no_grad()` to save memory
- Convert logits to probabilities with softmax

### 7. **Transfer Learning**
- Freeze early layers, train only last layers
- Replace output layer for new task
- Use lower learning rate for fine-tuning

### 8. **Best Practices**
- ✅ Create separate train/val/test splits
- ✅ Use DataLoaders for batching
- ✅ Monitor both train and validation loss
- ✅ Save best model, not last model
- ✅ Use meaningful variable names
- ✅ Add comments explaining complex logic
- ✅ Profile code for performance bottlenecks

---

## Common Pitfalls and Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| Model not learning | Wrong learning rate | Try 0.001 or 0.0001 |
| Out of memory | Batch size too large | Reduce batch size |
| NaN loss | Exploding gradients | Use gradient clipping or lower LR |
| Poor validation | Overfitting | Use dropout, data augmentation, regularization |
| Slow inference | Computing gradients | Remember `torch.no_grad()` |
| Wrong predictions | Not setting eval mode | Use `model.eval()` before inference |
| Can't resume training | Optimizer state lost | Save and load optimizer state |

---

## Further Learning Resources

1. **PyTorch Documentation:** https://pytorch.org/docs/
2. **CNN Visualization:** Understanding what each layer learns
3. **Batch Normalization:** Stabilize training and speed convergence
4. **Data Augmentation:** Artificially increase dataset diversity
5. **Hyperparameter Tuning:** Systematic optimization of learning rate, batch size, etc.
6. **Distributed Training:** Scale to multiple GPUs/TPUs
7. **Model Quantization:** Compress models for deployment
8. **Attention Mechanisms:** Build Transformers and advanced architectures

---

## Conclusion

You've now learned the complete workflow for building, training, and deploying PyTorch neural networks:

1. ✅ **Created** a CNN model with proper dimension calculations
2. ✅ **Trained** the model with validation and checkpointing
3. ✅ **Performed** inference on new data
4. ✅ **Resumed** training from checkpoints
5. ✅ **Applied** transfer learning techniques

These are the fundamental skills needed for most deep learning projects. Practice these concepts with different architectures (RNNs, Transformers, etc.) and datasets to deepen your understanding.

**Happy coding!** 🚀

---

*Last Updated: December 24, 2025*
