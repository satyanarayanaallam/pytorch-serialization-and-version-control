# 🧾 PyTorch Tensor Dimension Cheatsheet

Understanding tensor shapes is critical for debugging models and designing architectures.  
This cheatsheet summarizes the most common tensor dimensions in PyTorch, their meanings, and typical use cases.

---

## 1D Tensors
**Shape:** `(batch_size,)`  
**Example:**  
```python
y = torch.randint(0, 10, (1000,))
```

Use Case: Labels, 1D signals, single feature vectors.
---
2D Tensors
Shape: (batch_size, features)  
Example:

```python
X = torch.randn(32, 784)
```
Use Case: Tabular data, fully connected (dense) networks.
---
3D Tensors
Shape: (batch_size, sequence_length, features)  
Example:
```python
X = torch.randn(32, 50, 100)
```
Use Case: Sequential data (text, time‑series, RNNs).
---
4D Tensors
Shape: (batch_size, channels, height, width)  
Example:

X = torch.randn(32, 3, 224, 224)

Use Case: Image data for CNNs (RGB images, grayscale images).
---
5D Tensors
Shape: (batch_size, channels, depth, height, width)  
Example:

X = torch.randn(8, 1, 64, 64, 64)

Use Case: 3D data (medical imaging, volumetric scans, video frames).
---
Quick Reference Table


🔑 Key Tips
• Always define shapes as tuples: (1000,) not 1000.
• For CNNs, remember the order: (batch, channels, height, width).
• For RNNs/Transformers, use (batch, seq_len, features).
• Use .shape to debug tensors at each step in your model.