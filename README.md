# Foreteller 🔮👁️✨

A PyTorch implementation of an in-context tabular transformer encoder that combines efficient 2D attention mechanisms with latent processing for scalable pretraining of tabular foundation models. The architecture is basically a re-implementation of **ConTextTab**  as described in (https://www.arxiv.org/pdf/2506.10707) which itself is a modified version of the **TabPFN** model (see https://www.nature.com/articles/s41586-024-08328-6).


## 🏗️ Architecture Overview

The input is a preprocessed dataset of shape `(R, C, D)` with **R** rows, **C** columns and **D**-dimensional cell-embeddings. The target column has to be at position -1 (last column). Some rows are part of the context **K** (used to learn from) and some representing the **T** test-rows (the ones for which we want to predict a target column). Whereas the context has access to the target, it is masked for test-rows. 

```
┌─────┬─────┬─────┬─────┐
│ c1  │ c2  │ c3  │ t   │
├─────┼─────┼─────┼─────┤
│  .  │  .  │  .  │  .  │ ← K context rows
│  .  │  .  │  .  │  .  │ ← (target visible)
│  .  │  .  │  .  │  .  │
├─────┼─────┼─────┼─────┤
│  .  │  .  │  .  │  m  │ ← T Test rows
│  .  │  .  │  .  │  m  │ ← (target masked)  
│  .  │  .  │  .  │  m  │
└─────┴─────┴─────┴─────┘
```

The Tabular Transformer processes tabular data through a multi-stage pipeline. 

Given an input of shape `(B, (K + T), C, D)`, it processes the data using **2D attention**, alternating between row and column attention.

## Early layers

### Across-column attention, i.e. learning local feature dependencies/interactions. 

Each row is treated as a sequence of **C** cells using **full-attention** (every cell can attend to every other cell in each row) with **R** becoming the batch-dimension. 

The input is reshaped from `(B, R, C, D) -> ((B*R), C, D)`. In order to avoid batch-related OOM errors, the input is split into smaller chunks (see `max_batch_size_col_sequence`).

### Across-row attention, i.e. learning column-specific global statistics

Each column is treated as a sequence of **R** cells with **C** becoming the batch-dimension. 

The input is reshaped from `(B, R, C, D) -> ((B, C), R, D)`. In order to avoid sequence-length OOM errors, the input is processed in smaller chunks (see `max_batch_size_col_sequence`).

#### Downsampling using ISAB (Induced Set Attention Block)

Due to the **quadratic complexity**, the potentially large sequence (datasets with many rows) is first downsampled using **inducing points** as introduced here (https://arxiv.org/pdf/1810.00825). Specifically, we choose a **latent parameter tensor** of shape **(L, D)**. This latent tensor is concatenated with the **T test rows** of shape `(C, T, D)` along the sequence-dimension to get `(C, L + T, D)`. The attention complexity is reduced from **O(R²)** to **O((L + T)²)**, where **L << R**, making it computationally feasible for large datasets. 

#### Upsampling

The latent representation `(C, L + T, D)` is upsampled again to original input dimensionality `(C, R, D)`.

## Later layers

Subsequent layers apply **2D attention** without chunking on the **latent representation** directly, where the test-rows rely solely on the learned latent representation and do not have access to the context-rows anymore. 

## Output

The model eventually returns `(B, T, D)`, i.e. the contextualized cell-embeddings for the T test-rows of the target column. You can use those for downstream tasks by adding an output projection of your choice. 


## 🚀 Usage

### **Basic Configuration**

```python
from foreteller import Foreteller

config = {
    'd_in': 128,                         # Input embedding dimension
    'd_out': 256,                        # Output embedding dimension
    'num_heads': 8,                      # Number of attention heads
    'n_latents': 64,                     # Number of latent tokens
    'n_isab_layers': 2,                  # Number of 2D ISAB layers
    'n_mab_layers': 1,                   # Number of 2D MAB layers
    'dropout': 0.1,                      # Dropout probability
    'max_batch_size_col_sequence': 1024  # maximum pseudo-batch-size for cross-column attention
    'max_batch_size_col_sequence': 100   # maximum pseudo-batch-size for cross-row attention
}

model = Foreteller(config)
```

### **Input Format**

```python
# Input tensor: (Batch, Rows, Columns, Features)
x = torch.randn(2, 1000, 20, 128)  # 2 batches, 1000 rows, 20 columns, 128 features
n_context = 800                     # Number of context rows for masking

# Forward pass
output = model(x, n_context)        # Shape: (2, 200, 128)
```

## 🔄 Data Flow

### **Phase 1: 2D Feature Learning**
```
Input (B, R, C, D)
    ↓
Column Attention: (B*R, C, D) → MAB → (B*R, C, D)
    ↓
Row Attention: (B*C, R, D) → ISAB → (B*C, R, D)
    ↓
Output (B, R, C, D)
```

### **Phase 2: Compression**
```
Input (B, R, C, D)
    ↓
Reshape: (B*C, R, D)
    ↓
Downsample: (B*C, I+T, D)
    ↓
Reshape: (B, I+T, C, D)
```

### **Phase 3: Latent Processing**
```
Input (B, I+T, C, D)
    ↓
Column Attention: (B*(I+T), C, D) → MAB → (B*(I+T), C, D)
    ↓
Row Attention: (B*C, I+T, D) → MAB → (B*C, I+T, D)
    ↓
Output (B, I+T, C, D)
```


## 🎯 Key Features

### **Memory Efficiency**
- **ISAB Layers**: O(n) complexity for long sequences
- **Chunked Processing**: Configurable batch sizes for memory management
- **Latent Compression**: Reduce sequence length from R to I

### **Flexibility**
- **Hybrid Architecture**: Combine efficient and standard attention
- **Configurable Layers**: Adjust number of ISAB vs MAB layers
- **Context Awareness**: Support for masked attention patterns

### **Scalability**
- **2D Processing**: Handle tabular data naturally
- **Variable Dimensions**: Adapt to different table sizes
- **Batch Processing**: Efficient GPU utilization

## 🤝 Contributing

This repo provides a clean and concise implementation of the "engine" underlying current SOTA tabular foundation models. My hope is that it facilitates deeper understanding and easier adjustments based on new developments in the field. 

- **Attention Variants**: Implement different attention mechanisms
- **Architecture Modifications**: Explore different layer combinations
- **Optimization**: Improve memory efficiency and speed
- **Applications**: Adapt for specific tabular data domains

