# Storage Architecture Documentation

## Overview

This document describes the redesigned data-saving architecture for the EEG training and analysis pipeline. The design eliminates duplicate storage, uses memory-efficient formats, and clearly separates training artifacts from post-hoc analysis artifacts.

## Directory Structure

```
training_stats/
└── run_YYYYMMDD_HHMMSS/
    ├── epoch_statistics.csv           # Epoch-level metrics
    ├── predictions_eval.csv            # Per-sample predictions
    ├── confusion_matrices.json         # Confusion matrices
    ├── class_distributions.json        # Class distributions
    ├── training_curves.png             # Loss/accuracy plots
    ├── confusion_matrix.png            # Confusion matrix heatmap
    ├── metadata.json                   # Run metadata
    ├── training_summary.txt            # Human-readable summary
    ├── best_model.pth                  # Single authoritative checkpoint
    └── attention_maps/                 # Post-hoc analysis artifacts
        ├── {file_id}_attention.npz     # One file per sample
        ├── metadata.json                # Attention maps metadata
        └── manifest.json                # File listing
```

## Storage Formats

### Training Statistics (CSV/JSON)

**Epoch Statistics** (`epoch_statistics.csv`):
- Format: CSV (human-readable, easily loadable)
- Content: One row per epoch with:
  - Train/eval losses (total, classification, gaze)
  - Train/eval accuracy
  - F1 scores (macro, weighted)
  - Precision, recall, balanced accuracy
  - Learning rate
  - Timestamp

**Predictions** (`predictions_eval.csv`):
- Format: CSV
- Content: One row per sample with:
  - File identifier
  - True label
  - Predicted label
  - Probability distribution
  - Correctness flag
  - Timestamp

**Confusion Matrices** (`confusion_matrices.json`):
- Format: JSON (not pickle)
- Content: Serialized confusion matrices with labels

### Attention Maps (Compressed NPZ)

**Format**: `numpy.savez_compressed`

**Why NPZ instead of pickle?**
- Safer: No arbitrary code execution risk
- Faster: Optimized for NumPy arrays
- Smaller: Built-in compression
- Standard: Part of NumPy standard library
- Portable: Works across Python versions

**File Structure** (`{file_id}_attention.npz`):
```python
{
    'attention_map': ndarray,    # Shape: (22, 15000), dtype: float32
    'file_id': str,              # Original file identifier
    'shape': tuple,              # (22, 15000)
    'sampling_rate': float       # 50.0 Hz
}
```

**Loading Example**:
```python
data = np.load('sample_attention.npz', allow_pickle=False)
attention_map = data['attention_map']  # (22, 15000) float32
```

### Model Checkpoint (PyTorch)

**File**: `best_model.pth` (single authoritative checkpoint)

**Selection Criterion**: Lowest validation loss

**Content**:
```python
{
    'epoch': int,                    # Epoch number
    'model_state_dict': dict,        # Model parameters
    'optimizer_state_dict': dict,    # Optimizer state
    'train_loss': float,             # Training loss at this epoch
    'eval_loss': float,              # Validation loss (best)
    'eval_acc': float,               # Validation accuracy
    'eval_f1': float                 # Macro F1 score
}
```

## Invariants and Contracts

### Attention Map Invariants

1. **Shape**: `(n_channels, n_timepoints)` where:
   - `n_channels = 22` (EEG channels)
   - `n_timepoints = 15000` (50 Hz × 300 seconds)

2. **Data Type**: `float32` (4 bytes per value)
   - Total size per map: 22 × 15000 × 4 = 1.32 MB

3. **Value Range**: `[0, 1]` (normalized attention weights)
   - Values clipped to this range before saving
   - No negative values allowed

4. **Temporal Resolution**:
   - EEG sampling rate: 50 Hz
   - Time per sample: 20 ms
   - Total duration: 300 seconds (5 minutes)

5. **Alignment**:
   - Attention maps align with EEG data via file identifiers
   - Same temporal resolution as original EEG

### Normalization Rules

**Attention Maps**:
- Raw attention values may vary in range
- Always normalized to [0, 1] before saving
- Normalization: `clipped_value = np.clip(raw_value, 0, 1)`
- No additional scaling or mean centering

**EEG Data** (processed in dataset):
- Z-score normalization per channel
- Formula: `(x - mean) / std`
- Applied before model input

## What NOT to Save (and Why)

### ❌ Duplicate Attention Maps
- **Previous**: Stored in `stats_tracker.attention_maps` AND saved via pickle
- **Now**: Saved only once in `attention_maps/` directory
- **Why**: Eliminates 2x storage waste

### ❌ Batch-Level Statistics During Training
- **Previous**: Recorded every batch (thousands of entries)
- **Now**: Only epoch-level metrics
- **Why**: Batch stats are noisy and rarely useful for analysis

### ❌ Model Weights Per Epoch
- **Previous**: Saved weight statistics for every epoch
- **Now**: Only best model checkpoint saved
- **Why**: Weight history not needed for inference; wastes storage

### ❌ Pickle for Large Arrays
- **Previous**: Used `pickle.dump()` for attention maps
- **Now**: Use `numpy.savez_compressed()`
- **Why**: Pickle is slow, insecure, and not optimized for arrays

### ❌ Attention Maps During Training/Evaluation
- **Previous**: Extracted attention during eval, stored in memory
- **Now**: Collected only after training completes
- **Why**: Training doesn't need attention maps; saves RAM

### ❌ Multiple Model Checkpoints
- **Previous**: Could save multiple checkpoints (best acc, best loss, etc.)
- **Now**: Single checkpoint (best validation loss)
- **Why**: One authoritative model is sufficient

## Memory Efficiency

### During Training

**Before**:
- Stores attention maps in `stats_tracker.attention_maps` (grows unbounded)
- Peak RAM: ~1.32 MB × N samples (e.g., 660 MB for 500 samples)

**After**:
- No attention maps stored during training
- Peak RAM: Only current batch

### During Attention Collection

**Before**:
- Collected all attention maps in list
- Saved at end using pickle
- Peak RAM: 1.32 MB × N samples

**After**:
- Process one batch at a time
- Save immediately to disk
- Peak RAM: 1.32 MB × batch_size (e.g., 42 MB for batch_size=32)

## Loading Artifacts

### Load Training Statistics

```python
import pandas as pd

# Load epoch metrics
epochs_df = pd.read_csv('run_YYYYMMDD_HHMMSS/epoch_statistics.csv')

# Load predictions
preds_df = pd.read_csv('run_YYYYMMDD_HHMMSS/predictions_eval.csv')

# Load confusion matrices
import json
with open('run_YYYYMMDD_HHMMSS/confusion_matrices.json') as f:
    cms = json.load(f)
```

### Load Attention Maps

```python
import numpy as np
from pathlib import Path

# Load single attention map
data = np.load('attention_maps/sample_001_attention.npz', allow_pickle=False)
attention = data['attention_map']  # (22, 15000)
file_id = str(data['file_id'])

# Load all attention maps (memory-efficient)
attention_dir = Path('attention_maps')
for npz_file in attention_dir.glob('*_attention.npz'):
    data = np.load(npz_file, allow_pickle=False)
    attention = data['attention_map']
    # Process one at a time (low memory)
```

### Load Model Checkpoint

```python
import torch
from neurogate_gaze import NeuroGATE_Gaze_MultiRes

# Load model
model = NeuroGATE_Gaze_MultiRes(n_chan=22, n_outputs=2, original_time_length=15000)
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Access metadata
print(f"Epoch: {checkpoint['epoch']}")
print(f"Validation loss: {checkpoint['eval_loss']:.4f}")
print(f"Validation accuracy: {checkpoint['eval_acc']:.2f}%")
```

## Reproducibility

All artifacts are independently reloadable without re-running training:

1. **Training Statistics**: CSV/JSON files contain complete history
2. **Model Checkpoint**: Can reload exact model state
3. **Attention Maps**: Saved per-sample with file identifiers
4. **Predictions**: Per-sample predictions allow error analysis

## Comparison: Before vs After

| Aspect | Before | After |
|--------|--------|-------|
| Attention storage | Pickle + in-memory dict | Compressed NPZ (one file per sample) |
| Duplication | 2× (memory + disk) | 1× (disk only) |
| Format safety | Pickle (insecure) | NPZ (safe, no code execution) |
| Memory efficiency | Stores all maps in RAM | Batch-by-batch processing |
| Checkpoints | Multiple files | Single best_model.pth |
| Training stats | Pickle for some data | CSV + JSON only |
| Separation | Mixed artifacts | Clear separation |
| Reloadability | Some data hard to load | All data independently loadable |

## Metadata Bugs Fixed

1. **Incorrect attention map counts**:
   - Before: Metadata had wrong count due to duplicates
   - After: Accurate count from manifest

2. **Missing shape information**:
   - Before: Shape not stored with attention maps
   - After: Each NPZ file contains shape metadata

3. **Unclear file mappings**:
   - Before: No clear mapping between attention maps and source files
   - After: File identifier stored in each NPZ file + manifest.json

## Usage Example

```python
from updated_main_gaze import main

# Run training
best_acc, run_dir = main(
    lr=1e-4,
    epochs=50,
    batch_size=32,
    gaze_weight=0.3,
    gaze_loss_type='mse'
)

# Results automatically saved to:
# - training_stats/run_YYYYMMDD_HHMMSS/
#   - epoch_statistics.csv
#   - predictions_eval.csv
#   - best_model.pth
#   - attention_maps/*.npz
```

## Summary

This architecture achieves:
- ✅ No duplicate storage
- ✅ Memory-efficient processing
- ✅ Safe, standard formats (no pickle for arrays)
- ✅ Clear separation of concerns
- ✅ Single authoritative checkpoint
- ✅ All artifacts independently reloadable
- ✅ Proper invariants and contracts
- ✅ Fixed metadata bugs
