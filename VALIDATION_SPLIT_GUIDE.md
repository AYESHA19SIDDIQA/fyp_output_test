# Validation Split Feature Guide

## Overview

The training script now supports splitting the training data into training and validation sets, allowing you to create a temporary validation set from your training data without needing a separate evaluation directory.

## Features

1. **Automatic Train/Validation Split**: Split training data into training (90-95%) and validation (5-10%) sets
2. **Stratified Splitting**: Maintains class distribution across train and validation sets
3. **Validation Metrics**: Computes loss and accuracy on validation set at the end of each epoch
4. **Learning Rate Scheduling**: Uses validation loss to update ReduceLROnPlateau scheduler
5. **Best Model Selection**: Saves the best model checkpoint based on validation loss

## Usage

### Command-Line Arguments

#### Enable Validation Split (Default)
```bash
python updated_main_gaze.py --use-train-val-split --val-split 0.1
```

#### Disable Validation Split (Use Separate Eval Directory)
```bash
python updated_main_gaze.py --no-train-val-split
```

#### Adjust Validation Split Ratio
```bash
# Use 5% for validation
python updated_main_gaze.py --val-split 0.05

# Use 15% for validation
python updated_main_gaze.py --val-split 0.15
```

### Python API

```python
from updated_main_gaze import main

# Use 10% validation split (default)
main(
    lr=1e-4,
    epochs=50,
    batch_size=32,
    val_split=0.1,
    use_train_val_split=True
)

# Use 5% validation split
main(
    lr=1e-4,
    epochs=50,
    batch_size=32,
    val_split=0.05,
    use_train_val_split=True
)

# Disable validation split, use separate eval directory
main(
    lr=1e-4,
    epochs=50,
    batch_size=32,
    use_train_val_split=False
)
```

## Implementation Details

### Stratified Splitting

The implementation uses `sklearn.model_selection.train_test_split` with `stratify` parameter to ensure that the class distribution is maintained in both training and validation sets.

```python
train_indices, val_indices = train_test_split(
    range(len(full_trainset)),
    test_size=val_split,
    random_state=seed,
    stratify=full_labels
)
```

### Example Output

When validation split is enabled, you'll see output like:

```
BUILD DATALOADERS (FIXED)
  Main data_dir: /path/to/data
  Train directory: /path/to/data/train
  Validation split ratio: 0.1
  Use train/val split: True

  Splitting training data: 90% train, 10% validation
  Train indices: 900, Validation indices: 100
  Train label distribution: {0: 630, 1: 270}
  Validation label distribution: {0: 70, 1: 30}
```

### Validation Metrics

At the end of each epoch, the following metrics are computed on the validation set:
- Loss (classification loss + gaze loss)
- Accuracy
- Macro F1 score
- Weighted F1 score
- Precision
- Recall
- Balanced accuracy

Example epoch output:
```
Epoch 1 Summary:
  Train: Loss=0.5234 (CLS=0.4123, Gaze=0.1111) | Acc=78.45%
  Eval:  Loss=0.4987 (CLS=0.3876, Gaze=0.1111) | Acc=81.20% | 
         Balanced Acc=0.8034 | Macro F1=0.7945
```

### Learning Rate Scheduling

The ReduceLROnPlateau scheduler automatically uses validation loss to adjust the learning rate:

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',      # Minimize validation loss
    patience=2,      # Wait 2 epochs before reducing LR
    factor=0.1       # Reduce LR by 10x
)
```

At each epoch:
```python
metric_for_sched = eval_stats['loss']
scheduler.step(metric_for_sched)
```

### Best Model Checkpoint

The best model is saved based on validation loss:

```python
if eval_stats['loss'] < best_loss:
    best_loss = eval_stats['loss']
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_stats['loss'],
        'eval_loss': eval_stats['loss'],
        'eval_acc': eval_stats['acc'],
        'eval_f1': eval_stats['macro_f1'],
    }, checkpoint_path)
```

## Benefits

1. **No Separate Validation Data Required**: Use your existing training data without creating a separate validation directory
2. **Fair Evaluation**: Stratified splitting ensures balanced class distribution
3. **Reproducible**: Uses a fixed random seed for consistent splits across runs
4. **Flexible**: Easy to switch between split mode and separate directory mode
5. **Optimal Model Selection**: Automatically saves the best model based on validation performance

## Recommendations

- **For small datasets**: Use a larger validation split (10-15%) to ensure sufficient validation samples
- **For large datasets**: Use a smaller validation split (5-10%) to maximize training data
- **For imbalanced datasets**: The stratified splitting ensures each class is represented proportionally

## Example Complete Training Command

```bash
python updated_main_gaze.py \
    --lr 1e-4 \
    --epochs 50 \
    --batch-size 32 \
    --gaze-weight 0.3 \
    --gaze-loss-type mse \
    --val-split 0.1 \
    --use-train-val-split
```

This will:
1. Load the training data
2. Split it into 90% train / 10% validation (stratified)
3. Train for 50 epochs
4. Evaluate on validation set after each epoch
5. Update learning rate based on validation loss
6. Save the best model based on validation loss
7. Generate training curves and statistics

## Notes

- The validation split is performed once at the start of training and remains fixed throughout
- The split is deterministic (uses `random_state=42`) for reproducibility
- Original behavior (using separate train/eval directories) is preserved and can be enabled with `--no-train-val-split`
