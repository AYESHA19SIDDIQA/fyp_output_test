# Validation Split Feature - Quick Start

## What's New?

The training script now supports splitting your training data into training and validation sets automatically. This means you can:

✅ **Create a temporary validation set** from your training data (5-10%)  
✅ **Compute evaluation metrics** (loss, accuracy) at the end of each epoch  
✅ **Use validation loss** to update the learning rate scheduler  
✅ **Select the best model** checkpoint based on validation performance  

## Quick Start

### Default Usage (Recommended)
```bash
python updated_main_gaze.py
```
This will automatically:
- Split training data: 90% train, 10% validation
- Use stratified splitting to maintain class balance
- Evaluate on validation set after each epoch

### Custom Validation Split
```bash
# Use 5% for validation
python updated_main_gaze.py --val-split 0.05

# Use 15% for validation
python updated_main_gaze.py --val-split 0.15
```

### Disable Validation Split
If you want to use the original behavior (separate train/eval directories):
```bash
python updated_main_gaze.py --no-train-val-split
```

## Key Benefits

1. **No separate validation data needed** - Use your existing training data
2. **Stratified splitting** - Class distribution is maintained
3. **Automatic learning rate adjustment** - Scheduler uses validation loss
4. **Best model selection** - Saves model with lowest validation loss
5. **Reproducible** - Fixed random seed for consistent splits

## Example Output

When you run training with validation split:

```
BUILD DATALOADERS (FIXED)
  Splitting training data: 90% train, 10% validation
  Train indices: 900, Validation indices: 100
  Train label distribution: {0: 630, 1: 270}
  Validation label distribution: {0: 70, 1: 30}

Epoch 1 Summary:
  Train: Loss=0.5234 | Acc=78.45%
  Eval:  Loss=0.4987 | Acc=81.20% | Macro F1=0.7945
  ✓ Saved best model checkpoint (val_loss: 0.4987, epoch: 1)
```

## What Changed?

### Code Changes
- Added `val_split` parameter (default: 0.1 = 10%)
- Added `use_train_val_split` parameter (default: True)
- Integrated sklearn's stratified train/test split
- Uses torch Subset for efficient data splitting

### Backward Compatibility
✅ All existing code continues to work  
✅ Use `--no-train-val-split` to use original behavior  
✅ No breaking changes  

## Documentation

For detailed information, see:
- **[VALIDATION_SPLIT_GUIDE.md](VALIDATION_SPLIT_GUIDE.md)** - Complete documentation
- **[example_train_with_validation.py](example_train_with_validation.py)** - Usage examples

## Recommendations

| Dataset Size | Recommended Split |
|-------------|-------------------|
| Small (<500 samples) | 15% validation |
| Medium (500-2000 samples) | 10% validation |
| Large (>2000 samples) | 5% validation |

## Questions?

### Q: Will this affect my training results?
A: Yes, but positively! You now have reliable validation metrics to track overfitting and select the best model.

### Q: Can I still use my separate eval directory?
A: Yes! Use `--no-train-val-split` to use the original behavior.

### Q: Is the split random?
A: The split is stratified (maintains class distribution) and uses a fixed random seed (42) for reproducibility.

### Q: How does this affect learning rate scheduling?
A: The ReduceLROnPlateau scheduler now uses validation loss (instead of training loss) to adjust learning rate, which is a better practice.

## Technical Details

The implementation:
1. Loads the full training dataset
2. Splits it using `sklearn.model_selection.train_test_split` with stratification
3. Creates torch `Subset` objects for efficient indexing
4. Maintains all existing functionality (weighted sampling, data augmentation, etc.)
5. Computes metrics on validation set after each epoch
6. Updates scheduler and saves best model based on validation performance

---

**Ready to train?** Try it now:
```bash
python updated_main_gaze.py --epochs 50 --val-split 0.1
```
