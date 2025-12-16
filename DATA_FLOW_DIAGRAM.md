# Data Flow Diagram: Train/Val/Test Split

## Before Fix (WRONG - Data Leakage)

```
┌─────────────────────────────────────────────────────┐
│                    Data Directory                    │
└─────────────────────────────────────────────────────┘
                        │
          ┌─────────────┴─────────────┐
          │                           │
          ▼                           ▼
┌──────────────────┐        ┌──────────────────┐
│  train/ folder   │        │   eval/ folder   │
│   (training)     │        │  (TEST SET!)     │
└──────────────────┘        └──────────────────┘
          │                           │
          │                           │
          ▼                           ▼
┌──────────────────┐        ┌──────────────────┐
│  train_loader    │        │   eval_loader    │
│                  │        │  (TEST LOADER!)  │
└──────────────────┘        └──────────────────┘
          │                           │
          │                           │
          ▼                           ▼
┌──────────────────┐        ┌──────────────────┐
│  Training Loop   │        │ ❌ WRONG USAGE:  │
│                  │◄───────┤ - LR scheduling  │
│  Learn weights   │        │ - Checkpointing  │
│                  │        │ - Early stopping │
└──────────────────┘        └──────────────────┘
                                      │
                                      ▼
                            ┌──────────────────┐
                            │ ❌ DATA LEAKAGE! │
                            │ Test set used    │
                            │ for training     │
                            │ decisions!       │
                            └──────────────────┘
```

## After Fix (CORRECT - No Data Leakage)

```
┌─────────────────────────────────────────────────────┐
│                    Data Directory                    │
└─────────────────────────────────────────────────────┘
                        │
          ┌─────────────┴─────────────┐
          │                           │
          ▼                           ▼
┌──────────────────┐        ┌──────────────────┐
│  train/ folder   │        │   eval/ folder   │
│                  │        │  (TEST SET)      │
└──────────────────┘        └──────────────────┘
          │                           │
          │ Split 90/10               │ Separate
          │ (stratified)              │
          │                           │
    ┌─────┴─────┐                     │
    │           │                     │
    ▼           ▼                     ▼
┌────────┐  ┌────────┐      ┌──────────────────┐
│ Train  │  │  Val   │      │   Test Loader    │
│ 90%    │  │  10%   │      │                  │
└────────┘  └────────┘      │ (Kept separate!) │
    │           │            └──────────────────┘
    │           │                     │
    ▼           ▼                     │
┌────────┐  ┌────────┐               │
│ train_ │  │  val_  │               │
│ loader │  │ loader │               │
└────────┘  └────────┘               │
    │           │                     │
    │           │                     │
    ▼           ▼                     │
┌─────────────────────┐               │
│   Training Loop     │               │
│                     │               │
│ Learn from train_   │               │
│ loader              │               │
└─────────────────────┘               │
    │           │                     │
    │           │                     │
    ▼           ▼                     │
┌─────────┐ ┌─────────────────┐      │
│ Train   │ │ ✅ CORRECT:     │      │
│ Stats   │ │ Evaluate on     │      │
│         │ │ val_loader      │      │
└─────────┘ └─────────────────┘      │
                   │                  │
                   ▼                  │
          ┌─────────────────┐         │
          │ ✅ Use val_loss │         │
          │ for:            │         │
          │ - LR scheduling │         │
          │ - Checkpointing │         │
          │ - Early stopping│         │
          └─────────────────┘         │
                                      │
          Training completes ─────────┤
                                      │
                                      ▼
                            ┌──────────────────┐
                            │ ✅ FINAL EVAL:   │
                            │ Evaluate on      │
                            │ test_loader      │
                            │ (FIRST TIME!)    │
                            └──────────────────┘
                                      │
                                      ▼
                            ┌──────────────────┐
                            │ ✅ NO LEAKAGE!   │
                            │ Test set used    │
                            │ only for final   │
                            │ evaluation       │
                            └──────────────────┘
```

## Key Differences

### Before (WRONG)
```python
# Training loop
for epoch in range(epochs):
    train_stats = train_epoch(model, train_loader)
    
    # ❌ WRONG: Using test set for training decisions
    eval_stats = evaluate(model, eval_loader)  # Actually test set!
    scheduler.step(eval_stats['loss'])         # Test loss!
    if eval_stats['loss'] < best_loss:         # Test loss!
        save_checkpoint()
```

### After (CORRECT)
```python
# Training loop
for epoch in range(epochs):
    train_stats = train_epoch(model, train_loader)
    
    # ✅ CORRECT: Using validation set for training decisions
    val_stats = evaluate(model, val_loader)    # Validation set
    scheduler.step(val_stats['loss'])          # Validation loss
    if val_stats['loss'] < best_loss:          # Validation loss
        save_checkpoint()

# ✅ CORRECT: Test set used ONLY after training completes
test_stats = evaluate(model, test_loader)      # Test set (first time!)
print(f"Final Test Accuracy: {test_stats['acc']}")
```

## Data Statistics

### Before Fix
- Train samples: 1000 (from train/)
- Eval/Test samples: 300 (from eval/) ❌ Used for training decisions!
- **Problem**: Test set contaminated by training decisions

### After Fix
- Train samples: 900 (90% of train/)
- Validation samples: 100 (10% of train/) ✅ Used for training decisions
- Test samples: 300 (from eval/) ✅ Used only for final evaluation
- **Result**: Test set completely separate and uncontaminated

## Timeline of Test Set Usage

### Before (WRONG)
```
Epoch 1: Evaluate on test set → Use test loss for LR scheduling ❌
Epoch 2: Evaluate on test set → Use test loss for LR scheduling ❌
Epoch 3: Evaluate on test set → Use test loss for checkpoint ❌
...
Epoch 50: Evaluate on test set → Final results ❌
Result: Test set seen 50+ times, used for all decisions!
```

### After (CORRECT)
```
Epoch 1: Evaluate on val set → Use val loss for LR scheduling ✅
Epoch 2: Evaluate on val set → Use val loss for LR scheduling ✅
Epoch 3: Evaluate on val set → Use val loss for checkpoint ✅
...
Epoch 50: Evaluate on val set → Best model selected ✅
THEN: Evaluate on test set → Final results (FIRST TIME!) ✅
Result: Test set seen only once, never used for decisions!
```

## Why This Matters

### Data Leakage Analogy
Imagine you're a teacher creating an exam:

**Before Fix (WRONG):**
- You give students the final exam questions
- Students study specifically for those questions
- You use their practice scores to decide what to teach
- Final exam uses the same questions they've been practicing
- **Result**: Artificially high scores, but students haven't actually learned

**After Fix (CORRECT):**
- You give students practice questions (validation set)
- Students study for the practice questions
- You use practice scores to decide what to teach
- Final exam uses completely different questions (test set)
- **Result**: True measure of learning and understanding

### Impact on Results

**Before Fix:**
- Reported accuracy: 95% ❌ (But it's the test set used for tuning!)
- Actual generalization: Unknown
- Results: Cannot be trusted or published

**After Fix:**
- Validation accuracy: 92% (Used for model selection)
- Test accuracy: 89% ✅ (True measure of generalization)
- Results: Trustworthy and can be published

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Data Split** | 2-way (train/test) | 3-way (train/val/test) |
| **Test Set Role** | Training decisions ❌ | Final evaluation only ✅ |
| **Validation Set** | None ❌ | For training decisions ✅ |
| **LR Scheduling** | Based on test loss ❌ | Based on val loss ✅ |
| **Checkpoint Selection** | Based on test loss ❌ | Based on val loss ✅ |
| **Test Set Evaluations** | 50+ times ❌ | 1 time (at end) ✅ |
| **Data Leakage** | YES ❌ | NO ✅ |
| **Results Valid** | NO ❌ | YES ✅ |
| **Can Publish** | NO ❌ | YES ✅ |

The fix transforms this from an invalid experiment into a valid scientific study with trustworthy results.
