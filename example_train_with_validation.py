#!/usr/bin/env python3
"""
Example script demonstrating validation split usage
"""

# Example 1: Use validation split with default settings (10% validation)
def example_default_validation():
    """Train with 10% validation split"""
    # from updated_main_gaze import main  # Commented out for example purposes
    
    print("="*80)
    print("EXAMPLE 1: Training with 10% validation split (default)")
    print("="*80)
    
    # Note: This example won't actually run without data, but shows the API
    # main(
    #     lr=1e-4,
    #     epochs=50,
    #     batch_size=32,
    #     val_split=0.1,           # 10% for validation
    #     use_train_val_split=True  # Enable train/val split
    # )
    
    print("\nThis would:")
    print("  - Split training data: 90% train, 10% validation")
    print("  - Use stratified splitting to maintain class balance")
    print("  - Evaluate on validation set after each epoch")
    print("  - Use validation loss for learning rate scheduling")
    print("  - Save best model based on validation loss")

# Example 2: Use larger validation split (15%)
def example_larger_validation():
    """Train with 15% validation split"""
    # from updated_main_gaze import main  # Commented out for example purposes
    
    print("\n" + "="*80)
    print("EXAMPLE 2: Training with 15% validation split")
    print("="*80)
    
    # main(
    #     lr=1e-4,
    #     epochs=50,
    #     batch_size=32,
    #     val_split=0.15,          # 15% for validation
    #     use_train_val_split=True
    # )
    
    print("\nThis would:")
    print("  - Split training data: 85% train, 15% validation")
    print("  - More data for validation (better evaluation)")
    print("  - Less data for training (might affect performance)")

# Example 3: Use smaller validation split (5%)
def example_smaller_validation():
    """Train with 5% validation split"""
    # from updated_main_gaze import main  # Commented out for example purposes
    
    print("\n" + "="*80)
    print("EXAMPLE 3: Training with 5% validation split")
    print("="*80)
    
    # main(
    #     lr=1e-4,
    #     epochs=50,
    #     batch_size=32,
    #     val_split=0.05,          # 5% for validation
    #     use_train_val_split=True
    # )
    
    print("\nThis would:")
    print("  - Split training data: 95% train, 5% validation")
    print("  - More data for training (better learning)")
    print("  - Less data for validation (evaluation less reliable)")

# Example 4: Disable validation split (use separate eval directory)
def example_no_split():
    """Train without validation split"""
    # from updated_main_gaze import main  # Commented out for example purposes
    
    print("\n" + "="*80)
    print("EXAMPLE 4: Training without validation split (original behavior)")
    print("="*80)
    
    # main(
    #     lr=1e-4,
    #     epochs=50,
    #     batch_size=32,
    #     use_train_val_split=False  # Use separate eval directory
    # )
    
    print("\nThis would:")
    print("  - Use separate train/ and eval/ directories")
    print("  - No splitting of training data")
    print("  - Original behavior preserved")

# Command-line examples
def show_command_line_examples():
    """Show command-line usage examples"""
    print("\n" + "="*80)
    print("COMMAND-LINE USAGE EXAMPLES")
    print("="*80)
    
    examples = [
        {
            "description": "Default: 10% validation split",
            "command": "python updated_main_gaze.py --use-train-val-split --val-split 0.1"
        },
        {
            "description": "5% validation split",
            "command": "python updated_main_gaze.py --use-train-val-split --val-split 0.05"
        },
        {
            "description": "15% validation split",
            "command": "python updated_main_gaze.py --use-train-val-split --val-split 0.15"
        },
        {
            "description": "Disable validation split",
            "command": "python updated_main_gaze.py --no-train-val-split"
        },
        {
            "description": "Complete example with all parameters",
            "command": (
                "python updated_main_gaze.py \\\n"
                "    --lr 1e-4 \\\n"
                "    --epochs 50 \\\n"
                "    --batch-size 32 \\\n"
                "    --gaze-weight 0.3 \\\n"
                "    --gaze-loss-type mse \\\n"
                "    --val-split 0.1 \\\n"
                "    --use-train-val-split"
            )
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}: {example['description']}")
        print(f"  {example['command']}")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("VALIDATION SPLIT USAGE EXAMPLES")
    print("="*80)
    
    example_default_validation()
    example_larger_validation()
    example_smaller_validation()
    example_no_split()
    show_command_line_examples()
    
    print("\n" + "="*80)
    print("See VALIDATION_SPLIT_GUIDE.md for complete documentation")
    print("="*80)
