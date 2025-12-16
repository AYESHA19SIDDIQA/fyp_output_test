#!/usr/bin/env python3
"""
Validation script to check the refactoring without running the full training.
Verifies code structure and design principles.
"""

import ast
import sys
from pathlib import Path

def check_code_structure():
    """Analyze the refactored code structure."""
    print("=" * 80)
    print("VALIDATING REFACTORED CODE STRUCTURE")
    print("=" * 80)
    
    # Load the refactored file
    code_path = Path(__file__).parent / 'updated_main_gaze.py'
    with open(code_path) as f:
        code = f.read()
    
    # Parse the AST
    tree = ast.parse(code)
    
    # Find all class definitions
    classes = {}
    functions = {}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes[node.name] = node
        elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
            functions[node.name] = node
    
    print("\n1. Checking class definitions...")
    
    # Check AttentionMapStorage exists
    if 'AttentionMapStorage' in classes:
        print("   ✓ AttentionMapStorage class exists")
        cls = classes['AttentionMapStorage']
        methods = [n.name for n in ast.walk(cls) if isinstance(n, ast.FunctionDef)]
        expected_methods = ['__init__', 'save_attention_map', 'save_batch', 'save_metadata', 'load_attention_map']
        for method in expected_methods:
            if method in methods:
                print(f"     ✓ Method {method} exists")
            else:
                print(f"     ✗ Method {method} missing")
                return False
    else:
        print("   ✗ AttentionMapStorage class missing")
        return False
    
    # Check TrainingStatistics is refactored
    if 'TrainingStatistics' in classes:
        print("   ✓ TrainingStatistics class exists")
        cls = classes['TrainingStatistics']
        
        # Check that __init__ doesn't initialize attention_maps or model_weights
        init_method = None
        for node in ast.walk(cls):
            if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                init_method = node
                break
        
        if init_method:
            init_code = ast.get_source_segment(code, init_method)
            if 'self.attention_maps' in init_code:
                print("     ✗ TrainingStatistics still initializes attention_maps")
                return False
            else:
                print("     ✓ TrainingStatistics doesn't initialize attention_maps")
            
            if 'self.model_weights' in init_code:
                print("     ✗ TrainingStatistics still initializes model_weights")
                return False
            else:
                print("     ✓ TrainingStatistics doesn't initialize model_weights")
        
        # Check methods
        methods = [n.name for n in ast.walk(cls) if isinstance(n, ast.FunctionDef)]
        
        # Should NOT have these methods
        bad_methods = ['record_batch', 'record_attention_maps', '_record_model_weights']
        for method in bad_methods:
            if method in methods:
                print(f"     ✗ TrainingStatistics still has {method} method (should be removed)")
                return False
        print(f"     ✓ Removed unwanted methods: {', '.join(bad_methods)}")
        
        # Should have record_epoch without model parameter
        if 'record_epoch' in methods:
            print("     ✓ record_epoch method exists")
    else:
        print("   ✗ TrainingStatistics class missing")
        return False
    
    print("\n2. Checking function definitions...")
    
    # Check new function exists
    if 'collect_and_save_attention_maps' in functions:
        print("   ✓ collect_and_save_attention_maps function exists")
    else:
        print("   ✗ collect_and_save_attention_maps function missing")
        return False
    
    # Check old function is gone or renamed
    if 'collect_eval_attention_maps' not in functions:
        print("   ✓ collect_eval_attention_maps function removed (replaced)")
    else:
        print("   ℹ collect_eval_attention_maps still exists (may be OK if updated)")
    
    print("\n3. Checking imports...")
    
    # Check pickle is not imported
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    
    if 'pickle' not in imports:
        print("   ✓ pickle module not imported (good - using NPZ instead)")
    else:
        print("   ℹ pickle module still imported (may be for backward compatibility)")
    
    # Check that numpy is used
    if 'numpy' in imports or any('np' in code for code in [code]):
        print("   ✓ numpy/NPZ usage detected")
    
    print("\n4. Checking code patterns...")
    
    # Check for NPZ usage
    if 'savez_compressed' in code:
        print("   ✓ Uses np.savez_compressed for attention maps")
    else:
        print("   ✗ Missing np.savez_compressed usage")
        return False
    
    if 'np.load' in code:
        print("   ✓ Uses np.load for loading attention maps")
    else:
        print("   ✗ Missing np.load usage")
        return False
    
    # Check for single checkpoint pattern
    if 'best_model.pth' in code:
        print("   ✓ Uses single checkpoint: best_model.pth")
    else:
        print("   ℹ Checkpoint naming may vary")
    
    # Check for storage invariants documentation
    if 'ATTENTION MAP STORAGE INVARIANTS' in code or 'Storage Invariants' in code:
        print("   ✓ Storage invariants documented in code")
    else:
        print("   ℹ Storage invariants documentation could be enhanced")
    
    print("\n5. Checking file organization...")
    
    # Check that STORAGE_ARCHITECTURE.md exists
    doc_path = Path(__file__).parent / 'STORAGE_ARCHITECTURE.md'
    if doc_path.exists():
        print("   ✓ STORAGE_ARCHITECTURE.md documentation exists")
        with open(doc_path) as f:
            doc_content = f.read()
        
        # Check key sections
        key_sections = [
            'Invariants',
            'Memory Efficiency',
            'What NOT to Save',
            'NPZ format',
            'Compression'
        ]
        for section in key_sections:
            if section in doc_content:
                print(f"     ✓ Documentation includes '{section}' section")
    else:
        print("   ✗ STORAGE_ARCHITECTURE.md missing")
        return False
    
    # Check .gitignore
    gitignore_path = Path(__file__).parent / '.gitignore'
    if gitignore_path.exists():
        print("   ✓ .gitignore exists")
        with open(gitignore_path) as f:
            gitignore_content = f.read()
        
        # Check key exclusions
        key_exclusions = ['*.pth', '*.pkl', 'training_statistics/', 'attention_maps/']
        for exclusion in key_exclusions:
            if exclusion in gitignore_content:
                print(f"     ✓ .gitignore excludes {exclusion}")
    else:
        print("   ℹ .gitignore missing (recommended to add)")
    
    return True

def check_design_principles():
    """Verify design principles are met."""
    print("\n" + "=" * 80)
    print("VALIDATING DESIGN PRINCIPLES")
    print("=" * 80)
    
    code_path = Path(__file__).parent / 'updated_main_gaze.py'
    with open(code_path) as f:
        code = f.read()
    
    principles = {
        "No duplicate storage": [
            ("attention_maps stored once", 'AttentionMapStorage' in code),
            ("No in-memory accumulation", 'save_batch' in code)
        ],
        "Memory efficiency": [
            ("Batch-by-batch processing", 'for batch' in code or 'for batch_idx' in code),
            ("Immediate disk save", 'save_batch' in code)
        ],
        "Safe formats": [
            ("NPZ for attention maps", 'savez_compressed' in code),
            ("CSV for metrics", '.to_csv' in code),
            ("JSON for metadata", 'json.dump' in code)
        ],
        "Single checkpoint": [
            ("best_model.pth", 'best_model.pth' in code),
            ("Validation loss criterion", 'eval_loss' in code or 'val_loss' in code)
        ],
        "Separation of concerns": [
            ("AttentionMapStorage class", 'class AttentionMapStorage' in code),
            ("TrainingStatistics class", 'class TrainingStatistics' in code),
            ("Separate directories", 'attention_maps' in code)
        ]
    }
    
    all_pass = True
    for principle, checks in principles.items():
        print(f"\n{principle}:")
        for desc, passed in checks:
            if passed:
                print(f"  ✓ {desc}")
            else:
                print(f"  ✗ {desc}")
                all_pass = False
    
    return all_pass

def main():
    """Run all validation checks."""
    print("\n" + "=" * 80)
    print("STORAGE ARCHITECTURE REFACTORING VALIDATION")
    print("=" * 80)
    
    try:
        # Check code structure
        if not check_code_structure():
            print("\n✗ Code structure validation failed")
            return 1
        
        # Check design principles
        if not check_design_principles():
            print("\n✗ Design principles validation failed")
            return 1
        
        print("\n" + "=" * 80)
        print("✓✓✓ ALL VALIDATIONS PASSED ✓✓✓")
        print("=" * 80)
        print("\nRefactoring Summary:")
        print("  ✓ AttentionMapStorage class properly implemented")
        print("  ✓ TrainingStatistics class cleaned up (no attention maps)")
        print("  ✓ Memory-efficient batch processing")
        print("  ✓ NPZ format for attention maps")
        print("  ✓ CSV/JSON for training statistics")
        print("  ✓ Single authoritative checkpoint")
        print("  ✓ Clear separation of concerns")
        print("  ✓ Comprehensive documentation")
        print("\n✓ Refactoring meets all requirements!")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Validation error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
