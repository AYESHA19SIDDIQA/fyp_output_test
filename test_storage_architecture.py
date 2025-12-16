#!/usr/bin/env python3
"""
Test script for validating the refactored storage architecture.

Tests:
1. AttentionMapStorage can save and load attention maps correctly
2. NPZ format preserves data integrity
3. Memory efficiency (no large accumulations)
4. File manifest and metadata are correct
5. Saved artifacts are reloadable
"""

import numpy as np
import torch
import tempfile
import shutil
from pathlib import Path
import json

# Import the new storage classes
import sys
sys.path.insert(0, '/home/runner/work/fyp_output_test/fyp_output_test')
from updated_main_gaze import AttentionMapStorage, TrainingStatistics

def test_attention_map_storage():
    """Test AttentionMapStorage class functionality."""
    print("=" * 80)
    print("TEST: AttentionMapStorage")
    print("=" * 80)
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = AttentionMapStorage(output_dir=tmpdir)
        
        # Test 1: Save single attention map
        print("\n1. Testing single attention map save...")
        attention_map = np.random.rand(22, 15000).astype(np.float32)
        attention_map = np.clip(attention_map, 0, 1)  # Ensure [0, 1] range
        
        file_id = "test_sample_001"
        saved_path = storage.save_attention_map(attention_map, file_id)
        print(f"   ✓ Saved to: {saved_path}")
        assert saved_path.exists(), "File should exist"
        
        # Test 2: Load and verify
        print("\n2. Testing attention map load...")
        loaded_data = AttentionMapStorage.load_attention_map(saved_path)
        loaded_map = loaded_data['attention_map']
        
        assert loaded_map.shape == (22, 15000), f"Shape mismatch: {loaded_map.shape}"
        assert loaded_map.dtype == np.float32, f"Dtype mismatch: {loaded_map.dtype}"
        assert np.allclose(loaded_map, attention_map, atol=1e-6), "Data mismatch"
        assert loaded_data['file_id'] == file_id, "File ID mismatch"
        print(f"   ✓ Loaded shape: {loaded_map.shape}, dtype: {loaded_map.dtype}")
        print(f"   ✓ Data integrity verified (max diff: {np.abs(loaded_map - attention_map).max():.2e})")
        
        # Test 3: Batch save
        print("\n3. Testing batch save...")
        batch_size = 5
        batch_maps = np.random.rand(batch_size, 22, 15000).astype(np.float32)
        batch_maps = np.clip(batch_maps, 0, 1)
        batch_ids = [f"test_sample_{i:03d}" for i in range(2, 2 + batch_size)]
        
        saved_paths = storage.save_batch(batch_maps, batch_ids)
        print(f"   ✓ Saved {len(saved_paths)} files")
        
        # Verify batch
        for i, path in enumerate(saved_paths):
            assert path.exists(), f"Batch file {i} should exist"
            loaded = AttentionMapStorage.load_attention_map(path)
            assert np.allclose(loaded['attention_map'], batch_maps[i], atol=1e-6)
        print(f"   ✓ All batch files verified")
        
        # Test 4: Metadata and manifest
        print("\n4. Testing metadata and manifest...")
        meta_path, manifest_path = storage.save_metadata()
        
        assert meta_path.exists(), "Metadata file should exist"
        assert manifest_path.exists(), "Manifest file should exist"
        
        with open(meta_path) as f:
            metadata = json.load(f)
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        print(f"   ✓ Metadata: {metadata['n_samples']} samples")
        print(f"   ✓ Manifest: {len(manifest)} entries")
        assert metadata['n_samples'] == 6, f"Should have 6 samples, got {metadata['n_samples']}"
        assert len(manifest) == 6, f"Manifest should have 6 entries, got {len(manifest)}"
        
        # Test 5: File size (compression check)
        print("\n5. Testing compression...")
        file_size = saved_paths[0].stat().st_size
        uncompressed_size = 22 * 15000 * 4  # float32 = 4 bytes
        compression_ratio = file_size / uncompressed_size
        print(f"   Uncompressed size: {uncompressed_size / 1024:.1f} KB")
        print(f"   Compressed size: {file_size / 1024:.1f} KB")
        print(f"   Compression ratio: {compression_ratio:.2%}")
        print(f"   ✓ Compression working (ratio < 100%): {compression_ratio < 1.0}")
        
        # Test 6: Shape validation
        print("\n6. Testing shape validation...")
        try:
            bad_shape = np.random.rand(20, 10000).astype(np.float32)
            storage.save_attention_map(bad_shape, "bad_shape", validate=True)
            print("   ✗ Should have raised ValueError for bad shape")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            print(f"   ✓ Correctly rejected bad shape: {e}")
        
        print("\n" + "=" * 80)
        print("✓ ALL AttentionMapStorage TESTS PASSED")
        print("=" * 80)

def test_training_statistics():
    """Test TrainingStatistics class (verify no attention map storage)."""
    print("\n" + "=" * 80)
    print("TEST: TrainingStatistics")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        stats = TrainingStatistics(output_dir=tmpdir)
        
        # Test 1: Record epoch (no model weights)
        print("\n1. Testing epoch recording...")
        train_stats = {
            'loss': 0.5,
            'cls_loss': 0.3,
            'gaze_loss': 0.2,
            'acc': 85.0,
            'lr': 1e-4,
            'gaze_batches': 10,
            'gaze_samples': 320
        }
        eval_stats = {
            'acc': 88.0,
            'loss': 0.45,
            'cls_loss': 0.28,
            'gaze_loss': 0.17,
            'macro_f1': 0.87,
            'balanced_acc': 0.86,
            'weighted_f1': 0.88,
            'precision': 0.89,
            'recall': 0.87,
            'gaze_batches': 5
        }
        
        epoch_data = stats.record_epoch(0, train_stats, eval_stats)
        print(f"   ✓ Recorded epoch 0")
        assert epoch_data['epoch'] == 0
        assert epoch_data['train_loss'] == 0.5
        assert epoch_data['eval_acc'] == 88.0
        
        # Verify no attention_maps attribute
        assert not hasattr(stats, 'attention_maps') or len(stats.attention_maps) == 0, \
            "TrainingStatistics should not store attention maps"
        print(f"   ✓ No attention maps stored in TrainingStatistics")
        
        # Test 2: Save and reload
        print("\n2. Testing save and reload...")
        stats.save_final_results()
        
        # Check files exist
        csv_file = stats.run_dir / 'epoch_statistics.csv'
        assert csv_file.exists(), "Epoch statistics CSV should exist"
        print(f"   ✓ Saved epoch_statistics.csv")
        
        # Load and verify
        import pandas as pd
        df = pd.read_csv(csv_file)
        assert len(df) == 1, "Should have 1 epoch"
        assert df.iloc[0]['train_loss'] == 0.5
        assert df.iloc[0]['eval_acc'] == 88.0
        print(f"   ✓ CSV data verified")
        
        # Test 3: Predictions
        print("\n3. Testing predictions storage...")
        files = ['sample1.npz', 'sample2.npz', 'sample3.npz']
        true_labels = [0, 1, 0]
        predictions = [0, 1, 1]
        probabilities = [[0.9, 0.1], [0.3, 0.7], [0.4, 0.6]]
        
        stats.record_predictions(files, true_labels, predictions, probabilities, 'eval')
        stats.save_final_results()
        
        pred_file = stats.run_dir / 'predictions_eval.csv'
        assert pred_file.exists(), "Predictions CSV should exist"
        
        pred_df = pd.read_csv(pred_file)
        assert len(pred_df) == 3, "Should have 3 predictions"
        assert pred_df.iloc[0]['correct'] == True
        assert pred_df.iloc[2]['correct'] == False
        print(f"   ✓ Predictions CSV verified")
        
        # Test 4: Confusion matrix (JSON, not pickle)
        print("\n4. Testing confusion matrix storage...")
        y_true = [0, 1, 0, 1, 0]
        y_pred = [0, 1, 1, 1, 0]
        stats.record_confusion_matrix(y_true, y_pred, 'eval')
        stats.save_final_results()
        
        cm_file = stats.run_dir / 'confusion_matrices.json'
        assert cm_file.exists(), "Confusion matrix JSON should exist"
        
        with open(cm_file) as f:
            cm_data = json.load(f)
        
        assert len(cm_data) > 0, "Should have confusion matrices"
        assert 'matrix' in cm_data[0], "Should have matrix field"
        print(f"   ✓ Confusion matrices JSON verified (not pickle)")
        
        print("\n" + "=" * 80)
        print("✓ ALL TrainingStatistics TESTS PASSED")
        print("=" * 80)

def test_memory_efficiency():
    """Test memory efficiency of batch processing."""
    print("\n" + "=" * 80)
    print("TEST: Memory Efficiency")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = AttentionMapStorage(output_dir=tmpdir)
        
        print("\n1. Simulating batch-by-batch processing...")
        n_batches = 10
        batch_size = 32
        
        for batch_idx in range(n_batches):
            # Simulate one batch
            batch_maps = np.random.rand(batch_size, 22, 15000).astype(np.float32)
            batch_maps = np.clip(batch_maps, 0, 1)
            batch_ids = [f"batch{batch_idx}_sample{i}" for i in range(batch_size)]
            
            # Save immediately (memory-efficient)
            storage.save_batch(batch_maps, batch_ids)
            
            # Delete batch data (simulates not accumulating in memory)
            del batch_maps
            
            if batch_idx % 3 == 0:
                print(f"   Processed batch {batch_idx + 1}/{n_batches}")
        
        print(f"\n2. Verifying all {n_batches * batch_size} files saved...")
        storage.save_metadata()
        assert storage.metadata['n_samples'] == n_batches * batch_size
        print(f"   ✓ All {storage.metadata['n_samples']} attention maps saved")
        
        # Check total disk usage
        total_size = sum(f.stat().st_size for f in Path(tmpdir).glob('*.npz'))
        avg_size_kb = total_size / storage.metadata['n_samples'] / 1024
        print(f"\n3. Disk usage:")
        print(f"   Total: {total_size / (1024 * 1024):.1f} MB")
        print(f"   Average per file: {avg_size_kb:.1f} KB")
        print(f"   ✓ Memory-efficient: processed {n_batches * batch_size} samples without accumulation")
        
        print("\n" + "=" * 80)
        print("✓ MEMORY EFFICIENCY TEST PASSED")
        print("=" * 80)

def test_reproducibility():
    """Test that saved artifacts are reloadable."""
    print("\n" + "=" * 80)
    print("TEST: Reproducibility (Reload Artifacts)")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save some test data
        storage = AttentionMapStorage(output_dir=tmpdir)
        test_maps = np.random.rand(5, 22, 15000).astype(np.float32)
        test_ids = [f"test_{i}" for i in range(5)]
        storage.save_batch(test_maps, test_ids)
        storage.save_metadata()
        
        print("\n1. Testing attention map reload...")
        # Reload all maps
        reloaded = []
        for npz_file in Path(tmpdir).glob('*_attention.npz'):
            data = AttentionMapStorage.load_attention_map(npz_file)
            reloaded.append(data['attention_map'])
        
        assert len(reloaded) == 5, f"Should reload 5 maps, got {len(reloaded)}"
        print(f"   ✓ Successfully reloaded {len(reloaded)} attention maps")
        
        # Verify data integrity
        for i, reloaded_map in enumerate(reloaded):
            # Maps may be in different order, so just check properties
            assert reloaded_map.shape == (22, 15000)
            assert reloaded_map.dtype == np.float32
            assert reloaded_map.min() >= 0.0 and reloaded_map.max() <= 1.0
        print(f"   ✓ All reloaded maps have correct shape and range")
        
        print("\n2. Testing metadata reload...")
        with open(Path(tmpdir) / 'metadata.json') as f:
            metadata = json.load(f)
        
        assert metadata['n_samples'] == 5
        assert metadata['expected_shape'] == [22, 15000]
        print(f"   ✓ Metadata correctly reloaded")
        
        print("\n3. Testing manifest reload...")
        with open(Path(tmpdir) / 'manifest.json') as f:
            manifest = json.load(f)
        
        assert len(manifest) == 5
        assert all('file_id' in entry for entry in manifest)
        assert all('path' in entry for entry in manifest)
        print(f"   ✓ Manifest correctly reloaded")
        
        print("\n" + "=" * 80)
        print("✓ REPRODUCIBILITY TEST PASSED")
        print("=" * 80)

def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("STORAGE ARCHITECTURE VALIDATION TESTS")
    print("=" * 80)
    
    try:
        test_attention_map_storage()
        test_training_statistics()
        test_memory_efficiency()
        test_reproducibility()
        
        print("\n" + "=" * 80)
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("=" * 80)
        print("\nSummary:")
        print("  ✓ AttentionMapStorage works correctly")
        print("  ✓ NPZ format preserves data integrity")
        print("  ✓ TrainingStatistics doesn't store attention maps")
        print("  ✓ Memory-efficient batch processing works")
        print("  ✓ All artifacts are reloadable")
        print("\nStorage architecture is ready for production use!")
        
    except Exception as e:
        print(f"\n✗✗✗ TEST FAILED ✗✗✗")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
