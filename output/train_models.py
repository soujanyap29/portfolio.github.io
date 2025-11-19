#!/usr/bin/env python3
"""
Train and evaluate all model variants for protein localization
Usage: python train_models.py --data_dir /path/to/labeled/data --epochs 50
"""
import os
import sys
import argparse
import numpy as np
from glob import glob
from PIL import Image

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from training import ModelTrainer
from config import OUTPUT_DIR, IMAGE_SIZE, PROTEIN_CLASSES


def load_labeled_dataset(data_dir: str):
    """
    Load labeled dataset from directory structure:
    data_dir/
        class1/
            image1.tif
            image2.tif
        class2/
            image3.tif
            ...
    
    Args:
        data_dir: Path to root directory containing class subdirectories
        
    Returns:
        images: numpy array (N, H, W, C)
        labels: numpy array (N,)
    """
    print(f"Loading dataset from: {data_dir}")
    
    images = []
    labels = []
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d))]
    
    if not class_dirs:
        print(f"ERROR: No class directories found in {data_dir}")
        print("Expected structure: data_dir/class_name/image.tif")
        sys.exit(1)
    
    print(f"Found {len(class_dirs)} classes: {class_dirs}")
    
    for class_name in class_dirs:
        class_dir = os.path.join(data_dir, class_name)
        
        # Find all TIFF files
        tif_files = glob(os.path.join(class_dir, '*.tif')) + \
                   glob(os.path.join(class_dir, '*.tiff'))
        
        print(f"  {class_name}: {len(tif_files)} images")
        
        for tif_path in tif_files:
            try:
                # Load and resize image
                img = Image.open(tif_path).convert('RGB')
                img = img.resize(IMAGE_SIZE)
                img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                
                images.append(img_array)
                labels.append(class_name)
            except Exception as e:
                print(f"    Warning: Failed to load {tif_path}: {e}")
                continue
    
    if not images:
        print("ERROR: No images loaded!")
        sys.exit(1)
    
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"\nDataset loaded:")
    print(f"  Total images: {len(images)}")
    print(f"  Image shape: {images.shape}")
    print(f"  Classes: {np.unique(labels)}")
    
    return images, labels


def main():
    parser = argparse.ArgumentParser(
        description='Train protein localization models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all CNN models for 50 epochs
  python train_models.py --data_dir /path/to/data --epochs 50
  
  # Train only VGG16 for 30 epochs with batch size 16
  python train_models.py --data_dir /path/to/data --models vgg16 --epochs 30 --batch_size 16
  
  # Train with custom train-test split
  python train_models.py --data_dir /path/to/data --test_size 0.15 --val_size 0.15
        """
    )
    
    parser.add_argument('--data_dir', required=True,
                       help='Path to labeled dataset directory')
    parser.add_argument('--models', nargs='+', 
                       default=['vgg16', 'resnet50', 'efficientnet'],
                       choices=['vgg16', 'resnet50', 'efficientnet', 'all'],
                       help='Models to train')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Proportion of data for test set')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Proportion of remaining data for validation set')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--output_dir', default=None,
                       help='Output directory (defaults to config OUTPUT_DIR)')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = args.output_dir if args.output_dir else OUTPUT_DIR
    graphs_dir = os.path.join(output_dir, 'graphs')
    results_dir = os.path.join(output_dir, 'results', 'reports')
    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print("="*70)
    print("PROTEIN LOCALIZATION MODEL TRAINING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Data directory:  {args.data_dir}")
    print(f"  Models:          {', '.join(args.models)}")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Test size:       {args.test_size}")
    print(f"  Validation size: {args.val_size}")
    print(f"  Random seed:     {args.random_seed}")
    print(f"  Output dir:      {output_dir}")
    print()
    
    # Load dataset
    images, labels = load_labeled_dataset(args.data_dir)
    
    # Initialize trainer
    trainer = ModelTrainer(random_seed=args.random_seed)
    
    # Prepare data with train-val-test split
    print("\nPerforming train-validation-test split...")
    data = trainer.prepare_data(images, labels, 
                                test_size=args.test_size, 
                                val_size=args.val_size)
    
    print(f"\nData split:")
    print(f"  Train set:      {len(data['X_train'])} samples ({len(data['X_train'])/len(images)*100:.1f}%)")
    print(f"  Validation set: {len(data['X_val'])} samples ({len(data['X_val'])/len(images)*100:.1f}%)")
    print(f"  Test set:       {len(data['X_test'])} samples ({len(data['X_test'])/len(images)*100:.1f}%)")
    print(f"  Number of classes: {data['num_classes']}")
    
    # Determine models to train
    if 'all' in args.models:
        models_to_train = ['vgg16', 'resnet50', 'efficientnet']
    else:
        models_to_train = args.models
    
    # Train each model
    for model_name in models_to_train:
        print(f"\n{'='*70}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*70}")
        
        try:
            results = trainer.train_cnn_model(
                model_name, data, 
                epochs=args.epochs, 
                batch_size=args.batch_size
            )
            
            # Visualize training history
            trainer.visualize_training_history(model_name, save_dir=graphs_dir)
            
            print(f"\n✓ {model_name.upper()} training complete!")
            
        except Exception as e:
            print(f"\n✗ Error training {model_name}: {e}")
            continue
    
    # Create comparison visualizations
    print(f"\n{'='*70}")
    print("GENERATING COMPARISON VISUALIZATIONS")
    print(f"{'='*70}")
    
    trainer.visualize_all_metrics(save_dir=graphs_dir)
    
    # Save results
    results_path = os.path.join(results_dir, 'training_results.json')
    trainer.save_results(results_path)
    
    # Print final summary
    trainer.print_summary()
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\nOutputs saved to:")
    print(f"  Models:         {os.path.join(output_dir, 'models')}")
    print(f"  Visualizations: {graphs_dir}")
    print(f"  Results JSON:   {results_path}")
    print()


if __name__ == '__main__':
    main()
