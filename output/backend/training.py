"""
Model Training & Evaluation Pipeline
Performs reproducible train-test splits and comprehensive model evaluation
"""
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from cnn_model import VGG16Classifier, ResNetClassifier, EfficientNetClassifier
from gnn_model import GCNModel, GATModel, GraphSAGEModel
from evaluation import EvaluationMetrics
from config import PROTEIN_CLASSES, IMAGE_SIZE, OUTPUT_DIR


class ModelTrainer:
    """Comprehensive training and evaluation for all model variants"""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize trainer
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.set_random_seeds()
        self.results = {}
        
    def set_random_seeds(self):
        """Set random seeds for reproducibility"""
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
    
    def prepare_data(self, images: np.ndarray, labels: np.ndarray, 
                     test_size: float = 0.2, val_size: float = 0.1) -> Dict:
        """
        Perform train-validation-test split
        
        Args:
            images: Image data (N x H x W x C)
            labels: Labels (N,)
            test_size: Proportion for test set
            val_size: Proportion for validation set (from remaining data)
            
        Returns:
            Dictionary with split data
        """
        # Encode labels
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)
        
        # Train-test split
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            images, labels_encoded, 
            test_size=test_size, 
            random_state=self.random_seed,
            stratify=labels_encoded
        )
        
        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_size / (1 - test_size),
            random_state=self.random_seed,
            stratify=y_train_val
        )
        
        # Convert labels to one-hot for keras models
        num_classes = len(np.unique(labels_encoded))
        y_train_oh = keras.utils.to_categorical(y_train, num_classes)
        y_val_oh = keras.utils.to_categorical(y_val, num_classes)
        y_test_oh = keras.utils.to_categorical(y_test, num_classes)
        
        return {
            'X_train': X_train, 'y_train': y_train, 'y_train_oh': y_train_oh,
            'X_val': X_val, 'y_val': y_val, 'y_val_oh': y_val_oh,
            'X_test': X_test, 'y_test': y_test, 'y_test_oh': y_test_oh,
            'label_encoder': le,
            'num_classes': num_classes
        }
    
    def train_cnn_model(self, model_name: str, data: Dict, 
                       epochs: int = 50, batch_size: int = 32) -> Dict:
        """
        Train CNN model variant
        
        Args:
            model_name: 'vgg16', 'resnet50', or 'efficientnet'
            data: Data dictionary from prepare_data()
            epochs: Training epochs
            batch_size: Batch size
            
        Returns:
            Dictionary with training results
        """
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()} Model")
        print(f"{'='*60}")
        
        # Initialize model
        num_classes = data['num_classes']
        input_shape = (*IMAGE_SIZE, 3)
        
        if model_name.lower() == 'vgg16':
            model = VGG16Classifier(num_classes=num_classes, input_shape=input_shape)
        elif model_name.lower() == 'resnet50':
            model = ResNetClassifier(num_classes=num_classes, input_shape=input_shape)
        elif model_name.lower() == 'efficientnet':
            model = EfficientNetClassifier(num_classes=num_classes, input_shape=input_shape)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Train
        history = model.train(
            data['X_train'], data['y_train_oh'],
            data['X_val'], data['y_val_oh'],
            epochs=epochs, batch_size=batch_size
        )
        
        # Evaluate on all sets
        results = {
            'model_name': model_name,
            'model_type': 'CNN',
            'history': history.history,
            'train_metrics': self._evaluate_model(model, data['X_train'], data['y_train'], 'train'),
            'val_metrics': self._evaluate_model(model, data['X_val'], data['y_val'], 'val'),
            'test_metrics': self._evaluate_model(model, data['X_test'], data['y_test'], 'test'),
        }
        
        # Save model
        model_path = os.path.join(OUTPUT_DIR, 'models', f'{model_name}_trained.h5')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save_model(model_path)
        results['model_path'] = model_path
        
        self.results[model_name] = results
        return results
    
    def train_gnn_model(self, model_name: str, graph_data: List, 
                       labels: np.ndarray, epochs: int = 100) -> Dict:
        """
        Train GNN model variant
        
        Args:
            model_name: 'gcn', 'gat', or 'graphsage'
            graph_data: List of PyTorch Geometric Data objects
            labels: Labels for graphs
            epochs: Training epochs
            
        Returns:
            Dictionary with training results
        """
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()} Model")
        print(f"{'='*60}")
        
        # Split graph data
        indices = np.arange(len(graph_data))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=self.random_seed, stratify=labels
        )
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.125, random_state=self.random_seed,
            stratify=labels[train_idx]
        )
        
        # Initialize model
        num_classes = len(np.unique(labels))
        input_dim = graph_data[0].x.shape[1] if len(graph_data) > 0 else 11
        
        if model_name.lower() == 'gcn':
            model = GCNModel(input_dim=input_dim, hidden_dim=64, num_classes=num_classes)
        elif model_name.lower() == 'gat':
            model = GATModel(input_dim=input_dim, hidden_dim=64, num_classes=num_classes)
        elif model_name.lower() == 'graphsage':
            model = GraphSAGEModel(input_dim=input_dim, hidden_dim=64, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown GNN model: {model_name}")
        
        # Train model (implementation depends on GNN model class)
        # For now, we'll record the structure
        results = {
            'model_name': model_name,
            'model_type': 'GNN',
            'epochs': epochs,
            'num_graphs': len(graph_data),
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'test_size': len(test_idx),
        }
        
        self.results[model_name] = results
        return results
    
    def _evaluate_model(self, model, X: np.ndarray, y: np.ndarray, 
                       split: str) -> Dict:
        """
        Evaluate model on a dataset
        
        Args:
            model: Trained model
            X: Images
            y: True labels
            split: 'train', 'val', or 'test'
            
        Returns:
            Dictionary of metrics
        """
        # Get predictions
        y_pred = []
        for i in range(len(X)):
            pred_class, _ = model.predict(X[i])
            y_pred.append(pred_class)
        y_pred = np.array(y_pred)
        
        # Compute metrics
        metrics = EvaluationMetrics.compute_metrics(y, y_pred, PROTEIN_CLASSES)
        
        print(f"\n{split.upper()} SET METRICS:")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  F1-Score:    {metrics['f1']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        
        return metrics
    
    def visualize_training_history(self, model_name: str, save_dir: str = None):
        """
        Visualize training history (loss and accuracy curves)
        
        Args:
            model_name: Name of model to visualize
            save_dir: Directory to save plots
        """
        if model_name not in self.results:
            print(f"No results found for {model_name}")
            return
        
        results = self.results[model_name]
        if 'history' not in results:
            print(f"No training history found for {model_name}")
            return
        
        history = results['history']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(history['loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title(f'{model_name.upper()} - Training Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
        if 'val_accuracy' in history:
            axes[1].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title(f'{model_name.upper()} - Training Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{model_name}_training_history.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved training history plot: {save_path}")
        
        plt.close()
    
    def visualize_all_metrics(self, save_dir: str = None):
        """
        Create comprehensive visualization of all model metrics
        
        Args:
            save_dir: Directory to save plots
        """
        if not self.results:
            print("No results to visualize")
            return
        
        # Prepare data for comparison
        models = []
        train_acc, val_acc, test_acc = [], [], []
        train_f1, val_f1, test_f1 = [], [], []
        
        for model_name, result in self.results.items():
            if 'train_metrics' in result:
                models.append(model_name)
                train_acc.append(result['train_metrics']['accuracy'])
                val_acc.append(result['val_metrics']['accuracy'])
                test_acc.append(result['test_metrics']['accuracy'])
                train_f1.append(result['train_metrics']['f1'])
                val_f1.append(result['val_metrics']['f1'])
                test_f1.append(result['test_metrics']['f1'])
        
        if not models:
            print("No metrics available for visualization")
            return
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Accuracy comparison
        x = np.arange(len(models))
        width = 0.25
        
        axes[0, 0].bar(x - width, train_acc, width, label='Train', alpha=0.8)
        axes[0, 0].bar(x, val_acc, width, label='Validation', alpha=0.8)
        axes[0, 0].bar(x + width, test_acc, width, label='Test', alpha=0.8)
        axes[0, 0].set_xlabel('Model', fontsize=12)
        axes[0, 0].set_ylabel('Accuracy', fontsize=12)
        axes[0, 0].set_title('Accuracy Comparison Across Models', fontsize=14, fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        axes[0, 0].set_ylim([0, 1.0])
        
        # F1-Score comparison
        axes[0, 1].bar(x - width, train_f1, width, label='Train', alpha=0.8)
        axes[0, 1].bar(x, val_f1, width, label='Validation', alpha=0.8)
        axes[0, 1].bar(x + width, test_f1, width, label='Test', alpha=0.8)
        axes[0, 1].set_xlabel('Model', fontsize=12)
        axes[0, 1].set_ylabel('F1-Score', fontsize=12)
        axes[0, 1].set_title('F1-Score Comparison Across Models', fontsize=14, fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        axes[0, 1].set_ylim([0, 1.0])
        
        # Test set metrics comparison
        test_metrics = ['accuracy', 'precision', 'recall', 'f1', 'specificity']
        test_values = {model: [] for model in models}
        
        for model_name in models:
            result = self.results[model_name]
            for metric in test_metrics:
                test_values[model_name].append(result['test_metrics'][metric])
        
        x_metrics = np.arange(len(test_metrics))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            axes[1, 0].bar(x_metrics + i * width, test_values[model], width, 
                          label=model, alpha=0.8)
        
        axes[1, 0].set_xlabel('Metric', fontsize=12)
        axes[1, 0].set_ylabel('Score', fontsize=12)
        axes[1, 0].set_title('Test Set Metrics Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].set_xticks(x_metrics + width * (len(models) - 1) / 2)
        axes[1, 0].set_xticklabels(test_metrics, rotation=45, ha='right')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].set_ylim([0, 1.0])
        
        # Best model summary
        best_model_acc = max(models, key=lambda m: self.results[m]['test_metrics']['accuracy'])
        best_model_f1 = max(models, key=lambda m: self.results[m]['test_metrics']['f1'])
        
        summary_text = f"""
        BEST MODELS (Test Set)
        
        Highest Accuracy:
          {best_model_acc.upper()}
          Accuracy: {self.results[best_model_acc]['test_metrics']['accuracy']:.4f}
          
        Highest F1-Score:
          {best_model_f1.upper()}
          F1-Score: {self.results[best_model_f1]['test_metrics']['f1']:.4f}
          
        Overall Test Performance:
        """
        
        for model in models:
            summary_text += f"\n  {model.upper()}:"
            summary_text += f"\n    Acc: {self.results[model]['test_metrics']['accuracy']:.4f}"
            summary_text += f"  F1: {self.results[model]['test_metrics']['f1']:.4f}"
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                       verticalalignment='center')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'all_models_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison plot: {save_path}")
        
        plt.close()
    
    def save_results(self, save_path: str):
        """
        Save all training results to JSON
        
        Args:
            save_path: Path to save JSON file
        """
        # Convert numpy types to Python types for JSON serialization
        serializable_results = {}
        
        for model_name, result in self.results.items():
            serializable_result = {}
            for key, value in result.items():
                if key == 'history':
                    # Convert history values to lists
                    serializable_result[key] = {k: [float(v) for v in vals] 
                                               for k, vals in value.items()}
                elif isinstance(value, dict):
                    # Handle metrics dictionaries
                    serializable_result[key] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                               for k, v in value.items()}
                else:
                    serializable_result[key] = value
            
            serializable_results[model_name] = serializable_result
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nSaved training results to: {save_path}")
    
    def print_summary(self):
        """Print summary of all training results"""
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        for model_name, result in self.results.items():
            print(f"\n{model_name.upper()} ({result['model_type']})")
            print("-" * 40)
            
            if 'test_metrics' in result:
                metrics = result['test_metrics']
                print(f"  Test Accuracy:    {metrics['accuracy']:.4f}")
                print(f"  Test Precision:   {metrics['precision']:.4f}")
                print(f"  Test Recall:      {metrics['recall']:.4f}")
                print(f"  Test F1-Score:    {metrics['f1']:.4f}")
                print(f"  Test Specificity: {metrics['specificity']:.4f}")
            
            if 'model_path' in result:
                print(f"  Saved to: {result['model_path']}")


if __name__ == "__main__":
    print("Model Training & Evaluation Module")
    print("Use this module to train and evaluate all model variants")
    print("\nExample usage:")
    print("  trainer = ModelTrainer(random_seed=42)")
    print("  data = trainer.prepare_data(images, labels)")
    print("  trainer.train_cnn_model('vgg16', data)")
    print("  trainer.visualize_all_metrics(save_dir='output/graphs')")
