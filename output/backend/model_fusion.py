"""
Model fusion module for combining CNN and GNN predictions
"""
import numpy as np
from typing import Tuple, Dict, List


class ModelFusion:
    """Ensemble methods for combining multiple model predictions"""
    
    @staticmethod
    def late_fusion_average(cnn_probs: np.ndarray, gnn_probs: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Simple averaging of probabilities
        
        Args:
            cnn_probs: CNN probability distribution
            gnn_probs: GNN probability distribution
            
        Returns:
            Tuple of (predicted_class, fused_probabilities)
        """
        fused_probs = (cnn_probs + gnn_probs) / 2.0
        predicted_class = np.argmax(fused_probs)
        return predicted_class, fused_probs
    
    @staticmethod
    def late_fusion_weighted(cnn_probs: np.ndarray, gnn_probs: np.ndarray,
                            cnn_weight: float = 0.6, gnn_weight: float = 0.4) -> Tuple[int, np.ndarray]:
        """
        Weighted fusion of probabilities
        
        Args:
            cnn_probs: CNN probability distribution
            gnn_probs: GNN probability distribution
            cnn_weight: Weight for CNN predictions
            gnn_weight: Weight for GNN predictions
            
        Returns:
            Tuple of (predicted_class, fused_probabilities)
        """
        # Normalize weights
        total_weight = cnn_weight + gnn_weight
        cnn_weight /= total_weight
        gnn_weight /= total_weight
        
        fused_probs = cnn_weight * cnn_probs + gnn_weight * gnn_probs
        predicted_class = np.argmax(fused_probs)
        return predicted_class, fused_probs
    
    @staticmethod
    def late_fusion_max(cnn_probs: np.ndarray, gnn_probs: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Maximum probability fusion
        
        Args:
            cnn_probs: CNN probability distribution
            gnn_probs: GNN probability distribution
            
        Returns:
            Tuple of (predicted_class, fused_probabilities)
        """
        fused_probs = np.maximum(cnn_probs, gnn_probs)
        predicted_class = np.argmax(fused_probs)
        return predicted_class, fused_probs
    
    @staticmethod
    def late_fusion_geometric_mean(cnn_probs: np.ndarray, gnn_probs: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Geometric mean fusion
        
        Args:
            cnn_probs: CNN probability distribution
            gnn_probs: GNN probability distribution
            
        Returns:
            Tuple of (predicted_class, fused_probabilities)
        """
        fused_probs = np.sqrt(cnn_probs * gnn_probs)
        # Normalize
        fused_probs = fused_probs / np.sum(fused_probs)
        predicted_class = np.argmax(fused_probs)
        return predicted_class, fused_probs
    
    @staticmethod
    def voting_fusion(cnn_class: int, gnn_class: int, 
                     additional_classes: List[int] = None) -> int:
        """
        Majority voting fusion
        
        Args:
            cnn_class: CNN predicted class
            gnn_class: GNN predicted class
            additional_classes: Additional model predictions
            
        Returns:
            Voted class
        """
        votes = [cnn_class, gnn_class]
        if additional_classes:
            votes.extend(additional_classes)
        
        # Count votes
        from collections import Counter
        vote_counts = Counter(votes)
        predicted_class = vote_counts.most_common(1)[0][0]
        
        return predicted_class
    
    @staticmethod
    def confidence_based_fusion(cnn_probs: np.ndarray, gnn_probs: np.ndarray,
                               threshold: float = 0.8) -> Tuple[int, np.ndarray]:
        """
        Fusion based on confidence threshold
        
        If one model is very confident (max prob > threshold), use it.
        Otherwise, use weighted average.
        
        Args:
            cnn_probs: CNN probability distribution
            gnn_probs: GNN probability distribution
            threshold: Confidence threshold
            
        Returns:
            Tuple of (predicted_class, fused_probabilities)
        """
        cnn_max = np.max(cnn_probs)
        gnn_max = np.max(gnn_probs)
        
        if cnn_max > threshold and cnn_max > gnn_max:
            # Use CNN prediction
            predicted_class = np.argmax(cnn_probs)
            return predicted_class, cnn_probs
        elif gnn_max > threshold and gnn_max > cnn_max:
            # Use GNN prediction
            predicted_class = np.argmax(gnn_probs)
            return predicted_class, gnn_probs
        else:
            # Use weighted average
            return ModelFusion.late_fusion_weighted(cnn_probs, gnn_probs)
    
    @staticmethod
    def ensemble_predictions(models_probs: List[np.ndarray], 
                           method: str = "average",
                           weights: List[float] = None) -> Tuple[int, np.ndarray]:
        """
        Ensemble multiple model predictions
        
        Args:
            models_probs: List of probability distributions from different models
            method: Fusion method ('average', 'weighted', 'max', 'geometric')
            weights: Weights for each model (only for 'weighted' method)
            
        Returns:
            Tuple of (predicted_class, fused_probabilities)
        """
        if method == "average":
            fused_probs = np.mean(models_probs, axis=0)
        elif method == "weighted":
            if weights is None:
                weights = [1.0 / len(models_probs)] * len(models_probs)
            else:
                # Normalize weights
                weights = np.array(weights) / np.sum(weights)
            
            fused_probs = np.average(models_probs, axis=0, weights=weights)
        elif method == "max":
            fused_probs = np.max(models_probs, axis=0)
        elif method == "geometric":
            fused_probs = np.prod(models_probs, axis=0) ** (1.0 / len(models_probs))
            fused_probs = fused_probs / np.sum(fused_probs)
        else:
            raise ValueError(f"Unknown fusion method: {method}")
        
        predicted_class = np.argmax(fused_probs)
        return predicted_class, fused_probs


class AdaptiveFusion:
    """Adaptive fusion with learned weights"""
    
    def __init__(self, num_models: int = 2):
        """
        Initialize adaptive fusion
        
        Args:
            num_models: Number of models to fuse
        """
        self.num_models = num_models
        self.weights = np.ones(num_models) / num_models  # Initialize with equal weights
    
    def train_weights(self, all_probs: List[List[np.ndarray]], 
                     y_true: np.ndarray, lr: float = 0.01, epochs: int = 100):
        """
        Learn optimal fusion weights using gradient descent
        
        Args:
            all_probs: List of [model1_probs, model2_probs, ...] for each sample
            y_true: True labels
            lr: Learning rate
            epochs: Number of training epochs
        """
        n_samples = len(all_probs)
        
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(n_samples):
                # Get probabilities from all models for this sample
                sample_probs = all_probs[i]  # Shape: (num_models, num_classes)
                
                # Fuse with current weights
                fused_probs = np.average(sample_probs, axis=0, weights=self.weights)
                
                # Cross-entropy loss
                true_class = y_true[i]
                loss = -np.log(fused_probs[true_class] + 1e-10)
                total_loss += loss
                
                # Gradient (simplified)
                gradient = np.zeros(self.num_models)
                for j in range(self.num_models):
                    gradient[j] = -(sample_probs[j][true_class] - fused_probs[true_class])
                
                # Update weights
                self.weights -= lr * gradient
                
                # Normalize weights
                self.weights = np.clip(self.weights, 0, 1)
                self.weights = self.weights / np.sum(self.weights)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_samples:.4f}")
    
    def fuse(self, models_probs: List[np.ndarray]) -> Tuple[int, np.ndarray]:
        """
        Fuse predictions using learned weights
        
        Args:
            models_probs: List of probability distributions
            
        Returns:
            Tuple of (predicted_class, fused_probabilities)
        """
        fused_probs = np.average(models_probs, axis=0, weights=self.weights)
        predicted_class = np.argmax(fused_probs)
        return predicted_class, fused_probs
