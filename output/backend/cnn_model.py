"""
VGG16-based CNN classifier for protein localization
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from typing import List, Tuple


class VGG16Classifier:
    """VGG16-based deep CNN for protein localization classification"""
    
    def __init__(self, num_classes: int = 8, input_shape: Tuple[int, int, int] = (224, 224, 3)):
        """
        Initialize VGG16 classifier
        
        Args:
            num_classes: Number of protein localization classes
            input_shape: Input image shape
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = self._build_model()
    
    def _build_model(self) -> keras.Model:
        """
        Build VGG16-based model with transfer learning
        
        Returns:
            Keras model
        """
        # Load pre-trained VGG16
        base_model = VGG16(weights='imagenet', include_top=False, 
                          input_shape=self.input_shape)
        
        # Freeze early layers
        for layer in base_model.layers[:-4]:
            layer.trainable = False
        
        # Build classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: np.ndarray = None, y_val: np.ndarray = None,
             epochs: int = 50, batch_size: int = 32) -> keras.callbacks.History:
        """
        Train the model
        
        Args:
            X_train: Training images
            y_train: Training labels (one-hot encoded)
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        else:
            validation_data = None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, image: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Predict protein localization for an image
        
        Args:
            image: Input image (preprocessed)
            
        Returns:
            Tuple of (predicted_class_index, probability_distribution)
        """
        # Ensure batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Predict
        probabilities = self.model.predict(image, verbose=0)[0]
        predicted_class = np.argmax(probabilities)
        
        return predicted_class, probabilities
    
    def save_model(self, path: str):
        """Save model to file"""
        self.model.save(path)
    
    def load_model(self, path: str):
        """Load model from file"""
        self.model = keras.models.load_model(path)


class ResNetClassifier:
    """Alternative ResNet-based classifier"""
    
    def __init__(self, num_classes: int = 8, input_shape: Tuple[int, int, int] = (224, 224, 3)):
        """Initialize ResNet classifier"""
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = self._build_model()
    
    def _build_model(self) -> keras.Model:
        """Build ResNet50-based model"""
        from tensorflow.keras.applications import ResNet50
        
        base_model = ResNet50(weights='imagenet', include_top=False,
                             input_shape=self.input_shape)
        
        # Freeze early layers
        for layer in base_model.layers[:-10]:
            layer.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def predict(self, image: np.ndarray) -> Tuple[int, np.ndarray]:
        """Predict protein localization"""
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        probabilities = self.model.predict(image, verbose=0)[0]
        predicted_class = np.argmax(probabilities)
        
        return predicted_class, probabilities


class EfficientNetClassifier:
    """EfficientNet-based classifier for improved performance"""
    
    def __init__(self, num_classes: int = 8, input_shape: Tuple[int, int, int] = (224, 224, 3)):
        """Initialize EfficientNet classifier"""
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = self._build_model()
    
    def _build_model(self) -> keras.Model:
        """Build EfficientNetB0 model"""
        from tensorflow.keras.applications import EfficientNetB0
        
        base_model = EfficientNetB0(weights='imagenet', include_top=False,
                                    input_shape=self.input_shape)
        
        # Freeze base layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def predict(self, image: np.ndarray) -> Tuple[int, np.ndarray]:
        """Predict protein localization"""
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        probabilities = self.model.predict(image, verbose=0)[0]
        predicted_class = np.argmax(probabilities)
        
        return predicted_class, probabilities
