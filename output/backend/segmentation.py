"""
Segmentation module supporting U-Net, SLIC, Watershed, and Cellpose methods
"""
import numpy as np
import cv2
from skimage.segmentation import slic, watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

try:
    from cellpose import models
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False
    print("Warning: Cellpose not available. Install with: pip install cellpose")


class SegmentationModule:
    """Handles different segmentation methods including Cellpose"""
    
    def __init__(self, method: str = "CELLPOSE"):
        """
        Initialize segmentation module
        
        Args:
            method: Segmentation method ('CELLPOSE', 'UNET', 'SLIC', or 'WATERSHED')
        """
        self.method = method.upper()
        self.unet_model = None
        self.cellpose_model = None
        
        if self.method == "UNET":
            self.unet_model = self._build_unet()
        elif self.method == "CELLPOSE":
            if CELLPOSE_AVAILABLE:
                # Initialize Cellpose with cyto2 model (general purpose for cells and nuclei)
                self.cellpose_model = models.Cellpose(gpu=False, model_type='cyto2')
            else:
                print("Warning: Cellpose not available. Falling back to SLIC segmentation.")
                self.method = "SLIC"
    
    def segment(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform segmentation on image
        
        Args:
            image: Input image
            **kwargs: Method-specific parameters
            
        Returns:
            Segmented image mask
        """
        if self.method == "CELLPOSE":
            return self._segment_cellpose(image, **kwargs)
        elif self.method == "UNET":
            return self._segment_unet(image)
        elif self.method == "SLIC":
            return self._segment_slic(image, **kwargs)
        elif self.method == "WATERSHED":
            return self._segment_watershed(image)
        else:
            raise ValueError(f"Unknown segmentation method: {self.method}")
    
    def _segment_cellpose(self, image: np.ndarray, diameter: Optional[float] = 30.0, 
                         flow_threshold: float = 0.4, cellprob_threshold: float = 0.0) -> np.ndarray:
        """
        Segment using Cellpose neural network
        
        Args:
            image: Input image (2D grayscale or 3D RGB)
            diameter: Estimated cell/nucleus diameter in pixels (None for auto-detect)
            flow_threshold: Flow threshold (higher = more selective, default 0.4)
            cellprob_threshold: Cell probability threshold (default 0.0)
            
        Returns:
            Segmentation mask with labeled regions
        """
        if self.cellpose_model is None:
            raise ValueError("Cellpose model not initialized")
        
        # Ensure image is in correct format
        if len(image.shape) == 2:
            # Grayscale image
            img_for_cellpose = image
            channels = [[0, 0]]  # Grayscale
        elif len(image.shape) == 3:
            if image.shape[2] == 1:
                # Single channel
                img_for_cellpose = image[:, :, 0]
                channels = [[0, 0]]  # Grayscale
            elif image.shape[2] >= 3:
                # RGB or more channels
                img_for_cellpose = image
                channels = [[2, 1]]  # RGB (cytoplasm in green channel, nucleus in red)
            else:
                # Unexpected format, convert to grayscale
                img_for_cellpose = np.mean(image, axis=-1)
                channels = [[0, 0]]
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        # Normalize image to [0, 1] if needed
        if img_for_cellpose.max() > 1.0:
            img_for_cellpose = img_for_cellpose / 255.0
        
        # Run Cellpose segmentation
        try:
            masks, flows, styles, diams = self.cellpose_model.eval(
                img_for_cellpose,
                diameter=diameter,
                channels=channels,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold
            )
            return masks
        except Exception as e:
            print(f"Cellpose segmentation error: {e}")
            print("Falling back to SLIC segmentation")
            return self._segment_slic(image)
    
    def _build_unet(self, input_shape: Tuple[int, int, int] = (224, 224, 3)) -> keras.Model:
        """
        Build U-Net architecture for segmentation
        
        Args:
            input_shape: Input image shape
            
        Returns:
            U-Net model
        """
        inputs = keras.Input(shape=input_shape)
        
        # Encoder
        c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)
        
        c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)
        
        c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)
        
        c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
        c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
        p4 = layers.MaxPooling2D((2, 2))(c4)
        
        # Bottleneck
        c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
        c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
        
        # Decoder
        u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = layers.concatenate([u6, c4])
        c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
        c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
        
        u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, c3])
        c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
        c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
        
        u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = layers.concatenate([u8, c2])
        c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
        c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
        
        u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = layers.concatenate([u9, c1])
        c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
        c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
        
        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
        
        model = keras.Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
    
    def _segment_unet(self, image: np.ndarray) -> np.ndarray:
        """Segment using U-Net"""
        if self.unet_model is None:
            self.unet_model = self._build_unet()
        
        # Ensure correct shape
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        if len(image.shape) == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        
        # Add batch dimension
        if len(image.shape) == 3:
            image_batch = np.expand_dims(image, axis=0)
        else:
            image_batch = image
        
        # Predict
        mask = self.unet_model.predict(image_batch, verbose=0)
        mask = mask[0, :, :, 0]
        
        # Threshold
        mask = (mask > 0.5).astype(np.uint8)
        
        return mask
    
    def _segment_slic(self, image: np.ndarray, n_segments: int = 100, 
                     compactness: float = 10) -> np.ndarray:
        """
        Segment using SLIC superpixels
        
        Args:
            image: Input image
            n_segments: Number of superpixels
            compactness: Compactness parameter
            
        Returns:
            Superpixel segmentation mask
        """
        # Ensure image is in correct format
        if len(image.shape) == 2:
            image_rgb = np.stack([image] * 3, axis=-1)
        else:
            image_rgb = image
        
        # Apply SLIC
        segments = slic(image_rgb, n_segments=n_segments, compactness=compactness,
                       start_label=1, channel_axis=-1)
        
        return segments
    
    def _segment_watershed(self, image: np.ndarray) -> np.ndarray:
        """
        Segment using Watershed algorithm
        
        Args:
            image: Input image
            
        Returns:
            Watershed segmentation mask
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        
        # Mark the region of unknown with zero
        markers[unknown == 255] = 0
        
        # Apply watershed
        if len(image.shape) == 2:
            image_rgb = np.stack([image] * 3, axis=-1)
        else:
            image_rgb = image
        
        image_rgb = (image_rgb * 255).astype(np.uint8)
        markers = cv2.watershed(image_rgb, markers)
        
        return markers


def save_segmentation(image: np.ndarray, mask: np.ndarray, output_path: str):
    """
    Save segmentation result with overlay
    
    Args:
        image: Original image
        mask: Segmentation mask
        output_path: Path to save result
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    if len(image.shape) == 2:
        axes[0].imshow(image, cmap='gray')
    else:
        axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Segmentation mask
    axes[1].imshow(mask, cmap='nipy_spectral')
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    
    # Overlay
    if len(image.shape) == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    else:
        image_rgb = image
    
    axes[2].imshow(image_rgb)
    axes[2].imshow(mask, alpha=0.5, cmap='nipy_spectral')
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
