"""
Segmentation module with U-Net and SLIC implementations
"""
import numpy as np
import cv2
from skimage.segmentation import slic, mark_boundaries
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """U-Net architecture for image segmentation"""
    
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, out_channels, 1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return torch.sigmoid(self.out(dec1))


class SegmentationEngine:
    """Main segmentation engine with multiple methods"""
    
    def __init__(self, method='slic'):
        """
        Initialize segmentation engine
        
        Args:
            method: 'unet', 'slic', or 'watershed'
        """
        self.method = method
        if method == 'unet':
            self.unet_model = UNet()
            self.unet_model.eval()
    
    def segment_slic(self, image, n_segments=100, compactness=10, sigma=1):
        """
        SLIC superpixel segmentation
        
        Args:
            image: Input image
            n_segments: Number of superpixels
            compactness: Balances color proximity and space proximity
            sigma: Width of Gaussian smoothing
            
        Returns:
            Segmentation mask
        """
        segments = slic(image, n_segments=n_segments, 
                       compactness=compactness, sigma=sigma,
                       start_label=1)
        return segments
    
    def segment_watershed(self, image):
        """
        Watershed segmentation
        
        Args:
            image: Input image
            
        Returns:
            Segmentation mask
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        # Threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Distance transform
        distance = ndimage.distance_transform_edt(thresh)
        
        # Find peaks
        local_maxi = peak_local_max(distance, indices=False, min_distance=20,
                                    labels=thresh)
        
        # Marker labeling
        markers = ndimage.label(local_maxi)[0]
        
        # Watershed
        labels = watershed(-distance, markers, mask=thresh)
        
        return labels
    
    def segment_unet(self, image):
        """
        U-Net based segmentation
        
        Args:
            image: Input image
            
        Returns:
            Segmentation mask
        """
        # Convert to tensor
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
        
        with torch.no_grad():
            mask = self.unet_model(img_tensor)
        
        mask = mask.squeeze().numpy()
        return (mask > 0.5).astype(np.uint8)
    
    def segment(self, image, **kwargs):
        """
        Perform segmentation using selected method
        
        Args:
            image: Input image
            **kwargs: Method-specific parameters
            
        Returns:
            Segmentation mask
        """
        if self.method == 'slic':
            return self.segment_slic(image, **kwargs)
        elif self.method == 'watershed':
            return self.segment_watershed(image)
        elif self.method == 'unet':
            return self.segment_unet(image)
        else:
            raise ValueError(f"Unknown segmentation method: {self.method}")
    
    def visualize_segmentation(self, image, segments):
        """
        Create visualization of segmentation
        
        Args:
            image: Original image
            segments: Segmentation mask
            
        Returns:
            Visualization image
        """
        return mark_boundaries(image, segments, color=(1, 0, 0), mode='thick')
    
    def save_segmentation(self, segments, filepath):
        """
        Save segmentation mask
        
        Args:
            segments: Segmentation mask
            filepath: Output filepath
        """
        # Normalize and save
        seg_normalized = ((segments - segments.min()) / 
                         (segments.max() - segments.min()) * 255).astype(np.uint8)
        cv2.imwrite(filepath, seg_normalized)
