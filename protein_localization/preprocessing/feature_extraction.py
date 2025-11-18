"""
Feature extraction module for protein sub-cellular localization
Extracts spatial, morphological, intensity, and region-level features
"""
import numpy as np
import pandas as pd
from skimage import measure
from scipy.spatial.distance import cdist, pdist, squareform
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class FeatureExtractor:
    """Extract various features from segmented images"""
    
    def __init__(self):
        self.features = {}
    
    def extract_all_features(self, image: np.ndarray, masks: np.ndarray, 
                            channel_index: int = 0) -> pd.DataFrame:
        """
        Extract all features from an image and its masks
        
        Args:
            image: Original image
            masks: Segmentation masks
            channel_index: Which channel to extract intensity from
        
        Returns:
            DataFrame with all features
        """
        regions = measure.regionprops(masks, intensity_image=self._get_intensity_image(image, channel_index))
        
        if len(regions) == 0:
            return pd.DataFrame()
        
        features_list = []
        
        for region in regions:
            feature_dict = {}
            
            # Spatial features
            spatial = self.extract_spatial_features(region, masks)
            feature_dict.update(spatial)
            
            # Morphological features
            morphological = self.extract_morphological_features(region)
            feature_dict.update(morphological)
            
            # Intensity features
            intensity = self.extract_intensity_features(region, image, channel_index)
            feature_dict.update(intensity)
            
            # Region-level descriptors
            region_desc = self.extract_region_descriptors(region, masks)
            feature_dict.update(region_desc)
            
            feature_dict['label'] = region.label
            features_list.append(feature_dict)
        
        df = pd.DataFrame(features_list)
        
        # Add pairwise distances
        if len(regions) > 1:
            df = self._add_pairwise_distances(df)
        
        return df
    
    def extract_spatial_features(self, region, masks: np.ndarray) -> Dict:
        """Extract spatial features"""
        centroid = region.centroid
        bbox = region.bbox
        
        features = {
            'centroid_y': centroid[0] if len(centroid) > 0 else 0,
            'centroid_x': centroid[1] if len(centroid) > 1 else 0,
            'centroid_z': centroid[2] if len(centroid) > 2 else 0,
            'bbox_min_y': bbox[0],
            'bbox_min_x': bbox[1],
            'bbox_max_y': bbox[2] if len(bbox) > 2 else 0,
            'bbox_max_x': bbox[3] if len(bbox) > 3 else 0,
            'position_normalized_y': centroid[0] / masks.shape[0] if len(centroid) > 0 else 0,
            'position_normalized_x': centroid[1] / masks.shape[1] if len(centroid) > 1 else 0,
        }
        
        return features
    
    def extract_morphological_features(self, region) -> Dict:
        """Extract morphological features"""
        features = {
            'area': region.area,
            'perimeter': region.perimeter if hasattr(region, 'perimeter') else 0,
            'eccentricity': region.eccentricity if hasattr(region, 'eccentricity') else 0,
            'solidity': region.solidity if hasattr(region, 'solidity') else 0,
            'extent': region.extent if hasattr(region, 'extent') else 0,
            'major_axis_length': region.major_axis_length if hasattr(region, 'major_axis_length') else 0,
            'minor_axis_length': region.minor_axis_length if hasattr(region, 'minor_axis_length') else 0,
            'orientation': region.orientation if hasattr(region, 'orientation') else 0,
            'filled_area': region.filled_area if hasattr(region, 'filled_area') else region.area,
            'convex_area': region.convex_area if hasattr(region, 'convex_area') else region.area,
        }
        
        # Derived features
        if features['perimeter'] > 0:
            features['compactness'] = (4 * np.pi * features['area']) / (features['perimeter'] ** 2)
        else:
            features['compactness'] = 0
        
        if features['major_axis_length'] > 0:
            features['aspect_ratio'] = features['minor_axis_length'] / features['major_axis_length']
        else:
            features['aspect_ratio'] = 0
        
        return features
    
    def extract_intensity_features(self, region, image: np.ndarray, channel_index: int = 0) -> Dict:
        """Extract intensity-based features"""
        if not hasattr(region, 'intensity_image') or region.intensity_image is None:
            return {
                'mean_intensity': 0,
                'max_intensity': 0,
                'min_intensity': 0,
                'std_intensity': 0,
                'median_intensity': 0,
                'intensity_range': 0,
            }
        
        intensities = region.intensity_image[region.image]
        
        features = {
            'mean_intensity': np.mean(intensities),
            'max_intensity': np.max(intensities),
            'min_intensity': np.min(intensities),
            'std_intensity': np.std(intensities),
            'median_intensity': np.median(intensities),
            'intensity_range': np.max(intensities) - np.min(intensities),
            'intensity_q25': np.percentile(intensities, 25),
            'intensity_q75': np.percentile(intensities, 75),
            'intensity_iqr': np.percentile(intensities, 75) - np.percentile(intensities, 25),
        }
        
        # Histogram features
        hist, _ = np.histogram(intensities, bins=10, density=True)
        features['intensity_entropy'] = -np.sum(hist * np.log2(hist + 1e-10))
        features['intensity_skewness'] = self._calculate_skewness(intensities)
        features['intensity_kurtosis'] = self._calculate_kurtosis(intensities)
        
        return features
    
    def extract_region_descriptors(self, region, masks: np.ndarray) -> Dict:
        """Extract region-level descriptors"""
        features = {
            'euler_number': region.euler_number if hasattr(region, 'euler_number') else 0,
            'equivalent_diameter': region.equivalent_diameter if hasattr(region, 'equivalent_diameter') else 0,
        }
        
        # Neighborhood relationships
        # Count neighboring regions
        dilated_mask = self._dilate_mask(region, masks.shape)
        neighbors = np.unique(masks[dilated_mask & (masks != region.label) & (masks != 0)])
        features['num_neighbors'] = len(neighbors)
        
        return features
    
    def extract_channel_features(self, image: np.ndarray, masks: np.ndarray) -> pd.DataFrame:
        """
        Extract features for all channels in multi-channel image
        
        Args:
            image: Multi-channel image
            masks: Segmentation masks
        
        Returns:
            DataFrame with features from all channels
        """
        if len(image.shape) < 3:
            return self.extract_all_features(image, masks, 0)
        
        # Determine number of channels
        if image.shape[-1] < 10:  # Assume last dimension is channels
            num_channels = image.shape[-1]
            all_features = []
            
            for ch in range(num_channels):
                ch_features = self.extract_all_features(image, masks, ch)
                # Add channel prefix to column names
                ch_features = ch_features.add_prefix(f'ch{ch}_')
                all_features.append(ch_features)
            
            # Concatenate all channel features
            result = pd.concat(all_features, axis=1)
            return result
        else:
            return self.extract_all_features(image, masks, 0)
    
    @staticmethod
    def _get_intensity_image(image: np.ndarray, channel_index: int = 0) -> np.ndarray:
        """Get the intensity image for a specific channel"""
        if len(image.shape) == 2:
            return image
        elif len(image.shape) == 3:
            if image.shape[-1] < 10:  # Multi-channel
                return image[:, :, channel_index]
            else:  # Z-stack
                return image[image.shape[0] // 2, :, :]
        elif len(image.shape) == 4:  # 4D TIFF
            return image[0, 0, :, :]
        else:
            return image
    
    @staticmethod
    def _calculate_skewness(data: np.ndarray) -> float:
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    @staticmethod
    def _calculate_kurtosis(data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    @staticmethod
    def _dilate_mask(region, shape: Tuple) -> np.ndarray:
        """Create a dilated version of a region's mask"""
        from scipy.ndimage import binary_dilation
        mask = np.zeros(shape, dtype=bool)
        coords = region.coords
        for coord in coords:
            mask[tuple(coord)] = True
        dilated = binary_dilation(mask, iterations=2)
        return dilated
    
    def _add_pairwise_distances(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pairwise distance features"""
        if 'centroid_y' not in df.columns or 'centroid_x' not in df.columns:
            return df
        
        centroids = df[['centroid_y', 'centroid_x']].values
        
        # Calculate pairwise distances
        distances = squareform(pdist(centroids, metric='euclidean'))
        
        # For each region, calculate distance statistics to other regions
        df['min_distance_to_neighbor'] = np.min(distances + np.eye(len(distances)) * 1e10, axis=1)
        df['mean_distance_to_neighbors'] = np.sum(distances, axis=1) / (len(distances) - 1)
        df['max_distance_to_neighbor'] = np.max(distances, axis=1)
        
        return df


class FeatureStorage:
    """Store and manage extracted features"""
    
    def __init__(self, output_dir: str = './features'):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def save_features(self, features: pd.DataFrame, filename: str):
        """Save features to file"""
        import os
        filepath = os.path.join(self.output_dir, filename)
        
        # Save as CSV
        features.to_csv(filepath + '.csv', index=False)
        
        # Save as HDF5 for faster loading
        try:
            features.to_hdf(filepath + '.h5', key='features', mode='w')
        except:
            pass
        
        # Save as pickle for full Python object preservation
        features.to_pickle(filepath + '.pkl')
        
        print(f"Features saved to {filepath}")
    
    def load_features(self, filename: str) -> pd.DataFrame:
        """Load features from file"""
        import os
        filepath = os.path.join(self.output_dir, filename)
        
        # Try loading from HDF5 first (fastest)
        try:
            return pd.read_hdf(filepath + '.h5', key='features')
        except:
            pass
        
        # Try pickle
        try:
            return pd.read_pickle(filepath + '.pkl')
        except:
            pass
        
        # Fall back to CSV
        return pd.read_csv(filepath + '.csv')


if __name__ == "__main__":
    # Test feature extraction
    print("Testing feature extraction module...")
    
    # Create dummy data
    masks = np.zeros((100, 100), dtype=int)
    masks[20:40, 20:40] = 1
    masks[60:80, 60:80] = 2
    
    image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    extractor = FeatureExtractor()
    features = extractor.extract_all_features(image, masks)
    
    print(f"Extracted {len(features)} regions with {len(features.columns)} features each")
    print(features.head())
