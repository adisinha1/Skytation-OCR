"""
Image Preprocessing Module
Handles image preprocessing and augmentation for better detection
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import albumentations as A
from pathlib import Path


class ImagePreprocessor:
    """Handle image preprocessing for license plate detection"""
    
    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        """
        Initialize preprocessor
        
        Args:
            target_size: Target size for images (width, height)
        """
        self.target_size = target_size
        self.augmentation_pipeline = self._create_augmentation_pipeline()
    
    def _create_augmentation_pipeline(self):
        """Create augmentation pipeline for training"""
        return A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.MotionBlur(blur_limit=5, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Rotate(limit=15, p=0.5),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
        ])
    
    def resize_image(self, image: np.ndarray, 
                    maintain_aspect: bool = True) -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image
            maintain_aspect: Whether to maintain aspect ratio
            
        Returns:
            Resized image
        """
        if maintain_aspect:
            return self._resize_with_padding(image)
        else:
            return cv2.resize(image, self.target_size)
    
    def _resize_with_padding(self, image: np.ndarray) -> np.ndarray:
        """Resize image maintaining aspect ratio with padding"""
        h, w = image.shape[:2]
        target_w, target_h = self.target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Calculate padding
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        
        # Place resized image in center
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        return padded
    
    def enhance_plate_region(self, plate_image: np.ndarray) -> np.ndarray:
        """
        Enhance cropped license plate for better OCR
        
        Args:
            plate_image: Cropped license plate image
            
        Returns:
            Enhanced image
        """
        # Convert to grayscale
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.bilateralFilter(enhanced, 11, 17, 17)
        
        # Apply sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return sharpened
    
    def prepare_for_ocr(self, plate_image: np.ndarray) -> np.ndarray:
        """
        Prepare cropped plate image for OCR processing
        
        Args:
            plate_image: Cropped license plate image
            
        Returns:
            Processed image ready for OCR
        """
        # Enhance the image
        enhanced = self.enhance_plate_region(plate_image)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def augment_image(self, image: np.ndarray, 
                     bboxes: Optional[List] = None) -> Tuple[np.ndarray, List]:
        """
        Apply augmentation to image and bounding boxes
        
        Args:
            image: Input image
            bboxes: List of bounding boxes in format [x1, y1, x2, y2, class_id]
            
        Returns:
            Augmented image and bounding boxes
        """
        if bboxes is None:
            augmented = self.augmentation_pipeline(image=image)
            return augmented['image'], []
        
        # Convert bboxes to albumentations format
        bbox_labels = []
        class_labels = []
        
        for bbox in bboxes:
            # Normalize coordinates
            h, w = image.shape[:2]
            x1, y1, x2, y2 = bbox[:4]
            bbox_labels.append([x1/w, y1/h, x2/w, y2/h])
            if len(bbox) > 4:
                class_labels.append(bbox[4])
            else:
                class_labels.append(0)
        
        # Apply augmentation
        augmented = self.augmentation_pipeline(
            image=image,
            bboxes=bbox_labels,
            class_labels=class_labels
        )
        
        # Convert back to original format
        aug_bboxes = []
        h, w = image.shape[:2]
        for bbox, class_id in zip(augmented['bboxes'], augmented['class_labels']):
            x1, y1, x2, y2 = bbox
            aug_bboxes.append([x1*w, y1*h, x2*w, y2*h, class_id])
        
        return augmented['image'], aug_bboxes
    
    def batch_preprocess(self, image_paths: List[str], 
                        augment: bool = False) -> List[np.ndarray]:
        """
        Preprocess multiple images
        
        Args:
            image_paths: List of image paths
            augment: Whether to apply augmentation
            
        Returns:
            List of preprocessed images
        """
        processed_images = []
        
        for image_path in image_paths:
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            # Resize image
            resized = self.resize_image(image)
            
            # Apply augmentation if requested
            if augment:
                augmented, _ = self.augment_image(resized)
                processed_images.append(augmented)
            else:
                processed_images.append(resized)
        
        return processed_images
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        return image.astype(np.float32) / 255.0
    
    def denormalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Denormalize image back to [0, 255] range
        
        Args:
            image: Normalized image
            
        Returns:
            Denormalized image
        """
        return (image * 255).astype(np.uint8)


class PlateOrientationCorrector:
    """Correct orientation of detected license plates"""
    
    @staticmethod
    def correct_skew(plate_image: np.ndarray, max_angle: float = 45) -> np.ndarray:
        """
        Correct skew in license plate image
        
        Args:
            plate_image: Cropped license plate image
            max_angle: Maximum angle to correct
            
        Returns:
            Deskewed image
        """
        # Convert to grayscale if needed
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image.copy()
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return plate_image
        
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        
        # Adjust angle
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        
        # Limit correction angle
        if abs(angle) > max_angle:
            return plate_image
        
        # Get rotation matrix
        h, w = plate_image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate image
        rotated = cv2.warpAffine(
            plate_image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    @staticmethod
    def auto_crop_plate(plate_image: np.ndarray) -> np.ndarray:
        """
        Auto crop to remove excess borders from plate image
        
        Args:
            plate_image: License plate image
            
        Returns:
            Tightly cropped plate image
        """
        # Convert to grayscale if needed
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image.copy()
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return plate_image
        
        # Get bounding rectangle of all contours
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
        
        # Add small padding
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(plate_image.shape[1] - x, w + 2 * padding)
        h = min(plate_image.shape[0] - y, h + 2 * padding)
        
        # Crop image
        cropped = plate_image[y:y+h, x:x+w]
        
        return cropped


if __name__ == "__main__":
    # Example usage
    preprocessor = ImagePreprocessor()
    
    # Test with a sample image
    # image = cv2.imread("path/to/image.jpg")
    # processed = preprocessor.resize_image(image)
    # enhanced = preprocessor.enhance_plate_region(processed)
    
    print("Preprocessor module initialized successfully")