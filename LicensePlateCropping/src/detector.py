"""
License Plate Detection Module
Main module for detecting and cropping license plates from vehicle images
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import yaml
import torch
from typing import Tuple, List, Optional, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LicensePlateDetector:
    """Main class for license plate detection and cropping"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the detector with configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.device = self._setup_device()
        
        # Create necessary directories
        self._setup_directories()
        
        # Load model if exists
        self.load_model()
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_device(self) -> str:
        """Setup computing device (CPU/GPU)"""
        if self.config['training']['device'] == 'cuda' and torch.cuda.is_available():
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            return 'cuda'
        else:
            logger.info("Using CPU")
            return 'cpu'
    
    def _setup_directories(self):
        """Create necessary directories if they don't exist"""
        for path_key, path_value in self.config['paths'].items():
            Path(path_value).mkdir(parents=True, exist_ok=True)
    
    def load_model(self, model_path: Optional[str] = None):
        """
        Load a pre-trained model or initialize a new one
        
        Args:
            model_path: Optional path to specific model weights
        """
        if model_path is None:
            model_path = self.config['paths']['best_model']
        
        if Path(model_path).exists():
            logger.info(f"Loading existing model from {model_path}")
            self.model = YOLO(model_path)
        else:
            logger.info(f"Initializing new {self.config['model']['type']} model")
            self.model = YOLO(f"{self.config['model']['type']}.pt")
    
    def detect_plate(self, image_path: str) -> List[Dict]:
        """
        Detect license plates in an image
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of detection dictionaries with bounding boxes and confidence
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Run detection
        results = self.model(
            image,
            conf=self.config['model']['confidence_threshold'],
            iou=self.config['model']['iou_threshold'],
            device=self.device
        )
        
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    detection = {
                        'bbox': box.xyxy[0].cpu().numpy(),  # [x1, y1, x2, y2]
                        'confidence': box.conf[0].cpu().numpy(),
                        'class': int(box.cls[0].cpu().numpy()) if box.cls is not None else 0
                    }
                    detections.append(detection)
        
        return detections
    
    def crop_plate(self, image_path: str, bbox: np.ndarray, 
                   padding: int = 10) -> np.ndarray:
        """
        Crop license plate from image using bounding box
        
        Args:
            image_path: Path to input image
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            padding: Additional padding around the plate
            
        Returns:
            Cropped license plate image
        """
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        
        # Add padding and ensure coordinates are within image bounds
        x1 = max(0, int(bbox[0]) - padding)
        y1 = max(0, int(bbox[1]) - padding)
        x2 = min(w, int(bbox[2]) + padding)
        y2 = min(h, int(bbox[3]) + padding)
        
        cropped = image[y1:y2, x1:x2]
        
        return cropped
    
    def process_image(self, image_path: str, save_crops: bool = True) -> Dict:
        """
        Complete pipeline: detect and crop license plates from an image
        
        Args:
            image_path: Path to input image
            save_crops: Whether to save cropped images
            
        Returns:
            Dictionary with detection results and cropped images
        """
        # Detect plates
        detections = self.detect_plate(image_path)
        
        if not detections:
            logger.warning(f"No license plates detected in {image_path}")
            return {'image_path': image_path, 'detections': [], 'crops': []}
        
        # Sort by confidence and take the best detection
        # (assuming one plate per vehicle for now)
        best_detection = max(detections, key=lambda x: x['confidence'])
        
        # Crop the plate
        cropped_plate = self.crop_plate(image_path, best_detection['bbox'])
        
        # Save cropped plate if requested
        if save_crops:
            filename = Path(image_path).stem
            save_path = Path(self.config['paths']['cropped_plates']) / f"{filename}_plate.jpg"
            cv2.imwrite(str(save_path), cropped_plate)
            logger.info(f"Saved cropped plate to {save_path}")
        
        return {
            'image_path': image_path,
            'detection': best_detection,
            'cropped_plate': cropped_plate
        }
    
    def process_batch(self, image_folder: str, save_crops: bool = True) -> List[Dict]:
        """
        Process multiple images in a folder
        
        Args:
            image_folder: Path to folder containing images
            save_crops: Whether to save cropped images
            
        Returns:
            List of processing results
        """
        image_folder = Path(image_folder)
        image_files = list(image_folder.glob("*.jpg")) + \
                     list(image_folder.glob("*.png")) + \
                     list(image_folder.glob("*.jpeg"))
        
        results = []
        for image_path in image_files:
            try:
                result = self.process_image(str(image_path), save_crops)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                continue
        
        return results
    
    def visualize_detection(self, image_path: str, save_path: Optional[str] = None):
        """
        Visualize detection results on the image
        
        Args:
            image_path: Path to input image
            save_path: Optional path to save visualization
        """
        image = cv2.imread(image_path)
        detections = self.detect_plate(image_path)
        
        for det in detections:
            bbox = det['bbox']
            conf = det['confidence']
            
            # Draw bounding box
            cv2.rectangle(
                image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 255, 0),
                2
            )
            
            # Add confidence score
            label = f"Plate: {conf:.2f}"
            cv2.putText(
                image,
                label,
                (int(bbox[0]), int(bbox[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        if save_path:
            cv2.imwrite(save_path, image)
            logger.info(f"Saved visualization to {save_path}")
        
        return image
    
    def save_model(self, save_path: Optional[str] = None):
        """Save the current model weights"""
        if self.model is None:
            raise ValueError("No model to save")
        
        if save_path is None:
            save_path = self.config['paths']['best_model']
        
        # For YOLO models, we save the entire model
        self.model.save(save_path)
        logger.info(f"Model saved to {save_path}")


# Convenience functions for quick usage
def quick_detect(image_path: str, config_path: str = "config.yaml") -> Dict:
    """
    Quick function to detect and crop license plate from a single image
    
    Args:
        image_path: Path to input image
        config_path: Path to configuration file
        
    Returns:
        Detection results dictionary
    """
    detector = LicensePlateDetector(config_path)
    return detector.process_image(image_path)


if __name__ == "__main__":
    # Example usage
    detector = LicensePlateDetector()
    
    # Process a single image
    # result = detector.process_image("path/to/your/image.jpg")
    
    # Process multiple images
    # results = detector.process_batch("path/to/image/folder")
    
    logger.info("License Plate Detector initialized successfully")