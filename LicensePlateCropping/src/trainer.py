"""
Model Training Module
Handles training of the license plate detection model
"""

import os
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LicensePlateTrainer:
    """Handle training of license plate detection models"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize trainer with configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.model = None
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def prepare_dataset(self, images_path: str, labels_path: str,
                       split_ratio: Tuple[float, float, float] = (0.7, 0.2, 0.1)):
        """
        Prepare dataset for YOLO training
        
        Args:
            images_path: Path to images directory
            labels_path: Path to labels directory (YOLO format)
            split_ratio: Train/Val/Test split ratio
            
        Creates the following structure:
        data/
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── val/
        │   ├── images/
        │   └── labels/
        └── test/
            ├── images/
            └── labels/
        """
        # Create directories
        base_dir = Path("data")
        for split in ['train', 'val', 'test']:
            (base_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (base_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        images_dir = Path(images_path)
        image_files = list(images_dir.glob("*.jpg")) + \
                     list(images_dir.glob("*.png")) + \
                     list(images_dir.glob("*.jpeg"))
        
        # Shuffle and split
        np.random.shuffle(image_files)
        
        n_total = len(image_files)
        n_train = int(n_total * split_ratio[0])
        n_val = int(n_total * split_ratio[1])
        
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        # Copy files to respective directories
        for split, files in zip(['train', 'val', 'test'], 
                                [train_files, val_files, test_files]):
            logger.info(f"Preparing {split} set with {len(files)} images")
            
            for img_path in tqdm(files, desc=f"Copying {split} files"):
                # Copy image
                img_dest = base_dir / split / 'images' / img_path.name
                shutil.copy2(img_path, img_dest)
                
                # Copy corresponding label if exists
                label_path = Path(labels_path) / f"{img_path.stem}.txt"
                if label_path.exists():
                    label_dest = base_dir / split / 'labels' / label_path.name
                    shutil.copy2(label_path, label_dest)
        
        # Create dataset.yaml for YOLO
        self._create_dataset_yaml(base_dir)
        
        logger.info("Dataset preparation complete!")
        return base_dir
    
    def _create_dataset_yaml(self, base_dir: Path):
        """Create dataset configuration file for YOLO"""
        dataset_config = {
            'path': str(base_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': {
                0: 'license_plate'
            },
            'nc': 1  # Number of classes
        }
        
        yaml_path = base_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f)
        
        logger.info(f"Created dataset configuration at {yaml_path}")
    
    def train_model(self, dataset_yaml: str = "data/dataset.yaml",
                   resume: bool = False) -> YOLO:
        """
        Train the license plate detection model
        
        Args:
            dataset_yaml: Path to dataset configuration
            resume: Whether to resume from previous training
            
        Returns:
            Trained YOLO model
        """
        # Initialize or load model
        if resume and Path(self.config['paths']['best_model']).exists():
            logger.info("Resuming from previous training")
            self.model = YOLO(self.config['paths']['best_model'])
        else:
            model_name = f"{self.config['model']['type']}.pt"
            logger.info(f"Starting new training with {model_name}")
            self.model = YOLO(model_name)
        
        # Train the model
        results = self.model.train(
            data=dataset_yaml,
            epochs=self.config['training']['epochs'],
            batch=self.config['training']['batch_size'],
            imgsz=self.config['model']['input_size'],
            device=self.config['training']['device'],
            lr0=self.config['training']['learning_rate'],
            project='models/runs',
            name='license_plate_detection',
            exist_ok=True,
            save=True,
            save_period=10,  # Save checkpoint every 10 epochs
            patience=20,  # Early stopping patience
            verbose=True,
            plots=True
        )
        
        # Save the best model
        best_model_path = Path('models/runs/license_plate_detection/weights/best.pt')
        if best_model_path.exists():
            shutil.copy2(best_model_path, self.config['paths']['best_model'])
            logger.info(f"Best model saved to {self.config['paths']['best_model']}")
        
        return self.model
    
    def evaluate_model(self, model_path: Optional[str] = None,
                      dataset_yaml: str = "data/dataset.yaml") -> Dict:
        """
        Evaluate the trained model
        
        Args:
            model_path: Path to model weights
            dataset_yaml: Path to dataset configuration
            
        Returns:
            Dictionary with evaluation metrics
        """
        if model_path is None:
            model_path = self.config['paths']['best_model']
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load model
        model = YOLO(model_path)
        
        # Run validation
        metrics = model.val(
            data=dataset_yaml,
            split='test',
            batch=self.config['training']['batch_size'],
            device=self.config['training']['device']
        )
        
        # Extract key metrics
        results = {
            'precision': float(metrics.box.mp),  # Mean precision
            'recall': float(metrics.box.mr),     # Mean recall
            'mAP50': float(metrics.box.map50),   # mAP at IoU=0.5
            'mAP50-95': float(metrics.box.map),  # mAP at IoU=0.5:0.95
        }
        
        logger.info("Evaluation Results:")
        for metric, value in results.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return results
    
    def export_model(self, format: str = 'onnx') -> str:
        """
        Export model to different formats
        
        Args:
            format: Export format ('onnx', 'torchscript', 'tensorflow', etc.)
            
        Returns:
            Path to exported model
        """
        if self.model is None:
            self.model = YOLO(self.config['paths']['best_model'])
        
        # Export the model
        export_path = self.model.export(format=format)
        logger.info(f"Model exported to {export_path}")
        
        return export_path


class DatasetLabeler:
    """Helper class for labeling license plate datasets"""
    
    @staticmethod
    def convert_to_yolo_format(bbox: List[float], 
                              img_width: int, 
                              img_height: int,
                              class_id: int = 0) -> str:
        """
        Convert bounding box to YOLO format
        
        Args:
            bbox: [x1, y1, x2, y2] in pixel coordinates
            img_width: Image width
            img_height: Image height
            class_id: Class ID (0 for license plate)
            
        Returns:
            YOLO format string: "class_id x_center y_center width height"
        """
        x1, y1, x2, y2 = bbox
        
        # Calculate center coordinates and dimensions
        x_center = (x1 + x2) / 2 / img_width
        y_center = (y1 + y2) / 2 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    
    @staticmethod
    def create_label_file(image_path: str, bboxes: List[List[float]], 
                         output_dir: str = "data/annotations"):
        """
        Create YOLO format label file for an image
        
        Args:
            image_path: Path to image
            bboxes: List of bounding boxes
            output_dir: Directory to save label files
        """
        # Read image to get dimensions
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not read image: {image_path}")
            return
        
        height, width = img.shape[:2]
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create label file
        label_path = Path(output_dir) / f"{Path(image_path).stem}.txt"
        
        with open(label_path, 'w') as f:
            for bbox in bboxes:
                yolo_line = DatasetLabeler.convert_to_yolo_format(
                    bbox, width, height
                )
                f.write(yolo_line + '\n')
        
        logger.info(f"Created label file: {label_path}")
    
    @staticmethod
    def visualize_labels(image_path: str, label_path: str) -> np.ndarray:
        """
        Visualize YOLO labels on image
        
        Args:
            image_path: Path to image
            label_path: Path to YOLO label file
            
        Returns:
            Image with bounding boxes drawn
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        height, width = img.shape[:2]
        
        # Read labels
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            # Parse YOLO format
            class_id = int(parts[0])
            x_center = float(parts[1]) * width
            y_center = float(parts[2]) * height
            box_width = float(parts[3]) * width
            box_height = float(parts[4]) * height
            
            # Convert to corner coordinates
            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"Class {class_id}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img


class TrainingMonitor:
    """Monitor and log training progress"""
    
    def __init__(self, log_dir: str = "models/runs"):
        """
        Initialize training monitor
        
        Args:
            log_dir: Directory for training logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def get_latest_metrics(self) -> Dict:
        """Get metrics from the latest training run"""
        # Find the latest run directory
        run_dirs = [d for d in self.log_dir.iterdir() if d.is_dir()]
        if not run_dirs:
            return {}
        
        latest_run = max(run_dirs, key=lambda d: d.stat().st_mtime)
        
        # Read results.csv if exists
        results_file = latest_run / 'results.csv'
        if not results_file.exists():
            return {}
        
        import pandas as pd
        df = pd.read_csv(results_file)
        
        if df.empty:
            return {}
        
        # Get last row (latest epoch)
        last_row = df.iloc[-1]
        
        metrics = {
            'epoch': int(last_row.get('epoch', 0)),
            'train_loss': float(last_row.get('train/box_loss', 0)),
            'val_loss': float(last_row.get('val/box_loss', 0)),
            'precision': float(last_row.get('metrics/precision(B)', 0)),
            'recall': float(last_row.get('metrics/recall(B)', 0)),
            'mAP50': float(last_row.get('metrics/mAP50(B)', 0)),
            'mAP50-95': float(last_row.get('metrics/mAP50-95(B)', 0))
        }
        
        return metrics
    
    def print_training_summary(self):
        """Print a summary of the latest training run"""
        metrics = self.get_latest_metrics()
        
        if not metrics:
            logger.info("No training metrics found")
            return
        
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Epoch: {metrics['epoch']}")
        print(f"Train Loss: {metrics['train_loss']:.4f}")
        print(f"Val Loss: {metrics['val_loss']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"mAP@0.5: {metrics['mAP50']:.4f}")
        print(f"mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
        print("="*50 + "\n")


if __name__ == "__main__":
    # Example usage
    trainer = LicensePlateTrainer()
    
    # Prepare dataset (if you have images and labels)
    # trainer.prepare_dataset("data/raw_images", "data/annotations")
    
    # Train model
    # model = trainer.train_model()
    
    # Evaluate model
    # metrics = trainer.evaluate_model()
    
    logger.info("Trainer module initialized successfully")