"""
Main script for License Plate Detection System
Run this script to detect and crop license plates from vehicle images
"""

import argparse
import sys
from pathlib import Path
import cv2
import logging

# Add src to path
sys.path.append('src')

from detector import LicensePlateDetector
from preprocessor import ImagePreprocessor, PlateOrientationCorrector
from trainer import LicensePlateTrainer, DatasetLabeler, TrainingMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_single_image(image_path: str, visualize: bool = False):
    """
    Detect and crop license plate from a single image
    """
    detector = LicensePlateDetector()
    result = detector.process_image(image_path, save_crops=True)

    if result['detection']:
        print(f"\n✓ License plate detected!")
        print(f"  Confidence: {result['detection']['confidence']:.2%}")
        print(f"  Saved to: data/cropped_plates/")

        if visualize:
            viz_image = detector.visualize_detection(image_path)
            cv2.imshow("Detection Result", cv2.resize(viz_image, (800, 600)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print(f"\n✗ No license plate detected in {image_path}")


def detect_batch(folder_path: str):
    """
    Process multiple images in a folder
    """
    detector = LicensePlateDetector()
    results = detector.process_batch(folder_path, save_crops=True)
    successful = sum(1 for r in results if r.get('detection'))
    print(f"\nBatch Processing Complete!")
    print(f"  Total images: {len(results)}")
    print(f"  Successful detections: {successful}")
    print(f"  Success rate: {successful / len(results) * 100:.1f}%")
    print(f"  Cropped plates saved to: data/cropped_plates/")


def train_new_model(images_path: str, labels_path: str):
    """
    Train a new license plate detection model
    """
    trainer = LicensePlateTrainer()
    print("Preparing dataset...")
    dataset_path = trainer.prepare_dataset(images_path, labels_path)
    print("Starting training...")
    model = trainer.train_model(f"{dataset_path}/dataset.yaml")
    print("Evaluating model...")
    metrics = trainer.evaluate_model()
    monitor = TrainingMonitor()
    monitor.print_training_summary()
    print(f"\n✓ Training complete! Model saved to: {trainer.config['paths']['best_model']}")


def label_image_interactive(image_path: str):
    """
    Interactive tool to label license plates in a single image
    """
    print(f"\nLabeling: {image_path}")
    print("Instructions:")
    print("  1. Click and drag to draw bounding box around license plate")
    print("  2. Press 's' to save the label and move to next image")
    print("  3. Press 'r' to reset bounding boxes")
    print("  4. Press 'q' to quit labeling entirely")

    drawing = False
    start_point = None
    end_point = None
    bboxes = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start_point, end_point

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
            end_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            end_point = (x, y)
            if start_point != end_point:
                x1, y1 = start_point
                x2, y2 = end_point
                bboxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return False  # skip this image

    cv2.namedWindow('Label Image')
    cv2.setMouseCallback('Label Image', mouse_callback)

    while True:
        display = img.copy()
        if drawing and start_point and end_point:
            cv2.rectangle(display, start_point, end_point, (0, 255, 0), 2)
        for bbox in bboxes:
            cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        cv2.imshow('Label Image', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if bboxes:
                DatasetLabeler.create_label_file(image_path, bboxes)
                print(f"✓ Saved {len(bboxes)} label(s)")
            else:
                print("No bounding boxes to save")
            break  # move to next image
        elif key == ord('r'):
            bboxes = []
            print("Reset bounding boxes")
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return True  # quit entire labeling

    cv2.destroyAllWindows()
    return False


def label_images_in_folder(folder_path: str):
    """
    Label all images in a folder interactively, skipping already labeled images
    """
    image_folder = Path(folder_path)
    if not image_folder.is_dir():
        print(f"Error: {folder_path} is not a valid folder")
        return

    annotation_folder = Path("data/annotations")
    annotation_folder.mkdir(exist_ok=True)

    IMG_EXTS = ('.jpg', '.jpeg', '.png')
    images = sorted([f for f in image_folder.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS])
    print(f"Found {len(images)} images to label in {folder_path}")

    for img_path in images:
        txt_file = annotation_folder / f"{img_path.stem}.txt"
        if txt_file.exists():
            print(f"Skipping {img_path.name}, annotation already exists.")
            continue

        quit_labeling = label_image_interactive(str(img_path))
        if quit_labeling:
            print("Labeling process terminated by user")
            break


def enhance_plate(plate_image_path: str):
    """
    Enhance a cropped license plate image for better OCR
    """
    img = cv2.imread(plate_image_path)
    if img is None:
        print(f"Error: Could not read image {plate_image_path}")
        return

    preprocessor = ImagePreprocessor()
    corrector = PlateOrientationCorrector()
    corrected = corrector.correct_skew(img)
    auto_cropped = corrector.auto_crop_plate(corrected)
    enhanced = preprocessor.prepare_for_ocr(auto_cropped)

    output_path = Path(plate_image_path).parent / f"{Path(plate_image_path).stem}_enhanced.jpg"
    cv2.imwrite(str(output_path), enhanced)
    print(f"✓ Enhanced image saved to: {output_path}")

    cv2.imshow("Original", cv2.resize(img, (400, 200)))
    cv2.imshow("Enhanced", cv2.resize(enhanced, (400, 200)))
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="License Plate Detection System")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Detect license plates')
    detect_parser.add_argument('input', help='Path to image or folder')
    detect_parser.add_argument('--visualize', '-v', action='store_true', help='Visualize detection results')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train detection model')
    train_parser.add_argument('images', help='Path to training images')
    train_parser.add_argument('labels', help='Path to label files')

    # Label command
    label_parser = subparsers.add_parser('label', help='Label images interactively')
    label_parser.add_argument('input', help='Path to image or folder')

    # Enhance command
    enhance_parser = subparsers.add_parser('enhance', help='Enhance plate for OCR')
    enhance_parser.add_argument('plate', help='Path to cropped plate image')

    args = parser.parse_args()

    if args.command == 'detect':
        path = Path(args.input)
        if path.is_file():
            detect_single_image(str(path), args.visualize)
        elif path.is_dir():
            detect_batch(str(path))
        else:
            print(f"Error: {path} is not a valid file or directory")

    elif args.command == 'train':
        train_new_model(args.images, args.labels)

    elif args.command == 'label':
        path = Path(args.input)
        if path.is_file():
            label_image_interactive(str(path))
        elif path.is_dir():
            label_images_in_folder(str(path))
        else:
            print(f"Error: {path} is not a valid file or directory")

    elif args.command == 'enhance':
        enhance_plate(args.plate)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
