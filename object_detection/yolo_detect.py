import cv2
from ultralytics import YOLO
import os
from glob import glob


def process_image(image_path, model, output_dir):
    """
    Process a single image: detect objects and save results
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image {image_path}, skipping...")
        return False

    # Perform detection
    results = model(image)

    # Generate annotated image
    annotated_image = results[0].plot()

    # Save result
    image_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"detected_{image_name}")
    cv2.imwrite(output_path, annotated_image)

    return True


def process_folder(input_folder, model, output_dir="detection_results"):
    """
    Process all images in a folder
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created/verified: {output_dir}")

    # Get all image files in the input folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(input_folder, ext)))

    if not image_paths:
        print(f"No images found in folder: {input_folder}")
        print(f"Supported formats: {image_extensions}")
        return

    total_images = len(image_paths)
    print(f"Found {total_images} images for processing...")

    # Process each image
    success_count = 0
    for i, img_path in enumerate(image_paths, 1):
        print(f"Processing image {i}/{total_images}: {os.path.basename(img_path)}")
        if process_image(img_path, model, output_dir):
            success_count += 1

    # Print summary
    print("\n" + "=" * 50)
    print(f"Processing complete!")
    print(f"Total images: {total_images}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {total_images - success_count}")
    print(f"All results saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    # Configuration
    INPUT_FOLDER = "GoPro_images"  # Folder containing images to process
    OUTPUT_FOLDER = "detection_results"  # Folder to save results
    MODEL_NAME = "yolov8n.pt"  # YOLOv8 model to use

    # Load YOLOv8 model
    print(f"Loading YOLOv8 model: {MODEL_NAME}")
    try:
        model = YOLO(MODEL_NAME)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        exit(1)

    # Verify input folder exists
    if not os.path.isdir(INPUT_FOLDER):
        print(f"Error: Input folder '{INPUT_FOLDER}' does not exist")
        exit(1)

    # Process all images in the folder
    process_folder(INPUT_FOLDER, model, OUTPUT_FOLDER)