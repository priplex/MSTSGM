import os
import shutil


def extract_clear_images(txt_file_path, target_directory):
    """
    Extract clear images (gt paths) from RealBlur_J_test_list.txt, rename them and copy to target directory

    Parameters:
    txt_file_path: str - Full path (absolute or relative) of the RealBlur_J_test_list.txt file
    target_directory: str - Path of the target directory to store extracted clear images
    """
    # 1. Check if the txt file exists
    if not os.path.exists(txt_file_path):
        print(f"Error: txt file does not exist - {txt_file_path}")
        return

    # 2. Create target directory if it doesn't exist
    os.makedirs(target_directory, exist_ok=True)
    print(f"Target directory is ready: {target_directory}")

    # 3. Initialize counters for success/failure
    success_count = 0
    fail_count = 0
    failed_files = []

    # 4. Read the txt file and process each line
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        total_lines = len(lines)
        print(f"Successfully read {total_lines} lines, starting to extract clear images...")

        for line_idx, line in enumerate(lines, 1):
            # Remove whitespace characters from both ends of the line
            clean_line = line.strip()
            # Skip empty lines
            if not clean_line:
                print(f"Line {line_idx} is empty, skipping")
                continue

            # Split line data (each line contains 2 paths, split by space)
            path_parts = clean_line.split()
            if len(path_parts) != 2:
                print(f"Line {line_idx} has invalid format (not 2 paths), skipping: {clean_line}")
                fail_count += 1
                failed_files.append(f"Line {line_idx}: {clean_line}")
                continue

            # Get the first path (gt path = clear image)
            gt_image_path = path_parts[0]

            # Check if the original image file exists
            if not os.path.exists(gt_image_path):
                print(f"Line {line_idx}: Clear image does not exist - {gt_image_path}")
                fail_count += 1
                failed_files.append(gt_image_path)
                continue

            # ---------------------- Parse parameters for renaming ----------------------
            # Extract scene number (e.g., "230" from "scene230", "050" from "scene050")
            scene_start_idx = gt_image_path.find("scene")
            if scene_start_idx == -1:
                print(f"Line {line_idx}: Failed to extract scene number - {gt_image_path}")
                fail_count += 1
                failed_files.append(gt_image_path)
                continue
            # Scene number follows "scene" and ends at the next "/"
            scene_end_idx = gt_image_path.find("/", scene_start_idx + len("scene"))
            scene_num = gt_image_path[scene_start_idx + len("scene"): scene_end_idx]

            # Extract number from gt filename (e.g., "7" from "gt_7.png", "21" from "gt_21.png")
            gt_filename = os.path.basename(gt_image_path)  # Get filename (e.g., "gt_7.png")
            # Split by "_", take the last part and split by "." to get the number
            gt_num_parts = gt_filename.split("_")
            if len(gt_num_parts) < 2:
                print(f"Line {line_idx}: Failed to extract gt number - {gt_image_path}")
                fail_count += 1
                failed_files.append(gt_image_path)
                continue
            gt_num = gt_num_parts[-1].split(".")[0]  # e.g., "7.png" → ["7", "png"] → "7"

            # ---------------------- Generate new filename and copy ----------------------
            # New filename format: RealBlur_J_scene{scene_num}_{gt_num}.png
            new_filename = f"RealBlur_J_scene{scene_num}_{gt_num}.png"
            # Target path = target directory + new filename
            target_image_path = os.path.join(target_directory, new_filename)

            # Copy file (use copy2 to preserve original file metadata like creation time)
            try:
                shutil.copy2(gt_image_path, target_image_path)
                success_count += 1
                print(f"Line {line_idx}/{total_lines} Success: {gt_image_path} → {new_filename}")
            except Exception as e:
                print(f"Line {line_idx} Copy failed: {gt_image_path} → Error: {str(e)}")
                fail_count += 1
                failed_files.append(f"{gt_image_path} (Error: {str(e)})")

    # 5. Output final statistics
    print("\n" + "=" * 50)
    print(f"Processing completed! Total: {total_lines} lines")
    print(f"✅ Successfully extracted: {success_count} clear images")
    print(f"❌ Failed to extract: {fail_count} items")
    if failed_files:
        print("\nFailure details:")
        for fail_item in failed_files:
            print(f"  - {fail_item}")
    print(f"\nAll successfully extracted images are saved to: {os.path.abspath(target_directory)}")


# ---------------------- Please modify the following two parameters according to your actual situation ----------------------
# 1. Path of RealBlur_J_test_list.txt (absolute or relative)
# Examples:
#   Relative path (txt file and code in the same folder): "RealBlur_J_test_list.txt"
#   Absolute path (Windows): "C:/Users/YourName/Datasets/RealBlur_J_test_list.txt"
#   Absolute path (Mac/Linux): "/home/YourName/Datasets/RealBlur_J_test_list.txt"
TXT_FILE_PATH = "RealBlur_J_test_list.txt"

# 2. Path of target directory (to store extracted clear images)
# Example:
TARGET_DIRECTORY = "RealBlur_J_clear_images"
# ------------------------------------------------------------------------


# Execute the extraction function
if __name__ == "__main__":
    extract_clear_images(TXT_FILE_PATH, TARGET_DIRECTORY)