import os
import shutil
from tqdm import tqdm
import argparse
import os
import shutil
import argparse
import json

def load_dataset_info(json_path):
    """
    Load dataset information from a JSON file.
    :param json_path: Path to the dataset JSON file
    :return: Dictionary containing dataset details
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset information file '{json_path}' not found.")
    
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    return data

def separate_dataset(in_dataset, src_dir, dst_dir, json_path="dataset_info.json"):
    """
    Separates specific classes from a large dataset into a new folder.

    :param in_dataset: Dataset category (e.g., "Manzanita", "Bulrush", "Wild Rye")
    :param src_dir: Source dataset directory (containing images)
    :param dst_dir: Destination directory for the separated dataset
    :param json_filename: Name of the JSON file containing dataset metadata
    """
    dataset_info = load_dataset_info(json_path)

    if in_dataset not in dataset_info:
        print(f"Error: Unknown dataset '{in_dataset}'")
        return
    
    classes = dataset_info[in_dataset]["classes"]
    dst_path = os.path.join(dst_dir, in_dataset)

    # Create destination folder
    os.makedirs(dst_path, exist_ok=True)

    # Find all existing class directories in src_dir
    existing_classes = set(next(os.walk(src_dir))[1])

    for cls in classes:
        src_class_path = os.path.join(src_dir, cls)
        dst_class_path = os.path.join(dst_path, cls)

        if cls in existing_classes:
            print(f"Copying {src_class_path} -> {dst_class_path}")
            shutil.copytree(src_class_path, dst_class_path, dirs_exist_ok=True)
        else:
            print(f"Warning: Source folder '{src_class_path}' does not exist, skipping.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a subset from ImageNet dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in_dataset', default='Wrasse', type=str,
                        choices=['Bulrush', 'Wrasse', 'Wild_rye','Manzanita','Lichen'], help='in-distribution dataset')
    parser.add_argument('--src_dir', default='datasets/ImageNet_OOD_dataset/iNaturalist', type=str,
                        help='full path of iNaturalist')
    parser.add_argument('--dst_dir', default='datasets_temp', type=str,
                        help='root dir of in_dataset')
    parser.add_argument("--json_path", default="descriptions/inaturalist_species.json", type=str,
                        help="Name of the JSON file containing dataset metadata")

    args = parser.parse_args()
    separate_dataset(args.in_dataset, args.src_dir, args.dst_dir, args.json_path)
 