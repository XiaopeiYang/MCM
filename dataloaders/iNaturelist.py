import os
import json
import random
from collections import defaultdict
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# Default JSON file relative path
DEFAULT_DATASET_INFO_FILE = "./descriptions/inaturalist_species.json"

def random_split_in_out(root_dir, transform=None, do_rescale=False,
                        max_samples_per_class=100, num_in=2, seed=None,
                        dataset_info_file=DEFAULT_DATASET_INFO_FILE, dataset_key=None):
    """
    Randomly split classes in the dataset into "in" and "out" groups,
    and create datasets with optional JSON-based class name mapping.

    :param root_dir: Root directory of the dataset (each subfolder is a class)
    :param transform: Image transformations
    :param do_rescale: Whether to apply additional image adjustments (e.g., brightness)
    :param max_samples_per_class: Maximum number of images to load per class
    :param num_in: Number of classes to randomly select as "in-class"
    :param seed: Random seed for reproducibility (optional)
    :param dataset_info_file: JSON file path with class metadata (default is DEFAULT_DATASET_INFO_FILE)
    :param dataset_key: Key in the JSON file to look up (e.g., "Bulrush")
    :return: (in_dataset, out_dataset)
    """
    if seed is not None:
        random.seed(seed)

    # Get all subfolder names (each subfolder is a class)
    all_labels = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    if len(all_labels) < num_in:
        raise ValueError(f"Total classes ({len(all_labels)}) are fewer than num_in ({num_in}); cannot split.")

    # Shuffle and split the classes into in-class and out-class groups
    random.shuffle(all_labels)
    in_labels = all_labels[:num_in]
    out_labels = all_labels[num_in:]

    # Create the in and out datasets with JSON class name mapping if provided
    in_dataset = iNaturalistDataset(
        root_dir=root_dir,
        label_names=in_labels,
        transform=transform,
        do_rescale=do_rescale,
        max_samples_per_class=max_samples_per_class,
        dataset_info_file=dataset_info_file,
        dataset_key=dataset_key
    )
    out_dataset = iNaturalistDataset(
        root_dir=root_dir,
        label_names=out_labels,
        transform=transform,
        do_rescale=do_rescale,
        max_samples_per_class=max_samples_per_class,
        dataset_info_file=dataset_info_file,
        dataset_key=dataset_key
    )

    return in_dataset, out_dataset


class iNaturalistDataset(Dataset):
    def __init__(self, root_dir, label_names, transform=None, do_rescale=False,
                 max_samples_per_class=100, dataset_info_file=DEFAULT_DATASET_INFO_FILE, dataset_key=None):
        """
        Dataset for loading images from specified class folders, with optional
        JSON-based class name mapping.

        :param root_dir: Root directory containing class subfolders.
        :param label_names: List of folder names (classes) to load.
        :param transform: Image transformations (from torchvision).
        :param do_rescale: Whether to apply custom adjustments (e.g., brightness).
        :param max_samples_per_class: Maximum number of images per class.
        :param dataset_info_file: JSON file path with class metadata (default is DEFAULT_DATASET_INFO_FILE).
        :param dataset_key: Key in the JSON file to look up (e.g., "Bulrush"). If not provided,
                            the folder names are used as class names.
        """
        self.root_dir = root_dir
        self.label_names = label_names
        self.transform = transform
        self.do_rescale = do_rescale

        # Assign numeric labels to the provided class names.
        # For example, if label_names = ["06297_...", "06298_..."],
        # they will be assigned labels 0 and 1 respectively.
        self.label_mapping = {name: idx for idx, name in enumerate(label_names)}
        self.text_label = {idx: name for name, idx in self.label_mapping.items()}

        self.image_paths = []
        self.labels = []
        class_sample_count = defaultdict(int)

        # Traverse each specified class folder and load images
        for label in label_names:
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for image_file in os.listdir(label_dir):
                    if image_file.lower().endswith(('jpg', 'jpeg', 'png')):
                        if class_sample_count[label] < max_samples_per_class:
                            self.image_paths.append(os.path.join(label_dir, image_file))
                            self.labels.append(self.label_mapping[label])
                            class_sample_count[label] += 1

        # Build class names string list using JSON information if provided.
        if dataset_info_file is not None and dataset_key is not None:
            try:
                with open(dataset_info_file, 'r') as f:
                    info = json.load(f)
                if dataset_key in info:
                    dataset_info = info[dataset_key]
                    classes_json = dataset_info.get("classes", [])
                    common_names = dataset_info.get("common_names", [])
                    # Create a mapping from class identifier (from JSON) to common name.
                    json_mapping = {cls: common_names[i] for i, cls in enumerate(classes_json)}
                    # For each label in label_names, if found in the JSON mapping, use its common name.
                    self.class_names_str = [json_mapping.get(lbl, lbl) for lbl in label_names]
                else:
                    # If dataset_key not found, fall back to using the original folder names.
                    self.class_names_str = [self.text_label[i] for i in range(len(label_names))]
            except Exception as e:
                print(f"Failed to load dataset info from {dataset_info_file}: {e}")
                self.class_names_str = [self.text_label[i] for i in range(len(label_names))]
        else:
            self.class_names_str = [self.text_label[i] for i in range(len(label_names))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Optionally apply additional adjustments (e.g., brightness)
        if self.do_rescale:
            image = transforms.functional.adjust_brightness(image, 1.0)

        return image, label
