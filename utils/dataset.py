import csv
import math
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def generate_csv_dataset(path: Path) -> Path | None:
    """
    Create a csv file of the dataset stored in the input path.
    The input path must contains the images and labels folder.

    Args:
        path (pathlib.Path): A Path pointing to the dataset folder
    Returns:
        pathlib.Path: the path of the csv file representing the dataset (image-label mapping) or None if the input path is not valid or the corresponding sample of a label doesn't exist.
    """
    if not isinstance(path, Path):
        raise TypeError(f"path must be a pathlib.Path, got {type(path)}")
    if not path.exists():
        raise FileNotFoundError("path doesn't exist")
    if not path.is_dir():
        raise ValueError("path must point to a directory containing the dataset")

    # Get the images and labels folder paths
    images_path: Path = path / "images"
    labels_path: Path = path / "labels"

    if not images_path.exists() or not images_path.is_dir():
        raise ValueError("path must contain the 'images' folder. This folder must contain the dataset samples")
    if not labels_path.exists() or not labels_path.exists():
        raise ValueError("path must contain the 'labels' folder. This folder must contain a label for each dataset sample")

    # Get all label files
    labels: list[Path] = [label_path for label_path in labels_path.glob("*.txt")]

    if not labels:
        raise ValueError("No files found in 'path/labels")

    csv_dataset: Path = path / "dataset.csv"

    # Create the csv file representing the entire dataset
    with open(csv_dataset, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image_name", "label_file_name", "class", "obj_x", "obj_y", "obj_width", "obj_height"])
        writer.writeheader()

        try:
            for label in tqdm(labels, desc="Reading samples"):
                img_path: Path = images_path / f"{label.stem}.jpg"

                if not img_path.exists() or not img_path.is_file():
                    raise FileNotFoundError(f"The sample {label.stem}.jpg doesn't exist")

                with open(label, "r") as txt_file:
                    lines: list = txt_file.readlines()

                    # Store all objects for the i-th sample
                    for line in lines:
                        line: str
                        line = line.strip()
                        if not line or line == "\n":
                            # Skip empty lines
                            continue

                        pattern: re.Pattern = re.compile(r"(\d+)\s+(\d+.\d+)\s+(\d+.\d+)\s+(\d+.\d+)\s+(\d+.\d+)")
                        if pattern.match(line):
                            values = pattern.match(line)
                            
                            writer.writerow({
                                "image_name": img_path.name,
                                "label_file_name": label.name,
                                "class": values[1],
                                "obj_x": values[2],
                                "obj_y": values[3],
                                "obj_width": values[4],
                                "obj_height": values[5]
                            })
                        else:
                            raise Exception(f"The regex doesn't work for that label file: {label.name}")
        except FileNotFoundError:
            print("Unable to define the dataset in csv format. Some samples don't exist.")
            return None

    dataset: pd.DataFrame = pd.read_csv(csv_dataset, sep=",", dtype={"image_name": str, "label_file_name":str, "class":int, "obj_x":float, "obj_y":float, "obj_width":float, "obj_height":float})

    samples: list = dataset["image_name"].unique().tolist()

    assert len(samples) == len(labels)

    return csv_dataset
            

def split_dataset(dataset_path: Path, train_perc: float = 0.7, test_perc: float = 0.2, valid_perc: float = 0.1) -> dict[str, Path]:
    """
    Split the dataset in training, test, and valid sets based on input percentages.

    Args:
        dataset_path (pathlib.Path): A Path of the dataset in csv format
        train_perc (float): a percentage of the dataset to define the training set.
        test_perc (float): a percentage of the dataset to define the test set.
        valid_perc (float): a percentage of the dataset to define the validation set.
    Returns:
        dict: a dictionary containing the mapping between the sets and the corresponding csv file path.
    """
    if not isinstance(dataset_path, Path):
        raise TypeError(f"dataset_path must be a pathlib.Path, got {type(dataset_path)}")
    if not isinstance(train_perc, float):
        raise TypeError(f"train_perc must be a float, got {type(train_perc)}")
    if not isinstance(test_perc, float):
        raise TypeError(f"test_perc must be a float, got {type(test_perc)}")
    if not isinstance(valid_perc, float):
        raise TypeError(f"valid_perc must be a float, got {type(valid_perc)}")

    if not dataset_path.exists() or not dataset_path.is_file():
        raise FileNotFoundError("dataset csv file doesn't exist")

    if train_perc <= 0 or train_perc >= 1:
        raise ValueError("train_perc must be a percentage between 0.1 and 1.0")
    if test_perc <= 0 or test_perc >= 1:
        raise ValueError("test_perc must be a percentage between 0.1 and 1.0")
    if valid_perc <= 0 or valid_perc >= 1:
        raise ValueError("valid_perc must be a percentage between 0.1 and 1.0")
    if not math.isclose(train_perc + test_perc + valid_perc, 1.0, abs_tol= 1e-9):
        raise ValueError("the input percentages must sum up to 100%")

    # Load dataset
    dataset: pd.DataFrame = pd.read_csv(dataset_path, sep=",", dtype={"image_name": str, "label_file_name":str, "class":int, "obj_x":float, "obj_y":float, "obj_width":float, "obj_height":float})

    train_folder: Path = dataset_path.parent / "train"    
    train_path: Path = dataset_path.parent / "train.csv"

    test_folder: Path = dataset_path.parent / "test"
    test_path: Path = dataset_path.parent / "test.csv"

    valid_folder: Path = dataset_path.parent / "valid"
    valid_path: Path = dataset_path.parent / "valid.csv"

    for folder in [train_folder, test_folder, valid_folder]:
        folder.mkdir(exist_ok=True)

    # Get samples (unique images)
    images: list[str] = dataset["image_name"].unique().tolist()

    # Split the dataset into train, test, and valid sets
    train_imgs, remaining_imgs = train_test_split(images, train_size= train_perc, test_size= test_perc + valid_perc, random_state=42)
    valid_imgs, test_imgs = train_test_split(remaining_imgs, train_size= valid_perc, test_size= test_perc, random_state=42) 

    train_samples: pd.DataFrame = dataset[dataset["image_name"].isin(train_imgs)].copy()
    valid_samples: pd.DataFrame = dataset[dataset["image_name"].isin(valid_imgs)].copy()
    test_samples: pd.DataFrame = dataset[dataset["image_name"].isin(test_imgs)].copy()

    # Store csv files
    train_samples.to_csv(train_path, index=False)
    valid_samples.to_csv(valid_path, index=False)
    test_samples.to_csv(test_path, index=False)

    # Copy images to the corresponding folders
    for img in train_imgs:
        src_path: Path = dataset_path.parent / "images" / img

        shutil.copy2(src_path, train_folder)

    for img in test_imgs:
        src_path: Path = dataset_path.parent / "images" / img

        shutil.copy2(src_path, test_folder)

    for img in valid_imgs:
        src_path: Path = dataset_path.parent / "images" / img

        shutil.copy2(src_path, valid_folder)

    return {
        "train": train_path,
        "valid": valid_path,
        "test": test_path
    }

def plot_classes(set_path: Path, plot_title: str, out_dir: Path, file_name: str) -> None:
    """
    Plot the class distribution of the input sets

    Args:
        set_path (pathlib.Path): A path of the set csv file.
        plot_title (str): the title plot.
        out_dir (pathlib.Path): the output directory where the plot will be stored.
        file_name (str): the plot file name.
    Returns:
        None
    """
    if not isinstance(set_path, Path):
        raise TypeError(f"set_path must be a pathlib.Path, got {type(set_path)}")
    if not isinstance(plot_title, str):
        raise TypeError(f"plot_title must be a string, got {type(plot_title)}")
    if not isinstance(out_dir, Path):
        raise TypeError(f"out_path must be a pathlib.Path, got {type(out_dir)}")
    if not isinstance(file_name, str):
        raise TypeError(f"file_name must be a string, got {type(file_name)}")
    
    if not set_path.exists():
        raise FileNotFoundError("set csv file doesn't exist")
    if re.match(r"\w+\.\w+", file_name):
        raise ValueError(f"file_name must not contain the file exstension, got {file_name}")

    # Create the output folder and file path
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{file_name}.png"

    set_samples: pd.DataFrame = pd.read_csv(set_path, sep=",", dtype={"image_name": str, "label_file_name":str, "class":int, "obj_x":float, "obj_y":float, "obj_width":float, "obj_height":float})

    class_counts = set_samples["class"].value_counts().sort_index()

    plt.figure(figsize=(8, 6))
    plt.bar(class_counts.index.astype(str), class_counts.values)
    plt.title(plot_title)
    plt.xlabel("Classes")
    plt.ylabel("Amount")
    plt.savefig(out_file)
    plt.show()
    plt.close()
    