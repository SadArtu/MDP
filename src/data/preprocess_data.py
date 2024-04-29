import os
import shutil
from sklearn.model_selection import train_test_split


def split_data(
    source_dir: str, target_dir: str, train_size: float = 0.7, val_size: float = 0.15
) -> None:
    """
    Splits the data into training, validation, and test sets.

    Args:
    - source_dir (str): Path to the source directory containing class folders.
    - target_dir (str): Path to the target directory where split data will be saved.
    - train_size (float): Fraction of data to be used for training.
    - val_size (float): Fraction of data to be used for validation.
    """

    set_dirs = ["train", "val", "test"]
    for set_dir in set_dirs:
        for class_dir in os.listdir(source_dir):
            os.makedirs(os.path.join(target_dir, set_dir, class_dir), exist_ok=True)

    for class_dir in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_dir)
        images = [os.path.join(class_path, img) for img in os.listdir(class_path)]
        train, test = train_test_split(
            images, train_size=train_size + val_size, random_state=42
        )
        val, test = train_test_split(
            test, train_size=val_size / (1 - train_size), random_state=42
        )

        def copy_files(files: list, set_name: str) -> None:
            """Copies files to the specified dataset folder."""
            for file in files:
                shutil.copy(file, os.path.join(target_dir, set_name, class_dir))

        copy_files(train, "train")
        copy_files(val, "val")
        copy_files(test, "test")

        print(f"Finished splitting data for class: {class_dir}")


if __name__ == "__main__":
    source_directory = "/home/artur_176/CNN/CNN/datasets/raw/PlantVillage/"
    target_directory = "/home/artur_176/CNN/CNN/datasets/processed"
    split_data(source_directory, target_directory)
