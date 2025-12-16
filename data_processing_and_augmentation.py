import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional
import albumentations as A
import matplotlib.pyplot as plt

def show_multiple_dataset_samples(
    dataset: Dataset,
    num_samples: int = 8,
    title: str = "Dataset samples"
):
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))

    for i in range(num_samples):
        x = dataset[i % len(dataset)]
        x = x.permute(1, 2, 0).numpy()
        x = np.clip(x, 0, 1)

        axes[i].imshow(x, interpolation="nearest")
        axes[i].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


class ImageFolderDataset(Dataset):
    """
    PyTorch Dataset for loading images from a folder with Albumentations support.
    """

    def __init__(
        self,
        root_dir: str,
        transforms: Optional[A.Compose] = None,
        extensions: List[str] = [".jpg", ".jpeg", ".png"]
    ):
        self.root_dir = root_dir
        self.transforms = transforms
        self.extensions = extensions
        self.image_paths = self._collect_images()

    def _collect_images(self) -> List[str]:
        paths = [
            os.path.join(self.root_dir, f)
            for f in os.listdir(self.root_dir)
            if any(f.lower().endswith(ext) for ext in self.extensions)
        ]

        if not paths:
            raise RuntimeError(f"No images found in {self.root_dir}")

        return sorted(paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = self.image_paths[index]
        image = load_image(image_path)

        if self.transforms:
            image = self.transforms(image=image)["image"]

        return to_tensor(image)


def load_image(path: str) -> np.ndarray:
    """
    Loads an image from disk and converts it to RGB numpy array.
    """
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def to_tensor(image: np.ndarray) -> torch.Tensor:
    """
    Converts HWC uint8 image to CHW float32 torch tensor.
    """
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image)

def get_augmentation_pipeline(image_size=(128, 128)) -> A.Compose:
    return A.Compose([
        # Resize first to normalize scale
        A.Resize(image_size[0], image_size[1]),

        # Extend background safely before geometry
        A.PadIfNeeded(
            min_height=image_size[0],
            min_width=image_size[1],
            border_mode=cv2.BORDER_REFLECT_101,
            p=1.0
        ),

        # Gentle geometric transforms (NO black fill)
        A.Affine(
            scale=(0.97, 1.03),
            translate_percent=(0.02, 0.02),
            rotate=(-5, 5),
            mode=cv2.BORDER_CONSTANT,
            cval=255,
            p=0.6
        ),

        # Crop back to target size
        A.RandomCrop(image_size[0], image_size[1]),

        # Safe photometric changes
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.3
        ),
    ])

if __name__ == "__main__":
    dataset = ImageFolderDataset(
        root_dir="data_raw",
        transforms=get_augmentation_pipeline(image_size=(128, 128))
    )

    sample = dataset[0]
    print(sample.shape, sample.dtype)

    # test
    show_multiple_dataset_samples(dataset, num_samples=8)
