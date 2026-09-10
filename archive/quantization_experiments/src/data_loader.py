import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import constants as c


def _remap_targets(dataset):
    synset_to_imagenet = {
        folder_idx: c.IMAGENETTE_LABEL_MAP[synset]
        for synset, folder_idx in dataset.class_to_idx.items()
    }
    dataset.targets = [synset_to_imagenet[t] for t in dataset.targets]
    dataset.samples = [(p, synset_to_imagenet[t]) for p, t in dataset.samples]
    return dataset


def get_val_loader(batch_size=1):
    val_dir = os.path.join(c.DATASET_PATH, "imagenette2-320", "val")
    dataset = ImageFolder(root=val_dir, transform=c.IMAGE_TRANSFORM)
    dataset = _remap_targets(dataset)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=4, pin_memory=True)


def get_train_loader(batch_size=None):
    """Train split used for PTQ calibration."""
    if batch_size is None:
        batch_size = c.CALIB_BATCH_SIZE
    train_dir = os.path.join(c.DATASET_PATH, "imagenette2-320", "train")
    dataset = ImageFolder(root=train_dir, transform=c.IMAGE_TRANSFORM)
    dataset = _remap_targets(dataset)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=4, pin_memory=True)
