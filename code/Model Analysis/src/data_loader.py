"""
Loads the ImageNette validation set with correct ImageNet-index labels.
"""

import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import constants as c


def get_val_loader(batch_size=None):
    if batch_size is None:
        batch_size = c.VAL_BATCH_SIZE

    val_dir = os.path.join(c.DATASET_PATH, "imagenette2-320", "val")
    dataset = ImageFolder(root=val_dir, transform=c.IMAGE_TRANSFORM)

    # dataset.class_to_idx maps folder name -> 0..9. Remap those to the real
    # ImageNet class index for every sample, since pretrained ResNets predict
    # over 1000 classes, not 0..9.
    synset_to_imagenet = {}
    for synset, folder_idx in dataset.class_to_idx.items():
        synset_to_imagenet[folder_idx] = c.IMAGENETTE_LABEL_MAP[synset]

    new_targets = []
    for target in dataset.targets:
        new_targets.append(synset_to_imagenet[target])
    dataset.targets = new_targets

    new_samples = []
    for path, target in dataset.samples:
        new_samples.append((path, synset_to_imagenet[target]))
    dataset.samples = new_samples

    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=4, pin_memory=True)
