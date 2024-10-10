import torch
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import KFold
from .denodata import denodata

def get_data_loader(config, mode='train', val_split=0.2, kfold=False, n_splits=5):
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset']['n_threads']
    dataset = denodata(config, mode=mode)

    if mode == 'train':
        # Try prefetch_factor=4, persistent_workers=True in dataloader
        if kfold:
            kfold_loaders = []
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            indices = list(range(len(dataset)))

            for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
                train_dataset = Subset(dataset, train_idx)
                val_dataset = Subset(dataset, val_idx)

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True
                )
                kfold_loaders.append((train_loader, val_loader))
            return kfold_loaders

        else:
            total_size = len(dataset)
            val_size = int(total_size * val_split)
            train_size = total_size - val_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
            return train_loader, val_loader

    elif mode == 'test':
        test_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        return test_loader

    else:
        raise ValueError(f"Invalid mode: {mode}. Expected 'train' or 'test'.")
