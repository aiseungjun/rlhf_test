import os
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from IPython import display
import logging
from datetime import datetime

from utility import ImageDataset

class Trainer:
    def __init__(self, device, model, model_name, criterion, optimizer, scheduler, train_pairs, image_path, inference=False):
        self.device = device
        self.model = model.to(device)
        self.model_name = model_name
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_pairs = train_pairs
        self.image_path = image_path
        self.inference = inference
        self.best_loss = float('inf')
        
        # Logging setup
        logging.basicConfig(filename=f'{model_name}.log', level=logging.INFO, filemode='w')
        self.logger = logging.getLogger()

        # Log start time
        self.logger.info(f"Training started at {datetime.now()}")

        # Loss tracking for plotting
        self.train_losses = []
        self.val_losses = []

    def setup_datasets(self, use_kfold=False, n_splits=5, transform=None):
        if not use_kfold:
            # Train/Val split
            train_image_label_pairs = {k: self.train_pairs[k] for k in list(self.train_pairs.keys())[:800]}
            val_image_label_pairs = {k: self.train_pairs[k] for k in list(self.train_pairs.keys())[800:]}
            
            # Set transform = 'w' to change window
            train_dataset = ImageDataset(train_image_label_pairs, self.image_path, transform=transform)
            val_dataset = ImageDataset(val_image_label_pairs, self.image_path, transform=transform)

            self.train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            self.val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        else:
            # K-Fold setup
            self.kfold_loaders = []
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            indices = list(range(len(self.train_pairs)))

            # Precompute data loaders for each fold(run time will be low)
            for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
                train_dataset = Subset(ImageDataset(self.train_pairs, self.image_path, transform=transform), train_idx)
                val_dataset = Subset(ImageDataset(self.train_pairs, self.image_path, transform=transform), val_idx)

                train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

                # Store loaders for each fold
                self.kfold_loaders.append((train_loader, val_loader))

    def train_and_eval(self, num_epochs=100, kfold=False):
        for epoch in range(1, num_epochs + 1):
            if kfold:
                self._run_kfold_epoch(epoch, num_epochs)
            else:
                self._run_epoch(epoch, num_epochs)
    
            self.scheduler.step()
            self._plot_losses(epoch)
            
        plt.show()

    def _run_epoch(self, epoch, num_epochs):
        # Training step
        self.model.train()
        running_loss = 0.0
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            targets = targets.unsqueeze(1)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(self.train_loader.dataset)
        self.train_losses.append(epoch_loss)
        # print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}")

        # Validation step
        val_loss = self._validate()
        self.val_losses.append(val_loss)

        # Logging
        self.logger.info(f'Epoch {epoch}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Save model
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            os.makedirs('saved_model', exist_ok=True)
            torch.save(self.model.state_dict(), f'saved_model/{self.model_name}.pth')
            self.logger.info(f'Model saved with loss: {self.best_loss:.4f}')

        # Inference step for test data (only for check overfitting)
        if self.inference:
            test_loss = self._test()
            self.logger.info(f'Test Loss: {test_loss:.4f}')

    def _run_kfold_epoch(self, epoch, num_epochs):
        # Determine current fold for validation
        current_fold = epoch % len(self.kfold_loaders)
        
        # Use the precomputed train_loader and val_loader for the current fold
        self.train_loader = self.kfold_loaders[current_fold][0]
        self.val_loader = self.kfold_loaders[current_fold][1]

        self._run_epoch(epoch, num_epochs)

    def _validate(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                targets = targets.unsqueeze(1)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)

        val_loss = running_loss / len(self.val_loader.dataset)
        # print(f"Validation Loss: {val_loss:.4f}")
        return val_loss

    def _test(self):
        '''
        To check the model is overfitting. You can use inference_score function from utility.py to bulid this function.
        '''
        pass
        #test_loss = 0.0
        #return test_loss

    def _plot_losses(self, epoch):
        plt.clf()
        
        # Plot train and validation loss
        plt.plot(range(1, len(self.train_losses)+1), self.train_losses, label='Train Loss')
        plt.plot(range(1, len(self.val_losses)+1), self.val_losses, label='Val Loss')
        plt.title(f'Losses for {self.model_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlim([1, epoch])
        
        # Show and update the plot with pause
        plt.pause(0.001)
        plt.draw()
        display.clear_output(wait=True)
        # plt.savefig(f'{self.model_name}_loss_plot.png')
