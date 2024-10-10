import os
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from IPython import display
import logging
from datetime import datetime

from utility import get_criterion, get_optimizer, get_scheduler, Timer


class Trainer:
    def __init__(self, model, model_name, config, args=None, train_loader=None, val_loader=None, kfold_loaders=None, test_loader=None):
        self.device = torch.device('cuda')
        self.model = model.to(self.device)
        self.model_name = model_name
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.kfold_loaders = kfold_loaders
        self.test_loader = test_loader
        self.best_loss = float('inf')
        
        train_config = config['train']
        self.criterion = get_criterion(train_config['loss'], args)
        self.optimizer = get_optimizer(train_config['optimizer'], self.model)
        self.scheduler = get_scheduler(train_config['scheduler'], self.optimizer)
        self.epochs = train_config['epochs']
        
        # Logging setup
        logging.basicConfig(filename=f'{model_name}.log', level=logging.INFO, filemode='w')
        self.logger = logging.getLogger()

        # Log start time
        self.logger.info(f"Training started at {datetime.now()}")

        # Loss tracking for plotting
        self.train_losses = []
        self.val_losses = []

    def train_and_eval(self, kfold=False):
        num_epochs = self.epochs
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
        timer = Timer()
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(self.train_loader)
        self.train_losses.append(epoch_loss)
        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {epoch_loss:.4f}")

        # Validation step
        val_loss = self._validate()
        self.val_losses.append(val_loss)

        # Logging
        self.logger.info(f'Epoch {epoch}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Training Time: {timer.end()}')

        # Save model
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            os.makedirs('saved_model', exist_ok=True)
            torch.save(self.model.state_dict(), f'saved_model/{self.model_name}.pth')
            self.logger.info(f'Model saved with loss: {self.best_loss:.4f}')

        # Inference step for test data (only for check overfitting)
        # if self.self.test_loader is not None:
        #     test_loss = self._test()
        #     self.logger.info(f'Test Loss: {test_loss:.4f}')

    def _run_kfold_epoch(self, epoch, num_epochs, timer):
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
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()

        val_loss = running_loss / len(self.val_loader)
        print(f"Validation Loss: {val_loss:.4f}")
        return val_loss

    def _test(self):
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
