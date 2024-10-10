import os
import tifffile as tiff
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau

def show_image(x: int, img_list: list, mode, transform = False):
    if mode == 'train':
        image = tiff.imread(os.path.join('train/image', img_list[x]))
    elif mode == 'test':
        image = tiff.imread(os.path.join('test/images', img_list[x]))
        
    if transform == True:
        image = change_window(image=image, window=350, level=40)
        
    print(f'Image shape is {image.shape}')
    plt.imshow(image, cmap='gray')
    plt.title(img_list[x])
    plt.show()

def change_window(image, width, level):
    # Rescale image to Hounsfield units range (raw data digit -> HU)
    image_hu = image * 1400 - 1000
    
    min_value = level - width / 2
    max_value = level + width / 2
    return np.clip((image_hu - min_value) / (max_value - min_value), 0, 1)

class ImageDataset(Dataset):
    def __init__(self, image_label_pairs, image_dir, transform = None, transform_arg = None):
        self.image_label_pairs = list(image_label_pairs.items())
        self.image_dir = image_dir
        self.transform = transform  # change window -> w, transform -> t, both -> wt or tw
        self.transform_arg = transform_arg  # flip, rotate..
        
    def __len__(self):
        return len(self.image_label_pairs)
    
    def __getitem__(self, idx):
        image_n, label = self.image_label_pairs[idx]
        image_path = os.path.join(self.image_dir, image_n)
        image = tiff.imread(image_path)

        if self.transform is not None:
            if 'w' in self.transform:
                image = change_window(image=image, width=350, level=40) # change defalut values to change window of images
            if 't' in self.transform:
                image = self.transform_arg(image)
    
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.float32)
        
        return image, label

def check_dataloader(dataloader, num_batches=3):
    for i, (images, labels) in enumerate(dataloader):
        if i >= num_batches:
            break
        print(f"Batch {i+1}")
        for j in range(len(images)):
            print(f"Image {j+1}:")
            show_image(images[j], labels[j])

def inference_score(model, image_label_pairs, device, image_path, window=350, level=40, apply_window=False):
    '''
    Get test images' predicted score list.
    '''
    model.eval()    
    ori_images = []
    
    for image_n in image_label_pairs.keys():
        img = torch.tensor(tiff.imread(os.path.join(image_path, image_n)), dtype=torch.float32)
        
        if apply_window:
            img = change_window(image=img, width=window, level=level)
        
        ori_images.append(img.unsqueeze(0))
        
    infer_output = []
    with torch.no_grad():
        for img in ori_images:
            img = img.unsqueeze(0).to(device)
            output = model(img)
            output_value = output.item()
            
            if output_value > 4.0:
                infer_output.append(4.0)
            elif output_value < 0.0:
                infer_output.append(0.0)
            else:
                infer_output.append(round(output_value, 1))
    
    return infer_output

def calculate_metrics(y_true, y_pred):
    plcc, _ = pearsonr(y_true, y_pred)
    srocc, _ = spearmanr(y_true, y_pred)
    krocc, _ = kendalltau(y_true, y_pred)
    total = plcc + srocc + krocc
    return plcc, srocc, krocc, total

def plot_score_distributions(y_true, y_pred):
    '''
    Plot test images' score dist and difference dist compare with true score.
    '''
    # Plot score distribution
    plt.figure(figsize=(10, 5))
    plt.hist(y_true, color='red', alpha=0.2, label='original', bins=21)
    plt.hist(y_pred, color='green', alpha=0.2, label='inference', bins=21)
    plt.legend()
    plt.title('score_dist')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.show()

    # Plot score difference distribution
    score_diff = [a - b for a, b in zip(y_true, y_pred)]
    plt.figure(figsize=(10, 5))
    plt.hist(score_diff, bins=21, color='blue', alpha=0.7)
    plt.title('score_diff_dist')
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    plt.show()