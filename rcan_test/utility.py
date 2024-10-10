import os
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lrs
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from src.loss import Loss

def get_criterion(loss_name, args=None):
    if loss_name == 'l1':
        return nn.L1Loss()
    elif loss_name == 'l2':
        return nn.MSELoss()
    elif loss_name == 'vgg':
        pass
    elif loss_name == 'perceptual':
        pass
    elif loss_name == 'gan':
        pass
    elif loss_name == 'edsr':
        return Loss(args)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def get_optimizer(optimizer_config, model):
    lr = optimizer_config['lr']
    optimizer_name = optimizer_config['name']
    
    if optimizer_name == 'Adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        return optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'RMSprop':
        return optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == 'RAdam':
        return optim.RAdam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(scheduler_config, optimizer):
    if scheduler_config['name'] == 'StepLR':
        step_size = scheduler_config['step_size']
        gamma = scheduler_config['gamma']
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_config['name'] == 'CosineAnnealingLR':
        T_max = scheduler_config['T_max']
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_config['name']}")
    
    
def check_dataloader(dataloader, num_batches=1):
    for i, (dt_imgs, gt_imgs) in enumerate(dataloader):
        if i >= num_batches:
            break
        print(f"Batch size: {len(dt_imgs)}")
        print(f"Batch {i+1}")

        for ldct, ndct in zip(dt_imgs, gt_imgs):
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(ldct.squeeze(), cmap='gray')
            plt.title(f"Low-Dose CT")
            
            plt.subplot(1, 2, 2)
            plt.imshow(ndct.squeeze(), cmap='gray')
            plt.title(f"Normal-Dose CT")
            
            plt.show()
         
            
class Timer():
    def __init__(self):
        self.v = time.time()

    def start(self):
        self.v = time.time()

    def end(self):
        elapsed_time = time.time() - self.v
        minutes = int(elapsed_time // 60)  # 분 단위
        seconds = int(elapsed_time % 60)   # 초 단위 (소수점 없이)
        return f"{minutes} minutes {seconds} seconds"
        
        
def inference(device, model, test_loader):
    distorted_images_list = []
    predicted_images_list = []
    ground_truth_images_list = []

    model.eval()
    with torch.no_grad():
        for dt_imgs, gt_img in test_loader:
            # dt_imgs = (12, 1, 1, H, W) list, gt_img = (1, 1, H, W) tensor
            dt_imgs = torch.stack([dt.to(device) for dt in dt_imgs])  # to GPU and stack to input to model(model only take tensor, not a list)
            dt_imgs = dt_imgs.squeeze(2)  # dt_imgs = (12, 1, H, W) tensor
            gt_img = gt_img.squeeze(0).squeeze(0).to(device)  # gt_img = (H, W) tensor

            # 두 번째 리스트: 모델의 예측 결과 12개 저장
            predicted_images = model(dt_imgs)
            predicted_images_list.append(predicted_images.squeeze(1))  # (12, H, W) tensor

            # 첫 번째 리스트: 왜곡된 이미지 12개를 저장
            distorted_images_list.append(dt_imgs.squeeze(1))  # (12, H, W) tensor

            # 세 번째 리스트: 정상 이미지 (gt_img) 저장
            ground_truth_images_list.append(gt_img)

    return distorted_images_list, predicted_images_list, ground_truth_images_list


def inference_loss(predicted_images_list, ground_truth_images_list, loss='l1'):
    distortion_list = ['10_180', '10_360', '10_720', '25_180', '25_360', '25_720', '50_180', '50_360', '50_720', '100_180', '100_360', '100_720']
    criterion = get_criterion(loss)
    
    print(f"{'Distortion':<15} {f'{loss} Loss':<15}")
    print('-' * 40)
    
    distortion_losses = {distortion: 0.0 for distortion in distortion_list}
    distortion_counts = {distortion: 0 for distortion in distortion_list}
    
    for pred_images, gt_image in zip(predicted_images_list, ground_truth_images_list):
        for i, distortion in enumerate(distortion_list):
            pred_image = pred_images[i].unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
            gt_image_exp = gt_image.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
            loss_value = criterion(pred_image, gt_image_exp)
            distortion_losses[distortion] += loss_value.item()
            distortion_counts[distortion] += 1

    for distortion in distortion_list:
        avg_loss = distortion_losses[distortion] / distortion_counts[distortion]
        print(f"{distortion:<15} {avg_loss:<15.4f}")


def calculate_psnr_ssim(predicted_images_list, ground_truth_images_list):
    distortion_list = ['10_180', '10_360', '10_720', '25_180', '25_360', '25_720', '50_180', '50_360', '50_720', '100_180', '100_360', '100_720']

    print(f"{'Distortion':<15} {'Average PSNR':<15} {'Average SSIM':<15}")
    print('-' * 50)
    
    distortion_psnr = {distortion: 0.0 for distortion in distortion_list}
    distortion_ssim = {distortion: 0.0 for distortion in distortion_list}
    distortion_counts = {distortion: 0 for distortion in distortion_list}
    
    for pred_images, gt_image in zip(predicted_images_list, ground_truth_images_list):
        gt_image_np = gt_image.cpu().numpy()  # (H, W)
        
        for i, distortion in enumerate(distortion_list):
            pred_image_np = pred_images[i].cpu().numpy()  # (H, W)
            psnr_value = psnr(gt_image_np, pred_image_np, data_range=gt_image_np.max() - gt_image_np.min())
            ssim_value = ssim(gt_image_np, pred_image_np, data_range=gt_image_np.max() - gt_image_np.min())
            distortion_psnr[distortion] += psnr_value
            distortion_ssim[distortion] += ssim_value
            distortion_counts[distortion] += 1

    for distortion in distortion_list:
        avg_psnr = distortion_psnr[distortion] / distortion_counts[distortion]
        avg_ssim = distortion_ssim[distortion] / distortion_counts[distortion]
        print(f"{distortion:<15} {avg_psnr:<15.4f} {avg_ssim:<15.4f}")


def plot_images(distorted_images_list, predicted_images_list, ground_truth_images_list, distortion=None, idx=0):
    distortion_list = ['10_180', '10_360', '10_720', '25_180', '25_360', '25_720', 
                       '50_180', '50_360', '50_720', '100_180', '100_360', '100_720']
    
    if distortion is None:
        # Plot all distortions in a 2x13 grid, GT image in the last column
        plt.figure(figsize=(70, 15))

        for i, distortion in enumerate(distortion_list):
            distorted_image = distorted_images_list[idx][i].cpu().numpy()
            predicted_image = predicted_images_list[idx][i].cpu().numpy()
            ground_truth_image = ground_truth_images_list[idx].cpu().numpy()

            # Distorted Image (1st row)
            plt.subplot(2, 13, i + 1)
            plt.imshow(distorted_image.squeeze(), cmap='gray')
            plt.title(f"Distorted ({distortion})")
            plt.axis('off')

            # Predicted Image (2nd row)
            plt.subplot(2, 13, i + 14)
            plt.imshow(predicted_image.squeeze(), cmap='gray')
            plt.title(f"Predicted ({distortion})")
            plt.axis('off')

        # Ground Truth Image (last column, spanning both rows)
        plt.subplot(2, 13, 13)
        plt.imshow(ground_truth_image.squeeze(), cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')

        plt.subplot(2, 13, 26)
        plt.imshow(ground_truth_image.squeeze(), cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    else:
        # Check if distortion is valid
        if distortion not in distortion_list:
            raise ValueError(f"Invalid distortion: {distortion}. Must be one of {distortion_list}")
        
        # Plot the specific distortion as before
        distortion_idx = distortion_list.index(distortion)
        distorted_image = distorted_images_list[idx][distortion_idx].cpu().numpy()
        predicted_image = predicted_images_list[idx][distortion_idx].cpu().numpy()
        ground_truth_image = ground_truth_images_list[idx].cpu().numpy()

        plt.figure(figsize=(24, 8))

        # Distorted Image
        plt.subplot(1, 3, 1)
        plt.imshow(distorted_image.squeeze(), cmap='gray')
        plt.title(f"Distorted Image ({distortion})")
        plt.axis('off')

        # Predicted Image
        plt.subplot(1, 3, 2)
        plt.imshow(predicted_image.squeeze(), cmap='gray')
        plt.title("Predicted Image")
        plt.axis('off')

        # Ground Truth Image
        plt.subplot(1, 3, 3)
        plt.imshow(ground_truth_image.squeeze(), cmap='gray')
        plt.title("Ground Truth Image")
        plt.axis('off')

        plt.show()

def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch
    
    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer
    