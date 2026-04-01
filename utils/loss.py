import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import cv2

class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=loss_in+10*loss_grad
        return loss_total,loss_in,loss_grad

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()

def ssim(img1, img2, pixel_range=255, color_mode='rgb'):
    C1 = (0.01 * pixel_range)**2
    C2 = (0.03 * pixel_range)**2

    # transfer color channel to y
    if color_mode == 'rgb':
        img1 = (img1 * np.array([0.256789, 0.504129, 0.097906])).sum(axis=2) + 16 / 255 * pixel_range
        img2 = (img2 * np.array([0.256789, 0.504129, 0.097906])).sum(axis=2) + 16 / 255 * pixel_range
    elif color_mode == 'yuv':
        img1 = img1[:, 0, :, :]
        img2 = img2[:, 0, :, :]
    elif color_mode == 'y':
        img1 = img1
        img2 = img2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2, pixel_range=255):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2, pixel_range)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2, pixel_range))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2), pixel_range)
    else:
        raise ValueError('Wrong input image dimensions.')

class CosineSimilarity(nn.Module):
    def __init__(self, size=(64, 64), greyscale=False):
        super(CosineSimilarity, self).__init__()
        self.size = size
        self.greyscale = greyscale

    def forward(self, image1, image2):
        image1 = self.preprocess(image1)
        image2 = self.preprocess(image2)
        cos_sim = self.cosine_similarity(image1, image2)

        # loss = 1 - cos_sim
        return cos_sim.mean()

    def preprocess(self, image):
        image = F.interpolate(image, size=self.size, mode='bilinear', align_corners=False)
        if self.greyscale:
            image = image.mean(dim=1, keepdim=True)
        image = image.contiguous().view(image.size(0), -1)
        return image

    def cosine_similarity(self, image1, image2):
        norm1 = image1.norm(dim=1)
        norm2 = image2.norm(dim=1)
        cos_sim = (image1 * image2).sum(dim=1) / (norm1 * norm2 + 1e-8)
        return cos_sim

#version adaptation for PyTorch > 1.7.1
IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft

