# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''

from net import Encoder, Decoder,SpatialFeatureFusion,FrequencyFeatureFusion,FIBlock,MCBlock
from utils.dataset import H5Dataset
from utils.Evaluator import extract_amp_pha
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.loss import Fusionloss,CosineSimilarity
import kornia



'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''


os.environ['CUDA_VISIBLE_DEVICES'] = '3'
criteria_fusion = Fusionloss()
model_str = 'D2Fusion'

# . Set the hyper-parameters for training
num_epochs = 101 # total epoch
epoch_gap = 40 # epoches of Phase I 

lr = 1e-4
weight_decay = 0
batch_size = 8
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
# Coefficients of the loss function
coeff_mse_loss_VF = 1. # alpha1
coeff_mse_loss_IF = 1.5
coeff_decomp = 2.      # alpha2 and alpha4
coeff_tv = 5.
clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5


# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
Dual_Domain_Encoder = nn.DataParallel(Encoder()).to(device)
Dual_Domain_Decoder = nn.DataParallel(Decoder()).to(device)
FI_Block=FIBlock().to(device)
MC_Block=nn.DataParallel(MCBlock(channels=64)).to(device)
Spatial_Fuse_Layer = nn.DataParallel(SpatialFeatureFusion(dim=64)).to(device)
Frequency_Fuse_Layer = nn.DataParallel(FrequencyFeatureFusion(dim=64)).to(device)

# optimizer, scheduler and loss function
optimizer1 = torch.optim.Adam(
    Dual_Domain_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(
    Dual_Domain_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(
    Spatial_Fuse_Layer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(
    Frequency_Fuse_Layer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer5 = torch.optim.Adam(MC_Block.parameters(), lr=lr, weight_decay=weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)
scheduler5 = torch.optim.lr_scheduler.StepLR(optimizer5, step_size=optim_step, gamma=optim_gamma)


MSELoss = nn.MSELoss()  
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.ssim.SSIMLoss(11, reduction='mean')

cos=CosineSimilarity()




# data loader
trainloader = DataLoader(H5Dataset(r"data/MSRS_train_imgsize_128_stride_200.h5"),
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0)

loader = {'train': trainloader, }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")


'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()


for epoch in range(num_epochs):

    ''' train '''
    for i, (data_VIS, data_IR) in enumerate(loader['train']):
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
        Dual_Domain_Encoder.train()
        Dual_Domain_Decoder.train()
        Spatial_Fuse_Layer.train()
        Frequency_Fuse_Layer.train()
        MC_Block.train()

        Dual_Domain_Encoder.zero_grad()
        Dual_Domain_Decoder.zero_grad()
        Spatial_Fuse_Layer.zero_grad()
        Frequency_Fuse_Layer.zero_grad()
        MC_Block.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()
        optimizer5.zero_grad()



        if epoch < epoch_gap: #Phase I
            feature_V_S, feature_V_F, _,_,_ = Dual_Domain_Encoder(data_VIS)
            feature_I_S, feature_I_F, _,_,_ = Dual_Domain_Encoder(data_IR)
            data_VIS_hat, _ = Dual_Domain_Decoder(data_VIS, feature_V_S, feature_V_F)
            data_IR_hat, _ = Dual_Domain_Decoder(data_IR, feature_I_S, feature_I_F)

            #获取特征的幅度和相位
            feature_V_F_A, feature_I_F_A, feature_V_F_P, feature_I_F_P = extract_amp_pha(feature_V_F, feature_I_F)

            cos_loss_S = cos(feature_V_S, feature_I_S)
            cos_loss_F = cos(feature_V_F_A, feature_I_F_A) ** 2 + cos(feature_V_F_P, feature_I_F_P) ** 2

            mse_loss_V = 5 * Loss_ssim(data_VIS, data_VIS_hat) + MSELoss(data_VIS, data_VIS_hat)
            mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat) + MSELoss(data_IR, data_IR_hat)
            Gradient_loss = L1Loss(kornia.filters.SpatialGradient()(data_VIS),
                                   kornia.filters.SpatialGradient()(data_VIS_hat))

            loss_decomp = (cos_loss_F) / (1.01 + cos_loss_S)
            loss = coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * \
                   mse_loss_I + coeff_decomp * loss_decomp + coeff_tv * Gradient_loss

            loss.backward()
            nn.utils.clip_grad_norm_(
                Dual_Domain_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                Dual_Domain_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step()  
            optimizer2.step()
        else:  #Phase II
            feature_V_S, feature_V_F, feature_V, vis_amp, vis_pha = Dual_Domain_Encoder(data_VIS)
            feature_I_S, feature_I_F, feature_I, ir_amp, ir_pha = Dual_Domain_Encoder(data_IR)
            fused_ST = FI_Block(vis_amp, vis_pha, ir_pha,data_VIS)

            feature_F_S = Spatial_Fuse_Layer(feature_I_S+feature_V_S)
            feature_F_F = Frequency_Fuse_Layer(feature_I_F,feature_V_F)
            #space frequency fre_amp
            fused_y = MC_Block(fused_ST, feature_F_S, feature_F_F)

            data_Fuse, feature_F = Dual_Domain_Decoder(data_VIS, fused_y)

            mse_loss_V = 5*Loss_ssim(data_VIS, data_Fuse) + MSELoss(data_VIS, data_Fuse)
            mse_loss_I = 5*Loss_ssim(data_IR,  data_Fuse) + MSELoss(data_IR,  data_Fuse)
            #获取特征的幅度和相位
            feature_V_F_A, feature_I_F_A, feature_V_F_P, feature_I_F_P = extract_amp_pha(feature_V_F, feature_I_F)

            cos_loss_S = cos(feature_V_S, feature_I_S)
            cos_loss_F = cos(feature_V_F_A, feature_I_F_A) ** 2 + cos(feature_V_F_P, feature_I_F_P) ** 2

            cos_loss_A=cos(fused_ST, feature_F_S) ** 2 + cos(fused_ST, feature_F_F) ** 2

            loss_decomp = (cos_loss_F) / (1.01 + cos_loss_S)+0.5*cos_loss_A

            fusionloss, _,_  = criteria_fusion(data_VIS, data_IR, data_Fuse)

            loss = fusionloss + coeff_decomp * loss_decomp
            loss.backward()
            nn.utils.clip_grad_norm_(
                Dual_Domain_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                Dual_Domain_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                Spatial_Fuse_Layer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                Frequency_Fuse_Layer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                MC_Block.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step()
            optimizer3.step()
            optimizer4.step()
            optimizer5.step()

        # Determine approximate time left
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        if epoch < epoch_gap:
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] [mse_loss_V: %f] [mse_loss_I: %f] [loss_decomp: %f] [Gradient_loss: %f]  ETA: %.10s\n"
                % (
                    epoch,
                    num_epochs,
                    i,
                    len(loader['train']),
                    loss.item(),
                    mse_loss_V.item(),
                    mse_loss_I.item(),
                    loss_decomp.item(),
                    Gradient_loss.item(),
                    time_left,
                )
            )
        else:
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] [fusionloss: %f] [loss_decomp: %f] ETA: %.10s\n"
                % (
                    epoch,
                    num_epochs,
                    i,
                    len(loader['train']),
                    loss.item(),
                    fusionloss.item(),
                    loss_decomp.item(),
                    #loss_fre.item(),
                    time_left,
                )
            )

    # adjust the learning rate
    scheduler1.step()  
    scheduler2.step()
    if not epoch < epoch_gap:
        scheduler3.step()
        scheduler4.step()
        scheduler5.step()

    if optimizer1.param_groups[0]['lr'] <= 1e-6:
        optimizer1.param_groups[0]['lr'] = 1e-6
    if optimizer2.param_groups[0]['lr'] <= 1e-6:
        optimizer2.param_groups[0]['lr'] = 1e-6
    if optimizer3.param_groups[0]['lr'] <= 1e-6:
        optimizer3.param_groups[0]['lr'] = 1e-6
    if optimizer4.param_groups[0]['lr'] <= 1e-6:
        optimizer4.param_groups[0]['lr'] = 1e-6
    if optimizer5.param_groups[0]['lr'] <= 1e-6:
        optimizer5.param_groups[0]['lr'] = 1e-6

    if epoch in {80,90,100}:
        checkpoint = {
            'Dual_Domain_Encoder': Dual_Domain_Encoder.state_dict(),
            'Dual_Domain_Decoder': Dual_Domain_Decoder.state_dict(),
            'Spatial_Fuse_Layer': Spatial_Fuse_Layer.state_dict(),
            'Frequency_Fuse_Layer': Frequency_Fuse_Layer.state_dict(),
            'MC_Block': MC_Block.state_dict(),
        }
        torch.save(checkpoint, os.path.join("models/D2Fusion_" + timestamp + "_" + str(epoch) + '.pth'))





