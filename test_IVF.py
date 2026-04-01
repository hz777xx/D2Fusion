from net import Encoder, Decoder,SpatialFeatureFusion,FrequencyFeatureFusion,FIBlock,MCBlock
import os
import numpy as np
from utils.Evaluator import Evaluator
import torch
import torch.nn as nn
from utils.img_read_save import img_save,image_read_cv2
import warnings
import logging
import pandas as pd
import re

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ckpt_path=r"models/CDDFuse_09-09-16-02_100.pth"
for dataset_name in ["MSRS"]:
    print("\n"*2+"="*80)
    model_name="D2Fusion    "
    print("The test result of "+dataset_name+' :')
    test_folder=os.path.join('test_img',dataset_name)

    # 定义正则表达式模式
    date_pattern = r'(\d{2}-\d{2}-\d{2}-\d{2})'
    # 使用正则表达式搜索模式
    match = re.search(date_pattern, ckpt_path)
    data_str = match.group(1)

    test_out_folder = os.path.join('test_result', dataset_name+ data_str)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Dual_Domain_Encoder = nn.DataParallel(Encoder()).to(device)
    Dual_Domain_Decoder = nn.DataParallel(Decoder()).to(device)
    Spatial_Fuse_Layer = nn.DataParallel(SpatialFeatureFusion(dim=64)).to(device)
    Frequency_Fuse_Layer = nn.DataParallel(FrequencyFeatureFusion(dim=64)).to(device)

    ###########
    FI_Block = FIBlock().to(device)
    MC_Block=nn.DataParallel(MCBlock(channels=64)).to(device)#space compensate block


    Dual_Domain_Encoder.load_state_dict(torch.load(ckpt_path)['Dual_Domain_Encoder'])
    Dual_Domain_Decoder.load_state_dict(torch.load(ckpt_path)['Dual_Domain_Decoder'])
    Spatial_Fuse_Layer.load_state_dict(torch.load(ckpt_path)['Spatial_Fuse_Layer'])
    Frequency_Fuse_Layer.load_state_dict(torch.load(ckpt_path)['Frequency_Fuse_Layer'])
    MC_Block.load_state_dict(torch.load(ckpt_path)['MC_Block'])
    Dual_Domain_Encoder.eval()
    Dual_Domain_Decoder.eval()
    Spatial_Fuse_Layer.eval()
    Frequency_Fuse_Layer.eval()
    MC_Block.eval()

    with torch.no_grad():
        for img_name in os.listdir(os.path.join(test_folder,"ir")):

            data_IR=image_read_cv2(os.path.join(test_folder,"ir",img_name),mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0
            data_VIS = image_read_cv2(os.path.join(test_folder,"vi",img_name), mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0

            data_IR,data_VIS = torch.FloatTensor(data_IR),torch.FloatTensor(data_VIS)
            data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()

            feature_V_B, feature_V_D, feature_V, vis_amp, vis_pha = Dual_Domain_Encoder(data_VIS)
            feature_I_B, feature_I_D, feature_I, ir_amp, ir_pha = Dual_Domain_Encoder(data_IR)
            fused_a = FI_Block(vis_amp, vis_pha, ir_pha, data_VIS)

            feature_F_B = Spatial_Fuse_Layer(feature_I_B+feature_V_B)
            feature_F_D = Frequency_Fuse_Layer(feature_I_D, feature_V_D)
            fused_y = MC_Block(fused_a, feature_F_B, feature_F_D)
            data_Fuse, feature_F = Dual_Domain_Decoder(data_VIS, fused_y)

            data_Fuse=(data_Fuse-torch.min(data_Fuse))/(torch.max(data_Fuse)-torch.min(data_Fuse))
            fi = np.squeeze((data_Fuse * 255).cpu().numpy())
            fi = fi.astype(np.uint8)
            img_save(fi, img_name.split(sep='.')[0], test_out_folder)


    eval_folder=test_out_folder
    ori_img_folder=test_folder
    results = []

    for img_name in os.listdir(os.path.join(ori_img_folder,"ir")):
            ir = image_read_cv2(os.path.join(ori_img_folder,"ir", img_name), 'GRAY')
            vi = image_read_cv2(os.path.join(ori_img_folder,"vi", img_name), 'GRAY')
            fi = image_read_cv2(os.path.join(eval_folder, img_name.split('.')[0]+".png"), 'GRAY')

            metrics = {
                "Image Name": img_name,
                "EN": Evaluator.EN(fi),
                "SD": Evaluator.SD(fi),
                "SF": Evaluator.SF(fi),
                "MI": Evaluator.MI(fi, ir, vi),
                "SCD": Evaluator.SCD(fi, ir, vi),
                "VIFF": Evaluator.VIFF(fi, ir, vi),
                "Qabf": Evaluator.Qabf(fi, ir, vi),
                "SSIM": Evaluator.SSIM(fi, ir, vi),
                "AG": Evaluator.AG(fi),  # 添加AG
                "PSNR": Evaluator.PSNR(fi, ir, vi),  # 添加PSNR
                "CC": Evaluator.CC(fi, ir, vi),  # 添加CC
                "MSE": Evaluator.MSE(fi, ir, vi)  # 添加MSE
            }
            results.append(metrics)

    df = pd.DataFrame(results)
    metric_result = df.mean(numeric_only=True)
    print("\t\tEN\tSD\tSF\tMI\tSCD\tVIF\tQabf\tSSIM\tAG\tPSNR\tCC\tMSE")
    print(model_name + '\t' + str(np.round(metric_result["EN"], 4)) + '\t'
          + str(np.round(metric_result["SD"], 4)) + '\t'
          + str(np.round(metric_result["SF"], 4)) + '\t'
          + str(np.round(metric_result["MI"],4)) + '\t'
          + str(np.round(metric_result["SCD"], 4)) + '\t'
          + str(np.round(metric_result["VIFF"], 4)) + '\t'
          + str(np.round(metric_result["Qabf"], 4)) + '\t'
          + str(np.round(metric_result["SSIM"], 4)) + '\t'
          + str(np.round(metric_result["AG"], 4)) + '\t'
          + str(np.round(metric_result["PSNR"], 4)) + '\t'
          + str(np.round(metric_result["CC"], 4)) + '\t'
          + str(np.round(metric_result["MSE"], 4))
          )
    print("=" * 80)