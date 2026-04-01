# coding=gbk
from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureFusion,FIBlock,Interaction
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

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
ckpt_path=r"models/CDDFuse_09-09-16-02_110.pth"
for dataset_name in ["MRI_CT","MRI_PET","MRI_SPECT"]:
    print("\n"*2+"="*80)
    model_name="CDDFuse    "
    print("The test result of "+dataset_name+' :')
    test_folder=os.path.join('test_img',dataset_name)

    # 定义正则表达式模式
    date_pattern = r'(\d{2}-\d{2}-\d{2}-\d{2})'
    # 使用正则表达式搜索模式
    match = re.search(date_pattern, ckpt_path)
    data_str = match.group(1)

    test_out_folder = os.path.join('test_result', dataset_name+ data_str)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
    DetailFuseLayer = nn.DataParallel(DetailFeatureFusion(dim=64)).to(device)
    BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64)).to(device)
    ###########
    FIB = FIBlock().to(device)
    SCB=nn.DataParallel(Interaction(channels=64)).to(device)#space compensate block


    Encoder.load_state_dict(torch.load(ckpt_path)['DIDF_Encoder'])
    Decoder.load_state_dict(torch.load(ckpt_path)['DIDF_Decoder'])
    BaseFuseLayer.load_state_dict(torch.load(ckpt_path)['BaseFuseLayer'])
    DetailFuseLayer.load_state_dict(torch.load(ckpt_path)['DetailFuseLayer'])
    SCB.load_state_dict(torch.load(ckpt_path)['SCB'])
    Encoder.eval()
    Decoder.eval()
    BaseFuseLayer.eval()
    DetailFuseLayer.eval()
    SCB.eval()

    with torch.no_grad():
        for img_name in os.listdir(os.path.join(test_folder,dataset_name.split('_')[0])):

            data_IR=image_read_cv2(os.path.join(test_folder,dataset_name.split('_')[1],img_name),mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0
            data_VIS = image_read_cv2(os.path.join(test_folder,dataset_name.split('_')[0],img_name), mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0

            data_IR,data_VIS = torch.FloatTensor(data_IR),torch.FloatTensor(data_VIS)
            data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()

            # feature_V_B, feature_V_D, feature_V = Encoder(data_VIS)
            # feature_I_B, feature_I_D, feature_I = Encoder(data_IR)
            # feature_F_B = BaseFuseLayer(feature_V_B,feature_I_B)
            # feature_F_D = DetailFuseLayer(feature_V_D ,feature_I_D)
            # data_Fuse, _ = Decoder(data_VIS, feature_F_B, feature_F_D)
            feature_V_B, feature_V_D, feature_V, vis_amp, vis_pha = Encoder(data_VIS)
            feature_I_B, feature_I_D, feature_I, ir_amp, ir_pha = Encoder(data_IR)
            fused_a = FIB(vis_amp, vis_pha, ir_pha, data_VIS)

            feature_F_B = BaseFuseLayer(feature_I_B+feature_V_B)
            feature_F_D = DetailFuseLayer(feature_I_D, feature_V_D)
            fused_y = SCB(fused_a, feature_F_B, feature_F_D)
            #data_Fuse, feature_F = Decoder(data_VIS, feature_F_B, feature_F_D, fused_amp)
            data_Fuse, feature_F = Decoder(data_IR+data_VIS, fused_y)

            data_Fuse=(data_Fuse-torch.min(data_Fuse))/(torch.max(data_Fuse)-torch.min(data_Fuse))
            fi = np.squeeze((data_Fuse * 255).cpu().numpy())
            fi = fi.astype(np.uint8)  # 新加的
            img_save(fi, img_name.split(sep='.')[0], test_out_folder)


    eval_folder=test_out_folder
    ori_img_folder=test_folder
    results = []

    metric_result = np.zeros((12))
    for img_name in os.listdir(os.path.join(ori_img_folder,dataset_name.split('_')[0])):
            ir = image_read_cv2(os.path.join(ori_img_folder,dataset_name.split('_')[1], img_name), 'GRAY')
            vi = image_read_cv2(os.path.join(ori_img_folder,dataset_name.split('_')[0], img_name), 'GRAY')
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

            # 将结果保存为DataFrame
    df = pd.DataFrame(results)

    #保存为CSV文件
    output_csv = dataset_name + 'metrics_results.csv'
    df.to_csv(output_csv, index=False)

    print("Metrics for each image saved to", output_csv)

    metric_result = df.mean(numeric_only=True)
    print("\t\tEN\tSD\tSF\tMI\tSCD\tVIF\tQabf\tSSIM\tAG\tPSNR\tCC\tMSE")
    print(model_name + '\t' + str(np.round(metric_result["EN"], 2)) + '\t'
          + str(np.round(metric_result["SD"], 2)) + '\t'
          + str(np.round(metric_result["SF"], 2)) + '\t'
          + str(np.round(metric_result["MI"], 2)) + '\t'
          + str(np.round(metric_result["SCD"], 2)) + '\t'
          + str(np.round(metric_result["VIFF"], 2)) + '\t'
          + str(np.round(metric_result["Qabf"], 2)) + '\t'
          + str(np.round(metric_result["SSIM"], 2)) + '\t'
          + str(np.round(metric_result["AG"], 2)) + '\t'
          + str(np.round(metric_result["PSNR"], 2)) + '\t'
          + str(np.round(metric_result["CC"], 2)) + '\t'
          + str(np.round(metric_result["MSE"], 2))
          )
    print("=" * 80)