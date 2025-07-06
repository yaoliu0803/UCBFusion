# import python package
import os
import numpy as np
import warnings
import logging
from PIL import Image
import datetime

# import external package
import torch
import torch.nn as nn

# import internal package
from opts import TestOptions
from utils.Evaluator import Evaluator
from utils.dataprocess import img_save, image_read_cv2

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CDDFuse_path = r"models/CDDFuse_IVF.pth"
CDDFuse_MIF_path = r"models/CDDFuse_MIF.pth"


# for dataset_name in ["MRI_CT", "MRI_PET", "MRI_SPECT"]:


def Eval(opts):
    # for dataset_name in ["MRI_PET"]:
    for dataset_name in ["GFP_PCI"]:
        print("\n" * 2 + "=" * 80)
        print("The test result of " + dataset_name + " :")
        print("\t\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM")
        for ckpt_path in [CDDFuse_path, CDDFuse_MIF_path]:
            model_name = "20240330_0946"
            # model_name = ckpt_path.split('/')[-1].split('.')[0]
            test_folder = os.path.join('./datasets/cross_validation_GFP_PCI/fold_1/test')
            # print("test_folder=", test_folder)
            # test_out_folder = os.path.join('test_result', model_name)
            test_out_folder = os.path.join('./test_result/GFP_PCI/', model_name)
            # test_out_folder = os.path.join('test_result', dataset_name)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            eval_folder = test_out_folder
            ori_img_folder = test_folder
            resize_W = opts.W
            resize_H = opts.H

            ######################## Evaluator ########################
            metric_result = np.zeros((8))
            # log
            # eval_log_path = opts.eval_log_path + 'log_{}.txt'.format(datetime.datetime.now().strftime("%Y%m%d"))
            # # print("eval_log = ", eval_log_path)
            # data = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # with open(eval_log_path, "a") as f:
            #     # 记录评估日期
            #     write_info = f"Data: {data} \t Model_name: {model_name} \n"
            #     f.write(write_info)

            for img_name in os.listdir(os.path.join(ori_img_folder, dataset_name.split('_')[0])):
                img_name = img_name.split("_")[1]
                # print("img_name=", img_name)

                ir = image_read_cv2(
                    os.path.join(ori_img_folder, dataset_name.split('_')[1], "PCI_" + img_name),
                    resize_W, resize_H, 'GRAY')
                vi = image_read_cv2(
                    os.path.join(ori_img_folder, dataset_name.split('_')[0], "GFP_" + img_name),
                    resize_W, resize_H, 'GRAY')
                # fi_path = os.path.join(eval_folder, img_name.split('.')[0]+".png")
                # print("fi_path=", fi_path)
                fi = image_read_cv2(os.path.join(eval_folder, "GFP_" + img_name.split(".")[0] + ".png"), resize_W,
                                    resize_H, 'GRAY')
                metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
                                              , Evaluator.SF(fi), Evaluator.MI(fi, ir, vi)
                                              , Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi)
                                              , Evaluator.Qabf(fi, ir, vi), Evaluator.SSIM(fi, ir, vi)])

                # metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
                #                               , Evaluator.SF(fi), Evaluator.AG(fi), Evaluator.MI(fi, ir, vi)
                #                               , Evaluator.MSE(fi, ir, vi), Evaluator.CC(fi, ir, vi)
                #                               , Evaluator.PSNR(fi, ir, vi), Evaluator.SCD(fi, ir, vi)
                #                               , Evaluator.VIFF(fi, ir, vi), Evaluator.SSIM(fi, ir, vi)])

            metric_result /= len(os.listdir(eval_folder))

            # Save info to txt file
            # with open(eval_log_path, 'a') as f:
            #     # 记录各指标
            #     write_info = f"[img_name: {img_name}] EN: {np.round(metric_result[0], 2):.2f} SD: {np.round(metric_result[1], 2):.2f}" \
            #                  f"SF: {np.round(metric_result[2], 2):.2f} MI: {np.round(metric_result[3], 2):.2f} SCD: {np.round(metric_result[4], 2):.2f}" \
            #                  f"VIF: {np.round(metric_result[5], 2):.2f} Qabf: {np.round(metric_result[6], 2):.2f} SSIM: {np.round(metric_result[7], 2):.2f} \n"
            #     f.write(write_info)

            print(model_name + '\t' + str(np.round(metric_result[0], 2)) + '\t'
                  + str(np.round(metric_result[1], 2)) + '\t'
                  + str(np.round(metric_result[2], 2)) + '\t'
                  + str(np.round(metric_result[3], 2)) + '\t'
                  + str(np.round(metric_result[4], 2)) + '\t'
                  + str(np.round(metric_result[5], 2)) + '\t'
                  + str(np.round(metric_result[6], 2)) + '\t'
                  + str(np.round(metric_result[7], 2))
                  )
        print("=" * 80)


if __name__ == '__main__':
    opts = TestOptions().parse()
    Eval(opts)
