# import python package
import argparse
import os

# import external package
import torch

"""
    This script defines the procedure to parse the parameters
    ### Setup Main Environment
            Python >= 3.7
            PyTorch >= 1.4.0 + cu102 is recommended
            opencv-python = 4.5.1
            matplotlib
            pytorch_msssim
"""


def INFO(string):
    """
        1.融合结果的边缘细节得到细化，噪声得到有效抑制
        2.交叉注意融合（CAF）块，它通过交换键和查询值自适应地融合空间和频率域中两种模态的特征，然后计算空间和频率特征之间的交叉注意力分数，以进一步指导空间频率信息融合。CAF 块增强了不同模态的高频特征，从而可以保留融合图像中的细节
        3.不仅关注图像的共同特征，还注重图像的互补信息
    """

    print("[ Detail-Refinement-Enhance Fusion ] %s" % (string))


def presentParameters(args_dict):
    """
        Print the parameters setting line by line
        
        Arg:    args_dict   - The dict object which is transferred from argparse Namespace object
    """
    INFO("==================== Parameters ====================")
    for key in sorted(args_dict.keys()):
        INFO("{:>15} : {}".format(key, args_dict[key]))
    INFO("====================================================")


class TrainOptions:
    """
                                                    Argument Explaination
        ======================================================================================================================
                Symbol          Type            Default                         Explaination
        ----------------------------------------------------------------------------------------------------------------------
            --folder            Str         /datasets/train_images/         The folder path of train image path
            --seed              Int         3479                              random seed for training
            --lr                Int         1e-4                            learning_rate, default is 0.001
            --epoch             Int         200                             train number, default is 200
            --epoch_gap         Int         50                              epoches of Phase I, default is 50
            --batch_size        Int         10                               -
            --is_gray           bool        False                           Color (False) or Gray (True)
            --log_dir           Str         /train_log                      -
            --crop_size         Int         256                             input image size
            --log_interval      Int         50                              number of images after which the training loss is logged
            --resume            Str         None                            The path of pre-trained models
            --train_result      Str         /train_result                    The path of folder you want to store the result in
            --record_epoch      Int         50                              The period you want to store the result
        ----------------------------------------------------------------------------------------------------------------------
    """

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--folder', type=str, default="datasets/dref_20240611/")
        # parser.add_argument('--folder', type=str, default="datasets/train_images/PET_MRI/")
        # parser.add_argument('--folder', type=str, default="datasets/dref_20240611/")
        parser.add_argument('--seed', type=int, default=1314)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--epoch', type=int, default=200)
        parser.add_argument('--epoch_gap', type=int, default=200)
        parser.add_argument('--batch_size', type=int, default=13)
        parser.add_argument('--is_gray', type=bool, default=False)
        parser.add_argument('--log_interval', type=int, default=100)
        parser.add_argument('--log_dir', type=str, default='train_log/')
        parser.add_argument('--resume', type=str, default="None")
        parser.add_argument('--train_result', type=str, default="train_result/tiao/dfeb_cafb_recon")
        # parser.add_argument('--train_result', type=str, default="train_result/ablation_studies")
        parser.add_argument('--record_epoch', type=int, default=200)
        self.opts = parser.parse_args()
        self.opts.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.opts.device = 'cpu'

    def parse(self):
        # Print the parameter first
        presentParameters(vars(self.opts))

        # Create the folder
        det_name = self.opts.train_result
        image_folder_name = os.path.join(det_name, "image")
        model_folder_name = os.path.join(det_name, "checkpoint")
        if not os.path.exists(self.opts.log_dir):
            os.mkdir(self.opts.log_dir)
        if not os.path.exists(self.opts.train_result):
            os.mkdir(self.opts.train_result)
        if not os.path.exists(image_folder_name):
            os.mkdir(image_folder_name)
        if not os.path.exists(model_folder_name):
            os.mkdir(model_folder_name)

        return self.opts


###########################################################################################################################################
class TestOptions:
    """
                                                    Argument Explaination
        ======================================================================================================================
                Symbol          Type            Default                                 Explaination
        ----------------------------------------------------------------------------------------------------------------------
            --image1            Str         /datasets/test_images/image1       The path of RGB image
            --image2            Str         /datasets/test_images/image2       The path of Gray image
            --folder            Str         /datasets/test_images/             The folder path of test image path
            --models             Str         models.pth                        The path of pre-trained models
            --is_gray           bool        False                              Color (False) or Gray (True)
            --batch_size        Int         1                                  -(Test one picture at a time)
            --crop_size         Int         256                                input image size
            --eval_folder       Str         /test_result                       The path to store the fusing image
            --eval_log_path     Str         /eval_log                          The path of folder you want to evaluator the result
            --H                 Int         358                                The height of the result image
            --W                 Int         358                                The width of the result image
        ----------------------------------------------------------------------------------------------------------------------
    """

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--image1', type=str, required=False)
        parser.add_argument('--image2', type=str, required=False)
        # parser.add_argument('--folder', type=str, default="datasets/cross_validation_GFP_PCI/fold_1/test/")
        # parser.add_argument('--folder', type=str, default="datasets/test_images/drefusion_20240611/GFP_PCI/")
        parser.add_argument('--folder', type=str, default="datasets/test_images/show_0826/GFP_PCI/")
        # parser.add_argument('--folder', type=str, default="datasets/test_images/Medical/")
        # parser.add_argument('--folder', type=str, default="datasets/test_images/RoadScene/")
        # parser.add_argument('--model', type=str, default="train_result/checkpoint/20240325/model_200_GFP_PCI_latest_20240325_1703.pth") # [0.5, 0.5]
        # parser.add_argument('--model', type=str, default="train_result/checkpoint/20240326/model_200_GFP_PCI_latest_20240326_0913.pth") # [0.7, 0.3]
        # parser.add_argument('--model', type=str, default="train_result/checkpoint/20240426/model_200_GFP_PET_latest_20240426_1526.pth")  # [0.9, 0.1]
        # parser.add_argument('--model', type=str, default="train_result/checkpoint/20240422/model_200_GFP_PCI_latest_20240422_1113.pth")  # [0.9, 0.1]
        # parser.add_argument('--model', type=str, default="train_result/checkpoint/20240425/model_200_PET_MRI_latest_20240425_2001.pth")  # [0.5, 0.5]
        # parser.add_argument('--model', type=str, default="train_result/checkpoint/20240427/model_200_MSRS_VISIR_latest_20240427_1927.pth")  # [0.6, 0.4](500,329)
        # parser.add_argument('--model', type=str, default="train_result/checkpoint/20240427/model_200_MSRS_VISIR_latest_20240427_2240.pth")  # [0.7,0.3]  (600,480)
        # parser.add_argument('--model', type=str, default="train_result/checkpoint/20240428/model_200_MSRS_VISIR_latest_20240428_0030.pth")  # [0.5,0.5]  (600,480)
        # parser.add_argument('--model', type=str, default="train_result/tiao/checkpoint/20240613/model_200_GFP_PCI_latest_20240613_0947.pth")
        # parser.add_argument('--model', type=str,default="train_result/tiao/checkpoint/20240702/model_200_GFP_PCI_latest_20240702_1012.pth")
        parser.add_argument('--model', type=str,
                            default="train_result/tiao/dfeb_cafb_recon/checkpoint/20240722/model_200_GFP_PCI_latest_20240722_2032.pth")
        parser.add_argument('--is_gray', type=bool, default=False)
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--eval_folder', type=str, default='test_result/tiao/dfeb_cafb_recon/')
        # parser.add_argument('--eval_folder', type=str, default='test_result/ablation_studies')
        parser.add_argument('--eval_log_path', type=str, default='eval_log/')
        parser.add_argument('--W', type=int, default=358)
        parser.add_argument('--H', type=int, default=358)  # GFP_PCI=358/MRI_ET=256/Road(640,480)
        self.opts = parser.parse_args()
        self.opts.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def parse(self):
        presentParameters(vars(self.opts))

        # Create the folder
        if not os.path.exists(self.opts.folder):
            os.mkdir(self.opts.folder)
        if not os.path.exists(self.opts.eval_folder):
            os.mkdir(self.opts.eval_folder)
        if not os.path.exists(self.opts.eval_log_path):
            os.mkdir(self.opts.eval_log_path)

        return self.opts
