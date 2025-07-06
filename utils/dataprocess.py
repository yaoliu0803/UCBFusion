# import python package
import os
import cv2
import numpy as np
from pathlib import Path
from skimage.io import imsave
from PIL import Image

# import external package
import torch
from torchvision import transforms
import torch.utils.data as data
import torchvision.transforms.functional as TF
from torch.nn import functional


def transform_image(is_gray=False):
    if is_gray:
        tf_list = transforms.Compose(
            [
                # transforms.ToPILImage(),
                # transforms.Resize([358, 358]),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                # transforms.Normalize((0.485, 0.456, 0.406),
                #                      (0.229, 0.224, 0.225))
            ]
        )
    else:
        tf_list = transforms.Compose(
            [
                # transforms.ToPILImage(),    #  TypeError: Unexpected type <class 'numpy.ndarray'>
                # transforms.Resize([358, 358]),
                transforms.ToTensor(),
                # transforms.Normalize((0.485, 0.456, 0.406),
                #                      (0.229, 0.224, 0.225))
            ]
        )
    return tf_list


class MMFdataset(data.Dataset):
    """
        Multi-modality (MM) image datasets
    """

    def __init__(self, images_dir_path, transform):
        super(MMFdataset, self).__init__()
        self.train_dir_prefix = images_dir_path
        # self.GFP = os.listdir(self.train_dir_prefix + 'GFP/')
        # print("gfp_path=", self.GFP)
        gfp_folder = Path(self.train_dir_prefix + 'GFP/')
        self.gfp_list = [x for x in sorted(gfp_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        pci_folder = Path(self.train_dir_prefix + 'PCI/')
        self.pci_list = [x for x in sorted(pci_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.transform = transform


    def __len__(self):
        assert len(self.gfp_list) == len(self.pci_list)
        return len(self.gfp_list)

    def __getitem__(self, idx):
        # print(self.gfp_list[idx], self.pci_list[idx])
        gfp_image_path_name = str(self.gfp_list[idx])
        pci_image_path_name = str(self.pci_list[idx])
        # gfp_name = gfp_image_path_name.split("/")[4].split("_")[1]  # window下为split("\\")，linux下为split("/")
        # pci_name = pci_image_path_name.split("/")[4].split("_")[1]
        # 交叉验证路径：
        # gfp_name = gfp_image_path_name.split("/")[5].split("_")[1]
        # pci_name = pci_image_path_name.split("/")[5].split("_")[1]
        gfp_name = gfp_image_path_name.split("/")[4]
        pci_name = pci_image_path_name.split("/")[4]
        # print("gfp_name={}  pci_name={}".format(gfp_name, pci_name))
        assert gfp_name == pci_name, f"Mismatch ir:{gfp_name} vi:{pci_name}."

        image_gfp = cv2.imread(gfp_image_path_name)
        image_gfp_resize = cv2.resize(image_gfp, (358, 358))
        gfp_color = cv2.cvtColor(image_gfp_resize, cv2.IMREAD_COLOR)
        gfp_ycrcb = cv2.cvtColor(gfp_color, cv2.COLOR_BGR2YCrCb)
        gfp_y, img_cr, img_cb = cv2.split(gfp_ycrcb)

        image_pci = cv2.imread(pci_image_path_name, cv2.IMREAD_GRAYSCALE)
        image_pci_resize = cv2.resize(image_pci, (358, 358))
        gfp_y = gfp_y[np.newaxis, ...] / 255.0
        pci_y = image_pci_resize[np.newaxis, ...] / 255.0
        data_gfp_y, data_pci = torch.FloatTensor(gfp_y), torch.FloatTensor(pci_y)

        return data_gfp_y, data_pci, img_cr, img_cb  # 1,256,256


class MIFdataset(data.Dataset):
    """
        Multi-modality (MM) image datasets
    """

    def __init__(self, images_dir_path, transform):
        super(MIFdataset, self).__init__()
        self.train_dir_prefix = images_dir_path
        image1_folder = Path(self.train_dir_prefix + 'PET/')
        self.image1_list = [x for x in sorted(image1_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        image2_folder = Path(self.train_dir_prefix + 'MRI/')
        self.image2_list = [x for x in sorted(image2_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.transform = transform

    def __len__(self):
        assert len(self.image1_list) == len(self.image2_list)
        return len(self.image1_list)

    def __getitem__(self, idx):
        image1_path_name = str(self.image1_list[idx])
        image2_path_name = str(self.image2_list[idx])
        # print("image1_path_name=", image1_path_name.split(('/'))[4])
        # print("image2_path_name=", image2_path_name.split(('/'))[4])

        # MRI_PET
        # print(image1_path_name.split('/')) # train:4 test:5
        image1_name = image1_path_name.split(('/'))[4][0:2] # window下为split("\\")，linux下为split("/")
        image2_name = image2_path_name.split(('/'))[4][0:2]
        # print("image_name=", image_name)

        assert image1_name == image2_name, f"Mismatch ir:{image1_name} vi:{image2_name}."

        image1 = cv2.imread(image1_path_name)
        image1_resize = cv2.resize(image1, (256, 256))
        image1_ycrcb = cv2.cvtColor(image1_resize, cv2.COLOR_BGR2YCrCb)
        image1_y, img_cr, img_cb = cv2.split(image1_ycrcb)

        image2 = cv2.imread(image2_path_name, cv2.IMREAD_GRAYSCALE)
        image2_resize = cv2.resize(image2, (256, 256))

        gfp_y = image1_y[np.newaxis, ...] / 255.0
        pci_y = image2_resize[np.newaxis, ...] / 255.0
        data_imag1_y, data_image2 = torch.FloatTensor(gfp_y), torch.FloatTensor(pci_y)
        # data_imag1_y = self.transform(image1_y)
        # data_image2 = self.transform(image2_resize)

        return data_imag1_y, data_image2, img_cr, img_cb  # 1,256,256

class RoadSceneDataset(data.Dataset):
    """
        Multi-modality (MM) image datasets
    """

    def __init__(self, images_dir_path, transform):
        super(RoadSceneDataset, self).__init__()
        self.train_dir_prefix = images_dir_path
        # self.GFP = os.listdir(self.train_dir_prefix + 'GFP/')
        # print("gfp_path=", self.GFP)
        gfp_folder = Path(self.train_dir_prefix + 'VIS/')
        self.gfp_list = [x for x in sorted(gfp_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        pci_folder = Path(self.train_dir_prefix + 'IR/')
        self.pci_list = [x for x in sorted(pci_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.transform = transform


    def __len__(self):
        # print(len(self.gfp_list), self.gfp_list)
        # print(len(self.pci_list), self.pci_list)
        assert len(self.gfp_list) == len(self.pci_list)
        return len(self.gfp_list)

    def __getitem__(self, idx):
        gfp_image_path_name = str(self.gfp_list[idx])
        pci_image_path_name = str(self.pci_list[idx])
        # print("gfp_image_path_name[idx={}]:{}".format(idx, gfp_image_path_name))
        # print("pci_image_path_name[idx={}]:{}".format(idx, pci_image_path_name))
        # print("gfp_name=", gfp_image_path_name.split("/"))
        gfp_name = gfp_image_path_name.split("/")[5]  # window下为split("\\")，linux下为split("/")
        pci_name = pci_image_path_name.split("/")[5]
        # print("gfp_name[idx={}]:{}".format(idx, gfp_name))
        # print("pci_name[idx={}]:{}".format(idx, pci_name))
        assert gfp_name == pci_name, f"Mismatch ir:{gfp_name} vi:{pci_name}."

        image_gfp = cv2.imread(gfp_image_path_name)
        image_gfp_resize = cv2.resize(image_gfp, (640, 480))
        gfp_color = cv2.cvtColor(image_gfp_resize, cv2.IMREAD_COLOR)
        gfp_ycrcb = cv2.cvtColor(gfp_color, cv2.COLOR_BGR2YCrCb)
        gfp_y, img_cr, img_cb = cv2.split(gfp_ycrcb)

        image_pci = cv2.imread(pci_image_path_name, cv2.IMREAD_GRAYSCALE)
        image_pci_resize = cv2.resize(image_pci, (640, 480))
        gfp_y = gfp_y[np.newaxis, ...] / 255.0
        pci_y = image_pci_resize[np.newaxis, ...] / 255.0
        data_gfp_y, data_pci = torch.FloatTensor(gfp_y), torch.FloatTensor(pci_y)

        return data_gfp_y, data_pci, img_cr, img_cb  # 1,256,256

class Testdataset(data.Dataset):
    """
        需要根据测试图像文件夹修改名字
    """
    def __init__(self, images_dir_path, transform, resize_W, resize_H):
        super(Testdataset, self).__init__()
        self.resize_W = resize_W
        self.resize_H = resize_H
        self.transform = transform
        self.test_dir_prefix = images_dir_path
        image1_folder = Path(self.test_dir_prefix + 'GFP/')
        self.image1_list = [x for x in sorted(image1_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        image2_folder = Path(self.test_dir_prefix + 'PCI/')
        self.image2_list = [x for x in sorted(image2_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.transform = transform
        self.image_name_list = [str(x) for x in sorted(image1_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

    def __len__(self):
        assert len(self.image1_list) == len(self.image2_list)
        return len(self.image1_list)

    def __getitem__(self, idx):
        # GFP_PCI
        image_name = str(self.image_name_list[idx]).split("/")[4][0:6]
        # MRI_PET
        # image_name = str(self.image_name_list[idx]).split("/")[5].split(".")[0]
        # print("image_name=", image_name)
        image1_path_name = str(self.image1_list[idx])
        image2_path_name = str(self.image2_list[idx])
        # 彩色3通道图像处理(RGB->YCrCb)
        image1 = cv2.imread(image1_path_name)
        image1_resize = cv2.resize(image1, (self.resize_W, self.resize_H))
        image1_color = cv2.cvtColor(image1_resize, cv2.IMREAD_COLOR)
        image1_ycrcb = cv2.cvtColor(image1_color, cv2.COLOR_BGR2YCrCb)
        img1_y, img1_cr, img1_cb = cv2.split(image1_ycrcb)
        # 单通道图像处理方式(转灰度图)
        image2_gray = cv2.imread(image2_path_name, cv2.IMREAD_GRAYSCALE)
        image2_resize = cv2.resize(image2_gray, (self.resize_W, self.resize_H))
        # 将图像转为张量类型
        img1_y = self.transform(img1_y)
        img2 = self.transform(image2_resize)
        return img1_y, img2, img1_cr, img1_cb, image_name  # img1[YCrCb]:3,256,256  img2[Gray]:1,256,256


# https://github.com/Zhaozixiang1228/MMIF-CDDFuse/blob/main/utils/img_read_save.py
# def image_read_cv2(path, resize_W, resize_H, mode='RGB'):
#     assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
#     if mode == 'RGB':
#         img_BGR = cv2.imread(path).astype('float32')
#         img_BGR = cv2.resize(img_BGR, (resize_W, resize_H))
#         img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
#     elif mode == 'GRAY':
#         img_gary = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         img = cv2.resize(img_gary, (resize_W, resize_H))
#     elif mode == 'YCrCb':
#         img_BGR = cv2.imread(path).astype('float32')
#         img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
#     return img

def image_read_cv2(path, resize_W, resize_H, mode='RGB'):
    # img_BGR = cv2.imread(path).astype('float32')
    # print("path=", path)
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    img_BGR = cv2.imread(path)
    img_BGR = cv2.resize(img_BGR, (resize_W, resize_H))
    if img_BGR is None:
        print("Error: Unable to load image.")
    else:
        if mode == 'RGB':
            img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        elif mode == 'GRAY':
            img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
        elif mode == 'YCrCb':
            img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
        return img


def img_save(image, imagename, savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    imsave(os.path.join(savepath, "{}.jpg".format(imagename)), image)

def tensor_to_PIL(img_tensor, dirpath, filename):
    dirname = dirpath + os.sep
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for b in range(img_tensor.shape[0]):
        single_img_tensor = img_tensor[b]
        img = (single_img_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img[0])
        savename = dirname + filename + "_{}.jpg".format(b)
        img.save(savename)
        print("save img result = ", savename)

# https://github.com/wyh1210/BFFR/blob/master/mmseg/models/fuser/transfuser6.py
def RGB2YCrCb(input_im, device):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).to(device)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def YCrCb2RGB(input_im, device):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(device)
    temp = (im_flat + bias).mm(mat).to(device)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

# https://github.com/draymondbiao/LE2Fusion/blob/main/data_loader/common.py
def clamp(value, min=0., max=1.0):
    """
    将像素值强制约束在[0,1], 以免出现异常斑点
    :param value:
    :param min:
    :param max:
    :return:
    """
    return torch.clamp(value, min=min, max=max)


# initial VGG16 network
# https://github.com/hli1221/imagefusion-LRRNet/blob/main/utils.py#L104
def init_vgg16(vgg, model_dir):
	vgg_load = torch.load(model_dir)
	count = 0
	for name, param in vgg_load.items():
		if count >= 20:
			break
		if count == 0:
			vgg.conv1_1.weight.data = param
		if count == 1:
			vgg.conv1_1.bias.data = param
		if count == 2:
			vgg.conv1_2.weight.data = param
		if count == 3:
			vgg.conv1_2.bias.data = param

		if count == 4:
			vgg.conv2_1.weight.data = param
		if count == 5:
			vgg.conv2_1.bias.data = param
		if count == 6:
			vgg.conv2_2.weight.data = param
		if count == 7:
			vgg.conv2_2.bias.data = param

		if count == 8:
			vgg.conv3_1.weight.data = param
		if count == 9:
			vgg.conv3_1.bias.data = param
		if count == 10:
			vgg.conv3_2.weight.data = param
		if count == 11:
			vgg.conv3_2.bias.data = param
		if count == 12:
			vgg.conv3_3.weight.data = param
		if count == 13:
			vgg.conv3_3.bias.data = param

		if count == 14:
			vgg.conv4_1.weight.data = param
		if count == 15:
			vgg.conv4_1.bias.data = param
		if count == 16:
			vgg.conv4_2.weight.data = param
		if count == 17:
			vgg.conv4_2.bias.data = param
		if count == 18:
			vgg.conv4_3.weight.data = param
		if count == 19:
			vgg.conv4_3.bias.data = param
		count = count + 1
