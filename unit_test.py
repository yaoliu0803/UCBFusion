# import python package
import numpy as np
import os
from PIL import Image
import cv2

# import external package
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import kornia

# import internal package.
from model import DRFE_Module
from utils.dataprocess import img_save


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



def unit_test():
    gfp_image_path_name = "datasets/test_images/Medical/MRI_PET/PET/32.png"
    pci_image_path_name = "datasets/test_images/Medical/MRI_PET/MRI/32.png"
    # gfp_image_path_name = "datasets/train_images/GFP/04-C12-g.jpg"
    # pci_image_path_name = "datasets/train_images/PCI/04-C12-t.jpg"

    image_gfp = cv2.imread(gfp_image_path_name)
    gfp_ycrcb = cv2.cvtColor(image_gfp, cv2.COLOR_BGR2YCrCb)
    gfp_y, img_cr, img_cb = cv2.split(gfp_ycrcb)
    print("gfp_y={}, img_cr={}, img_cb={}".format(gfp_y.shape, img_cr.shape, img_cb.shape))

    # PCI = cv2.imread(pci_image_path_name)
    # pci_y = cv2.cvtColor(PCI, cv2.COLOR_BGR2GRAY)
    pci_y = cv2.imread(pci_image_path_name, cv2.IMREAD_GRAYSCALE)

    # tran = transforms.ToTensor()
    # gfp_y = tran(gfp_y).unsqueeze(0);
    # pci_y = tran(pci_y).unsqueeze(0);
    gfp_y = gfp_y[np.newaxis,np.newaxis, ...] / 255.0
    pci_y = pci_y[np.newaxis,np.newaxis, ...] / 255.0
    image1, image2 = torch.FloatTensor(gfp_y), torch.FloatTensor(pci_y)
    # tensor_to_PIL(image1, "./test/feature_map", "np_unit_test_pet_0314")
    # tensor_to_PIL(image2, "./test/feature_map", "np_unit_test_mri_0314")
    # image1 = gfp_y
    # image2 = pci_y
    print(image1.shape, image2.shape)

    model = DRFE_Module()
    # print(models)
    feature_out = model(image1.cpu(), image2.cpu())
    tensor_to_PIL(image2, "./test/feature_map", "np_unit_test_mri_0316")
    print("\n feature_eca shape={}".format(feature_out.shape))

    # tensor_to_PIL(feature_out, "./test/feature_map", "Image_ECAAttention_feature_out")
    # print(gfp_ycrcb[:, 2:, :, :].shape)
    # fusion_ycrcb = torch.cat(
    #     (feature_out.cpu(), gfp_ycrcb[:, 1:2, :, :].cpu(),
    #      gfp_ycrcb[:, 2:, :, :].cpu()),
    #     dim=1,
    # )

    # 保存结果方法1
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print("\ngfp_ycrcb =", gfp_ycrcb.shape)
    # fusion_ycrcb = torch.cat(
    #     (feature_out.cpu(), gfp_ycrcb[:, 1:2, :, :].cpu(),
    #      gfp_ycrcb[:, 2:, :, :].cpu()),
    #     dim=1,
    # )
    # fusion_image = YCrCb2RGB(fusion_ycrcb)
    # print(fusion_image.shape)
    #
    # image = fusion_image.squeeze().permute(1,2,0).cpu().detach().numpy()
    # print(image.shape)
    # image = Image.fromarray((image*255).astype(np.uint8))

    path_name = os.path.join("./test/", "feature_map/")
    if not os.path.exists(path_name):
        os.mkdir(path_name)
    # save_path = os.path.join(path_name, "04-C12-Image_aaaa_y.jpg")
    # image.save(save_path)

    # 保存结果方法2[输出粉色图像]
    # fused_img_Y = (feature_out - feature_out.min()) / (feature_out.max() - feature_out.min()) * 255
    # fused_img_Y = fused_img_Y.cpu().detach().numpy().squeeze(0)
    # print("fused_img_Y ={}, img1_CrCb={}".format(fused_img_Y.shape, img1_CrCb.shape))
    # fused_img = np.concatenate((fused_img_Y, img1_CrCb), axis=0)
    # print("\n fused_img=", fused_img.shape)
    # fused_img = np.transpose(fused_img, (1, 2, 0))
    # fused_img = (fused_img*255).astype(np.uint8)
    # fused_img = cv2.cvtColor(fused_img, cv2.COLOR_YCrCb2BGR)
    # cv2.imwrite(path_name+"04-C12-Image_2_YCrCb2BGR.jpg", fused_img),

    # 保存结果方法3[输出绿色图像√]
    # img_out_y = feature_out.cpu().detach().numpy()
    # print(img_out_y.shape) # (1, 1, 358, 358)
    # print(img_out_y[0].shape) # (1, 358, 358)
    # fused_img_Y = np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0])
    # print("\n fused_img_Y={} img_cr={}, img_cb={}".format(fused_img_Y.shape, img_cr.shape, img_cb.shape)) # fused_img_Y=(358, 358) img_cr=(358, 358), img_cb=(358, 358)
    # final_img = cv2.merge([fused_img_Y, img_cr, img_cb])  # YCrCb
    # final_img = cv2.cvtColor(final_img, cv2.COLOR_YCrCb2BGR)  # BGR
    # final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)  # RGB
    # cv2.imwrite(path_name + "MSRS_cv2.jpg", final_img)

    # 保存结果方法4[输出彩色图像√]
    img_out_y = feature_out.cpu().detach().numpy()
    fused_img_Y = np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0])
    ycrcb_fi = np.dstack((fused_img_Y, img_cr, img_cb))
    rgb_fi = cv2.cvtColor(ycrcb_fi, cv2.COLOR_YCrCb2RGB)
    print("rgb_fi=", rgb_fi.shape)
    img_save(rgb_fi, "MRI-PET-0316", path_name)

    print('Fusion {0} Sucessfully!'.format(path_name))

    # print("feature1_out shape={}, feature2_out shape={}".format(feature1_out.shape, feature2_out.shape))
    # y = models(image1, image2)
    # print('output shape:', y.shape)
    # assert y.shape == (1, 1, 358, 358), 'output shape (1,1,358, 358) is expected!'
    print('test ok!')

# if __name__ == "__main__":
    # unit_test()
