# import python package
import numpy as np
import datetime
import time
import cv2
import os

# import external package
import torch
from torch.utils.data import DataLoader

# import internal package
from opts import TestOptions
from model import DRFE_Module
from utils.dataprocess import transform_image, MIFdataset, MMFdataset, RoadSceneDataset, image_read_cv2, img_save, tensor_to_PIL


# https://github.com/Zhaozixiang1228/MMIF-CDDFuse/blob/main/test_IVF.py
def inference(opts):
    # data loader
    # dataset_name = "VIS_IR"
    # dataset_name = "MRI_PET"
    dataset_name = "GFP_PCI"
    # image_dir_path = opts.folder + "{}/".format(dataset_name)
    image_dir_path = opts.folder
    print("\nimage_dir_path=", image_dir_path)
    assert os.path.exists(image_dir_path), "{} path does not exist.".format(image_dir_path)
    eval_folder = opts.eval_folder
    print("\nimage_dir_path=", eval_folder)
    assert os.path.exists(eval_folder), "{} path does not exist.".format(eval_folder)
    # transform = transform_image(is_gray=opts.is_gray)
    # test_dataset = MMFdataset(images_dir_path=image_dir_path, transform=transform)  # MIFdataset, MMFdataset, RoadSceneDataset
    # test_num = len(test_dataset)
    # print("test_image_number = ", test_num)
    # test_loader = DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=0)

    # Load the pre-trained models
    print('Loading pre-trained models......')
    model = DRFE_Module().to(opts.device)
    checkpoint = torch.load(opts.model)
    model.load_state_dict(checkpoint['models'])
    model.to(opts.device)
    print("====>" + opts.model)
    print("Resuming, Load pre-train models finish...")
    # models.eval()

    ########################### MIF Image List ##########################
    img1_name_list = []
    img2_name_list = []
    for img_name in os.listdir(os.path.join(image_dir_path, "GFP")):
        # img_name = img_name.split(".")[0][0:6]
        img1_name_list.append(img_name)

    for img_name in os.listdir(os.path.join(image_dir_path, "PCI")):
        # img_name = img_name.split(".")[0][0:6]
        img2_name_list.append(img_name)

    # print(img1_name_list)
    # print(img2_name_list)
    ############################# Fuse MIF ############################
    print('\n====>> Star Test fuse image......')
    save_path = os.path.join(opts.eval_folder, '{}/{}'.format(dataset_name, datetime.datetime.now().strftime("%Y%m%d_%H%M")))
    with torch.no_grad():
        for img_name in os.listdir(os.path.join(image_dir_path, dataset_name.split('_')[0])):
            # print("img_name=", img_name.split("_")[1])
            print("img_name=", img_name)
            # MRI-PET
            # data_mri = image_read_cv2(os.path.join(image_dir_path, dataset_name.split('_')[0], img_name),resize_W=opts.W, resize_H=opts.H, mode='GRAY')[np.newaxis, np.newaxis, ...] / 255.0
            # data_pet_y = cv2.split(image_read_cv2(os.path.join(image_dir_path, dataset_name.split('_')[1], img_name),resize_W=opts.W, resize_H=opts.H, mode='YCrCb'))[0][np.newaxis, np.newaxis, ...] / 255.0
            # data_pet_BGR = cv2.imread(os.path.join(image_dir_path, dataset_name.split('_')[1], img_name))
            # VIS_IR
            # data_mri = image_read_cv2(os.path.join(image_dir_path, dataset_name.split('_')[1], img_name),resize_W=opts.W, resize_H=opts.H, mode='GRAY')[np.newaxis, np.newaxis, ...] / 255.0
            # data_pet_y = cv2.split(image_read_cv2(os.path.join(image_dir_path, dataset_name.split('_')[0], img_name),resize_W=opts.W, resize_H=opts.H, mode='YCrCb'))[0][np.newaxis, np.newaxis, ...] / 255.0
            # data_pet_BGR = cv2.imread(os.path.join(image_dir_path, dataset_name.split('_')[0], img_name))
            # GFP-PCI
            # data_mri = image_read_cv2(os.path.join(image_dir_path, dataset_name.split('_')[1], "PCI_"+img_name.split("_")[1]), resize_W=opts.W, resize_H=opts.H, mode='GRAY')[np.newaxis, np.newaxis, ...] / 255.0
            # data_pet_y = cv2.split(image_read_cv2(os.path.join(image_dir_path, dataset_name.split('_')[0], "GFP_"+img_name.split("_")[1]), resize_W=opts.W, resize_H=opts.H, mode='YCrCb'))[0][np.newaxis, np.newaxis, ...] / 255.0
            # data_pet_BGR = cv2.imread(os.path.join(image_dir_path, dataset_name.split('_')[0], "GFP_"+img_name.split("_")[1]))
            # data_pet_BGR = cv2.resize(data_pet_BGR, (opts.W, opts.H))
            # GFP-PCI
            data_mri = image_read_cv2(os.path.join(image_dir_path, dataset_name.split('_')[1], img_name),
                           resize_W=opts.W, resize_H=opts.H, mode='GRAY')[np.newaxis, np.newaxis, ...] / 255.0
            data_pet_y = cv2.split(image_read_cv2(os.path.join(image_dir_path, dataset_name.split('_')[0], img_name), resize_W=opts.W, resize_H=opts.H, mode='YCrCb'))[0][np.newaxis, np.newaxis, ...] / 255.0
            data_pet_BGR = cv2.imread(os.path.join(image_dir_path, dataset_name.split('_')[0], img_name))
            data_pet_BGR = cv2.resize(data_pet_BGR, (opts.W, opts.H))
            _, img_cr, img_cb = cv2.split(cv2.cvtColor(data_pet_BGR, cv2.COLOR_BGR2YCrCb))
            data_mri, data_pet_y = torch.FloatTensor(data_mri), torch.FloatTensor(data_pet_y)
            data_mri, data_pet_y = data_mri.to(opts.device), data_pet_y.to(opts.device)
            # tensor_to_PIL(data_mri, "./test/feature_map", "t_MRI_0323_n")
            # tensor_to_PIL(data_pet_y, "./test/feature_map", "t_PCI_0323_n")
            fused_img_Y = model(data_pet_y, data_mri)
            # tensor_to_PIL(fused_img_Y, "./test/feature_map", "t_fusion_0323_y")
            data_Fuse = (fused_img_Y - torch.min(fused_img_Y)) / (torch.max(fused_img_Y) - torch.min(fused_img_Y))
            fi = np.squeeze((data_Fuse * 255).cpu().numpy())
            fi = fi.astype(np.uint8)
            ycrcb_fi = np.dstack((fi, img_cr, img_cb))
            rgb_fi = cv2.cvtColor(ycrcb_fi, cv2.COLOR_YCrCb2RGB)
            # save_image_name = img_name.split(".")[0] + "_" + datetime.datetime.now().strftime("%Y%m%d")
            save_image_name = img_name.split(".")[0]
            img_save(rgb_fi, save_image_name, save_path)
            print('Fusion {}/{} Sucessfully!'.format(save_path, save_image_name))

        print('Finished testing!')


if __name__ == '__main__':
    opts = TestOptions().parse()
    inference(opts)
