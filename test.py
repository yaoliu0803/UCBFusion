# import python package
import numpy as np
import datetime
import time
import cv2
import os

# import external package
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

# import internal package
from opts import TestOptions
from models import DRFE_Module
from utils.Evaluator import Evaluator
from utils.loss import Fusionloss
from utils.dataprocess import transform_image, Testdataset, image_read_cv2


def inference(opts):
    # data loader
    image_dir_path = opts.folder
    assert os.path.exists(image_dir_path), "{} path does not exist.".format(image_dir_path)
    transform = transform_image(is_gray=opts.is_gray)
    test_dataset = Testdataset(images_dir_path=image_dir_path, transform=transform, resize_W=opts.W, resize_H=opts.H)
    test_num = len(test_dataset)
    print("\ntest_image_number = ", test_num)
    test_loader = DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=0)

    # Load the pre-trained models
    print('Loading pre-trained models......')
    model = DRFE_Module().to(opts.device)
    checkpoint = torch.load(opts.model)
    model.load_state_dict(checkpoint['models'])
    model.to(opts.device)
    print("Resuming, Load pre-train models finish...")
    model.eval()

    ############################# Fuse ############################
    print('\n====>> Star Test fuse image......')
    img_name_list = []
    with torch.no_grad():
        # time
        start_time = time.time()
        fuse_time = []
        for i, (img1_y, img2, img1_cr, img1_cb, image_name) in enumerate(test_loader):
            print('Processing picture No.{} '.format((i + 1)))
            img1_y, img2, img1_cr, img1_cb = img1_y.to(opts.device), img2.to(opts.device), img1_cr.to(opts.device), img1_cb.to(opts.device)
            # img1_y, img2 = img1_y.to(opts.device), img2.to(opts.device)
            # img1_y, img2 = torch.FloatTensor(img1_y), torch.FloatTensor(img2)
            # img1_y, img2 = img1_y.to(opts.device), img2.to(opts.device)

            # get fusion image
            fusion_img_y = model(img1_y, img2)

            # Save the image
            img_out_y = fusion_img_y.cpu().detach().numpy()
            fused_img_Y = np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0])
            img1_cr = img1_cr[0].cpu().detach().numpy()
            img1_cb = img1_cb[0].cpu().detach().numpy()
            final_img = cv2.merge([fused_img_Y, img1_cr, img1_cb])  # YCrCb
            final_img = cv2.cvtColor(final_img, cv2.COLOR_YCrCb2BGR)  # BGR
            final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)  # RGB
            save_path = os.path.join(opts.eval_folder, 'GFP_PCI/{}'.format(datetime.datetime.now().strftime("%Y%m%d")))
            # print("save_path", image_name[0])   # image_name is a tuple
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_image_name = image_name[0] + "_" + datetime.datetime.now().strftime("%Y%m%d") + ".png"
            cv2.imwrite("{}/{}".format(save_path, save_image_name), final_img)
            img_name_list.append(save_image_name)
            print('Fusion {} Sucessfully!'.format(save_image_name))

        end_time = time.time()
        fuse_time.append(end_time - start_time)
        print('Finished testing!')

    ######################## Evaluator ########################
    print("\n" * 2 + "=" * 80)
    model_name = "DREFusion"
    print("The test result of [GFP_PCI] :")
    eval_folder = opts.eval_folder
    ori_img_folder = image_dir_path
    metric_result = np.zeros((8))

    # log
    # eval_log_path = opts.eval_log_path + 'log_{}.txt'.format(datetime.datetime.now().strftime("%Y%m%d"))
    # print("eval_log = ", eval_log_path)
    # data = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # with open(eval_log_path, "a") as f:
    #     # 记录评估日期
    #     write_info = f"Data: {data} \t Model_name: {model_name} \n"
    #     f.write(write_info)

    for img_name in img_name_list:
        # print("fusion_img_name=", img_name)
        fuse_img = img_name.split("_")[0]
        # print("fuse_img=", fuse_img)
        # GFP-PCI
        img1_name = img_name.split("_")[0] + "-g.jpg"
        img2_name = img_name.split("_")[0] + "-t.jpg"
        save_image_path = os.path.join(opts.eval_folder, 'GFP_PCI/{}'.format(datetime.datetime.now().strftime("%Y%m%d")))
        image1_path = save_image_path + img1_name
        image2_path = save_image_path + img2_name
        print("image1_name=", image1_path)
        print("image2_name=", image2_path)
        image1_y = image_read_cv2(image1_path, opts.W, opts.H, 'GRAY')
        image2 = image_read_cv2(image2_path, opts.W, opts.H, 'GRAY')
        fi = image_read_cv2(os.path.join(save_image_path, fuse_img), opts.W, opts.H, 'GRAY')
        # print("shape = ", fi.shape, image1_y.shape, image2.shape)
        metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
                                      , Evaluator.SF(fi), Evaluator.MI(fi, image1_y, image2)
                                      , Evaluator.SCD(fi, image1_y, image2), Evaluator.VIFF(fi, image1_y, image2)
                                      , Evaluator.Qabf(fi, image1_y, image2), Evaluator.SSIM(fi, image1_y, image2)])

        # Save info to txt file
        # with open(eval_log_path, 'a') as f:
        #     # 记录各指标
        #     write_info = f"[model_name: {model_name}] EN: {str(np.round(metric_result[0], 2)):.2f} SD: {str(np.round(metric_result[1], 2)):.2f}" \
        #                  f"SF: {str(np.round(metric_result[2], 2)):.2f} MI: {str(np.round(metric_result[3], 2)):.2f} SCD: {str(np.round(metric_result[4], 2)):.2f}" \
        #                  f"VIF: {str(np.round(metric_result[5], 2)):.2f} Qabf: {str(np.round(metric_result[6], 2)):.2f} SSIM: {str(np.round(metric_result[7], 2)):.2f} \n"
        #     f.write(write_info)

    metric_result /= len(os.listdir(eval_folder))
    print("\t\t\t EN \t SD\t\t SF\t\tMI \t\tSCD \tVIF \tQabf \tSSIM")
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
    inference(opts)
