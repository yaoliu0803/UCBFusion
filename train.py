# import python package
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
from PIL import Image
import statistics
import numpy as np
import datetime
import random
import cv2
import time
import os

# import external package
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.autograd import Variable

# import internal package
from model import DRFE_Module
from opts import TrainOptions
from utils.loss import Fusionloss, Average_loss
from utils.dataprocess import transform_image, MMFdataset, MIFdataset, RoadSceneDataset, img_save, tensor_to_PIL



def train(opts):
    # Set random seedl
    seed = opts.seed
    # seed = np.range(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print("=======>seed :", seed)

    # GPU is True
    device = opts.device
    print("\nusing {} device.".format(device))

    # Data loader
    assert os.path.exists(opts.folder), "{} path does not exist.".format(opts.folder)
    # image_dir_path = opts.folder + "PET_MRI/"     # MIFdataset
    # image_dir_path = opts.folder + "VIS_IR/"  # RoadSceneDataset/MSRS
    image_dir_path = opts.folder + "GFP_PCI/"     # MMFdataset
    # image_dir_path = opts.folder
    print("image_dir_path=", image_dir_path)
    transform = transform_image(is_gray=opts.is_gray)
    train_dataset = MMFdataset(images_dir_path=image_dir_path, transform=transform)  # MMFdataset, MIFdataset, RoadSceneDataset
    train_num = len(train_dataset)
    print("train_total_num = ", train_num)
    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=0)

    # Build the models and Compute network paramerters
    model = DRFE_Module().to(opts.device)
    # print(models)
    print("Initial Detail-Refinement-Enhance Fusion Model Finished")
    print("Train_network have  {}  parameters in total".format(sum(x.numel() for x in model.parameters())))


    # Load pre-train models
    if os.path.exists(opts.resume):
        checkpoint = torch.load(opts.resume)
        model.load_state_dict(checkpoint['models'])
        print("Resuming, Load pre-train models finish...")

    # Create loss function and optimiz42er
    criterion = Fusionloss(opts.device)
    # optimizer = Adam(model.parameters(), lr=opts.lr, betas=(0.9, 0.999))
    optimizer = Adam(model.parameters(), lr=opts.lr)

    ########################## Train loop #########################
    print('\n###########===>Training Begins [epochs:{}] <====############'.format(opts.epoch))
    # num_epochs = trange(opts.epoch)
    batch = opts.batch_size
    # Record train time
    train_begin_time = datetime.datetime.now()

    print('\nStart training......')

    # log
    # strain_path = opts.log_dir + 'log_loss_{}seed_{}epoch_{}.txt'.format(seed, opts.epoch, datetime.datetime.now().strftime("%Y%m%d_%H%M"))
    # print("log_path = ", strain_path)
    # with open(strain_path, "a") as f:
    #     # 记录随机种子
    #     write_info = f"Seed: {seed} \n"
    #     f.write(write_info)


    Loss_texture_plt = []
    Loss_ssim_plt = []
    Loss_all_plt = []
    for epoch in range(0, opts.epoch):
        epoch = epoch + 1  # epoch start at 1
        # num_epochs.set_description("epoch:%d  batch:%d" % (epoch, opts.batch_size))
        Loss_texture =[]
        Loss_ssim = []
        Loss_all = []
        fuse_time = []
        for i, (data_gfp_y, data_pci, img_cr, img_cb) in enumerate(train_loader):
            model.train()
            # print("data_gfp_y={}, data_pci={}".format(data_gfp_y.shape, data_pci.shape))
            # print("img_cr={}, img_cb={}".format(img_cr.shape, img_cb.shape))

            # Extract the luminance(Y channel) and move to computation device
            data_gfp_y, data_pci, img_cr, img_cb = data_gfp_y.to(opts.device), data_pci.to(opts.device), img_cr.to(opts.device), img_cb.to(opts.device)
            data_gfp_y = Variable(data_gfp_y, requires_grad=False)
            data_pci = Variable(data_pci, requires_grad=False)
            # network forward
            optimizer.zero_grad()

            # get fusion image
            start_time = time.time()
            fusion_img_y = model(data_gfp_y, data_pci)

            end_time = time.time()
            fuse_time.append(end_time - start_time)

            # Calculate loss
            loss_total, loss_texture, loss_ssim = criterion(data_gfp_y, data_pci, fusion_img_y)
            Loss_texture.append(loss_texture.item())
            Loss_ssim.append(loss_ssim.item())
            Loss_all.append(loss_total.item())

            # Update the parameters
            loss_total.backward()
            optimizer.step()

        ########################### Loss Analysis ##########################
        # Print training loss information
        loss_texture_avg = np.mean(Loss_texture)
        loss_ssim_avg = np.mean(Loss_ssim)
        loss_total_avg = np.mean(Loss_all)
        print('Epoch==>{}\t loss_total_avg: {:.4f}, loss_texture_avg：{:.4f}, loss_ssim_avg: {:.4f}'.format(epoch, loss_total_avg, loss_texture_avg, loss_ssim_avg))

        Loss_texture_plt.append(loss_texture_avg)
        Loss_ssim_plt.append(loss_ssim_avg)
        Loss_all_plt.append(loss_total_avg)

        # Time recoder
        # std = statistics.stdev(fuse_time[1:])
        # mean = statistics.mean(fuse_time[1:])
        # print(f'fuse std time: {std:.2f}')
        # print(f'fuse avg time: {mean:.2f}')
        # print('fps (equivalence): {:.2f}'.format(1. / mean))


        # Save info to txt file
        # with open(strain_path, 'a') as f:
        #     # f.write(Loss_file + '\r\n')
        #     # 记录每个epoch对应的train_loss、lr以及验证集各指标
        #     write_info = f"[epoch: {epoch}] loss_total_avg: {loss_total_avg:.4f} lr: {opts.lr:.6f}\t" \
        #                  f"loss_texture_avg: {loss_texture_avg:.3f} loss_ssim_avg: {loss_ssim_avg:.3f} \n"
        #     f.write(write_info)

        # Save the training image
        if epoch % opts.record_epoch == 0:
        # if epoch == 1:
            #   结果图像保存方法1
            save_path = os.path.join(opts.train_result, 'image/{}'.format(datetime.datetime.now().strftime("%Y%m%d")))
            # save_path = os.path.join(opts.train_result, 'image/{}/visual'.format(datetime.datetime.now().strftime("%Y%m%d")))
            save_image_name = "{}_{}_{}".format(str(epoch), datetime.datetime.now().strftime("%Y%m%d_%H%M"), "y")
            img1_feature1_name = "{}_{}_{}".format(str(epoch), datetime.datetime.now().strftime("%Y%m%d_%H%M"), "img1_sub")
            img2_feature1_name = "{}_{}_{}".format(str(epoch), datetime.datetime.now().strftime("%Y%m%d_%H%M"), "img2_sub")
            img1_dense_name = "{}_{}_{}".format(str(epoch), datetime.datetime.now().strftime("%Y%m%d_%H%M"), "fuse_cat")
            img2_dense_name = "{}_{}_{}".format(str(epoch), datetime.datetime.now().strftime("%Y%m%d_%H%M"), "img2_cat")
            # img1_sub, img2_sub
            # tensor_to_PIL(fusion_img_y, save_path, save_image_name)
            # tensor_to_PIL(img1_sub, save_path, img1_feature1_name)
            # tensor_to_PIL(img2_sub, save_path, img2_feature1_name)
            # tensor_to_PIL(fuse_cat, save_path, img1_dense_name)
            # tensor_to_PIL(img2_cat, save_path, img2_dense_name)

            img_out_y = fusion_img_y.detach().cpu().numpy()
            fused_img_Y = np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0])
            img_cr = img_cr[0].cpu().detach().numpy()
            img_cb = img_cb[0].cpu().detach().numpy()
            ycrcb_fi = np.dstack((fused_img_Y, img_cr, img_cb))
            rgb_fi = cv2.cvtColor(ycrcb_fi, cv2.COLOR_YCrCb2RGB)
            # if not os.path.exists(save_path):
            #     os.makedirs(save_path)
            # cv2.imwrite("{}/{}_{}.png".format(save_path, str(epoch), datetime.datetime.now().strftime("%Y%m%d_%H%M")), final_img)
            # img_save(rgb_fi, save_image_name, save_path)
            # print('Fusion {0} Sucessfully!'.format(save_path))

        # Save models
        if epoch % opts.record_epoch == 0:
            model_name = "model_{}_{}_latest_{}.pth".format(str(epoch), "GFP_PCI", datetime.datetime.now().strftime("%Y%m%d_%H%M"))
            # model_name = "model_{}_{}_latest_{}.pth".format(str(epoch), "PET_MRI", datetime.datetime.now().strftime("%Y%m%d_%H%M"))
            save_checkpoint = {
                'models': model.state_dict(),
                'optimizers': optimizer.state_dict(),
                'epoch': opts.epoch,
                'batch_index': opts.batch_size
            }
            save_model_path = os.path.join(opts.train_result, 'checkpoint/{}'.format(datetime.datetime.now().strftime("%Y%m%d")))
            if not os.path.exists(save_model_path):
                # 如果路径不存在，创建文件夹
                os.makedirs(save_model_path)
            save_model_path = os.path.join(save_model_path, model_name)
            torch.save(save_checkpoint, save_model_path)
            print("Done, trained models saved at", save_model_path)

    ####################### Record train time ########################
    train_end_time = datetime.datetime.now()
    training_duration = (train_end_time - train_begin_time)
    total_seconds = training_duration.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    print(f"\nTrain all use time: {hours} hours {minutes} minutes {seconds} seconds")

    # Plot the loss function curve
    plt.figure(figsize=[12, 8])
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.subplot(2, 3, 1), plt.plot(range(len(Loss_all_plt)), Loss_all_plt), plt.title('total_loss')
    plt.subplot(2, 3, 2), plt.plot(range(len(Loss_texture_plt)), Loss_texture_plt), plt.title('texture_loss')
    plt.subplot(2, 3, 3), plt.plot(range(len(Loss_ssim_plt)), Loss_ssim_plt), plt.title('ssim_loss')
    plt.tight_layout()
    plt.savefig(os.path.join(opts.log_dir, 'curve_per_epoch_loss_{}_{}.png'.format(opts.epoch, datetime.datetime.now().strftime("%Y%m%d_%H%M"))))  # 保存训练损失曲线图片
    plt.show()  # 显示曲线
    print("Finished Training!")


if __name__ == '__main__':
    opts = TrainOptions().parse()
    train(opts)
