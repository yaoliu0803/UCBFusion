# import python package
import pytorch_msssim

# import external package
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
 Parts of these codes are from: 
    https://github.com/Linfeng-Tang/SeAFusion/blob/main/loss.py
    https://github.com/Zhaozixiang1228/MMIF-CDDFuse/blob/main/utils/loss.py
    https://github.com/GeoVectorMatrix/Dif-Fusion/blob/main/models/fs_loss.py
'''

# plot
def Average_loss(Iter_per_epoch, loss):
    # return [sum(loss[i*Iter_per_epoch:(i+1)*Iter_per_epoch])/Iter_per_epoch for i in range(int(len(loss)/Iter_per_epoch))]
    return [sum(loss[i*Iter_per_epoch:(i+1)*Iter_per_epoch])/Iter_per_epoch for i in range(int(loss/Iter_per_epoch))]


# =====================================================================
# Calculate loss
#   l1_loss = torch.nn.L1Loss()
#   mse_loss = torch.nn.MSELoss().to(opts.device)
#   ssim_loss = pytorch_msssim.msssim
# =====================================================================
# https://github.com/mnawfal29/APWNet/blob/main/loss.py
# loss_total=alpha*loss_in+loss_grad
class Fusionloss(nn.Module):
    def __init__(self, device):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy(device)
        self.ssim = pytorch_msssim.msssim
        self.device = device

    def forward(self, image1_y, image2, fusion_image):
        alpha = 1
        # GFP_PCI 1:0, MRI_PET:0.5:0.5, MSRS:0.6:0.4
        w1_ssim = 1
        w2_ssim = 0
        # w2_ssim = 0
        ssim_weight = 10
        # detail loss
        loss_ssim1 = w1_ssim * self.ssim(fusion_image, image2, normalize=True)
        loss_ssim2 = w2_ssim * self.ssim(fusion_image, image1_y, normalize=True)
        loss_ssim = loss_ssim1 + loss_ssim2
        loss_ssim += ssim_weight * (1 - loss_ssim)
        # gradient loss
        image1_grad = self.sobelconv(image1_y)
        image2_grad = self.sobelconv(image2)
        generate_img_grad = self.sobelconv(fusion_image)
        x_grad_joint = torch.max(image1_grad, image2_grad)
        loss_texture = alpha * F.l1_loss(x_grad_joint, generate_img_grad)
        loss_total = loss_texture + loss_ssim
        return loss_total, loss_texture, loss_ssim


class Sobelxy(nn.Module):
    def __init__(self, device):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).to(device) # .cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).to(device)

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


def gradient(x):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        laplace = [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
        kernel = torch.FloatTensor(laplace).unsqueeze(0).unsqueeze(0).to(device)
        return F.conv2d(x, kernel, stride=1, padding=1)

def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(
        torch.sum(img2 ** 2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()


def gradient(input):
    """
    求图像梯度, sobel算子
    :param input:
    :return:
    """

    filter1 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1)
    filter2 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1)
    filter1.weight.data = torch.tensor([
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.]
    ]).reshape(1, 1, 3, 3).cuda()
    # ]).reshape(1, 1, 3, 3).to(device)

    filter2.weight.data = torch.tensor([
        [1., 2., 1.],
        [0., 0., 0.],
        [-1., -2., -1.]
    ]).reshape(1, 1, 3, 3).cuda()
    # ]).reshape(1, 1, 3, 3).to(device)

    g1 = filter1(input)
    g2 = filter2(input)
    image_gradient = torch.abs(g1) + torch.abs(g2)
    return image_gradient


