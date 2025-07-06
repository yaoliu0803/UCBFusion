import os


def rename_images(folder_path):
    # 获取文件夹中的图像文件列表
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

    # 按照相对顺序对图像文件进行排序
    image_files.sort()

    # 重命名图像文件
    for index, image_file in enumerate(image_files, start=1):
        _, ext = os.path.splitext(image_file)
        # new_name = "GFP_" + str(index) + ext
        new_name = "PCI_" + str(index) + ".jpg"
        # print("new_name=", new_name)
        os.rename(os.path.join(folder_path, image_file), os.path.join(folder_path, new_name))
        print(f'Renamed {image_file} to {new_name}')


# 指定您的原始文件夹路径
folder_path = '../image/test/GFP_PCI/PCI'

# 调用函数进行重命名
rename_images(folder_path)
