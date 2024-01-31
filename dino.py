# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import os
# import numpy as np

# # 加载预训练的DINO模型
# #dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
# #dino_model.eval()
# dino_model = torch.hub.load('facebookresearch/dinov2', "dinov2_vitb14")
# dino_model.cuda()
# state_dict = torch.load("pretrained_model/dinov2_vitb14_pretrain.pth", map_location='cpu')
# state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
# state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
# msg = dino_model.load_state_dict(state_dict, strict=False)
# print('Pretrained weights found at pretrained_model/dinov2_vitb14_pretrain.pth and loaded with msg: {}'.format(msg))
# dino_model.eval()

# # 定义图像预处理
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # 函数：提取图像特征
# # def extract_features(image_path):
# #     image = Image.open(image_path)
# #     image = preprocess(image).unsqueeze(0)
# #     with torch.no_grad():
# #         features = dino_model(image)
# #     return features.squeeze(0)
# def extract_features(image_path):
#     image = Image.open(image_path).convert("RGB")  # 将图像转换为RGB
#     image = preprocess(image).unsqueeze(0)
#     with torch.no_grad():
#         features = dino_model(image.cuda())
#     return features.squeeze(0)


# # 计算参考图像集的特征
# reference_dir = "pic_DALL-E_all/"
# reference_features = []
# for filename in os.listdir(reference_dir):
#     if filename.endswith(('.png', '.jpg', '.jpeg')):
#         path = os.path.join(reference_dir, filename)
#         features = extract_features(path)
#         reference_features.append(features)
# reference_features = torch.stack(reference_features)

# # 归一化特征
# reference_features = torch.nn.functional.normalize(reference_features, dim=1)

# # 计算测试图像的特征并找出最相似的参考图像
# test_dir = "pic_test/"
# for test_filename in os.listdir(test_dir):
#     if test_filename.endswith(('.png', '.jpg', '.jpeg')):
#         test_path = os.path.join(test_dir, test_filename)
#         test_features = extract_features(test_path)
#         test_features = torch.nn.functional.normalize(test_features, dim=0)

#         # 计算余弦相似度
#         similarities = torch.mm(test_features.unsqueeze(0), reference_features.t())

#         # 获取最相似图像的索引
#         _, top_indices = similarities.topk(5)
#         print(f"Test Image: {test_filename}")
#         for i, index in enumerate(top_indices.squeeze(0)):
#             print(f"{i+1}: Similar Image: {os.listdir(reference_dir)[index]}")


import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# 加载预训练的DINO模型
dino_model = torch.hub.load('facebookresearch/dinov2', "dinov2_vitb14")
dino_model.eval()

# 定义图像预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 函数：提取图像特征
def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = dino_model(image)
    return features.squeeze(0)

# 计算参考图像集的特征
reference_dir = "pic_DALL-E_all/"
reference_features = []
reference_filenames = []
for filename in os.listdir(reference_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        path = os.path.join(reference_dir, filename)
        features = extract_features(path)
        reference_features.append(features)
        reference_filenames.append(filename)
reference_features = torch.stack(reference_features)
print("1")

# 归一化特征
reference_features = torch.nn.functional.normalize(reference_features, dim=1)
print("2")

# 计算测试图像的特征并找出最相似的参考图像
test_dir = "pic_test/"
for test_filename in os.listdir(test_dir):
    print("3")
    print(test_filename)
    if test_filename.endswith(('.png', '.jpg', '.JPEG')):
        print("4")
        test_path = os.path.join(test_dir, test_filename)
        test_features = extract_features(test_path)
        test_features = torch.nn.functional.normalize(test_features, dim=0)

        # 计算余弦相似度
        similarity = torch.mm(test_features.unsqueeze(0), reference_features.t())
        top_similarities, top_indices = similarity.topk(3)

        # 打印结果
        print(f"Test Image: {test_filename}")
        for i, index in enumerate(top_indices.squeeze(0)):
            print(f"{i+1}: Similar Image: {reference_filenames[index]} with similarity score: {top_similarities.squeeze(0)[i].item()}")
        print("-----------")
