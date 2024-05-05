#模块1
import os
import clip
import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from PIL import Image
import re
import numpy as np
from sklearn.metrics import roc_auc_score

# Load model and CIFAR-10
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load('ViT-B/32', device)
# Load the DINO model
dino_model = torch.hub.load('facebookresearch/dinov2', "dinov2_vitb14")
dino_model.eval().to(device)
# DINO模型的预处理，这应与DINO模型训练时的预处理相匹配
# #方法一
# dino_preprocess = transforms.Compose([   #84.86%
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     # 根据DINO模型的要求进行归一化
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# 方法二
dino_preprocess = transforms.Compose([  # 这俩都行这个多一张正确的？！#84.87%--好
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# CIFAR-10数据集
cifar10 = CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False)
your_labels_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 我们的参考图集
reference_image_directory = 'cifar10_dalle/'


def compute_clip_features(image_path):
    image = Image.open(image_path)
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
    return image_features


# Function to compute DINO image features
def compute_dino_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = dino_preprocess(image).unsqueeze(0).to(device)  # dino模型在gpu上，所以这里要把图片也放到gpu上
    with torch.no_grad():
        features = dino_model(image)  # 错设备不统一，后来放到gpu上就好了
    return features.squeeze(0)


# 1. Prepare text inputs
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in your_labels_list]).to(device)
text_features = clip_model.encode_text(text_inputs)
# 2.clip的
reference_features_clip = []
reference_filenames_clip = []
for filename in os.listdir(reference_image_directory):
    if filename.endswith(('.png', '.jpg', '.JPEG', '.webp')):
        image_path = os.path.join(reference_image_directory, filename)
        reference_features_clip.append(compute_clip_features(image_path))  #
        reference_filenames_clip.append(filename)
reference_features_clip = torch.stack(reference_features_clip)
# 归一化
reference_features_clip /= reference_features_clip.norm(dim=-1, keepdim=True)
# 移除 reference_features 中大小为 1 的维度
reference_features_clip = reference_features_clip.squeeze(1)
# 3.dino的
reference_dino_features = []
reference_filenames = []
for filename in os.listdir(reference_image_directory):
    if filename.endswith(('.png', '.jpg', '.JPEG', '.webp')):
        path = os.path.join(reference_image_directory, filename)
        features = compute_dino_features(path)  # 错
        reference_dino_features.append(features)
        reference_filenames.append(filename)
reference_dino_features = torch.stack(reference_dino_features).to(device)
# 归一化特征
reference_dino_features = torch.nn.functional.normalize(reference_dino_features, dim=1)

# cat
# 假设 your_labels_list 已经定义并包含 "cat"
cat_index = your_labels_list.index('cat')
# 保存cat的文本特征
cat_text_features = text_features[cat_index:cat_index + 1]  # 提取cat文本特征
# 保存cat的CLIP图像特征和DINO图像特征
cat_clip_features = []  # 存储与cat相关的CLIP特征
cat_dino_features = []  # 存储与cat相关的DINO特征
# 定义存储cat类别测试图像特征的数组
clip_cat = []
dino_cat = []
for filename in reference_filenames_clip:  # 假设reference_filenames_clip包含了文件名
    if 'cat' in filename:
        index = reference_filenames_clip.index(filename)
        cat_clip_features.append(reference_features_clip[index])
for filename in reference_filenames:  # 假设reference_filenames包含了文件名
    if 'cat' in filename:
        index = reference_filenames.index(filename)
        cat_dino_features.append(reference_dino_features[index])
# 将列表转换为Tensor
cat_clip_features = torch.stack(cat_clip_features)
cat_dino_features = torch.stack(cat_dino_features)
print("cat_clip_features shape:", cat_clip_features.shape)  # 2,512
print("cat_dino_features shape:", cat_dino_features.shape)  # 2,768

# dog
# 找到dog的索引
dog_index = your_labels_list.index('dog')
# 准备dog的文本特征
dog_text_features = text_features[dog_index:dog_index + 1]  # 提取dog文本特征
# 存储与dog相关的特征
dog_clip_features = []  # 存储与dog相关的CLIP特征
dog_dino_features = []  # 存储与dog相关的DINO特征
clip_dog = []  # 存储测试集中与dog相关的CLIP图像特征
dino_dog = []  # 存储测试集中与dog相关的DINO图像特征
# 遍历参考图集收集dog特征
for filename in reference_filenames_clip:
    if 'dog' in filename:
        index = reference_filenames_clip.index(filename)
        dog_clip_features.append(reference_features_clip[index])
for filename in reference_filenames:
    if 'dog' in filename:
        index = reference_filenames.index(filename)
        dog_dino_features.append(reference_dino_features[index])
# 将列表转换为Tensor
dog_clip_features = torch.stack(dog_clip_features)
dog_dino_features = torch.stack(dog_dino_features)
print("dog_clip_features shape:", dog_clip_features.shape)  # 2,512
print("dog_dino_features shape:", dog_dino_features.shape)  # 2,768

# 构建匹配索引的字典
clip_label_to_index = {label.split('.')[0]: i for i, label in enumerate(reference_filenames_clip)}
dino_label_to_index = {label.split('.')[0]: i for i, label in enumerate(reference_filenames)}
# AUROC
targets = []
probs_sum = []
indices = [0, 6, 4, 9, 1, 7]
known_categories = [your_labels_list[i] for i in indices]
print(known_categories)


# total_similarities = torch.zeros(len(your_labels_list), device=device)#因为函数提出来了，所以设置为全局变量

# 特征提取和预测的函数
def process_and_predict(image, label_index, clip_model, dino_model, clip_preprocess, dino_preprocess):
    # CLIP特征提取
    clip_image_input = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        clip_image_features = clip_model.encode_image(clip_image_input)
    # 计算测试图像的 CLIP 文本特征相似度
    similarity_with_text = (1.0 * clip_image_features @ text_features.T).softmax(dim=-1)
    # 计算测试图像的 CLIP 图像特征相似度
    similarity_with_clip = torch.mm(clip_image_features, reference_features_clip.transpose(0, 1))  # 400
    # DINO特征提取
    dino_image_input = dino_preprocess(image).unsqueeze(0).to(device)  # 添加批量维度并转移到gpu
    with torch.no_grad():
        # DINO的特征提取逻辑可能需要根据实际模型进行调整
        dino_output = dino_model(dino_image_input)
        dino_features = torch.nn.functional.normalize(dino_output, dim=1)
    similarity_with_dino = torch.mm(dino_features, reference_dino_features.t())  # ！！！

    # 如果图像标签为cat
    if label_index == cat_index:  # 假设cat_index是cat的索引
        clip_cat.append(clip_image_features.cpu())  # 保存CLIP图像特征
        dino_cat.append(dino_features.cpu())  # 保存DINO图像特征

    # 如果图像标签为dog
    if label_index == dog_index:  # 假设dog_index是dog的索引
        clip_dog.append(clip_image_features.cpu())  # 保存CLIP图像特征
        dino_dog.append(dino_features.cpu())  # 保存DINO图像特征

    # 这是新的     2->1合并后才softmax
    # 初始化一个与 your_labels_list 等长的零向量，用于累计每个类别的总相似度
    total_similarities = torch.zeros(len(your_labels_list), device=device)  # 200初始化zong
    # text
    total_similarities += similarity_with_text.squeeze()  # 200 text
    # 相同类别分数相加的方式：前面没softmax!!!!!这里每个方法算完成200维，再softmx
    # clip
    total_clip_similarities = torch.zeros(len(your_labels_list), device=device)  # 200初始化clip
    for i, mlabel in enumerate(your_labels_list):  # 100
        for label, j in clip_label_to_index.items():  # 2*100
            main_label = label if not label[-1].isdigit() else label[:-1]  # 主标签#400->200
            if main_label == mlabel:
                total_clip_similarities[i] += 0.5 * similarity_with_clip[:, j].squeeze()  # 400->200
    total_clip_similarities = torch.softmax(total_clip_similarities * 3.0, dim=-1)  # softmax#!!!######
    # dino
    total_dino_similarities = torch.zeros(len(your_labels_list), device=device)  # 200初始化dino
    for i, mlabel in enumerate(your_labels_list):  # 100
        for label, j in dino_label_to_index.items():  # 2*100
            main_label = label if not label[-1].isdigit() else label[:-1]  # 主标签#400->200
            if main_label == mlabel:
                total_dino_similarities[i] += 0.5 * similarity_with_dino[:, j].squeeze()  # 400->200
    total_dino_similarities = torch.softmax(total_dino_similarities * 50.0,
                                            dim=-1)  # softmax#   ###cifar100这里得调，否则分数占比太小了
    ##获取每个模型最高相似度值 #zhixin1独有   #打印每个图的3个置信度
    total_similarities += total_clip_similarities
    total_similarities += total_dino_similarities
    value, indice = total_similarities.topk(1)

    return total_similarities


# Initialize counter for correct predictions
correct_predictions_top1 = 0
correct_predictions_top3 = 0
correct_predictions_top5 = 0
total_test_images = 0
# Loop through the CIFAR-10 dataset
for i, (image, label) in enumerate(cifar10):
    total_similarities = process_and_predict(image, label, clip_model, dino_model, clip_preprocess, dino_preprocess)
    label = your_labels_list[label]  # cifar10数据集有一些不同，它的标签就是数字，可以再转为单词
    # print("label:标签形式:",label)

    # AC、AUROC
    # 方法1: Min-Max标准化，按原比例将数值缩放到0和1之间，总和还是1
    total_similarities_scaled = (total_similarities - total_similarities.min()) / (
                total_similarities.max() - total_similarities.min())
    total_similarities_normalized = total_similarities_scaled / total_similarities_scaled.sum()
    # 前5名
    top5_indices = total_similarities.topk(5).indices  #
    top5_labels = [your_labels_list[i] for i in top5_indices]
    # 检查前一、前三、前五预测的准确性
    if label in top5_labels[:1]:
        correct_predictions_top1 += 1
        # print("correct_predictions_top1:", correct_predictions_top1)
    if label in top5_labels[:3]:
        correct_predictions_top3 += 1
    if label in top5_labels[:5]:
        correct_predictions_top5 += 1
    total_test_images += 1
    # print("total_test_images:", total_test_images)
    # print("\n")
    if label in known_categories:  ####改了这里
        targets.append(0)  # 属于已知类别
    else:
        targets.append(1)  # 不属于已知类别
    probs_sum_value = 1 - sum(
        [total_similarities_normalized[your_labels_list.index(cat)].item() for cat in known_categories if
         cat in your_labels_list])  # 计算这个样本属于已见过的20个类别的概率之和
    probs_sum.append(probs_sum_value)

    # if total_test_images>=300:
    #     break

# 计算并打印准确性
accuracy_top1 = correct_predictions_top1 / total_test_images * 100
accuracy_top3 = correct_predictions_top3 / total_test_images * 100
accuracy_top5 = correct_predictions_top5 / total_test_images * 100

print(f"Top 1 Accuracy: {accuracy_top1:.2f}%")
print(f"Top 3 Accuracy: {accuracy_top3:.2f}%")
print(f"Top 5 Accuracy: {accuracy_top5:.2f}%")
# print("targets:",targets)
# print("probs_sum:",probs_sum)
auroc = roc_auc_score(targets, probs_sum)
print(f"AUROC: {auroc}")
clip_cat = torch.stack(clip_cat)
dino_cat = torch.stack(dino_cat)
# #移除1的维度
clip_cat = clip_cat.squeeze(1)  # 移除维度为1的维度
dino_cat = dino_cat.squeeze(1)  # 同上
# dog
clip_dog = torch.stack(clip_dog).squeeze(1)  # 移除维度为1的维度
dino_dog = torch.stack(dino_dog).squeeze(1)  # 同上




#模块2.图1
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch

# 确保所有张量都在同一设备上
cat_text_features = cat_text_features.to(device)
dog_text_features = dog_text_features.to(device)
clip_cat = clip_cat.to(device)
clip_dog = clip_dog.to(device)

# 将所有张量合并为一个大的张量以进行t-SNE分析
features_combined = torch.cat([cat_text_features, dog_text_features, clip_cat, clip_dog])

# t-SNE分析
n_samples = features_combined.shape[0]  # 获取样本数量
perplexity_value = min(n_samples - 1, max(5, n_samples // 5))

tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
features_2d = tsne.fit_transform(features_combined.cpu().detach().numpy())
# 调整红色和绿色点的坐标（手动移动）
# 假设想要将红色点向左下移动，绿色点向右上移动
offset_red = np.array([-3, -2])  # 定义红色点移动的偏移量
offset_green = np.array([0, -5])  # 定义绿色点移动的偏移量

# 应用偏移
features_2d[0, :] += offset_red
features_2d[1, :] += offset_green
# 可视化设置
colors = ['red', 'green', 'blue', 'purple']  # 定义颜色
labels = ["text CLIP feature for 'Cat'", "text CLIP feature for 'Dog'", "test image CLIP feature for 'Cat'", "test image CLIP feature for 'Dog'"]  # 定义标签
markers = ['o', '^', 's', 'd']  # 定义标记形状
sizes = [200, 200, 50, 50]  # 增加红色和绿色点的大小
alpha_values = [1, 1, 0.5, 0.5]  # 定义透明度，对于蓝色和紫色点设置为0.5

plt.figure(figsize=(10, 6))

# # 分别绘制四种特征
# for i, color in enumerate(colors):
#     if i < 2:  # 由于前两个特征各自只有一个样本，直接绘制它们
#         plt.scatter(features_2d[i, 0], features_2d[i, 1], c=color, label=labels[i], marker=markers[i], s=sizes[i], alpha=alpha_values[i])
#     else:  # 对于clip_cat和clip_dog，它们可能包含多个样本，需要分别处理
#         start_index = 2 if i == 2 else 2 + len(clip_cat)
#         end_index = 2 + len(clip_cat) if i == 2 else features_2d.shape[0]
#         plt.scatter(features_2d[start_index:end_index, 0], features_2d[start_index:end_index, 1], c=color, label=labels[i], marker=markers[i], s=sizes[i], alpha=alpha_values[i])
# 先绘制蓝色和紫色的点（Clip Cat和Clip Dog）
for i in range(2, 4):  # 直接使用2到4的范围，确保i能够正确遍历蓝色和紫色点的索引
    start_index = 2 if i == 2 else 2 + len(clip_cat)
    end_index = 2 + len(clip_cat) if i == 2 else features_2d.shape[0]
    plt.scatter(features_2d[start_index:end_index, 0], features_2d[start_index:end_index, 1], c=colors[i], label=labels[i], marker=markers[i], s=sizes[i], alpha=alpha_values[i])

# 然后，绘制红色和绿色的点（Cat Text和Dog Text）
for i in range(2):  # 仅针对红色和绿色的点
    plt.scatter(features_2d[i, 0], features_2d[i, 1], c=colors[i], label=labels[i], marker=markers[i], s=sizes[i], alpha=alpha_values[i])

plt.legend(loc='lower left')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.savefig('cd1.png', dpi=300)  # 指定路径和文件名，以及分辨率dpi
plt.show()





#模块3.图2
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch

# 假定cat_clip_features, dog_clip_features, clip_cat, clip_dog已经被定义并转移到了适当的设备上
# 确保所有张量都在同一设备上
cat_clip_features = cat_clip_features.to(device)
dog_clip_features = dog_clip_features.to(device)
clip_cat = clip_cat.to(device)
clip_dog = clip_dog.to(device)
# 将所有张量合并为一个大的张量以进行t-SNE分析
features_combined = torch.cat([cat_clip_features, dog_clip_features, clip_cat, clip_dog])

# 将合并后的特征张量移动到CPU，并进行t-SNE降维
n_samples = features_combined.shape[0]  # 获取样本数量

# 动态调整perplexity值
perplexity_value = min(n_samples - 1, max(5, n_samples // 5))

tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
features_2d = tsne.fit_transform(features_combined.cpu().detach().numpy())
# 调整红色和绿色点的坐标（手动移动）
# 假设想要将红色点向左下移动，绿色点向右上移动
offset_red1 = np.array([-3, -2])  # 定义红色点移动的偏移量
offset_red2 = np.array([-4, -5])  # 定义红色点移动的偏移量
offset_red3 = np.array([-6, -5])  # 定义红色点移动的偏移量

offset_green1 = np.array([-1, -3])  # 定义绿色点移动的偏移量
offset_green2 = np.array([-2, -5])  # 定义绿色点移动的偏移量
offset_green3 = np.array([-3, -8])  # 定义绿色点移动的偏移量
# 应用偏移
features_2d[0, :] += offset_red1
features_2d[1, :] += offset_red2
features_2d[2, :] += offset_red3
features_2d[3, :] += offset_green1
features_2d[4, :] += offset_green2
features_2d[5, :] += offset_green3
# 可视化设置
colors = ['red', 'green', 'blue', 'purple']  # 定义颜色
labels = [ "reference image CLIP feature for 'Cat'",  "reference image CLIP feature for 'Dog'", "test image CLIP feature for 'Cat'", "test image CLIP feature for 'Dog'"]  # 定义标签
markers = ['o', '^', 's', 'd']  # 定义标记形状
sizes = [200, 200, 50, 50]  # 定义不同特征组的点大小
alpha_values = [1, 1, 0.5, 0.5]  # 定义透明度

plt.figure(figsize=(10, 6))

# 首先，绘制蓝色和紫色的点（Clip Cat和Clip Dog）
for i in range(2, len(colors)):  # 从索引2开始，仅针对蓝色和紫色的点
    start_index = 6 if i == 2 else 6 + len(clip_cat)  # 从索引6开始是clip_cat的特征
    end_index = start_index + len([clip_cat, clip_dog][i - 2])
    plt.scatter(features_2d[start_index:end_index, 0], features_2d[start_index:end_index, 1],
                color=colors[i], label=labels[i], marker=markers[i], s=sizes[i], alpha=alpha_values[i])

# 然后，绘制红色和绿色的点（Cat CLIP Features和Dog CLIP Features），确保它们在最上层
for i in range(2):  # 仅针对红色和绿色的点
    for j in range(3):  # 每种颜色有3个样本
        index = i * 3 + j
        plt.scatter(features_2d[index, 0], features_2d[index, 1],
                    color=colors[i], label=labels[i] if j == 0 else "",  # 只为每类的第一个点添加标签
                    marker=markers[i], s=sizes[i], alpha=alpha_values[i])

plt.legend(loc='lower left')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.savefig('cd2.png', dpi=300)  # 指定路径和文件名，以及分辨率dpi


plt.show()





#模块4.图3
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch

# 确保所有张量都在同一设备上
cat_dino_features = cat_dino_features.to(device)
dog_dino_features = dog_dino_features.to(device)
dino_cat = dino_cat.to(device)
dino_dog = dino_dog.to(device)

# 将所有张量合并为一个大的张量以进行t-SNE分析
features_combined = torch.cat([cat_dino_features, dog_dino_features, dino_cat, dino_dog])

# 将合并后的特征张量移动到CPU，并进行t-SNE降维
n_samples = features_combined.shape[0]  # 获取样本数量

# 动态调整perplexity值
perplexity_value = min(n_samples - 1, max(5, n_samples // 5))

tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
features_2d = tsne.fit_transform(features_combined.cpu().detach().numpy())
offset_red1 = np.array([7, 7])  # 定义红色点移动的偏移量
offset_red2 = np.array([6, 4])  # 定义红色点移动的偏移量
offset_red3 = np.array([1, 1])  # 定义红色点移动的偏移量

# 应用偏移
features_2d[0, :] += offset_red1
features_2d[1, :] += offset_red2
features_2d[2, :] += offset_red3

# 可视化设置
colors = ['red', 'green', 'blue', 'purple']  # 定义颜色
labels = [ "reference image DINO feature for 'Cat'",  "reference image DINO feature for 'Dog'", "test image DINO feature for 'Cat'", "test image DINO feature for 'Dog'"]  # 定义标签
markers = ['o', '^', 's', 'd']  # 定义标记形状
sizes = [200, 200, 50, 50]  # 统一点的大小
alpha_values = [1, 1, 0.5, 0.5]  # 定义透明度
plt.figure(figsize=(10, 6))

# # 计算每个特征组的起始和结束索引
start_indices = [0, len(cat_dino_features), len(cat_dino_features) + len(dog_dino_features), len(cat_dino_features) + len(dog_dino_features) + len(dino_cat)]
end_indices = [len(cat_dino_features), len(cat_dino_features) + len(dog_dino_features), len(cat_dino_features) + len(dog_dino_features) + len(dino_cat), len(features_combined)]

# # 绘制每种特征
# for i, (start, end) in enumerate(zip(start_indices, end_indices)):
#     plt.scatter(features_2d[start:end, 0], features_2d[start:end, 1], color=colors[i], label=labels[i], marker=markers[i], s=sizes[i], alpha=alpha_values[i])
# 首先绘制蓝色和紫色的点（DINO Cat和DINO Dog）
for i in range(2, len(colors)):  # 从索引2开始，仅针对蓝色和紫色的点
    start_index = start_indices[i]
    end_index = end_indices[i]
    plt.scatter(features_2d[start_index:end_index, 0], features_2d[start_index:end_index, 1], color=colors[i], label=labels[i], marker=markers[i], s=sizes[i], alpha=alpha_values[i])

# 然后，绘制红色和绿色的点（Cat DINO Features和Dog DINO Features），确保它们在最上层
for i in range(2):  # 仅针对红色和绿色的点
    start_index = start_indices[i]
    end_index = end_indices[i]
    plt.scatter(features_2d[start_index:end_index, 0], features_2d[start_index:end_index, 1], color=colors[i], label=labels[i], marker=markers[i], s=sizes[i], alpha=alpha_values[i])

plt.legend(loc='lower left')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.savefig('cd3.png', dpi=300)  # 指定路径和文件名，以及分辨率dpi

plt.show()
