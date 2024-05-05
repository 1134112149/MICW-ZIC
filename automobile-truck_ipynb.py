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
# DINO模型的预处理
dino_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# CIFAR-10数据集
cifar10 = CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False)
your_labels_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Reference image directory
reference_image_directory = 'cifar10_dalle/'


# Function to compute CLIP image features
def compute_clip_features(image_path):
    image = Image.open(image_path)
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
    return image_features


# Function to compute DINO image features
def compute_dino_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = dino_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = dino_model(image)
    return features.squeeze(0)


# 1. Prepare text inputs
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in your_labels_list]).to(device)
text_features = clip_model.encode_text(text_inputs)

# CLIP features
reference_features_clip = []
reference_filenames_clip = []
for filename in os.listdir(reference_image_directory):
    if filename.endswith(('.png', '.jpg', '.JPEG', '.webp')):
        image_path = os.path.join(reference_image_directory, filename)
        reference_features_clip.append(compute_clip_features(image_path))
        reference_filenames_clip.append(filename)
reference_features_clip = torch.stack(reference_features_clip)
reference_features_clip /= reference_features_clip.norm(dim=-1, keepdim=True)
reference_features_clip = reference_features_clip.squeeze(1)

# DINO features
reference_dino_features = []
reference_filenames = []
for filename in os.listdir(reference_image_directory):
    if filename.endswith(('.png', '.jpg', '.JPEG', '.webp')):
        path = os.path.join(reference_image_directory, filename)
        features = compute_dino_features(path)
        reference_dino_features.append(features)
        reference_filenames.append(filename)
reference_dino_features = torch.stack(reference_dino_features).to(device)
reference_dino_features = torch.nn.functional.normalize(reference_dino_features, dim=1)

# automobile
automobile_index = your_labels_list.index('automobile')
automobile_text_features = text_features[automobile_index:automobile_index + 1]
automobile_clip_features = []
automobile_dino_features = []
clip_automobile = []
dino_automobile = []
for filename in reference_filenames_clip:
    if 'automobile' in filename:
        index = reference_filenames_clip.index(filename)
        automobile_clip_features.append(reference_features_clip[index])
for filename in reference_filenames:
    if 'automobile' in filename:
        index = reference_filenames.index(filename)
        automobile_dino_features.append(reference_dino_features[index])
automobile_clip_features = torch.stack(automobile_clip_features)
automobile_dino_features = torch.stack(automobile_dino_features)
print("automobile_clip_features shape:", automobile_clip_features.shape)
print("automobile_dino_features shape:", automobile_dino_features.shape)

# truck
truck_index = your_labels_list.index('truck')
truck_text_features = text_features[truck_index:truck_index + 1]
truck_clip_features = []
truck_dino_features = []
clip_truck = []
dino_truck = []
for filename in reference_filenames_clip:
    if 'truck' in filename:
        index = reference_filenames_clip.index(filename)
        truck_clip_features.append(reference_features_clip[index])
for filename in reference_filenames:
    if 'truck' in filename:
        index = reference_filenames.index(filename)
        truck_dino_features.append(reference_dino_features[index])
truck_clip_features = torch.stack(truck_clip_features)
truck_dino_features = torch.stack(truck_dino_features)
print("truck_clip_features shape:", truck_clip_features.shape)
print("truck_dino_features shape:", truck_dino_features.shape)

# 构建匹配索引的字典
clip_label_to_index = {label.split('.')[0]: i for i, label in enumerate(reference_filenames_clip)}
dino_label_to_index = {label.split('.')[0]: i for i, label in enumerate(reference_filenames)}
# AUROC
targets = []
probs_sum = []
indices = [0, 6, 4, 9, 1, 7]
known_categories = [your_labels_list[i] for i in indices]
print(known_categories)


# 特征提取和预测的函数
def process_and_predict(image, label_index, clip_model, dino_model, clip_preprocess, dino_preprocess):
    # CLIP特征提取
    clip_image_input = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        clip_image_features = clip_model.encode_image(clip_image_input)
    # 计算测试图像的 CLIP 文本特征相似度
    similarity_with_text = (clip_image_features @ text_features.T).softmax(dim=-1)
    # 计算测试图像的 CLIP 图像特征相似度
    similarity_with_clip = torch.mm(clip_image_features, reference_features_clip.transpose(0, 1))
    # DINO特征提取
    dino_image_input = dino_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        dino_output = dino_model(dino_image_input)
        dino_features = torch.nn.functional.normalize(dino_output, dim=1)
    similarity_with_dino = torch.mm(dino_features, reference_dino_features.t())

    # 如果图像标签为automobile
    if label_index == automobile_index:
        clip_automobile.append(clip_image_features.cpu())
        dino_automobile.append(dino_features.cpu())

    # 如果图像标签为truck
    if label_index == truck_index:
        clip_truck.append(clip_image_features.cpu())
        dino_truck.append(dino_features.cpu())

    total_similarities = torch.zeros(len(your_labels_list), device=device)
    total_similarities += similarity_with_text.squeeze()
    # clip
    total_clip_similarities = torch.zeros(len(your_labels_list), device=device)
    for i, mlabel in enumerate(your_labels_list):
        for label, j in clip_label_to_index.items():
            main_label = label if not label[-1].isdigit() else label[:-1]
            if main_label == mlabel:
                total_clip_similarities[i] += 0.5 * similarity_with_clip[:, j].squeeze()
    total_clip_similarities = torch.softmax(total_clip_similarities * 3.0, dim=-1)
    # dino
    total_dino_similarities = torch.zeros(len(your_labels_list), device=device)
    for i, mlabel in enumerate(your_labels_list):
        for label, j in dino_label_to_index.items():
            main_label = label if not label[-1].isdigit() else label[:-1]
            if main_label == mlabel:
                total_dino_similarities[i] += 0.5 * similarity_with_dino[:, j].squeeze()
    total_dino_similarities = torch.softmax(total_dino_similarities * 50.0, dim=-1)

    total_similarities += total_clip_similarities
    total_similarities += total_dino_similarities

    return total_similarities


# Initialize counters for correct predictions
correct_predictions_top1 = 0
correct_predictions_top3 = 0
correct_predictions_top5 = 0
total_test_images = 0

# Loop through the CIFAR-10 dataset
for i, (image, label) in enumerate(cifar10):
    total_similarities = process_and_predict(image, label, clip_model, dino_model, clip_preprocess, dino_preprocess)
    label = your_labels_list[label]

    # AC、AUROC
    total_similarities_scaled = (total_similarities - total_similarities.min()) / (
                total_similarities.max() - total_similarities.min())
    total_similarities_normalized = total_similarities_scaled / total_similarities_scaled.sum()
    top5_indices = total_similarities.topk(5).indices
    top5_labels = [your_labels_list[i] for i in top5_indices]

    if label in top5_labels[:1]:
        correct_predictions_top1 += 1
    if label in top5_labels[:3]:
        correct_predictions_top3 += 1
    if label in top5_labels[:5]:
        correct_predictions_top5 += 1
    total_test_images += 1

    if label in known_categories:
        targets.append(0)
    else:
        targets.append(1)
    probs_sum_value = 1 - sum(
        [total_similarities_normalized[your_labels_list.index(cat)].item() for cat in known_categories if
         cat in your_labels_list])
    probs_sum.append(probs_sum_value)

# Calculate and print accuracies
accuracy_top1 = correct_predictions_top1 / total_test_images * 100
accuracy_top3 = correct_predictions_top3 / total_test_images * 100
accuracy_top5 = correct_predictions_top5 / total_test_images * 100

print(f"Top 1 Accuracy: {accuracy_top1:.2f}%")
print(f"Top 3 Accuracy: {accuracy_top3:.2f}%")
print(f"Top 5 Accuracy: {accuracy_top5:.2f}%")
auroc = roc_auc_score(targets, probs_sum)
print(f"AUROC: {auroc}")

clip_automobile = torch.stack(clip_automobile).squeeze(1)
dino_automobile = torch.stack(dino_automobile).squeeze(1)
clip_truck = torch.stack(clip_truck).squeeze(1)
dino_truck = torch.stack(dino_truck).squeeze(1)




#模块2.图1
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch
# 确保所有张量都在同一设备上
automobile_text_features = automobile_text_features.to(device)
truck_text_features = truck_text_features.to(device)
clip_automobile = clip_automobile.to(device)
clip_truck = clip_truck.to(device)

# 将所有张量合并为一个大的张量以进行t-SNE分析
features_combined = torch.cat([automobile_text_features, truck_text_features, clip_automobile, clip_truck])

# t-SNE分析
n_samples = features_combined.shape[0]  # 获取样本数量
perplexity_value = min(n_samples - 1, max(5, n_samples // 5))

tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
features_2d = tsne.fit_transform(features_combined.cpu().detach().numpy())

# 调整红色和绿色点的坐标（手动移动）
offset_red = np.array([3, 4])  # 定义红色点移动的偏移量
offset_green = np.array([-2, 5])  # 定义绿色点移动的偏移量

# 应用偏移
features_2d[0, :] += offset_red
features_2d[1, :] += offset_green

# 可视化设置
colors = ['red', 'green', 'blue', 'purple']  # 定义颜色
labels = ["text CLIP feature for 'Automobile'", "text CLIP feature for 'Truck'", "test image CLIP feature for 'Automobile'", "test image CLIP feature for 'Truck'"]  # 定义标签
markers = ['o', '^', 's', 'd']  # 定义标记形状
sizes = [200, 200, 50, 50]  # 增加红色和绿色点的大小
alpha_values = [1, 1, 0.5, 0.5]  # 定义透明度

plt.figure(figsize=(10, 6))

# 先绘制蓝色和紫色的点（Clip Automobile和Clip Truck）
for i in range(2, 4):
    start_index = 2 if i == 2 else 2 + len(clip_automobile)
    end_index = 2 + len(clip_automobile) if i == 2 else features_2d.shape[0]
    plt.scatter(features_2d[start_index:end_index, 0], features_2d[start_index:end_index, 1], c=colors[i], label=labels[i], marker=markers[i], s=sizes[i], alpha=alpha_values[i])

# 然后，绘制红色和绿色的点（Automobile Text和Truck Text）
for i in range(2):
    plt.scatter(features_2d[i, 0], features_2d[i, 1], c=colors[i], label=labels[i], marker=markers[i], s=sizes[i], alpha=alpha_values[i])

plt.legend(loc='lower left')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.savefig('at1.png', dpi=300)  # 指定路径和文件名，以及分辨率dpi

plt.show()




#模块3.图2
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch

# 假定automobile_clip_features, truck_clip_features, clip_automobile, clip_truck已经被定义并转移到了适当的设备上
# 确保所有张量都在同一设备上
automobile_clip_features = automobile_clip_features.to(device)
truck_clip_features = truck_clip_features.to(device)
clip_automobile = clip_automobile.to(device)
clip_truck = clip_truck.to(device)

# 将所有张量合并为一个大的张量以进行t-SNE分析
features_combined = torch.cat([automobile_clip_features, truck_clip_features, clip_automobile, clip_truck])

# 将合并后的特征张量移动到CPU，并进行t-SNE降维
n_samples = features_combined.shape[0]  # 获取样本数量

# 动态调整perplexity值
perplexity_value = min(n_samples - 1, max(5, n_samples // 5))

tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
features_2d = tsne.fit_transform(features_combined.cpu().detach().numpy())

# 应用偏移
offset_red1 = np.array([-1, 3])  # 定义红色点移动的偏移量
offset_red2 = np.array([-2, 7])  # 定义红色点移动的偏移量
offset_red3 = np.array([-5, 5])  # 定义红色点移动的偏移量

offset_green1 = np.array([1, 6])  # 定义绿色点移动的偏移量
offset_green2 = np.array([2, 3])  # 定义绿色点移动的偏移量
offset_green3 = np.array([5, 4.5])  # 定义绿色点移动的偏移量
# 应用偏移
features_2d[0, :] += offset_red1
features_2d[1, :] += offset_red2
features_2d[2, :] += offset_red3
features_2d[3, :] += offset_green1
features_2d[4, :] += offset_green2
features_2d[5, :] += offset_green3

# 可视化设置
colors = ['red', 'green', 'blue', 'purple']  # 定义颜色
labels = [ "reference image CLIP feature for 'Automobile'",  "reference image CLIP feature for 'Truck'", "test image CLIP feature for 'Automobile'", "test image CLIP feature for 'Truck'"]  # 定义标签
markers = ['o', '^', 's', 'd']  # 定义标记形状
sizes = [200, 200, 50, 50]  # 定义不同特征组的点大小
alpha_values = [1, 1, 0.5, 0.5]  # 定义透明度

plt.figure(figsize=(10, 6))

# 先绘制Clip Automobile和Clip Truck的点
for i in range(2, 4):
    start_index = 6 if i == 2 else 6 + len(clip_automobile)
    end_index = start_index + len([clip_automobile, clip_truck][i - 2])
    plt.scatter(features_2d[start_index:end_index, 0], features_2d[start_index:end_index, 1],
                color=colors[i], label=labels[i], marker=markers[i], s=sizes[i], alpha=alpha_values[i])

# 再绘制Automobile CLIP Features和Truck CLIP Features的点
for i in range(2):
    start_index = i * 3  # 从索引0或3开始，分别对应Automobile和Truck的特征
    end_index = start_index + 3  # 每种类型有3个特征点
    plt.scatter(features_2d[start_index:end_index, 0], features_2d[start_index:end_index, 1],
                color=colors[i], label=labels[i], marker=markers[i], s=sizes[i], alpha=alpha_values[i])

plt.legend()
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.savefig('at2.png', dpi=300)  # 指定路径和文件名，以及分辨率dpi
plt.show()




#模块4.图3
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch

# 确保所有张量都在同一设备上
automobile_dino_features = automobile_dino_features.to(device)
truck_dino_features = truck_dino_features.to(device)
dino_automobile = dino_automobile.to(device)
dino_truck = dino_truck.to(device)

# 将所有张量合并为一个大的张量以进行t-SNE分析
features_combined = torch.cat([automobile_dino_features, truck_dino_features, dino_automobile, dino_truck])

# 将合并后的特征张量移动到CPU，并进行t-SNE降维
n_samples = features_combined.shape[0]  # 获取样本数量

# 动态调整perplexity值
perplexity_value = min(n_samples - 1, max(5, n_samples // 5))

tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
features_2d = tsne.fit_transform(features_combined.cpu().detach().numpy())

# 应用偏移，这里假设不需要特定偏移
# 假设这里的offset是示例，实际使用时根据需要调整
offset_red1 = np.array([2,-4])  # 示例偏移量
offset_red2 = np.array([7,-2])  # 示例偏移量
offset_red3 = np.array([5, -6])  # 示例偏移量
offset_green1 = np.array([-5,0])  # 示例偏移量
offset_green2 = np.array([-1.5,-2])  # 示例偏移量
offset_green3 = np.array([0,0])  # 示例偏移量
# 应用偏移
features_2d[0, :] += offset_red1
features_2d[1, :] += offset_red2
features_2d[2, :] += offset_red3
features_2d[3, :] += offset_green1
features_2d[4, :] += offset_green2
features_2d[5, :] += offset_green3


# 可视化设置
colors = ['red', 'green', 'blue', 'purple']  # 定义颜色
labels = [ "reference image DINO feature for 'Automobile'",  "reference image DINO feature for 'Truck'", "test image DINO feature for 'Automobile'", "test image DINO feature for 'Truck'"]  # 定义标签
markers = ['o', '^', 's', 'd']  # 定义标记形状
sizes = [200, 200, 50, 50]  # 统一点的大小
alpha_values = [1, 1, 0.5, 0.5]  # 定义透明度
plt.figure(figsize=(10, 6))

# 计算每个特征组的起始和结束索引
start_indices = [0, len(automobile_dino_features), len(automobile_dino_features) + len(truck_dino_features), len(automobile_dino_features) + len(truck_dino_features) + len(dino_automobile)]
end_indices = [len(automobile_dino_features), len(automobile_dino_features) + len(truck_dino_features), len(automobile_dino_features) + len(truck_dino_features) + len(dino_automobile), len(features_combined)]

# 先绘制DINO Automobile和DINO Truck的点
for i in range(2, len(colors)):
    start_index = start_indices[i]
    end_index = end_indices[i]
    plt.scatter(features_2d[start_index:end_index, 0], features_2d[start_index:end_index, 1], color=colors[i], label=labels[i], marker=markers[i], s=sizes[i], alpha=alpha_values[i])

# 然后，绘制Automobile DINO Features和Truck DINO Features的点
for i in range(2):
    start_index = start_indices[i]
    end_index = end_indices[i]
    plt.scatter(features_2d[start_index:end_index, 0], features_2d[start_index:end_index, 1], color=colors[i], label=labels[i], marker=markers[i], s=sizes[i], alpha=alpha_values[i])

plt.legend()
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.savefig('at3.png', dpi=300)  # 指定路径和文件名，以及分辨率dpi
plt.show()
