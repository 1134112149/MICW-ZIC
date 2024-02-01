import os
import clip
import torch
from PIL import Image
import numpy as np
import re

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Specify the directories containing the images
reference_image_directory = 'pic_DALL-E_all/'  # 第一个目录
test_image_directory = 'pic_test/'            # 第二个目录

# Function to compute image features
def compute_image_features(image_path):
    image = Image.open(image_path)
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)#clip提取特征
    return image_features

# Compute features for each image in the reference directory计算参考图像集的特征
reference_features = []
reference_filenames = []
for filename in os.listdir(reference_image_directory):
    if filename.endswith(('.png', '.jpg', '.JPEG')):
        image_path = os.path.join(reference_image_directory, filename)
        reference_features.append(compute_image_features(image_path))
        reference_filenames.append(filename)
# Normalize the reference features
reference_features = torch.stack(reference_features)
reference_features /= reference_features.norm(dim=-1, keepdim=True)

# 移除 reference_features 中大小为 1 的维度
reference_features = reference_features.squeeze(1)
print("Updated reference features shape:", reference_features.shape)

# 从 val_annotations.txt 读取ImageNet ID
def load_image_ids(filename):
    image_ids = {}
    with open(filename, "r") as file:
        for line in file:
            parts = line.strip().split("\t")
            image_name = parts[0]
            image_id = parts[1]
            image_ids[image_name] = image_id
    return image_ids

# 从 imagenet_id_to_label.txt 映射ID到类别名称
def map_ids_to_labels(filename):
    id_to_label = {}
    with open(filename, "r") as file:
        for line in file:
            parts = line.strip().split(" ")
            image_id = parts[0]
            label = " ".join(parts[1:])  # 类别名称可能包含空格
            id_to_label[image_id] = label
    return id_to_label

image_ids = load_image_ids("val_annotations.txt")
id_to_label = map_ids_to_labels("imagenet_id_to_label.txt")

# 比较预测类别和实际类别
def check_prediction(predicted_label, actual_label):   
    # 移除非字母字符，并将所有字母转换为小写
    clean_predicted = re.sub(r'[^a-zA-Z]', '', predicted_label).lower()
    clean_actual = re.sub(r'[^a-zA-Z]', '', actual_label).lower()
    print("clean_predicted:",clean_predicted)## dalleahighlydetailedandrealisticimageofawatertowertheimageshouldfocusonthewatertowersdistinctivefeaturessuchasitslargecylindricpng
    print("clean_actual:",clean_actual)##watertower
    return clean_actual in clean_predicted

# 初始化准确预测的计数
correct_predictions = 0

# Process each test image and find top 5 similar images from the reference directory
for test_filename in os.listdir(test_image_directory):
    if test_filename.endswith(('.png', '.jpg', '.JPEG')):
        test_image_path = os.path.join(test_image_directory, test_filename)
        # Compute features for the test image
        test_features = compute_image_features(test_image_path)
        test_features = test_features.squeeze(0)  # 确保 test_features 是一维向量
        test_features /= test_features.norm(dim=-1, keepdim=True)

        # 确保 test_features 是 [1, feature_dim]
        test_features = test_features.unsqueeze(0)
        print("Test features shape:", test_features.shape)

        # 计算余弦相似度
        similarity = torch.mm(test_features, reference_features.transpose(0, 1))
        top_similarities, top_indices = similarity.topk(1)#不能是1以上的数啊！？最后一版输出可以啦嘿嘿
        
        # 检查预测类别是否包含实际类别
        predicted_labels = [reference_filenames[index] for index in top_indices.squeeze(0)]
        actual_image_id = image_ids.get(test_filename)
        actual_label = id_to_label.get(actual_image_id, "")
        print("actual_label:",actual_label)##
        if any(check_prediction(label, actual_label) for label in predicted_labels):
            correct_predictions += 1
            print("correct_predictions:",correct_predictions)##
        
        # 打印结果,可以是topk(3)了嘿嘿
        print(f"Test Image: {test_filename}")
        for i, index in enumerate(top_indices.view(-1)):
            # 转换为整数索引
            index = index.item()
            print(f"{i+1}: {reference_filenames[index]} with similarity score: {top_similarities.view(-1)[i].item()}")
        print("-----------")


# 计算准确率
accuracy = correct_predictions / len(os.listdir(test_image_directory))
print(f"Accuracy: {accuracy * 100:.2f}%")
