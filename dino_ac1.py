import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import re

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


# 归一化特征
reference_features = torch.nn.functional.normalize(reference_features, dim=1)


# 计算测试图像的特征并找出最相似的参考图像
test_dir = "pic_test/"

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

# 遍历测试图像
for test_filename in os.listdir(test_dir):
    if test_filename.endswith(('.png', '.jpg', '.JPEG')):
        test_path = os.path.join(test_dir, test_filename)
        test_features = extract_features(test_path)
        test_features = torch.nn.functional.normalize(test_features, dim=0)

        # 计算余弦相似度
        similarity = torch.mm(test_features.unsqueeze(0), reference_features.t())
        top_similarities, top_indices = similarity.topk(1)

        # 检查预测类别是否包含实际类别
        predicted_labels = [reference_filenames[index] for index in top_indices.squeeze(0)]
        actual_image_id = image_ids.get(test_filename)
        actual_label = id_to_label.get(actual_image_id, "")
        print("actual_label:",actual_label)##
        if any(check_prediction(label, actual_label) for label in predicted_labels):
            correct_predictions += 1
            print("correct_predictions:",correct_predictions)##
            
        # 打印结果
        print(f"Test Image: {test_filename}")
        for i, index in enumerate(top_indices.squeeze(0)):
            print(f"{i+1}: Similar Image: {reference_filenames[index]} with similarity score: {top_similarities.squeeze(0)[i].item()}")
        print("-----------")

# 计算准确率
accuracy = correct_predictions / len(os.listdir(test_dir))
print(f"Accuracy: {accuracy * 100:.2f}%")
