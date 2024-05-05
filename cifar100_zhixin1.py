import os
import clip
import torch
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from PIL import Image
import re
import numpy as np
from sklearn.metrics import roc_auc_score

# Load model and CIFAR-100
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
    #方法二
dino_preprocess = transforms.Compose([#这俩都行这个多一张正确的？！#84.87%--好
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# CIFAR-100数据集
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
your_labels_list = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 
    'worm'
]


#我们的参考图集
reference_image_directory = 'cifar100_dalle/'

def compute_clip_features(image_path):
    image = Image.open(image_path)
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
    return image_features

# Function to compute DINO image features
def compute_dino_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = dino_preprocess(image).unsqueeze(0).to(device)#dino模型在gpu上，所以这里要把图片也放到gpu上
    with torch.no_grad():
        features = dino_model(image)#错设备不统一，后来好了
    return features.squeeze(0)
# Prepare text inputs
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)
text_features = clip_model.encode_text(text_inputs)
#2.clip的                
reference_features_clip = []
reference_filenames_clip = []
for filename in os.listdir(reference_image_directory):
    if filename.endswith(('.png', '.jpg', '.JPEG','.webp')):
        image_path = os.path.join(reference_image_directory, filename)
        reference_features_clip.append(compute_clip_features(image_path))#
        reference_filenames_clip.append(filename)
reference_features_clip = torch.stack(reference_features_clip)
# 归一化
reference_features_clip /= reference_features_clip.norm(dim=-1, keepdim=True)
# 移除 reference_features 中大小为 1 的维度
reference_features_clip = reference_features_clip.squeeze(1)
print("Updated reference features shape:", reference_features_clip.shape)
#3.dino的
reference_dino_features = []
reference_filenames = []
for filename in os.listdir(reference_image_directory):
    if filename.endswith(('.png', '.jpg', '.JPEG','.webp')):
        path = os.path.join(reference_image_directory, filename)
        features = compute_dino_features(path)#错
        reference_dino_features.append(features)
        reference_filenames.append(filename)
reference_dino_features = torch.stack(reference_dino_features).to(device)
# 归一化特征
reference_dino_features = torch.nn.functional.normalize(reference_dino_features, dim=1)    

# 构建匹配索引的字典
clip_label_to_index = {label.split('.')[0]: i for i, label in enumerate(reference_filenames_clip)}
dino_label_to_index = {label.split('.')[0]: i for i, label in enumerate(reference_filenames)}

#AUROC
targets = []
probs_sum = []
indices = [30, 25, 1, 9, 8, 0, 46, 52, 49, 71]
known_categories = [your_labels_list[i] for i in indices]
print(known_categories)

# total_similarities = torch.zeros(len(your_labels_list), device=device)#因为函数提出来了，所以设置为全局变量

# 特征提取和预测的函数
def process_and_predict(image, clip_model, dino_model, clip_preprocess, dino_preprocess):
    # CLIP特征提取
    clip_image_input = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        clip_image_features = clip_model.encode_image(clip_image_input)
        # clip_image_features /= clip_image_features.norm(dim=-1, keepdim=True)#先不norm
    # 计算测试图像的 CLIP 文本特征相似度
    similarity_with_text = (1.0*clip_image_features @ text_features.T).softmax(dim=-1)
    # 计算测试图像的 CLIP 图像特征相似度
    similarity_with_clip = torch.mm(clip_image_features, reference_features_clip.transpose(0, 1))#400
    # DINO特征提取
    dino_image_input = dino_preprocess(image).unsqueeze(0).to(device)# 添加批量维度并转移到gpu
    with torch.no_grad():
        # DINO的特征提取逻辑可能需要根据实际模型进行调整
        dino_output = dino_model(dino_image_input)
        dino_features = torch.nn.functional.normalize(dino_output, dim=1)
    similarity_with_dino = torch.mm(dino_features, reference_dino_features.t())#！！！
   

    #这是新的     2->1合并后才softmax  
 # 初始化一个与 your_labels_list 等长的零向量，用于累计每个类别的总相似度
    total_similarities = torch.zeros(len(your_labels_list), device=device)#200初始化zong
    #text
    max_similarity_text, _ = similarity_with_text.topk(1)#没变
    total_similarities +=max_similarity_text.item() *  similarity_with_text.squeeze()#200 text
# 相同类别分数相加的方式：前面没softmax!!!!!这里每个方法算完成200维，再softmx
    #clip
    total_clip_similarities = torch.zeros(len(your_labels_list), device=device)#200初始化clip
    for i, mlabel in enumerate(your_labels_list):#200
        for label,j in clip_label_to_index.items():#400
            main_label = label if not label[-1].isdigit() else label[:-1]  # 主标签#400->200
            if main_label == mlabel:
                total_clip_similarities[i] += 0.5*similarity_with_clip[:, j].squeeze()#400->200
    total_clip_similarities = torch.softmax(total_clip_similarities*3.0, dim=-1)#softmax#!!!
    #dino
    total_dino_similarities = torch.zeros(len(your_labels_list), device=device)#200初始化dino
    for i, mlabel in enumerate(your_labels_list):#200
        for label,j in dino_label_to_index.items():#400
            main_label = label if not label[-1].isdigit() else label[:-1]  # 主标签#400->200
            if main_label == mlabel:
                total_dino_similarities[i] += 0.5*similarity_with_dino[:, j].squeeze()#400->200
    total_dino_similarities = torch.softmax(total_dino_similarities*50.0, dim=-1)#softmax
##获取每个模型最高相似度值 #zhixin1独有   #打印每个图的3个置信度
    max_similarity_clip, _ = total_clip_similarities.topk(1)#变了
    max_similarity_dino, _ = total_dino_similarities.topk(1)#变了
    total_similarities +=max_similarity_clip.item() *  total_clip_similarities
    total_similarities +=max_similarity_dino.item() *  total_dino_similarities
    
    # # 找到总相似度最高的五个类别
    # top_categories_indices = total_similarities.topk(5).indices#######
    # top_categories_labels = [your_labels_list[i] for i in top_categories_indices]
    # # 打印前五个类别及其相似度
    # print("Top 5 Categories based on Total Similarity Scores:")
    # for index in top_categories_indices:
    #     print(f"Category: {your_labels_list[index]}, Total Similarity Score: {total_similarities[index].item()}")
                
    values, indices = total_similarities.topk(1)
    return total_similarities


# Initialize counter for correct predictions
correct_predictions_top1 = 0
correct_predictions_top3 = 0
correct_predictions_top5 = 0
total_test_images = 0
# 初始化标签统计字典
label_stats = {label: {'correct': 0, 'total': 0} for label in your_labels_list}

# Loop through the CIFAR-100 dataset
for i, (image, label) in enumerate(cifar100):
    total_similarities = process_and_predict(image, clip_model, dino_model, clip_preprocess, dino_preprocess)
    # values, indices, total_similarities = process_and_predict(image, clip_model, dino_model, clip_preprocess, dino_preprocess)     
    # predicted_label = indices[0].item() # Get the index of the highest prediction
    # if predicted_label == label: # Check if the prediction matches the label
    #     correct_predictions += 1
    #     print("correct_predictions::",correct_predictions)    
    
    #AUROC
    # print("total_similarities:", total_similarities)#原始的，三者的和，会超过1
    # 方法1: Min-Max标准化，按原比例将数值缩放到0和1之间，总和还是1
    total_similarities_scaled = (total_similarities - total_similarities.min()) / (total_similarities.max() - total_similarities.min())
    total_similarities_normalized = total_similarities_scaled / total_similarities_scaled.sum()
    
    label = your_labels_list[label]#cifar100数据集有一些不同，它的标签就是数字，可以再转为单词
    # print("label:标签:",label)
    #前5名
    top5_indices = total_similarities.topk(5).indices#
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
    if label in known_categories:####改了这里
        targets.append(0)  # 属于已知类别
    else:
        targets.append(1)  # 不属于已知类别
    probs_sum_value =1- sum([total_similarities_normalized[your_labels_list.index(cat)].item() for cat in known_categories if cat in your_labels_list])# 计算这个样本属于已见过的20个类别的概率之和
    probs_sum.append(probs_sum_value)
    

    top1_index = total_similarities.topk(1).indices.item()
    predicted_label = your_labels_list[top1_index]  # 预测标签
    # 更新标签统计
    label_stats[label]['total'] += 1
    if predicted_label == label:
        label_stats[label]['correct'] += 1
    # if total_test_images>=500:
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



# 计算并打印每个标签的预测正确率
for label, stats in label_stats.items():
    if stats['total'] > 0:  # 避免除以零
        accuracy = stats['correct'] / stats['total'] * 100
        print(f'{label}: Accuracy = {accuracy:.2f}% ({stats["correct"]}/{stats["total"]})')

# 打印整体预测正确性
accuracy_top1 = correct_predictions_top1 / total_test_images * 100
print(f"Overall Top 1 Accuracy: {accuracy_top1:.2f}%")
