
import os
import clip
import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from PIL import Image
import re
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

# Load model and CIFAR-10
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load('ViT-B/32', device)
# Load the DINO model
dino_model = torch.hub.load('facebookresearch/dinov2', "dinov2_vitb14")
dino_model.eval().to(device)

dino_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

cifar10 = CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False)
your_labels_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#
reference_image_directory = 'cifar10_dalle/'  ####

def entropy(probabilities):

    return -torch.sum(probabilities * torch.log(probabilities + 1e-6), dim=-1)


def compute_clip_features(image_path):
    image = Image.open(image_path)
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
    return image_features


# Function to compute DINO image features
def compute_dino_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = dino_preprocess(image).unsqueeze(0).to(device)  #
    with torch.no_grad():
        features = dino_model(image)  #
    return features.squeeze(0)


# Prepare text inputs
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar10.classes]).to(device)
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
#
reference_features_clip = reference_features_clip.squeeze(1)
print("Updated reference features shape:", reference_features_clip.shape)
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
reference_dino_features = torch.nn.functional.normalize(reference_dino_features, dim=1)


# Build a dictionary that matches the index
clip_label_to_index = {label.split('.')[0]: i for i, label in enumerate(reference_filenames_clip)}
dino_label_to_index = {label.split('.')[0]: i for i, label in enumerate(reference_filenames)}

# AUROC
targets = []
probs_sum = []
indices = [8, 6, 1, 9, 0, 7]
known_categories = [your_labels_list[i] for i in indices]

print(known_categories)

def process_and_predict(image, clip_model, dino_model, clip_preprocess, dino_preprocess):

    clip_image_input = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        clip_image_features = clip_model.encode_image(clip_image_input)



    similarity_with_text = (1.0 * clip_image_features @ text_features.T).softmax(dim=-1)  #
    similarity_with_clip = torch.mm(clip_image_features, reference_features_clip.transpose(0, 1))  # 400
    dino_image_input = dino_preprocess(image).unsqueeze(0).to(device)  #
    with torch.no_grad():
        # DINO的特征提取逻辑可能需要根据实际模型进行调整
        dino_output = dino_model(dino_image_input)
        dino_features = torch.nn.functional.normalize(dino_output, dim=1)
    similarity_with_dino = torch.mm(dino_features, reference_dino_features.t())  #


    total_similarities = torch.zeros(len(your_labels_list), device=device)  #
    # text
    total_similarities += (1 / (text_entropy.item() + 1e-6)) * similarity_with_text.squeeze()  #

    # clip
    total_clip_similarities = torch.zeros(len(your_labels_list), device=device)  # 2
    for i, mlabel in enumerate(your_labels_list):  # 200
        for label, j in clip_label_to_index.items():  # 400
            main_label = label if not label[-1].isdigit() else label[:-1]  #
            if main_label == mlabel:
                total_clip_similarities[i] += 0.5 * similarity_with_clip[:, j].squeeze()  # 400->200
    total_clip_similarities = torch.softmax(total_clip_similarities * 2.0, dim=-1)  #
    clip_entropy = entropy(total_clip_similarities)
    # dino
    total_dino_similarities = torch.zeros(len(your_labels_list), device=device)  #
    for i, mlabel in enumerate(your_labels_list):  #
        for label, j in dino_label_to_index.items():  #
            main_label = label if not label[-1].isdigit() else label[:-1]  #
            if main_label == mlabel:
                total_dino_similarities[i] += 0.5 * similarity_with_dino[:, j].squeeze()  # 400->200
    total_dino_similarities = torch.softmax(total_dino_similarities * 30.0, dim=-1)  # softmax

    dino_entropy = entropy(total_dino_similarities)
    dino_entropy = dino_entropy.to(device)  #

    total_similarities += (1 / (clip_entropy.item() + 1e-6)) * total_clip_similarities
    total_similarities += (1 / (dino_entropy.item() + 1e-6)) * total_dino_similarities

    values, indices = total_similarities.topk(1)
    return total_similarities


# Initialize counter for correct predictions
correct_predictions_top1 = 0
correct_predictions_top3 = 0
correct_predictions_top5 = 0
total_test_images = 0
#
label_stats = {label: {'correct': 0, 'total': 0} for label in your_labels_list}
zx_text = []
zx_clip = []
zx_dino = []
# Loop through the CIFAR-10 dataset
for i, (image, label) in enumerate(cifar10):
    total_similarities = process_and_predict(image, clip_model, dino_model, clip_preprocess, dino_preprocess)

    total_similarities_scaled = (total_similarities - total_similarities.min()) / (
                total_similarities.max() - total_similarities.min())
    total_similarities_normalized = total_similarities_scaled / total_similarities_scaled.sum()

    label = your_labels_list[label]  #

    top5_indices = total_similarities.topk(5).indices  #
    top5_labels = [your_labels_list[i] for i in top5_indices]

    if label in top5_labels[:1]:
        correct_predictions_top1 += 1
        # print("correct_predictions_top1:", correct_predictions_top1)
    if label in top5_labels[:3]:
        correct_predictions_top3 += 1
    if label in top5_labels[:5]:
        correct_predictions_top5 += 1
    total_test_images += 1
    print("total_test_images:", total_test_images)
    # print("\n")
    if label in known_categories:  ####改
        targets.append(0)  #
    else:
        targets.append(1)  #
    probs_sum_value = 1 - sum(
        [total_similarities_normalized[your_labels_list.index(cat)].item() for cat in known_categories if
         cat in your_labels_list])  #
    probs_sum.append(probs_sum_value)

    top1_index = total_similarities.topk(1).indices.item()
    predicted_label = your_labels_list[top1_index]  #
    #
    label_stats[label]['total'] += 1
    if predicted_label == label:
        label_stats[label]['correct'] += 1
    # if total_test_images>=50:
    #     break


accuracy_top1 = correct_predictions_top1 / total_test_images * 100
accuracy_top3 = correct_predictions_top3 / total_test_images * 100
accuracy_top5 = correct_predictions_top5 / total_test_images * 100

print(f"Top 1 Accuracy: {accuracy_top1:.2f}%")
print(f"Top 3 Accuracy: {accuracy_top3:.2f}%")
print(f"Top 5 Accuracy: {accuracy_top5:.2f}%")

auroc = roc_auc_score(targets, probs_sum)
print(f"AUROC: {auroc}")


for label, stats in label_stats.items():
    if stats['total'] > 0:
        accuracy = stats['correct'] / stats['total'] * 100
        print(f'{label}: Accuracy = {accuracy:.2f}% ({stats["correct"]}/{stats["total"]})')


accuracy_top1 = correct_predictions_top1 / total_test_images * 100
print(f"Overall Top 1 Accuracy: {accuracy_top1:.2f}%")