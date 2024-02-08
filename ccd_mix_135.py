import os
import clip
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import re

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load('ViT-B/32', device)

# Load the DINO model
dino_model = torch.hub.load('facebookresearch/dinov2', "dinov2_vitb14")
dino_model.eval()

# Define image preprocessing for DINO model
dino_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Specify the directories containing the images
reference_image_directory = 'pic_DALL-E_all/'  # Directory for reference images
test_image_directory = 'pic_test/val'  # Directory for test images


# test_image_directory = 'hh'#测试一下

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
    image = dino_preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = dino_model(image)
    return features.squeeze(0)


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

# Initialize variables for accuracy calculation
correct_predictions = 0
total_test_images = 0
your_labels_list = ['gondola', 'sewing_machine', 'sea_slug', 'lesser_panda', 'espresso', 'neck_brace', 'beaker',
                    'sandal', 'binoculars', 'turnstile', 'poncho', 'brain_coral', 'bullet_train', 'volleyball',
                    'monarch', 'lakeside', 'spider_web', 'barn', 'European_fire_salamander', 'police_van', 'goldfish',
                    'bullfrog', 'tailed_frog', 'American_alligator', 'boa_constrictor', 'trilobite', 'scorpion',
                    'black_widow', 'tarantula', 'centipede', 'goose', 'koala', 'jellyfish', 'snail', 'slug',
                    'American_lobster', 'spiny_lobster', 'black_stork', 'king_penguin', 'albatross', 'dugong',
                    'Chihuahua', 'Yorkshire_terrier', 'golden_retriever', 'Labrador_retriever', 'German_shepherd',
                    'standard_poodle', 'tabby', 'Persian_cat', 'Egyptian_cat', 'cougar', 'lion', 'brown_bear',
                    'ladybug', 'fly', 'bee', 'grasshopper', 'walking_stick', 'cockroach', 'mantis', 'dragonfly',
                    'sulphur_butterfly', 'sea_cucumber', 'guinea_pig', 'hog', 'ox', 'bison', 'bighorn', 'gazelle',
                    'Arabian_camel', 'orangutan', 'chimpanzee', 'baboon', 'African_elephant', 'abacus', 'academic_gown',
                    'altar', 'apron', 'backpack', 'bannister', 'barbershop', 'barrel', 'basketball', 'bathtub',
                    'beach_wagon', 'beacon', 'beer_bottle', 'bikini', 'birdhouse', 'bow_tie', 'brass', 'broom',
                    'bucket', 'butcher_shop', 'candle', 'cannon', 'cardigan', 'cash_machine', 'CD_player', 'chain',
                    'chest', 'Christmas_stocking', 'cliff_dwelling', 'computer_keyboard', 'confectionery',
                    'convertible', 'crane', 'dam', 'desk', 'dining_table', 'drumstick', 'dumbbel', 'flagpole',
                    'fountain', 'freight_car', 'frying_pan', 'fur_coat', 'gasmask', 'go-kart', 'hourglass', 'iPod',
                    'jinrikisha', 'kimono', 'lampshade', 'lawn_mower', 'lifeboat', 'limousine', 'magnetic_compass',
                    'maypole', 'military_uniform', 'miniskirt', 'moving_van', 'nail', 'obelisk', 'oboe', 'organ',
                    'parking_meter', 'pay-phone', 'picket_fence', 'pill_bottle', 'plunger', 'pole', 'pop_bottle',
                    "potter's_wheel", 'projectile', 'punching_bag', 'reel', 'refrigerator', 'remote_control',
                    'rocking_chair', 'rugby_ball', 'school_bus', 'scoreboard', 'snorkel', 'sock', 'sombrero',
                    'space_heater', 'sports_car', 'steel_arch_bridge', 'stopwatch', 'sunglasses', 'suspension_bridge',
                    'swimming_trunks', 'syringe', 'teapot', 'teddy', 'thatch', 'torch', 'tractor', 'triumphal_arch',
                    'trolleybus', 'umbrella', 'vestment', 'viaduct', 'water_jug', 'water_tower', 'wok', 'wooden_spoon',
                    'comic_book', 'plate', 'guacamole', 'ice_cream', 'ice_lolly', 'pretzel', 'mashed_potato',
                    'cauliflower', 'bell_pepper', 'mushroom', 'orange', 'lemon', 'banana', 'pomegranate', 'meat_loaf',
                    'pizza', 'potpie', 'alp', 'cliff', 'coral_reef', 'seashore', 'acorn']
# Iterate through each image in the test directory and its subdirectories

# 1. Define the text inputs
text_inputs = torch.cat([clip.tokenize(f"a photo of a {label}") for label in your_labels_list]).to(device)
# Calculate features
with torch.no_grad():
    text_features = clip_model.encode_text(text_inputs)
# 2.
reference_features_clip = []
reference_filenames_clip = []
for filename in os.listdir(reference_image_directory):
    if filename.endswith(('.png', '.jpg', '.JPEG')):
        image_path = os.path.join(reference_image_directory, filename)
        reference_features_clip.append(compute_clip_features(image_path))  #
        reference_filenames_clip.append(filename)
reference_features_clip = torch.stack(reference_features_clip)
# 归一化
reference_features_clip /= reference_features_clip.norm(dim=-1, keepdim=True)
# 移除 reference_features 中大小为 1 的维度
reference_features_clip = reference_features_clip.squeeze(1)
print("Updated reference features shape:", reference_features_clip.shape)
# 3.dino的
reference_dino_features = []
reference_filenames = []
for filename in os.listdir(reference_image_directory):
    if filename.endswith(('.png', '.jpg', '.JPEG')):
        path = os.path.join(reference_image_directory, filename)
        features = compute_dino_features(path)  #
        reference_dino_features.append(features)
        reference_filenames.append(filename)
reference_dino_features = torch.stack(reference_dino_features)
# 归一化特征
reference_dino_features = torch.nn.functional.normalize(reference_dino_features, dim=1)


def check_prediction(predicted_label, actual_label):
    # Remove non-alphabet characters and convert to lowercase
    clean_predicted = re.sub(r'[^a-zA-Z]', '', predicted_label).lower()
    clean_actual = re.sub(r'[^a-zA-Z]', '', actual_label).lower()
    # print("clean_predicted:", clean_predicted)
    # print("clean_actual:", clean_actual)
    return clean_actual == clean_predicted


correct_predictions_top1 = 0
correct_predictions_top3 = 0
correct_predictions_top5 = 0
total_test_images = 0
# --------------------------------------------------
# 之前已经计算好了 text_features, reference_features_clip 和 reference_dino_features
# 构建匹配索引的字典
clip_label_to_index = {label.split('.')[0]: i for i, label in enumerate(reference_filenames_clip)}
dino_label_to_index = {label.split('.')[0]: i for i, label in enumerate(reference_filenames)}
# 测试一下
# 输出 clip_label_to_index 字典的内容
for label, index in clip_label_to_index.items():
    print(f"Label: {label}, Index: {index}")

# 初始化累加器，用于存储每个类别的总相似度得分
category_scores = torch.zeros(len(your_labels_list), device=device)

# 对每个测试图像进行处理
for dirpath, _, filenames in os.walk(test_image_directory):
    for test_filename in filenames:
        if test_filename.endswith(('.png', '.jpg', '.JPEG')):
            test_image_path = os.path.join(dirpath, test_filename)

            # 计算测试图像的 CLIP 文本特征相似度
            image_features = compute_clip_features(test_image_path)
            similarity_with_text = (image_features @ text_features.T).softmax(dim=-1)
            # 计算测试图像的 CLIP 图像特征相似度
            similarity_with_clip = torch.mm(image_features, reference_features_clip.transpose(0, 1))  # .softmax(dim=-1)
            similarity_with_clip = F.softmax(similarity_with_clip * 10.0, dim=-1)
            # 计算测试图像的 DINO 图像特征相似度
            dino_features = compute_dino_features(test_image_path)
            similarity_with_dino = torch.mm(dino_features.unsqueeze(0), reference_dino_features.t())  # .softmax(dim=-1)
            similarity_with_dino = F.softmax(similarity_with_dino, dim=-1)

            # 获取每个模型最高相似度值及对应的索引
            # max_similarity_text, top_indices_text = similarity_with_text.topk(5)
            # max_similarity_clip, top_indices_clip = similarity_with_clip.topk(5)
            # max_similarity_dino, top_indices_dino = similarity_with_dino.topk(5)
            # # 根据索引获取对应的类别标签
            # top_labels_text = [[your_labels_list[index.item()] for index in indices] for indices in top_indices_text]
            # top_labels_clip = [[reference_filenames_clip[index.item()] for index in indices] for indices in
            #                    top_indices_clip]
            # top_labels_dino = [[reference_filenames[index.item()] for index in indices] for indices in top_indices_dino]
            # 打印预测到的前5名类别标签及其对应的相似度值
            # print("Top 5 Predicted Labels with Text:")
            # for labels, similarities in zip(top_labels_text, max_similarity_text):
            #     for label, similarity in zip(labels, similarities):
            #         print(f"Label: {label}, Similarity: {similarity.item()}")
            # print("\nTop 5 Predicted Labels with CLIP:")
            # for labels, similarities in zip(top_labels_clip, max_similarity_clip):
            #     for label, similarity in zip(labels, similarities):
            #         print(f"Label: {label}, Similarity: {similarity.item()}")
            # print("\nTop 5 Predicted Labels with DINO:")
            # for labels, similarities in zip(top_labels_dino, max_similarity_dino):
            #     for label, similarity in zip(labels, similarities):
            #         print(f"Label: {label}, Similarity: {similarity.item()}")


            total_similarities = torch.zeros(len(your_labels_list), device=device)
            # 使用 similarity_with_text 的结果，因为它直接对应于 your_labels_list 的索引
            total_similarities += similarity_with_text.squeeze()

            for i, label in enumerate(your_labels_list):
                if label in clip_label_to_index:
                    clip_index = clip_label_to_index[label]
                    # 累加 similarity_with_clip 的得分
                    total_similarities[i] += similarity_with_clip[:, clip_index].squeeze()
                if label in dino_label_to_index:
                    dino_index = dino_label_to_index[label]
                    # 累加 similarity_with_dino 的得分
                    total_similarities[i] += similarity_with_dino[:, dino_index].squeeze()
            # 找到总相似度最高的五个类别
            top_categories_indices = total_similarities.topk(5).indices  #######
            top_categories_labels = [your_labels_list[i] for i in top_categories_indices]
            # 打印前五个类别及其相似度
            print("Top 5 Categories based on Total Similarity Scores:")
            for index in top_categories_indices:
                print(
                    f"Category: {your_labels_list[index]}, Total Similarity Score: {total_similarities[index].item()}")

            # 获取实际标签
            actual_image_id = image_ids.get(test_filename)
            actual_label = id_to_label.get(actual_image_id, "")
            actual_label = re.sub(r"^\d+\s+", "", actual_label)
            print("actual_label:", actual_label)

            # 检查前一、前三、前五预测的准确性
            if actual_label in top_categories_labels[:1]:
                correct_predictions_top1 += 1
                print("correct_predictions_top1:", correct_predictions_top1)
            if actual_label in top_categories_labels[:3]:
                correct_predictions_top3 += 1
            if actual_label in top_categories_labels[:5]:
                correct_predictions_top5 += 1

            total_test_images += 1
            print("total_test_images:", total_test_images)

# 计算并打印准确性
accuracy_top1 = correct_predictions_top1 / total_test_images * 100
accuracy_top3 = correct_predictions_top3 / total_test_images * 100
accuracy_top5 = correct_predictions_top5 / total_test_images * 100

print(f"Top 1 Accuracy: {accuracy_top1:.2f}%")
print(f"Top 3 Accuracy: {accuracy_top3:.2f}%")
print(f"Top 5 Accuracy: {accuracy_top5:.2f}%")


