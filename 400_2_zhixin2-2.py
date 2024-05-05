import os
import clip
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import re
import math
import numpy as np
from sklearn.metrics import roc_auc_score

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


# test_image_directory = 'hh/'#测试一下

# 计算每个模型输出概率向量的信息熵entropy
def entropy(probabilities):
    # epsilon = 1e-10  # 避免log(0)造成的NaN
    # probabilities = torch.clamp(probabilities, epsilon, 1.0)  # 将概率限制在 [epsilon, 1.0] 之间
    return -torch.sum(probabilities * torch.log(probabilities + 1e-6), dim=-1)


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
                    'convertible', 'crane', 'dam', 'desk', 'dining_table', 'drumstick', 'dumbbell', 'flagpole',
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
    if filename.endswith(('.png', '.jpg', '.JPEG', '.webp')):
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
    if filename.endswith(('.png', '.jpg', '.JPEG', '.webp')):
        path = os.path.join(reference_image_directory, filename)
        features = compute_dino_features(path)  #
        reference_dino_features.append(features)
        reference_filenames.append(filename)
reference_dino_features = torch.stack(reference_dino_features)
# 归一化特征
reference_dino_features = torch.nn.functional.normalize(reference_dino_features, dim=1)

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
# for label, index in clip_label_to_index.items():
#     print(f"Label: {label}, Index: {index}")
# 初始化累加器，用于存储每个类别的总相似度得分
# category_scores = torch.zeros(len(your_labels_list), device=device)
# 初始化 targets 和 probs_sum 列表###第二种auroc
targets = []
probs_sum = []
known_categories = ['gondola', 'sewing_machine', 'sea_slug', 'lesser_panda', 'espresso', 'neck_brace', 'beaker',
                    'sandal', 'binoculars', 'turnstile', 'poncho', 'brain_coral', 'bullet_train', 'volleyball',
                    'monarch', 'lakeside', 'spider_web', 'barn', 'European_fire_salamander', 'police_van']
# 10000*3置信度
zx_text = []
zx_clip = []
zx_dino = []
# 对每个测试图像进行处理
for dirpath, _, filenames in os.walk(test_image_directory):
    for test_filename in filenames:
        if test_filename.endswith(('.png', '.jpg', '.JPEG', '.webp')):
            test_image_path = os.path.join(dirpath, test_filename)

            # 计算测试图像的 CLIP 文本特征相似度
            image_features = compute_clip_features(test_image_path)
            similarity_with_text = (image_features @ text_features.T).softmax(dim=-1)  # 200
            # 计算测试图像的 CLIP 图像特征相似度
            similarity_with_clip = torch.mm(image_features, reference_features_clip.transpose(0, 1))  # 400
            # similarity_with_clip = F.softmax(similarity_with_clip*2.0, dim=-1)#####改这里！！！目前2.0最好
            # 计算测试图像的 DINO 图像特征相似度
            dino_features = compute_dino_features(test_image_path)
            similarity_with_dino = torch.mm(dino_features.unsqueeze(0), reference_dino_features.t())  # 400
            # similarity_with_dino = F.softmax(similarity_with_dino, dim=-1)
            # 计算text的熵entropy
            text_entropy = entropy(similarity_with_text)  ####
            # clip_entropy = entropy(similarity_with_clip)
            # dino_entropy = entropy(similarity_with_dino)
            # #             看看过程！！全！索引2+标签3+分数1
            #            # # 获取每个模型最高相似度值1及对应的索引2
            #             max_similarity_text, top_indices_text = similarity_with_text.topk(5)
            #             max_similarity_clip, top_indices_clip = similarity_with_clip.topk(5)
            #             max_similarity_dino, top_indices_dino = similarity_with_dino.topk(5)
            #             # 打印每个模型的最高相似度值对应的索引
            #             print(f"Top Indices with Text: {top_indices_text.tolist()}")#
            #             print(f"Top Indices with CLIP: {top_indices_clip.tolist()}")#
            #             print(f"Top Indices with DINO: {top_indices_dino.tolist()}")#
            #             # 根据索引获取对应的类别标签3
            #             top_labels_text = [[your_labels_list[index.item()] for index in indices] for indices in top_indices_text]
            #             top_labels_clip = [[reference_filenames_clip[index.item()] for index in indices] for indices in top_indices_clip]
            #             top_labels_dino = [[reference_filenames[index.item()] for index in indices] for indices in top_indices_dino]
            #             # 打印预测到的前三名类别标签及其对应的相似度值
            #             print("Top 5 Predicted Labels with Text:")
            #             for labels, similarities in zip(top_labels_text, max_similarity_text):
            #                 for label, similarity in zip(labels, similarities):
            #                     print(f"Label: {label}, Similarity: {similarity.item()}")
            #             print("\nTop 5 Predicted Labels with CLIP:")
            #             for labels, similarities in zip(top_labels_clip, max_similarity_clip):
            #                 for label, similarity in zip(labels, similarities):
            #                     print(f"Label: {label}, Similarity: {similarity.item()}")
            #             print("\nTop 5 Predicted Labels with DINO:")
            #             for labels, similarities in zip(top_labels_dino, max_similarity_dino):
            #                 for label, similarity in zip(labels, similarities):
            #                     print(f"Label: {label}, Similarity: {similarity.item()}")

            # . 在这里添加总相似度的计算（按照类别标签相同的进行相加），并输出总相似度前5名的类别标签和相似度的值
            # similarity_with_text, similarity_with_clip, similarity_with_dino 已经计算完毕
            # 初始化一个与 your_labels_list 等长的全零张量，用于累计每个类别的总相似度
            # total_similarities = torch.zeros(len(your_labels_list), device=device)
            # 在这里将其他张量移动到相同的设备  ----这个方法2可以不用下面这个吧
            # similarity_with_text = similarity_with_text.to(device)
            # similarity_with_clip = similarity_with_clip.to(device)
            # dino_entropy = dino_entropy.to(device)# 确保 dino_entropy和similarity_with_dino 与 total_similarities 在相同设备上
            # similarity_with_dino = similarity_with_dino.to(device)
            # print("text_entropy:",text_entropy)#16
            # print("clip_entropy:",clip_entropy)#16
            # print("dino_entropy:",dino_entropy)#32
            # #方法一：熵取倒数
            # total_similarities += 1/(text_entropy + 1e-6 ) * similarity_with_text.squeeze()#200
            # for label,i in clip_label_to_index.items():#400
            #     main_label = label
            #     if label[-1].isdigit():#标签去掉最后一个数字，变成主标签
            #         main_label = label[:-1]
            #     clip_index = clip_label_to_index[main_label]
            #     dino_index = dino_label_to_index[main_label]
            #     #在这里定义text_index：是由your_labels_list按照main_label得到的索引值
            #     text_index = your_labels_list.index(main_label)
            #     # print("total_similarities shape:", total_similarities.shape)
            #     # print("total_similarities[text_index] shape:", total_similarities[text_index].shape)
            #     # print("clip_entropy shape:", clip_entropy.shape)
            #     # print("similarity_with_clip[:, i] shape:", similarity_with_clip[:, i].shape)
            #     #转换一下float16--32，不转也行，但要额外加.squeeze().unsqueeze(0)
            #     total_similarities = total_similarities.float()  # 将 total_similarities 转换为 float32
            #     clip_entropy = clip_entropy.float()  # 将 clip_entropy 转换为 float32
            #     similarity_with_clip_i = similarity_with_clip[:, i].squeeze().unsqueeze(0).float()  #转32
            #     dino_entropy = dino_entropy.float()  # 将 clip_entropy 转换为 float32
            #     similarity_with_dino_i = similarity_with_dino[:, i].squeeze().unsqueeze(0).float()  #转32
            #     # print("text_entropy:",text_entropy)#16因为没改这个，无所谓
            #     # print("clip_entropy:",clip_entropy)#32
            #     # print("dino_entropy:",dino_entropy)#32
            #     total_similarities[text_index] = total_similarities[text_index].unsqueeze(0) + (1/(clip_entropy + 1e-6 ) * similarity_with_clip_i) #400->200
            #     total_similarities[text_index] = total_similarities[text_index].unsqueeze(0) + (1/(dino_entropy + 1e-6 ) * similarity_with_dino_i) #400->200
            # #方法二：熵作指数.这个怎么没上面那个那么麻烦？
            # total_similarities += math.exp(-text_entropy)  * similarity_with_text.squeeze()#200
            # for label,i in clip_label_to_index.items():#400
            #     main_label = label
            #     if label[-1].isdigit():#标签去掉最后一个数字，变成主标签
            #         main_label = label[:-1]
            #     clip_index = clip_label_to_index[main_label]
            #     dino_index = dino_label_to_index[main_label]
            #     #在这里定义text_index：是由your_labels_list按照main_label得到的索引值
            #     text_index = your_labels_list.index(main_label)
            #     total_similarities[text_index] += math.exp(-clip_entropy) * similarity_with_clip[:, i].squeeze()#400->200#i！！！！！
            #     total_similarities[text_index] += math.exp(-dino_entropy) * similarity_with_dino[:, i].squeeze()#400->200
            #     # print("total_similarities shape:", total_similarities.shape)
            #     # print("total_similarities[text_index] shape:", total_similarities[text_index].shape)
            #     # print("clip_entropy shape:", clip_entropy.shape)
            #     # print("similarity_with_clip[:, i] shape:", similarity_with_clip[:, i].shape)

            # 这是新的
            # 初始化一个与 your_labels_list 等长的零向量，用于累计每个类别的总相似度
            total_similarities = torch.zeros(len(your_labels_list), device=device)  # 200初始化zong
            total_similarities += math.exp(-text_entropy) * similarity_with_text.squeeze()  # 200
            ##相同类别分数相加的方式：前面没softmax!!!!!这里每个方法算完成200维，再softmx
            total_clip_similarities = torch.zeros(len(your_labels_list), device=device)  # 200初始化clip
            total_dino_similarities = torch.zeros(len(your_labels_list), device=device)  # 200初始化dino
            for i, mlabel in enumerate(your_labels_list):  # 200
                for label, j in clip_label_to_index.items():  # 400
                    main_label = label if not label[-1].isdigit() else label[:-1]  # 主标签#400->200
                    if main_label == mlabel:
                        total_clip_similarities[i] += 0.5 * similarity_with_clip[:, j].squeeze()  # 400->200
            total_clip_similarities = torch.softmax(total_clip_similarities * 2.0, dim=-1)  # softmax
            for i, mlabel in enumerate(your_labels_list):  # 200
                for label, j in dino_label_to_index.items():  # 400
                    main_label = label if not label[-1].isdigit() else label[:-1]  # 主标签#400->200
                    if main_label == mlabel:
                        total_dino_similarities[i] += 0.5 * similarity_with_dino[:, j].squeeze()  # 400->200
            total_dino_similarities = torch.softmax(total_dino_similarities, dim=-1)  # softmax
            # 计算另外两个的熵entropy
            clip_entropy = entropy(total_clip_similarities)
            # print("total_clip_similarities::",total_clip_similarities)
            # print("clip_entropy::",clip_entropy)
            dino_entropy = entropy(total_dino_similarities)
            ## #zhixin2-2独有   #打印每个图的3个置信度
            zx_text.append(math.exp(-text_entropy))
            zx_clip.append(math.exp(-clip_entropy))
            zx_dino.append(math.exp(-dino_entropy))
            total_similarities += math.exp(-clip_entropy) * total_clip_similarities
            total_similarities += math.exp(-dino_entropy) * total_dino_similarities

            # 找到总相似度最高的五个类别
            top_categories_indices = total_similarities.topk(5).indices  #######
            top_categories_labels = [your_labels_list[i] for i in top_categories_indices]
            # 打印前五个类别及其相似度，这个一般留着打印看
            # print("Top 5 Categories based on Total Similarity Scores:")
            # for index in top_categories_indices:
            #     print(f"Category: {your_labels_list[index]}, Total Similarity Score: {total_similarities[index].item()}")

            # 选择总相似度得分最高的前五个类别
            top5_indices = total_similarities.topk(5).indices  # 重复了吧
            top5_labels = [your_labels_list[i] for i in top5_indices]
            # print("???top5_labels:", top5_labels)

            # 获取实际标签
            actual_image_id = image_ids.get(test_filename)
            actual_label = id_to_label.get(actual_image_id, "")
            actual_label = re.sub(r"^\d+\s+", "", actual_label)
            print("actual_label:", actual_label)

            # 检查前一、前三、前五预测的准确性
            if actual_label in top5_labels[:1]:
                correct_predictions_top1 += 1
                print("correct_predictions_top1:", correct_predictions_top1)
            if actual_label in top5_labels[:3]:
                correct_predictions_top3 += 1
                # print("correct_predictions_top3:", correct_predictions_top3)
            if actual_label in top5_labels[:5]:
                correct_predictions_top5 += 1
                # print("correct_predictions_top5:", correct_predictions_top5)

            total_test_images += 1
            print("total_test_images:", total_test_images)

            # print("total_similarities:", total_similarities)#原始的，三者的和，会超过1
            # 方法1: Min-Max标准化，按原比例将数值缩放到0和1之间，总和还是1
            total_similarities_scaled = (total_similarities - total_similarities.min()) / (
                        total_similarities.max() - total_similarities.min())
            total_similarities_normalized = total_similarities_scaled / total_similarities_scaled.sum()
            # print("total_similarities_normalized:", total_similarities_normalized)#标准化后的
            # # 获取最高相似度的类别 # top_category = top_categories_labels[0]
            # 检查实际标签（非预测标签）是否属于已知类别
            if actual_label in known_categories:  ####改了这里
                targets.append(0)  # 属于已知类别
            else:
                targets.append(1)  # 不属于已知类别
            probs_sum_value = 1 - sum(
                [total_similarities_normalized[your_labels_list.index(cat)].item() for cat in known_categories if
                 cat in your_labels_list])  # 计算这个样本属于已见过的20个类别的概率之和
            probs_sum.append(probs_sum_value)

    #         if total_test_images >= 10:
    #             break
    # if total_test_images >= 10:
    #     break

print("zx_text/clip/dino:")
print(zx_text)
print(zx_clip)
print(zx_dino)
# 将结果保存到文件中
with open('zhixin2-2.txt', 'w') as f:
    f.write("Max Similarity with Text:\n")
    f.write("\n".join(map(str, zx_text)))
    f.write("\n\n")

    f.write("Max Similarity with Clip:\n")
    f.write("\n".join(map(str, zx_clip)))
    f.write("\n\n")

    f.write("Max Similarity with Dino:\n")
    f.write("\n".join(map(str, zx_dino)))
print("结果已保存到zhixin2-2.txt文件中。")
# 计算并打印准确性
accuracy_top1 = correct_predictions_top1 / total_test_images * 100
accuracy_top3 = correct_predictions_top3 / total_test_images * 100
accuracy_top5 = correct_predictions_top5 / total_test_images * 100

print(f"Top 1 Accuracy: {accuracy_top1:.2f}%")
print(f"Top 3 Accuracy: {accuracy_top3:.2f}%")
print(f"Top 5 Accuracy: {accuracy_top5:.2f}%")

# 所以第一个参数里面到底有几个0
num_zeros = targets.count(0)
print("Number of zeros in targets:", num_zeros)
# print("targets:",targets)
# print("probs_sum:",probs_sum)
auroc = roc_auc_score(targets, probs_sum)
print(f"AUROC: {auroc}")

