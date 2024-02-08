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
# test_image_directory = 'pic_test/val'  # Directory for test images
test_image_directory = 'hh'  # 测试一下


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
reference_features_clip = []
reference_filenames_clip = []
for filename in os.listdir(reference_image_directory):
    if filename.endswith(('.png', '.jpg', '.JPEG')):
        image_path = os.path.join(reference_image_directory, filename)
        reference_features_clip.append(compute_clip_features(image_path))
        reference_filenames_clip.append(filename)
reference_features_clip = torch.stack(reference_features_clip)
# 归一化
reference_features_clip /= reference_features_clip.norm(dim=-1, keepdim=True)
# 移除 reference_features 中大小为 1 的维度
reference_features_clip = reference_features_clip.squeeze(1)
print("Updated reference features shape:", reference_features_clip.shape)

reference_dino_features = []
reference_filenames = []
for filename in os.listdir(reference_image_directory):
    if filename.endswith(('.png', '.jpg', '.JPEG')):
        path = os.path.join(reference_image_directory, filename)
        features = compute_dino_features(path)
        reference_dino_features.append(features)
        reference_filenames.append(filename)
reference_dino_features = torch.stack(reference_dino_features)
# 归一化特征
reference_dino_features = torch.nn.functional.normalize(reference_dino_features, dim=1)


def check_prediction(predicted_label, actual_label):
    # Remove non-alphabet characters and convert to lowercase
    clean_predicted = re.sub(r'[^a-zA-Z]', '', predicted_label).lower()
    clean_actual = re.sub(r'[^a-zA-Z]', '', actual_label).lower()
    return clean_actual in clean_predicted


correct_predictions = 0
total_test_images = 0
for dirpath, _, filenames in os.walk(test_image_directory):
    for test_filename in filenames:
        if test_filename.endswith(('.png', '.jpg', '.JPEG')):
            test_image_path = os.path.join(dirpath, test_filename)

            # 1. Compute CLIP features for text-image alignment
            # Define the text inputs
            text_inputs = torch.cat([clip.tokenize(f"a photo of a {label}") for label in your_labels_list]).to(device)
            # Calculate features
            with torch.no_grad():
                text_features = clip_model.encode_text(text_inputs)
            image_features = compute_clip_features(test_image_path)
            similarity = (image_features @ text_features.T).softmax(dim=-1)

            # 2. Compute CLIP features for image-image alignment
            # image_features_2 = compute_clip_features(test_image_path)
            test_features = compute_clip_features(test_image_path)  ##删
            test_features = test_features.squeeze(0)
            test_features /= test_features.norm(dim=-1, keepdim=True)
            test_features = test_features.unsqueeze(0)
            # Compute features for each image in the reference directory计算参考图像集的特征
            similarity_clip = torch.mm(test_features, reference_features_clip.transpose(0, 1))  # 余弦相似度
            similarity_probabilities_clip = F.softmax(similarity_clip * 100.0, dim=1)

            # 3. Compute DINO features for image-image alignment
            dino_features = compute_dino_features(test_image_path)
            # reference_dino_features = torch.load('reference_dino_features.pt')
            similarity_dino = torch.mm(dino_features.unsqueeze(0), reference_dino_features.t())
            similarity_probabilities_dino = torch.nn.functional.softmax(similarity_dino, dim=1)

            # Choose the prediction with the highest similarity probability
            max_similarity_prob_clip = similarity.max()  # ？我改了
            max_similarity_prob_clip_idx = similarity.argmax()

            max_similarity_prob_clip_2 = similarity_probabilities_clip.max()
            max_similarity_prob_clip_idx_2 = similarity_probabilities_clip.argmax()

            max_similarity_prob_dino = similarity_probabilities_dino.max()
            max_similarity_prob_dino_idx = similarity_probabilities_dino.argmax()

            print(f"Test Image: {test_filename}")
            top_probabilities, top_indices = similarity_probabilities_clip.topk(1)  ##
            top_probabilities, top_indices = similarity_probabilities_dino.topk(1)  ##换顺序！！！就好了
            if max_similarity_prob_clip > max_similarity_prob_clip_2 and max_similarity_prob_clip > max_similarity_prob_dino:
                predicted_label = your_labels_list[max_similarity_prob_clip_idx]
                # CLIP text-image alignment has the highest similarity
                values, indices = similarity.max(1)  # topk
                # print("Prediction from CLIP text-image alignment:")
                # for value, index in zip(values, indices):
                #     print(f"Label: {your_labels_list[index]}, Similarity: {value.item()}")
            elif max_similarity_prob_clip_2 > max_similarity_prob_clip and max_similarity_prob_clip_2 > max_similarity_prob_dino:
                # CLIP image-image alignment has the highest similarity
                top_probabilities, top_indices = similarity_probabilities_clip.topk(1)
                predicted_label = reference_filenames[top_indices.view(-1)[0]]
                # reference_filenames = os.listdir(reference_image_directory)#前面已经有了的
                # print("Prediction from CLIP image-image alignment:")
                # for i, index in enumerate(top_indices.view(-1)):
                #     index = index.item()
                #     print(
                #         f"{i + 1}: {reference_filenames[index]} with similarity probability: {top_probabilities.view(-1)[i].item()}")
            else:
                # DINO image-image alignment has the highest similarity
                top_probabilities, top_indices = similarity_probabilities_dino.topk(1)  ##换顺序！！！就好了
                predicted_label = reference_filenames[top_indices.squeeze(0)[0].item()]
                print("Prediction from DINO image-image alignment:")
                for i, index in enumerate(top_indices.squeeze(0)):
                    # 确保 index 是一个整数
                    index = index.item()  # 此处将 tensor 转换为 integer
                    # print(f"Index: {index}, Type of Index: {type(index)}")  # 测试一下
                    # probability = top_probabilities.squeeze(0)[i].item()  # 同样确保概率是一个数字
                    # print(
                    #     f"{i + 1}: Similar Image: {reference_filenames[index]} with similarity probability: {probability}")

            actual_image_id = image_ids.get(test_filename)
            actual_label = id_to_label.get(actual_image_id, "")
            print("actual_label:", actual_label)

            # Check if the predicted label matches the actual label
            if check_prediction(predicted_label, actual_label):
                correct_predictions += 1
                print("correct_predictions:", correct_predictions)
            total_test_images += 1
            print("-----------")

# Calculate and print accuracy
accuracy = correct_predictions / total_test_images
print("total_test_images:", total_test_images)
print(f"Accuracy: {accuracy * 100:.2f}%")



