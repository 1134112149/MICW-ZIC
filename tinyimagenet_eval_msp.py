import argparse
import torch
import os
from dataloaders.ZO_Clip_loaders import tinyimage_single_isolated_class_loader, tiny_single_isolated_class_dino_loader, tinyimage_semantic_spit_generator
from clip.simple_tokenizer import SimpleTokenizer as clip_tokenizer
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import json
from datetime import datetime
import sys
from utils_.utils_ import Logger, compute_oscr
from utils_.clip_utils import tokenize_for_clip
from utils_.dino_utils import extract_features
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser("tiny imagenet eval")
parser.add_argument("--gpu_devices", default=1, help="gpu device")
parser.add_argument("--k_images", default=10, type=int)
parser.add_argument("--save_dir", default="output/tiny_imagenet")
parser.add_argument("--image_path", default="output/self_debug/keep")

#CLIP的强项在于其能够处理和理解图像及其相关文本，
#DINO擅长在没有大量标注数据的情况下从图像中提取有用的特征

#函数可能用于解码图像，计算与模型性能相关的指标
def image_decoder(clip_model, dino_model, stored_features, k_images, device, image_loaders, split, chatgpt_manual_similar_label, detailed_labels=None):
    seen_labels = split[:20]
    if detailed_labels is not None:
        seen_descriptions = [f"This is a photo of a {detailed_labels[label]}" for label in seen_labels]
    else:
        seen_descriptions = [f"This is a photo of a {label}" for label in seen_labels]

    n_seen = sum([len(image_loaders[label]) for label in seen_labels])
    n_unseen = sum([len(image_loaders[label]) for label in split[20:]])
    targets = torch.tensor(n_seen*[0] + n_unseen*[1])
    

    clip_ood_probs_sum = []
    dino_ood_probs_sum = []
    clip_closeset_probs_sum = []
    dino_closeset_probs_sum = []
    closeset_labels_list = []
    for i, semantic_label in tqdm(enumerate(split)):
        if semantic_label in seen_labels:
            close_set = True
            print(f"{semantic_label} is a closed-set class.")#
        else:
            close_set = False
            print(f"{semantic_label} is not a closed-set class.")#
        loader = image_loaders[semantic_label]
        

        # Prepare for dino
        total_labels = seen_labels + chatgpt_manual_similar_label
        total_features = []
        for i in total_labels:
            if i in stored_features: 
                feats = stored_features[i]
            else: 
                feats = None
            if feats is not None:
                if feats.shape[0] < k_images:
                    k = feats.shape[0]
                    k_short = k_images - k
                    n = k_short // k
                    p = k_short % k
                    stack_feat = [feats for n_ in range(n + 1)]
                    stack_feat.append(feats[:p, ...])
                    feats = torch.cat(stack_feat, dim=0)
                    assert feats.shape[0] == k_images
                total_features.append(feats)
            else:
                if i in seen_labels: 
                    raise NotImplementedError("no image for class {}".format(i))
        total_features = torch.cat(total_features, dim=0)
        total_features = total_features.t() # (d, k_images * k_class)

        for idx, image in enumerate(loader):
            
            # CLIP Alignment
            all_desc = seen_descriptions + [f"This is a photo of a {label}" for label in chatgpt_manual_similar_label]
            all_desc_ids = tokenize_for_clip(all_desc, cliptokenizer)

            with torch.no_grad():
                image_feature = clip_model.encode_image(image.cuda()).float()
                image_feature /= image_feature.norm(dim=-1, keepdim=True)
                text_features = clip_model.encode_text(all_desc_ids.cuda()).float()
                text_features /= text_features.norm(dim=-1, keepdim=True)
            zeroshot_probs = (100.0 * image_feature @ text_features.T).softmax(dim=-1).squeeze()

            clip_ood_prob_sum = zeroshot_probs[:20].detach().cpu().numpy()
            clip_ood_probs_sum.append(clip_ood_prob_sum)

            # DINO Alignment
            with torch.no_grad():
                image = image.cuda()
                feats = dino_model(image)
                feats = torch.nn.functional.normalize(feats, dim=1, p=2)

            # softmax then take sum (avg) for each class
            zeroshot_probs_dino = (100.0 * feats @ total_features)
            zeroshot_probs_dino_cls = zeroshot_probs_dino.split(k_images, dim=-1)
            zeroshot_probs_dino_cls = torch.tensor([torch.mean(zeroshot_probs_dino_cls[i]) for i in range(len(total_labels))]).softmax(dim=-1).squeeze()
            ood_prob_sum_dino = zeroshot_probs_dino_cls[:len(seen_labels)].detach().cpu().numpy()
            dino_ood_probs_sum.append(ood_prob_sum_dino)

            if close_set:
                # CLIP
                with torch.no_grad():
                    seen_desc_ids = tokenize_for_clip(seen_descriptions, cliptokenizer)
                    seen_text_feature = clip_model.encode_text(seen_desc_ids.cuda()).float()
                    seen_text_feature /= seen_text_feature.norm(dim=-1, keepdim=True)
                clip_closeset_probs = (100.0 * image_feature @ seen_text_feature.T).softmax(dim=-1).squeeze()
                clip_closeset_probs_sum.append(clip_closeset_probs.detach().cpu().numpy())
                closeset_labels_list.append(seen_labels.index(semantic_label))

                # DINO
                closeset_probs_dino = (100.0 * feats @ total_features[:, :len(seen_labels*k_images)])
                closeset_probs_dino_per_cls_ = closeset_probs_dino.split(k_images, dim=1)
                closeset_probs_dino_per_cls = torch.tensor([torch.mean(closeset_probs_dino_per_cls_[i]) for i in range(len(seen_labels))])
                closeset_probs_dino = closeset_probs_dino_per_cls.softmax(dim=-1).squeeze()
                dino_closeset_probs_sum.append(closeset_probs_dino.detach().cpu().numpy())


    prob = 0.6
    ood_probs_sum_ = [a * prob + b * (1 - prob) for (a, b) in zip(clip_ood_probs_sum, dino_ood_probs_sum)]
    ood_probs_sum = [1 - max(ood_probs_sum_[ii]) for ii in range(len(ood_probs_sum_))]
    auc_sum = roc_auc_score(np.array(targets), np.squeeze(ood_probs_sum))

    closeset_probs_sum = [a * prob + b * (1 - prob) for (a, b) in zip(clip_closeset_probs_sum, dino_closeset_probs_sum)]

    closeset_preds_list = []
    for closeset_prob in closeset_probs_sum:
        closeset_pred = np.argmax(closeset_prob, axis=-1)
        closeset_pred_label = seen_labels[closeset_pred]
        closeset_preds_list.append(seen_labels.index(closeset_pred_label))
    closeset_preds_list = np.array(closeset_preds_list)
    closeset_labels_list = np.array(closeset_labels_list)
    oscr = compute_oscr(np.squeeze(ood_probs_sum)[:n_seen], np.squeeze(ood_probs_sum)[n_seen:], closeset_preds_list, closeset_labels_list)
    print('AUROC = {}, OSCR = {}'.format(auc_sum, oscr))


    return auc_sum, oscr

if __name__ == '__main__':
    args = parser.parse_args()#解析命令行输入的参数
    k_images = args.k_images#获取命令行参数中指定的图像数量
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#设置设备为CUDA如果可用，否则使用CPU

    save_dir = args.save_dir#获取保存目录的路径
    if not os.path.exists(save_dir): #如果保存目录不存在，则创建它。
        os.makedirs(save_dir)
    time_str = datetime.strftime(datetime.now(), '%Y-%m-%d-%H:%M:%S') #获取当前时间的字符串表示。
    sys.stdout = Logger(os.path.join(args.save_dir, 'eval_{}.log'.format(time_str)))#设置日志log输出到指定文件。即result
    print('settings:')
    print(args)

    # prepare dino model
    dino_model = torch.hub.load('facebookresearch/dinov2', "dinov2_vitb14") #从torch hub加载DINO模型
    dino_model.cuda() #将模型移至CUDA

    state_dict = torch.load("pretrained_model/dinov2_vitb14_pretrain.pth", map_location='cpu') #加载预训练权重
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = dino_model.load_state_dict(state_dict, strict=False)
    print('Pretrained weights found at pretrained_model/dinov2_vitb14_pretrain.pth and loaded with msg: {}'.format(msg))
    dino_model.eval() #将模型设置为评估模式

    # prepare clip model
    # initialize tokenizers for clip and bert, these two use different tokenizers
    clip_model = torch.jit.load("pretrained_model/ViT-B-32.pt").to(device).eval() #初始化CLIP模型。
    cliptokenizer = clip_tokenizer() #初始化用于CLIP模型的tokenizer，这个tokenizer可能是本地定义的或从某个库加载的，用于处理文本输入以适配CLIP模型的需求
#下面几行从JSON文件中加载标签，创建虚拟标签和已见标签的列表。
    chatgpt_dict = json.load(open('chat_json/tinyimagenet.json')) #从JSON文件中加载标签

    chatgpt_labels = []#用于存储已见标签
    virtual_labels = []#用于存储虚拟标签
    for i in range(5): #循环遍历5个不同的标签组，并填充这两个列表：
        virtual_labels.append(chatgpt_dict[str(i)])
        chatgpt_labels += chatgpt_dict[str(i)]
    chatgpt_laXZbels = list(set(chatgpt_labels))

    all_seen_labels = [] #用于存储所有已见标签。
    semantic_splits, _ = tinyimage_semantic_spit_generator() #生成Tiny ImageNet数据集的语义分割。
    for split in semantic_splits:
        all_seen_labels += split[:20]
        print("split[:20]:", split[:20])  ###
    all_seen_labels = list(set(all_seen_labels))

    print("Final all_seen_labels:", all_seen_labels)###

    stored_features_list = []#存储特征的列表。得先为每个标签组提取特征：
    for i in range(5):
        labels = virtual_labels[i] #gpt生成的虚拟的标签，可能不准，后面会删
        labels += semantic_splits[i][:20]#将semantic_splits列表中第i个元素（一个标签列表）的前20个标签添加到labels列表中。
        #处理每个虚拟标签组，为其生成数据加载器，并从DINO模型中提取特征：
        image_root = os.path.join(args.image_path, str(i)) #生成了图像的本地路径
        classes = os.listdir(image_root) #列出image_root路径下的所有文件（或文件夹），这些可能代表不同的类别。
        for l in labels:
            if l not in classes: #检查labels中的每个标签l是否存在于classes列表中。
                print("Removing label not found in classes:", l)  # 打印出将要被移除的标签
                labels.remove(l)
        tiny_dino_loaders, tiny_dino_labels = tiny_single_isolated_class_dino_loader(tiny_dino_labels=labels, root=image_root) #调用一个函数来为每个标签创建一个特定的数据加载器，以便于从DINO模型中提取特征
                                            # 传入当前处理的labels和image_root路径。返回两个对象：tiny_dino_loaders（一个包含数据加载器的字典，针对每个标签）和tiny_dino_labels（处理过的标签列表）
        stored_features = {} #初始化一个空字典，用于存储每个标签的特征
        for idx_lable, semantic_label in enumerate(tiny_dino_labels): #遍历tdl的每个标签。前者是索引，后者是标签名
            print("Extracting features {} {}/{}".format(semantic_label, idx_lable, len(tiny_dino_labels))) #打印当前正在提取特征的标签名，以及它在列表中的位置和总标签数
            if semantic_label not in tiny_dino_loaders: #检查semantic_label是否有对应的数据加载器在tiny_dino_loaders中
                continue                        #如果没有对应的加载器，跳过当前循环的剩余部分，进入下一个循环
            #有对应的加载器，则进行特征提取！：
            stored_features[semantic_label] = extract_features(dino_model, tiny_dino_loaders[semantic_label], k_images) #传入DINO模型、特定标签的数据加载器和图像数量k_images，提取特征。将提取的特征存储在stored_features字典中，以标签名semantic_label为键
        print("Finish storing features")
        stored_features_list.append(stored_features)   #将存储了当前标签组特征的字典stored_features添加到stored_features_list列表中。


    splits, detailed_labels, tinyimg_loaders = tinyimage_single_isolated_class_loader() #它可能加载Tiny ImageNet数据集的某些部分，并返回三个对象：splits（数据分割列表）、detailed_labels（详细标签列表）和tinyimg_loaders（数据加载器）
    print('seen splits:')
    for split in splits:
        print(split[:20])
    
    auc_scores = []
    oscr_scores = []

    for index, split in enumerate(splits): #遍历splits列表中的每个分割。index是当前分割的索引，split是当前分割本身
        chatgpt_labels = virtual_labels[index] #获取与当前分割对应的virtual_labels中的标签集合
        stored_features = stored_features_list[index] #获取与当前分割对应的特征集合，这些特征之前已经被提取并存储在stored_features_list中（前面有，这里是分析个例）
        auc_list_sum_per_split, oscr_list_sum_per_split = image_decoder(clip_model=clip_model,
                                                                                    dino_model=dino_model, 
                                                                                    stored_features=stored_features,
                                                                                    k_images=k_images,
                                                                                    device=device, 
                                                                                    image_loaders=tinyimg_loaders,
                                                                                    split=split, 
                                                                                    chatgpt_manual_similar_label=chatgpt_labels,
                                                                                    detailed_labels=None)

        auc_scores.append(auc_list_sum_per_split) #将计算得到的AUROC和OSCR值分别添加到auc_scores和oscr_scores列表中
        oscr_scores.append(oscr_list_sum_per_split)

    prob = 0.6

    print('Average over 5 splits:')
    print(' AUROC: {} +/- {}, {}'.format(np.mean(auc_scores), np.std(auc_scores), auc_scores))
    print(' OSCR: {} +/- {}, {}'.format(np.mean(oscr_scores), np.std(oscr_scores), oscr_scores))
