import os
import math
import torch
import clip
from . import data_utils

from tqdm import tqdm
from torch.utils.data import DataLoader

PM_SUFFIX = {"max":"_max", "avg":""}

def save_target_activations(target_model, dataloader, save_name, device = "cuda"):
    _make_save_dir(save_name)
    save_names = {}
    for target_layer in ["out"]:
        save_names[target_layer] = save_name.format(target_layer)
        
    if _all_saved(save_names):
        return
    
    all_features = []
    with torch.no_grad():
        for images, _ in tqdm(dataloader):
            features = target_model(images.to(device))
            all_features.append(features.cpu())

    for target_layer in ["out"]:
        torch.save(torch.cat(all_features), save_names[target_layer])

    #free memory
    del all_features
    torch.cuda.empty_cache()
    return

def save_clip_image_features(model, dataloader, save_name, device = "cuda"):
    _make_save_dir(save_name)
    all_features = []
    
    if os.path.exists(save_name):
        return
    
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        for images, _ in tqdm(dataloader):
            features = model.encode_image(images.to(device))
            all_features.append(features.cpu())
    torch.save(torch.cat(all_features), save_name)
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return

def save_clip_text_features(model, text, save_name, batch_size=1000):
    
    if os.path.exists(save_name):
        return
    _make_save_dir(save_name)
    text_features = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(text)/batch_size))):
            text_features.append(model.encode_text(text[batch_size*i:batch_size*(i+1)]))
    text_features = torch.cat(text_features, dim=0)
    torch.save(text_features, save_name)
    del text_features
    torch.cuda.empty_cache()
    return

def save_activations(d_probe, concepts, device, save_dir, model_dir, dataset_dir):
    target_save_name, clip_save_name, text_save_name = get_save_names("ViT-B/16", "dinov2", 
                                                                    "{}", d_probe, "concept_set", 
                                                                      "avg", save_dir)
    save_names = {"clip": clip_save_name, "text": text_save_name}
    for target_layer in ["out"]:
        save_names[target_layer] = target_save_name.format(target_layer)

    if _all_saved(save_names):
        return

    clip_model, clip_preprocess = clip.load("ViT-B/16", device=device, download_root=model_dir)

    target_model, target_preprocess = data_utils.get_target_model(device, model_dir)
    #setup data
    data_c = data_utils.get_data(d_probe, dataset_dir, clip_preprocess)
    data_t = data_utils.get_data(d_probe, dataset_dir, target_preprocess)

    text = clip.tokenize(["{}".format(word) for word in concepts]).to(device)

    save_clip_text_features(clip_model, text, text_save_name, 256)

    save_clip_image_features(clip_model, data_c, clip_save_name, device)
    save_target_activations(target_model, data_t, target_save_name, device)
    
    return

def get_save_names(clip_name, target_name, target_layer, d_probe, concept_set, pool_mode, save_dir):
    
    if target_name.startswith("clip_"):
        target_save_name = "{}/{}_{}.pt".format(save_dir, d_probe, target_name.replace('/', ''))
    else:
        target_save_name = "{}/{}_{}_{}{}.pt".format(save_dir, d_probe, target_name, target_layer,
                                                 PM_SUFFIX[pool_mode])
    clip_save_name = "{}/{}_clip_{}.pt".format(save_dir, d_probe, clip_name.replace('/', ''))
    concept_set_name = (concept_set.split("/")[-1]).split(".")[0]
    text_save_name = "{}/{}_{}.pt".format(save_dir, concept_set_name, clip_name.replace('/', ''))
    
    return target_save_name, clip_save_name, text_save_name

def _all_saved(save_names):
    """
    save_names: {layer_name:save_path} dict
    Returns True if there is a file corresponding to each one of the values in save_names,
    else Returns False
    """
    for save_name in save_names.values():
        if not os.path.exists(save_name):
            return False
    return True

def _make_save_dir(save_name):
    """
    creates save directory if one does not exist
    save_name: full save path
    """
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return

def get_accuracy_cbm(model, dataloader, device):
    correct = 0
    total = 0
    for images, labels in tqdm(dataloader):
        with torch.no_grad():
            #outs = target_model(images.to(device))
            outs, _ = model(images.to(device))
            pred = torch.argmax(outs, dim=1)
            correct += torch.sum(pred.cpu()==labels)
            total += len(labels)
    return correct/total
