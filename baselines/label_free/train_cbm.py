import torch
import os
import random
from . import utils
from . import similarity
import datetime
import json

from .glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from .utils import save_activations
from .data_utils import get_targets_only
from torch.utils.data import DataLoader, TensorDataset

from cemcd.data import awa

def train_cbm_and_save(results_dir, dataset_dir, model_dir, dataset, device):
    clip_cutoff = 0.26
    interpretability_cutoff = 0.45
    lam = 0.0002
    n_iters = 5000
    proj_batch_size = 50000
    proj_steps = 1000
    saga_batch_size = 256

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    similarity_fn = similarity.cos_similarity_cubed_single

    d_train = dataset + "_train"
    d_val = dataset + "_val"

    if dataset == "cub":
        n_classes = 200
        with open("label_free/data/concept_sets/cub_filtered.txt", "r") as f:
            concepts = f.read().split("\n")
    elif dataset == "awa":
        n_classes = 50
        concepts = awa.SELECTED_CONCEPT_SEMANTICS
    elif dataset == "shapes":
        n_classes = 48
        concepts = [
            "Shape is a square",
            "Shape is a circle",
            "Shape is a triangle",
            "Shape is a hexagon",
            "Shape is red",
            "Shape is green",
            "Shape is blue",
            "Shape is purple",
            "Background is red",
            "Background is green",
            "Background is blue",
            "Background is purple",
        ]
    else:
        print("Unrecognised datasets. Valid datasets are cub, awa and shapes.")
        return

    #save activations and get save_paths
    for d_probe in [d_train, d_val]:
        save_activations(d_probe=d_probe, concepts=concepts, 
                               device=device, save_dir=results_dir, model_dir=model_dir, dataset_dir=dataset_dir)

    target_save_name, clip_save_name, text_save_name = utils.get_save_names("ViT-B/16", "dinov2", 
                                            "out", d_train, "concept_set", "avg", results_dir)
    val_target_save_name, val_clip_save_name, text_save_name = utils.get_save_names("ViT-B/16", "dinov2",
                                            "out", d_val, "concept_set", "avg", results_dir)

    #load features
    with torch.no_grad():
        target_features = torch.load(target_save_name, map_location="cpu").float()
        
        val_target_features = torch.load(val_target_save_name, map_location="cpu").float()
    
        image_features = torch.load(clip_save_name, map_location="cpu").float()
        image_features /= torch.norm(image_features, dim=1, keepdim=True)

        val_image_features = torch.load(val_clip_save_name, map_location="cpu").float()
        val_image_features /= torch.norm(val_image_features, dim=1, keepdim=True)

        text_features = torch.load(text_save_name, map_location="cpu").float()
        text_features /= torch.norm(text_features, dim=1, keepdim=True)
        
        clip_features = image_features @ text_features.T
        val_clip_features = val_image_features @ text_features.T

        del image_features, text_features, val_image_features
    
    #filter concepts not activating highly
    highest = torch.mean(torch.topk(clip_features, dim=0, k=5)[0], dim=0)

    for i, concept in enumerate(concepts):
        if highest[i]<=clip_cutoff:
            print("Deleting {}, CLIP top5:{:.3f}".format(concept, highest[i]))
    concepts = [concepts[i] for i in range(len(concepts)) if highest[i]>clip_cutoff]
    
    #save memory by recalculating
    del clip_features
    with torch.no_grad():
        image_features = torch.load(clip_save_name, map_location="cpu").float()
        image_features /= torch.norm(image_features, dim=1, keepdim=True)

        text_features = torch.load(text_save_name, map_location="cpu").float()[highest>clip_cutoff]
        text_features /= torch.norm(text_features, dim=1, keepdim=True)

        clip_features = image_features @ text_features.T
        del image_features, text_features

    val_clip_features = val_clip_features[:, highest>clip_cutoff]
    
    print(f"NUMBER OF CONCEPTS: {len(concepts)}")
    #learn projection layer
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(concepts),
                                 bias=False).to(device)
    opt = torch.optim.Adam(proj_layer.parameters(), lr=1e-3)
    
    indices = [ind for ind in range(len(target_features))]

    best_val_loss = float("inf")
    best_step = 0
    best_weights = None
    proj_batch_size = min(proj_batch_size, len(target_features))
    for i in range(proj_steps):
        batch = torch.LongTensor(random.sample(indices, k=proj_batch_size))
        outs = proj_layer(target_features[batch].to(device).detach())
        loss = -similarity_fn(clip_features[batch].to(device).detach(), outs)
        
        loss = torch.mean(loss)
        loss.backward()
        opt.step()
        if i%50==0 or i==proj_steps-1:
            with torch.no_grad():
                val_output = proj_layer(val_target_features.to(device).detach())
                val_loss = -similarity_fn(val_clip_features.to(device).detach(), val_output)
                val_loss = torch.mean(val_loss)
            if i==0:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
                print("Step:{}, Avg train similarity:{:.4f}, Avg val similarity:{:.4f}".format(best_step, -loss.cpu(),
                                                                                               -best_val_loss.cpu()))
                
            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
            else: #stop if val loss starts increasing
                break
        opt.zero_grad()
        
    proj_layer.load_state_dict({"weight":best_weights})
    print("Best step:{}, Avg val similarity:{:.4f}".format(best_step, -best_val_loss.cpu()))
    
    #delete concepts that are not interpretable
    with torch.no_grad():
        outs = proj_layer(val_target_features.to(device).detach())
        sim = similarity_fn(val_clip_features.to(device).detach(), outs)
        interpretable = sim > interpretability_cutoff
        
    for i, concept in enumerate(concepts):
        if sim[i]<=interpretability_cutoff:
            print("Deleting {}, Iterpretability:{:.3f}".format(concept, sim[i]))
    
    concepts = [concepts[i] for i in range(len(concepts)) if interpretable[i]]
    
    del clip_features, val_clip_features
    
    W_c = proj_layer.weight[interpretable]
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(concepts), bias=False)
    proj_layer.load_state_dict({"weight":W_c})
    
    train_targets = get_targets_only(d_train, dataset_dir)
    val_targets = get_targets_only(d_val, dataset_dir)
    
    with torch.no_grad():
        train_c = proj_layer(target_features.detach())
        val_c = proj_layer(val_target_features.detach())
        
        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True)
        
        train_c -= train_mean
        train_c /= train_std
        
        train_y = torch.LongTensor(train_targets)
        indexed_train_ds = IndexedTensorDataset(train_c, train_y)

        val_c -= train_mean
        val_c /= train_std
        
        val_y = torch.LongTensor(val_targets)

        val_ds = TensorDataset(val_c,val_y)


    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=saga_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=saga_batch_size, shuffle=False)

    # Make linear model and zero initialize
    linear = torch.nn.Linear(train_c.shape[1],n_classes).to(device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    
    STEP_SIZE = 0.1
    ALPHA = 0.99
    metadata = {}
    metadata['max_reg'] = {}
    metadata['max_reg']['nongrouped'] = lam

    # Solve the GLM path
    output_proj = glm_saga(linear, indexed_train_loader, STEP_SIZE, n_iters, ALPHA, epsilon=1, k=1,
                      val_loader=val_loader, do_zero=False, metadata=metadata, n_ex=len(target_features), n_classes = n_classes)
    W_g = output_proj['path'][0]['weight']
    b_g = output_proj['path'][0]['bias']
    
    save_name = os.path.join(results_dir, f"{dataset}_cbm_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}")
    os.mkdir(save_name)
    torch.save(train_mean, os.path.join(save_name, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_name, "proj_std.pt"))
    torch.save(W_c, os.path.join(save_name ,"W_c.pt"))
    torch.save(W_g, os.path.join(save_name, "W_g.pt"))
    torch.save(b_g, os.path.join(save_name, "b_g.pt"))

    with open(os.path.join(save_name, "concepts.txt"), 'w') as f:
        f.write(concepts[0])
        for concept in concepts[1:]:
            f.write('\n'+concept)

    with open(os.path.join(save_name, "metrics.txt"), 'w') as f:
        out_dict = {}
        for key in ('lam', 'lr', 'alpha', 'time'):
            out_dict[key] = float(output_proj['path'][0][key])
        out_dict['metrics'] = output_proj['path'][0]['metrics']
        nnz = (W_g.abs() > 1e-5).sum().item()
        total = W_g.numel()
        out_dict['sparsity'] = {"Non-zero weights":nnz, "Total weights":total, "Percentage non-zero":nnz/total}
        json.dump(out_dict, f, indent=2)

    return save_name
