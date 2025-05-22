# Adapted from https://github.com/mertyg/post-hoc-cbm

import argparse
import os
import sys
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
import clip

from post_hoc.concepts import ConceptBank
from post_hoc.models import PosthocLinearCBM, PosthocHybridCBM
from post_hoc.training_tools import load_or_compute_projections
from post_hoc import learn_concepts_multimodal
from post_hoc.concepts import EasyDict
from post_hoc.train_pcbm import run_linear_probe
from post_hoc.train_pcbm_h import train_hybrid

from cemcd.data import mnist, awa, cub, shapes

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=str)
    parser.add_argument("--results-dir", required=True, type=str)
    parser.add_argument("--dataset-dir", required=True, type=str)
    parser.add_argument("--dataset", required=True, type=str)
    return parser.parse_args()

def main(args, concept_bank, backbone, datasets):
    train_loader = datasets.train_dl(no_concepts=True)
    test_loader = datasets.test_dl(no_concepts=True)
    num_classes = datasets.n_tasks

    # Initialize the PCBM module.
    posthoc_layer = PosthocLinearCBM(concept_bank, backbone_name="ViT-L_14", idx_to_class=None, n_classes=num_classes)
    posthoc_layer = posthoc_layer.to(args.device)

    # We compute the projections and save to the output directory. This is to save time in tuning hparams / analyzing projections.
    train_embs, train_projs, train_lbls, test_embs, test_projs, test_lbls = load_or_compute_projections(args, backbone, posthoc_layer, train_loader, test_loader)

    run_info, weights, bias = run_linear_probe(args, (train_projs, train_lbls), (test_projs, test_lbls))
    
    # Convert from the SGDClassifier module to PCBM module.
    posthoc_layer.set_weights(weights=weights, bias=bias)

    # Sorry for the model path hack. Probably i'll change this later.
    model_path = os.path.join(args.out_dir,
                              f"pcbm_{args.dataset}__lam:{args.lam}__alpha:{args.alpha}__seed:{args.seed}.ckpt")
    torch.save(posthoc_layer, model_path)

    posthoc_layer = posthoc_layer.eval()

    hybrid_model_path = model_path.replace("pcbm_", "pcbm-hybrid_")
        
    train_loader = DataLoader(TensorDataset(torch.tensor(train_embs).float(), torch.tensor(train_lbls).long()), batch_size=256, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(test_embs).float(), torch.tensor(test_lbls).long()), batch_size=256, shuffle=False)

    # Initialize PCBM-h
    hybrid_model = PosthocHybridCBM(posthoc_layer)
    hybrid_model = hybrid_model.to(args.device)
    
    # Initialize the optimizer
    hybrid_optimizer = torch.optim.Adam(hybrid_model.residual_classifier.parameters(), lr=args.lr)
    hybrid_model.residual_classifier = hybrid_model.residual_classifier.float()
    hybrid_model.bottleneck = hybrid_model.bottleneck.float()
    
    # Train PCBM-h
    hybrid_run_info = train_hybrid(args, train_loader, test_loader, hybrid_model, hybrid_optimizer, num_classes)

    torch.save(hybrid_model, hybrid_model_path)

    print("Final standard test acc: ", run_info["test_acc"] / 100)
    print("Final hybrid test acc: ", hybrid_run_info["test_acc"].avg)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parse_arguments()

    options = EasyDict()
    options.seed = 42
    options.lr = 1e-3
    options.lam = 1e-5
    options.alpha = 0.99
    options.num_workers = 7
    options.batch_size = 256
    options.backbone_name = "clip:ViT-L_14"
    options.device = device
    options.concept_bank = "concept_bank"
    options.dataset = args.dataset
    options.out_dir = args.results_dir
    options.num_epochs = 20
    options.lr = 0.01
    options.l2_penalty = 0.001

    model, clip_preprocess = clip.load("ViT-L/14", device=device, download_root=args.model_dir)

    if args.dataset == "awa":
        datasets = awa.AwADatasets(dataset_dir=args.dataset_dir)
        concept_list = awa.SELECTED_CONCEPT_SEMANTICS
    elif args.dataset == "cub":
        datasets = cub.CUBDatasets(dataset_dir=args.dataset_dir)
        concept_list = cub.CONCEPT_SEMANTICS
    elif args.dataset == "mnist":
        datasets = mnist.MNISTDatasets(n_digits=2, max_digit=6, dataset_dir=args.dataset_dir)
        concept_list = [
            "Top digit is 0",
            "Top digit is 1",
            "Top digit is 2",
            "Top digit is 3",
            "Top digit is 4",
            "Top digit is 5",
            "Top digit is 6",
            "Bottom digit is 0",
            "Bottom digit is 1",
            "Bottom digit is 2",
            "Bottom digit is 3",
            "Bottom digit is 4",
            "Bottom digit is 5",
            "Bottom digit is 6",
        ]
    elif args.dataset == "shapes":
        datasets = shapes.ShapesDatasets(dataset_dir=args.dataset_dir)
        concept_list = [
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
        print("Invalid dataset: valid options are awa, cub, mnist, shapes.")
        sys.exit()

    datasets.train_img_transform = clip_preprocess
    datasets.val_test_img_transform = clip_preprocess

    learn_conceptbank_options = EasyDict()
    learn_conceptbank_options.out_dir = args.results_dir
    learn_conceptbank_options.backbone_name = ""
    learn_conceptbank_options.recurse = ""

    learn_concepts_multimodal.learn_conceptbank(learn_conceptbank_options, concept_list, "", model)

    with open(os.path.join(args.results_dir, "multimodal_concept___recurse:.pkl"), "rb") as f:
        all_concepts = pickle.load(f)
    all_concept_names = list(all_concepts.keys())
    concept_bank = ConceptBank(all_concepts, device)

    backbone = model.eval()
    backbone = backbone.to(device)
    main(options, concept_bank, backbone, datasets)
