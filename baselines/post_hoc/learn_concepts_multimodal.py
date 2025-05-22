import os
import pickle
import torch
import clip
import numpy as np
from tqdm import tqdm

@torch.no_grad()
def learn_conceptbank(args, concept_list, scenario, model):
    concept_dict = {}
    for concept in tqdm(concept_list):
        # Note: You can try other forms of prompting, e.g. "photo of {concept}" etc. here.
        text = clip.tokenize(f"{concept}").to("cuda")
        text_features = model.encode_text(text).cpu().numpy()
        text_features = text_features/np.linalg.norm(text_features)
        # store concept vectors in a dictionary. Adding the additional terms to be consistent with the
        # `ConceptBank` class (see `concepts/concept_utils.py`).
        concept_dict[concept] = (text_features, None, None, 0, {})

    print(f"# concepts: {len(concept_dict)}")
    concept_dict_path = os.path.join(args.out_dir, f"multimodal_concept_{args.backbone_name}_{scenario}_recurse:{args.recurse}.pkl")
    pickle.dump(concept_dict, open(concept_dict_path, 'wb'))
    print(f"Dumped to : {concept_dict_path}")
