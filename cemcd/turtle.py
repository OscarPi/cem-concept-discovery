# Adapted from https://github.com/mlbio-epfl/turtle

from tqdm import trange
import numpy as np
import torch
import torch.nn.functional as F

def run_turtle(Zs, k, warm_start=False, gamma=10., epochs=6000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_tr = Zs[0].shape[0]
    feature_dims = [Z.shape[1] for Z in Zs]
    batch_size = min(10000, n_tr)

    # Define task encoder
    task_encoder = [torch.nn.utils.weight_norm(torch.nn.Linear(d, k)).to(device) for d in feature_dims] 

    def task_encoding(Zs):
        assert len(Zs) == len(task_encoder)
        # Generate labeling by the average of $\sigmoid(\theta \phi(x))$, Eq. (9) in the paper
        label_per_space = [F.softmax(task_phi(z), dim=1) for task_phi, z in zip(task_encoder, Zs)] # shape of (K, N, k)
        labels = torch.mean(torch.stack(label_per_space), dim=0) # shape of (N, k)
        return labels, label_per_space
    
    # we use Adam optimizer for faster convergence, other optimziers such as SGD could also work
    optimizer = torch.optim.Adam(sum([list(task_phi.parameters()) for task_phi in task_encoder], []), lr=0.001, betas=(0.9, 0.999))

    # Define linear classifiers for the inner loop
    def init_inner():
        W_in = [torch.nn.Linear(d, k).to(device) for d in feature_dims] 
        inner_opt = torch.optim.Adam(sum([list(W.parameters()) for W in W_in], []), lr=0.001, betas=(0.9, 0.999))
    
        return W_in, inner_opt
    
    W_in, inner_opt = init_inner()

    # start training
    iters_bar = trange(epochs, leave=False)
    for _ in iters_bar:
        optimizer.zero_grad()
        # load batch of data
        indices = np.random.choice(n_tr, size=batch_size, replace=False)
        Zs_tr = [torch.from_numpy(Z_train[indices]).to(device) for Z_train in Zs]

        labels, label_per_space = task_encoding(Zs_tr)

        # init inner
        if not warm_start: 
            # cold start, re-init every time
            W_in, inner_opt = init_inner()
        # else, warm start, keep previous 

        # inner loop: update linear classifiers
        for _ in range(10):
            inner_opt.zero_grad()
            # stop gradient by "labels.detach()" to perform first-order hypergradient approximation, i.e., Eq. (13) in the paper
            loss = sum([F.cross_entropy(w_in(z_tr), labels.detach()) for w_in, z_tr in zip(W_in, Zs_tr)])
            loss.backward()
            inner_opt.step()

        # update task encoder
        optimizer.zero_grad()
        pred_error = sum([F.cross_entropy(w_in(z_tr).detach(), labels) for w_in, z_tr in zip(W_in, Zs_tr)])

        # entropy regularization 
        entr_reg = sum([torch.special.entr(l.mean(0)).sum() for l in label_per_space])
        
        # final loss, Eq. (12) in the paper
        (pred_error - gamma * entr_reg).backward()
        optimizer.step()

    labels, _ = task_encoding([torch.from_numpy(Z).to(device) for Z in Zs])
    preds = labels.argmax(dim=1).detach().cpu().numpy()

    return preds, float(pred_error), lambda Zs: task_encoding([torch.from_numpy(Z).to(device) for Z in Zs])[0]
