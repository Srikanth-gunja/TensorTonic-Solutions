import torch

def gradient_accumulation(w_init, micro_batches, lr, accum_steps):
    w = w_init.clone().float()
    grad_buf = torch.zeros_like(w)          # the "notebook"
    last_mean_grad = torch.zeros_like(w)

    for i, (X, t) in enumerate(micro_batches, start=1):
        X = torch.as_tensor(X, dtype=torch.float32)
        t = torch.as_tensor(t, dtype=torch.float32)

        pred = w @ X                         # scalar prediction
        grad_buf += 2 * (pred - t) * X       # accumulate, don't update yet

        if i % accum_steps == 0:             # notebook full -> take ONE step
            last_mean_grad = grad_buf / accum_steps
            w = w - lr * last_mean_grad
            grad_buf = torch.zeros_like(w)   # empty the notebook

    return w, last_mean_grad