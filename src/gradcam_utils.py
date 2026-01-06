import torch
import torch.nn.functional as F
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.acts = None
        self.grads = None
        self._hooks()

    def _hooks(self):
        def fwd_hook(module, inp, out):
            self.acts = out

        def bwd_hook(module, grad_in, grad_out):
            self.grads = grad_out[0]

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    @torch.no_grad()
    def _normalize(self, cam):
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

    def generate(self, x, class_idx):
        self.model.zero_grad()
        logits = self.model(x)
        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        acts = self.acts[0]     # [C,H,W]
        grads = self.grads[0]   # [C,H,W]

        weights = grads.mean(dim=(1, 2))  # [C]
        cam = torch.zeros(acts.shape[1:], device=acts.device)

        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()
        cam = self._normalize(cam)
        return cam
