
import os
from skimage import io
import numpy as np
import torch
from torch.utils.data import Dataset

import geotorch



class DenoisingNet_MLP_NST(torch.nn.Module):
    def __init__(
        self,
        patch_size,
        D_in,
        H_1,
        H_2,
        H_3,
        D_out_lam,
        T,
        min_v,
        max_v,
        Dict_init,
        c_init,
        w_init,
        device,
    ):

        super(DenoisingNet_MLP_NST, self).__init__()
        self.patch_size = patch_size

        self.T = T
        self.min_v = min_v
        self.max_v = max_v

        q, l = Dict_init.shape
        soft_comp = torch.zeros(l).to(device)
        Identity = torch.eye(l).to(device)

        self.soft_comp = soft_comp
        self.Identity = Identity
        self.device = device

        self.Dict = torch.nn.Parameter(Dict_init)

        # Nic: impose grassmanian constraint on dictionary
        print('Imposing grassmannian constraint on dictionary')
        geotorch.grassmannian(self, "Dict")
        self.Dict.data = Dict_init  # Reinitialize

        self.c = torch.nn.Parameter(c_init)
        self.unfold = torch.nn.Unfold(kernel_size=(self.patch_size, self.patch_size))

        self.linear1 = torch.nn.Linear(D_in, H_1, bias=True)
        self.linear2 = torch.nn.Linear(H_1, H_2, bias=True)
        self.linear3 = torch.nn.Linear(H_2, H_3, bias=True)
        self.linear4 = torch.nn.Linear(H_3, D_out_lam, bias=True)

        self.w = torch.nn.Parameter(w_init)

    def soft_thresh(self, x, l):
        return torch.sign(x) * torch.max(torch.abs(x) - l, self.soft_comp)

    def forward(self, x):                                                       # EXPLANATIONS

        N, C, w, h = x.shape

        unfold = self.unfold(x)                                                 # Extract 8x8 patches from images, vectorize them, put everything in a tensor. Unfold = y
        N, d, number_patches = unfold.shape
        unfold = unfold.transpose(1, 2)                                         # Final size = batch_size x num_vectors x n

        lin = self.linear1(unfold).clamp(min=0)                                 # MLP to estimate this lam value (scalar), step size lambda
        lin = self.linear2(lin).clamp(min=0)
        lin = self.linear3(lin).clamp(min=0)
        lam = self.linear4(lin)

        l = lam / self.c                                                        # l = the soft threshold.  self.c approximates L = sigma_max (D^T D), lam = lambda = step size? Overall S_lambda/L
        y = torch.matmul(unfold, self.Dict)                                     # Compute D^T y   (transposed)
        S = self.Identity - (1 / self.c) * self.Dict.t().mm(self.Dict)          # Compute I - 1/L * D^T D
        S = S.t()                                                               # Nic: isn't it symmetrical? Why need to transpose? Checked with np.linalg.norm(S.cpu().detach().numpy() - S.t().cpu().detach().numpy())

        z = self.soft_thresh(y, l)                                              # First iteration: initial previous solution = 0, soft-threshold the product D^T y
        for t in range(self.T):                                                 # Successive iterations (layers):
            z = self.soft_thresh(torch.matmul(z, S) + (1 / self.c) * y, l)      #   z = x = S( (I - 1/L D^T D) x + 1/L D^T y)
                                                                                # z is the output sparse code

        x_pred = torch.matmul(z, self.Dict.t())                                 # Final approximated reconstruction = D * x
        x_pred = torch.clamp(x_pred, min=self.min_v, max=self.max_v)            # Saturate at -1 and +1, probably because input images a like this
        x_pred = self.w * x_pred                       # w size = 64            # Weight each of the output pixel of the patch. These are learned weights!
        x_pred = x_pred.transpose(1, 2)                # size 18 x 14641 x 64   

        normalize = torch.ones(N, number_patches, d)   # size 18 x 14641 x 64   # Prepare
        normalize = normalize.to(self.device)
        normalize = self.w * normalize
        normalize = normalize.transpose(1, 2)          # size 18 x 64 x 14641

        fold = torch.nn.functional.fold(                                        # Overlap again all patches (with weighted pixels) into a common image
            x_pred, output_size=(w, h), kernel_size=(self.patch_size, self.patch_size)     
        )

        norm = torch.nn.functional.fold(                                        # Overlap the weights just the same
            normalize,
            output_size=(w, h),
            kernel_size=(self.patch_size, self.patch_size),
        )

        res = fold / norm                                                       # Reconstruct the image by dividing back to weights
                                                                                # This all seems like an elaborate way of blending the patches into a common image, without edges

        return res