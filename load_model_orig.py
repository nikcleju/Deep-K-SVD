"""
"""
import glob
import os
import numpy as np
from scipy import linalg
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import Deep_KSVD

import NST_models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#=========================================
# Parameters
model_class = "Deep_KSVD.DenoisingNet_MLP_2"
model_folder = 'out_orig_MLP2'
model_name_template = 'model_epoch1_iter{}_trainloss*_testloss*.pth'
images_dir = 'gray'
file_train_name = "train_gray.txt"
file_test_name  = "test_gray.txt"
#=========================================

# Overcomplete Discrete Cosinus Transform:
patch_size = 8
m = 16
Dict_init = Deep_KSVD.Init_DCT(patch_size, m)
Dict_init = Dict_init.to(device)

# Squared Spectral norm:
c_init = linalg.norm(Dict_init.cpu(), ord=2) ** 2
c_init = torch.FloatTensor((c_init,))
c_init = c_init.to(device)

# Average weight:
w_init = torch.normal(mean=1, std=1 / 10 * torch.ones(patch_size ** 2)).float()
w_init = w_init.to(device)

w_1_init = torch.normal(mean=1, std=1 / 10 * torch.ones(patch_size ** 2)).float()
w_1_init = w_1_init.to(device)
w_2_init = torch.normal(mean=1, std=1 / 10 * torch.ones(patch_size ** 2)).float()
w_2_init = w_2_init.to(device)

# Deep-KSVD:
D_in, H_1, H_2, H_3, D_out_lam, T, min_v, max_v = 64, 128, 64, 32, 1, 7, -1, 1

if model_class == "Deep_KSVD.DenoisingNet_MLP":
    model = Deep_KSVD.DenoisingNet_MLP(
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
    )
elif model_class == "Deep_KSVD.DenoisingNet_MLP_2":
    model = Deep_KSVD.DenoisingNet_MLP_2(
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
        w_1_init,
        w_2_init,
        device,
    )



# Test image names:
file_test = open(file_test_name, "r")
onlyfiles_test = []
for e in file_test:
    onlyfiles_test.append(e[:-1])

# Rescaling in [-1, 1]:
mean = 255 / 2
std = 255 / 2
data_transform = transforms.Compose(
    [Deep_KSVD.Normalize(mean=mean, std=std), Deep_KSVD.ToTensor()]
)
# Noise level:
sigma = 25

# Test Dataset:
my_Data_test = Deep_KSVD.mydataset_full_images(
    root_dir=images_dir, image_names=onlyfiles_test, sigma=sigma, transform=data_transform
)

dataloader_test = DataLoader(my_Data_test, batch_size=1, shuffle=False, num_workers=0)


# List PSNR:
with open(os.path.join(model_folder, "list_test_PSNR_all.csv"), "w") as fall:

    model_numbers = np.arange(start=6000, step=180000, stop=648001)
    for model_n in tqdm(model_numbers, desc="Evaluating models"):

        model_name = [name for name in glob.glob(os.path.join(model_folder, model_name_template.format(model_n)))][0]

        model.load_state_dict(torch.load(model_name, map_location="cpu"))
        model.to(device)

        file_to_print = open(os.path.join(model_folder, "list_test_PSNR_{}.csv".format(model_n)), "w")
        file_to_print.write(str(device) + "\n")
        file_to_print.flush()

        with open(os.path.join(model_folder, "list_test_PSNR_{}.txt".format(model_n)), "wb") as fp:

            with torch.no_grad():
                list_PSNR = []
                list_PSNR_init = []
                PSNR = 0
                for k, (image_true, image_noise) in enumerate(dataloader_test, 0):

                    image_true_t = image_true[0, 0, :, :]
                    image_true_t = image_true_t.to(device)

                    image_noise_0 = image_noise[0, 0, :, :]
                    image_noise_0 = image_noise_0.to(device)

                    image_noise_t = image_noise.to(device)
                    image_restored_t = model(image_noise_t)
                    image_restored_t = image_restored_t[0, 0, :, :]

                    PSNR_init = 10 * torch.log10(
                        4 / torch.mean((image_true_t - image_noise_0) ** 2)
                    )
                    file_to_print.write("Init:" + " " + str(PSNR_init) + "\n")
                    file_to_print.flush()

                    list_PSNR_init.append(PSNR_init)

                    PSNR = 10 * torch.log10(
                        4 / torch.mean((image_true_t - image_restored_t) ** 2)
                    )
                    PSNR = PSNR.cpu()
                    file_to_print.write("Test:" + " " + str(PSNR) + "\n")
                    file_to_print.flush()

                    list_PSNR.append(PSNR)

                    # imsave("im_noisy_"+str(q)+'.pdf',image_noise_0)
                    # imsave("im_restored_"+str(q)+'.pdf',image_restored_t)

            mean = np.mean(list_PSNR)
            file_to_print.write("FINAL" + " " + str(mean) + "\n")
            file_to_print.flush()
            pickle.dump(list_PSNR, fp)

        fall.write(', '.join((model_name, str(mean))) + "\n")
        fall.flush()
