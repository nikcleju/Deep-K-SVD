"""
"""
import glob
import os
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import Deep_KSVD

import NST_models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate_model(
        net_name,
        images_dir='gray',
        file_test_name= "test_gray.txt",    
        patch_size=8,
        m=16,
        sigma=25,
        model_folder='.',        
        model_name_template = 'model_epoch{}_iter{}_trainloss*_testloss*.pth',
        epochs=3,
        epoch_start=0,
        iter_stop=None,
        iter_step=None,
        eval_output_folder = 'eval',
        save_images_index=[],
        do_flush=False    # flush after every write in the text file
    ):

    #=========================================
    # Parameters

    #model_class = "NST_models.DenoisingNet_MLP_NST_Orth2"
    #model_folder = 'results_NST_Orth2_small_3p'
    #epochs = 3
    #epoch_start = 0
    #iter_stop = 324001

    #model_name_template = 'model_epoch{}_iter{}_trainloss*_testloss*.pth'


    #=======================================
    # BSD 500 - 68
    #=======================================
    # images_dir = 'gray'
    # file_train_name = "train_gray.txt"
    # file_test_name  = "test_gray.txt"
    #=======================================
    #=======================================
    # BSD 50 - 7
    #=======================================
    #images_dir = 'gray_small'
    #file_train_name = "train_gray_small.txt"
    #file_test_name  = "test_gray_small.txt"
    #=========================================


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
    #sigma = 25

    # Test Dataset:
    my_Data_test = Deep_KSVD.mydataset_full_images(
        root_dir=images_dir, image_names=onlyfiles_test, sigma=sigma, transform=data_transform
    )

    dataloader_test = DataLoader(my_Data_test, batch_size=1, shuffle=False, num_workers=0)


    # Overcomplete Discrete Cosinus Transform:
    #patch_size = 8
    #m = 16
    Dict_init = Deep_KSVD.Init_DCT(patch_size, m)
    U_init, S_init, VT_init = torch.linalg.svd(Dict_init, compute_uv=True, full_matrices=False)   # Compute SVD
    Dict_init = Dict_init.to(device)
    U_init = U_init.to(device)
    S_init = S_init.to(device)
    VT_init = VT_init.to(device)

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
    #D_in, H_1, H_2, H_3, D_out_lam, T, min_v, max_v = 64, 128, 64, 32, 1, 7, -1, 1
    D_in, H_1, H_2, H_3, D_out_lam, T, min_v, max_v = 64, 128, 64, 32, 1, 5, -1, 1

    # Nicolae Cleju: Use our network
    if net_name == 'DenoisingNet_MLP':
        model = Deep_KSVD.DenoisingNet_MLP(
                        patch_size = patch_size,
                        D_in = D_in,
                        H_1 = H_1,
                        H_2 = H_2,
                        H_3 = H_3,
                        D_out_lam = D_out_lam,
                        T = T,
                        min_v = min_v,
                        max_v = max_v,
                        Dict_init = Dict_init,
                        c_init = c_init,
                        w_init = w_init,
                        device = device,
        )
    elif net_name == 'DenoisingNet_MLP_NST_Orth':
        model = NST_models.DenoisingNet_MLP_NST_Orth(    
                        patch_size = patch_size,
                        D_in = D_in,
                        H_1 = H_1,
                        H_2 = H_2,
                        H_3 = H_3,
                        D_out_lam = D_out_lam,
                        T = T,
                        min_v = min_v,
                        max_v = max_v,
                        U_init = U_init,
                        S_init = S_init,    
                        VT_init = VT_init,
                        c_init = c_init,
                        w_init = w_init,
                        device = device,
        )
    elif net_name == 'DenoisingNet_MLP_NST_Orth2':
        model = NST_models.DenoisingNet_MLP_NST_Orth(
                        patch_size = patch_size,
                        D_in = D_in,
                        H_1 = H_1,
                        H_2 = H_2,
                        H_3 = H_3,
                        D_out_lam = D_out_lam,
                        T = T,
                        min_v = min_v,
                        max_v = max_v,
                        U_init = U_init,
                        S_init = S_init,    
                        VT_init = VT_init,
                        c_init = c_init,
                        w_1_init = w_1_init,
                        w_2_init = w_2_init,
                        device = device,
                    )


    # Make eval output folder, if not existing
    os.makedirs(eval_output_folder, exist_ok=True)

    # List PSNR:
    with open(os.path.join(eval_output_folder, "list_test_PSNR_all.csv"), "w") as fall:

        for epoch in range(epoch_start, epochs):  # loop over the dataset multiple times

            #iterations = np.arange(start=6000, step=6000, stop=iter_stop)
            iterations = np.arange(start=iter_step, step=iter_step, stop=iter_stop)
            for iter_n in tqdm(iterations, desc="Epoch {}, evaluating models".format(epoch+1)):

                model_name = [name for name in glob.glob(os.path.join(model_folder, model_name_template.format(epoch+1, iter_n)))][0]

                model.load_state_dict(torch.load(model_name, map_location="cpu"))
                model.to(device)

                file_to_print = open(os.path.join(eval_output_folder, "list_test_PSNR_{}_{}.txt".format(epoch+1, iter_n)), "w")
                file_to_print.write(str(device) + "\n")
                if do_flush:
                     file_to_print.flush()

                with open(os.path.join(eval_output_folder, "list_test_PSNR_{}_{}.pickle".format(epoch+1, iter_n)), "wb") as fp:

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
                            if do_flush:
                                file_to_print.flush()

                            list_PSNR_init.append(PSNR_init)

                            PSNR = 10 * torch.log10(
                                4 / torch.mean((image_true_t - image_restored_t) ** 2)
                            )
                            PSNR = PSNR.cpu()
                            if do_flush:
                                file_to_print.flush()

                            list_PSNR.append(PSNR)

                            # imsave("im_noisy_"+str(q)+'.pdf',image_noise_0)
                            # imsave("im_restored_"+str(q)+'.pdf',image_restored_t)
                            if k in save_images_index:
                                im_noisy_filename    = os.path.join(eval_output_folder, "im_noisy_{}.pdf".format(k))
                                im_restored_filename = os.path.join(eval_output_folder, "im_restored_{}_ep{}_it{}.pdf".format(k, epoch+1, iter_n))
                                if not os.path.exists(im_noisy_filename):
                                    plt.imsave(im_noisy_filename, image_noise_0.cpu(), cmap='gray', vmin=-1, vmax=1)
                                plt.imsave(im_restored_filename, image_restored_t.cpu(), cmap='gray', vmin=-1, vmax=1)

                    mean = np.mean(list_PSNR)
                    file_to_print.write("FINAL" + " " + str(mean) + "\n")
                    if do_flush:
                        file_to_print.flush()
                    pickle.dump(list_PSNR, fp)

                #fall.write(', '.join((model_name, str(mean))) + "\n")
                # Write all individual values also
                fall.write(', '.join((model_name, 
                                    ', '.join([str(x.item()) for x in list_PSNR_init]), 
                                    ', '.join([str(x.item()) for x in list_PSNR]),
                                    str(mean))) + "\n")
                if do_flush:
                    fall.flush()
