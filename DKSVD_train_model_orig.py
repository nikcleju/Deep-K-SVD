"""
"""

import datetime
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import Deep_KSVD
from scipy import linalg

import NST_models

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
images_dir = 'gray_small'
file_train_name = "train_gray_small.txt"
file_test_name  = "test_gray_small.txt"
#=======================================

save_folder = 'out_orig_MLP_small'



#=======================================

# List of the test image names BSD68:
file_test = open(file_test_name, "r")
onlyfiles_test = []
for e in file_test:
    onlyfiles_test.append(e[:-1])

# List of the train image names:
file_train = open(file_train_name, "r")
onlyfiles_train = []
for e in file_train:
    onlyfiles_train.append(e[:-1])

# Rescaling in [-1, 1]:
mean = 255 / 2
std = 255 / 2
data_transform = transforms.Compose(
    [Deep_KSVD.Normalize(mean=mean, std=std), Deep_KSVD.ToTensor()]
)

# Noise level:
sigma = 25
# Sub Image Size:
sub_image_size = 128
# Training Dataset:
my_Data_train = Deep_KSVD.mydataset_sub_images(
    root_dir=images_dir,
    image_names=onlyfiles_train,
    sub_image_size=sub_image_size,
    sigma=sigma,
    transform=data_transform,
)
# Test Dataset:
my_Data_test = Deep_KSVD.mydataset_full_images(
    root_dir=images_dir, image_names=onlyfiles_test, sigma=sigma, transform=data_transform
)

# Dataloader of the test set:
if images_dir == 'gray':
    num_images_test = 5
    indices_test = np.random.randint(0, 68, num_images_test).tolist()
elif images_dir == 'gray_small':
    indices_test = list(range(7))
else:
    raise ValueError("Unknown image directory!")

my_Data_test_sub = torch.utils.data.Subset(my_Data_test, indices_test)
dataloader_test = DataLoader(
    my_Data_test_sub, batch_size=1, shuffle=False, num_workers=0
)

# Dataloader of the training set:
batch_size = 16
dataloader_train = DataLoader(
    my_Data_train, batch_size=batch_size, shuffle=True, num_workers=0
)

# Create a file to see the output during the training:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialization:
patch_size = 8
m = 16
Dict_init = Deep_KSVD.Init_DCT(patch_size, m)
U_init, S_init, VT_init = torch.linalg.svd(Dict_init, compute_uv=True, full_matrices=False)   # Compute SVD
Dict_init = Dict_init.to(device)
U_init = U_init.to(device)
S_init = S_init.to(device)
VT_init = VT_init.to(device)


c_init = linalg.norm(Dict_init.cpu(), ord=2) ** 2
c_init = torch.FloatTensor((c_init,))
c_init = c_init.to(device)

w_init = torch.normal(mean=1, std=1 / 10 * torch.ones(patch_size ** 2)).float()
w_init = w_init.to(device)

w_1_init = torch.normal(mean=1, std=1 / 10 * torch.ones(patch_size ** 2)).float()
w_1_init = w_1_init.to(device)
w_2_init = torch.normal(mean=1, std=1 / 10 * torch.ones(patch_size ** 2)).float()
w_2_init = w_2_init.to(device)

D_in, H_1, H_2, H_3, D_out_lam, T, min_v, max_v = 64, 128, 64, 32, 1, 5, -1, 1

# Single MLP 
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

# Two MLP
# model = Deep_KSVD.DenoisingNet_MLP_2(
#     patch_size,
#     D_in,
#     H_1,
#     H_2,
#     H_3,
#     D_out_lam,
#     T,
#     min_v,
#     max_v,
#     Dict_init,
#     c_init,
#     w_1_init,
#     w_2_init,
#     device,
# )

model.to(device)

# Construct our loss function and an Optimizer:
criterion = torch.nn.MSELoss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

start = time.time()
epochs = 3
running_loss = 0.0

print_every = 100
save_every_print = 60

# Save everything in this folder

os.makedirs(save_folder, exist_ok=True)

# Set save file path
file_to_print_name_template = os.path.join(save_folder, "results_training_{}.csv")
i = 1
while os.path.exists(file_to_print_name_template.format(i)):
    i += 1
file_to_print_name = file_to_print_name_template.format(i)

# Initialize starting params
epoch_start = 0
i_start = 0

# Load from previous file
#start_training_from = None  
start_training_from = "out_orig_MLP_small/checkpoint_epoch2_iter132000_trainloss0.0072421858_testloss0.0060635836.pth.tar"
#start_training_from = "out_orig_MLP_small/checkpoint_epoch1_iter180000_trainloss0.0072173813_testloss0.0060881705.pth.tar"
#start_training_from = "out_orig_MLP_small/checkpoint_epoch1_iter150000_trainloss0.0071791534_testloss0.0060924115.pth.tar"
#start_training_from = "out_orig_MLP/checkpoint_epoch1_iter426000_trainloss0.0062626188_testloss0.0067728135.pth.tar"
#start_training_from = "checkpoint_epoch1_iter648000_trainloss0.0062827535_testloss0.0067506767.pth.tar"
if start_training_from is not None:
    checkpoint = torch.load(start_training_from)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_start = checkpoint['epoch']
    i_start = checkpoint['current_iter']

# Write header info
file_to_print = open(file_to_print_name, "w")
file_to_print.write(str(datetime.datetime.now()) + "\n")
file_to_print.write(str(device) + "\n")
file_to_print.write("start_training_from: " + str(start_training_from) + "\n")
file_to_print.write("epoch_start: " + str(epoch_start) + "\n")
file_to_print.write("i_start: " + str(i_start) + "\n")
file_to_print.flush()

# Train
train_losses, test_losses = [], []
for epoch in range(epoch_start, epochs):  # loop over the dataset multiple times

    # For the first epoch only, start from i_start, so consume the first `i_start` in the data generator
    file_to_print.write(str(datetime.datetime.now()) + ": ")
    file_to_print.write('Skipping {} batches...\n'.format(i_start))
    file_to_print.flush()

    enumerator = enumerate(dataloader_train)
    if epoch == epoch_start:
        for ii in range(i_start):
            next(enumerator)

    file_to_print.write(str(datetime.datetime.now()) + ": ")
    file_to_print.write('Start iterations...\n')
    file_to_print.flush()

    #for i, (sub_images, sub_images_noise) in enumerate(dataloader_train, start=i_start):   # start= changes only the indexing, does not actually skip the elements
    for i, (sub_images, sub_images_noise) in enumerator:   # start= changes only the indexing, does not actually skip the elements
        # get the inputs
        sub_images, sub_images_noise = (
            sub_images.to(device),
            sub_images_noise.to(device),
        )

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(sub_images_noise)
        loss = criterion(outputs, sub_images)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % print_every == print_every - 1:  # print every x mini-batches

            # Compute train_loss current
            train_loss = running_loss / print_every

            train_losses.append(train_loss)

            end = time.time()
            time_curr = end - start
            file_to_print.write("time:" + " " + str(time_curr) + "\n")
            start = time.time()

            #file_to_print.write("norm of Dict - Identity = {}\n".format(  ))

            # Compute test loss
            with torch.no_grad():
                test_loss = 0

                for patches_t, patches_noise_t in dataloader_test:
                    patches, patches_noise = (
                        patches_t.to(device),
                        patches_noise_t.to(device),
                    )
                    outputs = model(patches_noise)
                    loss = criterion(outputs, patches)
                    test_loss += loss.item()

                test_loss = test_loss / len(dataloader_test)
            test_losses.append(test_loss)

            end = time.time()
            time_curr = end - start
            file_to_print.write("time:" + " " + str(time_curr) + "\n")
            start = time.time()

            # Print info in file
            s = "%s: [%d, %d, batchnum=%d] loss_train: %f, loss_test: %f" % (
                str(datetime.datetime.now()),
                epoch + 1,
                (i + 1) * batch_size,
                (i + 1),
                train_loss,
                test_loss,
                )
            s = s + "\n"
            file_to_print.write(s)
            file_to_print.flush()
            running_loss = 0.0

        #if i % (10 * print_every) == (10 * print_every) - 1:
        if i % (save_every_print * print_every) == (save_every_print * print_every) - 1:
            model_name  = "model_epoch{}_iter{}_trainloss{:.10f}_testloss{:.10f}.pth".format(epoch+1, i+1, train_loss, test_loss)
            model_savefile = os.path.join(save_folder, model_name)

            checkpoint_name  = "checkpoint_epoch{}_iter{}_trainloss{:.10f}_testloss{:.10f}.pth.tar".format(epoch+1, i+1, train_loss, test_loss)
            checkpoint_savefile = os.path.join(save_folder, checkpoint_name)

            # Save model only, for inference
            torch.save(model.state_dict(), model_savefile)

            #losses_name = "losses_{}.npz".format(i+1)
            # np.savez(
            #     losses_name, train=np.array(test_losses), test=np.array(train_losses)
            # )

            # Checkpoint everything, for subsequent training
            torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'dataloader_train': dataloader_train,
                            'current_iter': i+1   # next iteration to be run
                        }, 
                        checkpoint_savefile)


file_to_print.write("Finished Training")
