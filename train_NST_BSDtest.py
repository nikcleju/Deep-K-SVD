
from DKSVD_train_model import run_train

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

#save_folder = 'results_NST_Orth_small_3p'
save_folder = 'results_NST_Orth2_small_3p'

run_train(
    net_name='DenoisingNet_MLP_NST_Orth',
    images_dir='gray_small',
    file_train_name="train_gray_test.txt",
    file_test_name= "test_gray_test.txt",
    #num_images_test=5,  # should be 5 in general
    num_images_test=1,  # should be 5 in general
    save_folder='newres_NST_Orth_small_3ep_test',
    batch_size=9,
    patch_size=8,
    m=16,
    sigma=25,
    sub_image_size=128,
    epochs=3,
    print_every=100,
    #save_every_print=60,
    save_every_print=10,
    start_training_from=None 
)