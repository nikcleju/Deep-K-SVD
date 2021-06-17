
from evaluate_model import evaluate_model

evaluate_model(
  # Network type
    net_name='DenoisingNet_MLP',
  # Folders
    model_folder='newres_gray_DKSVD_MLP1',    
    eval_output_folder = 'newres_gray_DKSVD_MLP1_eval',
  # Test images
    images_dir='gray',
    file_test_name= "test_gray.txt",
  # Noise & patch parameters
    patch_size=8,
    m=16,
    sigma=25,
  # Saved models
    model_name_template = 'model_epoch{}_iter{}_trainloss*_testloss*.pth',
    epochs=3,
    epoch_start=0,
    iter_step=50000,
    iter_stop=1850000,
  # Evaluation outputs
    save_images_index=[0, 10],  # can be 'all'
)