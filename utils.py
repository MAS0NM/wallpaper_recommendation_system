import os
import torch
import sys
import logging
import glob
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging

def save_checkpoint(model, optimizer, iteration, checkpoint_path):
    logger.info(f'Saving model and optimizer state at iteration {iteration}')
    torch.save({
        'model': model,
        'iteration': iteration,
        'optimizer': optimizer.state_dict(),
        },
        os.path.join(checkpoint_path, f'cv_{iteration}.pth')
    )
    
def load_checkpoint(checkpoint_path, optimizer=None):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    saved_model = checkpoint_dict['model']
    return saved_model, iteration, optimizer

def get_latest_checkpoint(dir_path, regex='cv_*.pth'):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    x = f_list[-1]
    print(f'loading latest checkpoint {x}')
    return x