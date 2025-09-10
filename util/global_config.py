import random, os, torch
import numpy as np

def setup_seed(seed=123):
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed) 
    os.environ['PYTHONHASHSEED'] = str(seed)     
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False

def get_datadir():
    cur_path = os.path.abspath(os.path.dirname(__file__))
    return cur_path[:cur_path.find('DISKRec-SIGIR')] + 'DISKRec-SIGIR/data'