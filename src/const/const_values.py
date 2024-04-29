import os
import torch

# Path to the root directory of the project
ROOT = os.path.dirname(os.path.abspath(__file__).replace('/src/const', ''))

# cpu oe gpu
DEVICE_TYPE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(DEVICE_TYPE)

if __name__ == '__main__':
    print(ROOT)
