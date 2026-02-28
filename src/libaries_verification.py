import librosa
import numpy as np
import scipy
import pandas as pd
import sklearn
import xgboost
import torch
import parselmouth

print('librosa:', librosa.__version__)
print('numpy:', np.__version__)
print('scipy:', scipy.__version__)
print('pandas:', pd.__version__)
print('sklearn:', sklearn.__version__)
print('xgboost:', xgboost.__version__)
print('torch:', torch.__version__)
print('parselmouth:', parselmouth.__version__)
print()
print('M1 MPS GPU available:', torch.backends.mps.is_available())
print()
print('ALL LIBRARIES OK ✅')
