@echo off
call conda activate cv_conda
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -c "import numpy; print('NumPy:', numpy.__version__)"
