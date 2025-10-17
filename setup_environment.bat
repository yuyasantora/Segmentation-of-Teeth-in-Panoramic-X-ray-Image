@echo off
REM Setup script for teeth segmentation environment

echo Activating conda environment...
call conda activate teeth_segmentation

echo.
echo Installing PyTorch with CUDA support (GPU)...
echo If you don't have NVIDIA GPU, change this to CPU version
echo.

REM For CUDA 11.8 (most compatible)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

REM For CPU only, uncomment below and comment above:
REM conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

echo.
echo Installing other dependencies...
pip install scikit-image opencv-python pillow numpy scipy imutils pandas requests natsort tqdm matplotlib torchinfo

echo.
echo Setup complete!
echo.
echo To use this environment:
echo   conda activate teeth_segmentation
echo.
