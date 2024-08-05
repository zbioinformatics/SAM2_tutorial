# SAM2_tutorial
This step requires a GPU machine to work.
First, let's install anaconda
```python
https://www.anaconda.com/download/success
```
Then, let's install cuda 11.8
```python
https://developer.nvidia.com/cuda-11-8-0-download-archive
```
In the anaconda prompt, you can check the cuda using
```python
nvidia-smi
```
This one would display the GPU status as follows, you can have a different GPU model and CUDA version depends on your system.

Then open anaconda prompt, and create our virtual envioronment. 
```python
conda create -n SAM2_test python=3.11
```
Then activate virtual environment
```
conda activate SAM2_test
```
Get the SAM2 github files

```python
git clone https://github.com/facebookresearch/segment-anything-2.git
```
nevigate to the segment-anything-2 folder
```python
cd segment-anything-2
```
Install required packages
``` python
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install hydra-core
pip install tqdm
pip install matplotlib
pip install opencv-python
pip install ninja
pip install imageio
```

Then use the installation command, make sure that the the working directory is in the segment-anything-2 folder
```python
python setup.py build_ext --inplace
```

To test if the installation is successful or not, you can enter the python environment using 
```python
python
```
then type in
```python
import sam2
```
If you do not see any error message, it means you have successully installed SAM2


Download the SAM2 large model
```python
https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```
and put the .pt file in the "...\segment-anything-2\checkpoints" folder

Now you can run the script and make predictions.
