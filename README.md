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
Mon Aug  5 12:21:34 2024
Mon Aug 5 12:21:34 2024
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 536.23 Driver Version: 536.23 CUDA Version: 12.2 |
|-----------------------------------------+----------------------+----------------------|
| GPU Name TCC/WDDM | Bus-Id Disp.A | Volatile Uncorr. ECC |
| Fan Temp Perf Pwr
/Cap | Memory-Usage | GPU-Util Compute M. |
| | | MIG M. |
|=========================================+======================+======================|
| 0 NVIDIA GeForce RTX 3070 WDDM | 00000000:01:00.0 Off | N/A |
| 0% 42C P8 7W / 220W | 0MiB / 8192MiB | 0% Default |
| | | N/A |
+-----------------------------------------+----------------------+----------------------+
| 1 NVIDIA GeForce RTX 3070 WDDM | 00000000:4B:00.0 On | N/A |
| 0% 54C P8 24W / 220W | 6404MiB / 8192MiB | 14% Default |
| | | N/A |
+-----------------------------------------+----------------------+----------------------+

python version 3.11
CUDA version 11.8

```python
git clone https://github.com/facebookresearch/segment-anything-2.git
```

```python
cd segment-anything-2
```

``` python
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install hydra-core
pip install tqdm
pip install matplotlib
pip install opencv-python
pip install ninja
pip install imageio
```

then use the installation command
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





The SAM for video model can be downloaded through 
```python
https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```
