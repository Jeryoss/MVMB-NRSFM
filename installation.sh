conda create --name nrsfm python==3.12.0 -y
conda activate nrsfm
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install nvidia/label/cuda-11.8.0::cuda-toolkit -y
pip install numpy scikit-image scikit-learn pandas matplotlib plotly
python3 -m ipykernel install --sys-prefix --name HPC_NRSFM_PYTORCH_ENV_IPYPARALLEL_CUDA --display-name HPC_NRSFM_PYTORCH_ENV_IPYPARALLEL_CUDA
python3 -m pip install tensorboard

conda install jupyterlab