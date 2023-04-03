# Dive-into-Tevatron

"Dive-into-Tevatron" includes my learning notes on using Tevatron to reproduce a dense retrieval model, specifically the bi-encoder model on the MS-MARCO passage ranking dataset.


>
> Tevatron is a simple and efficient toolkit for training and running dense retrievers with deep language models. The toolkit has a modularized design for easy research; a set of command line tools are also provided for fast development and testing. A set of easy-to-use interfaces to Huggingface's state-of-the-art pre-trained transformers ensures Tevatron's superior performance.

For more information, please refer to the original repository: [texttron/tevatron: Tevatron - A flexible toolkit for dense retrieval research and development. (github.com)](https://github.com/texttron/tevatron)

# Installation

This section shows how to install the required libraries on Windows 10.

```
conda create -n tevatron python=3.7.0
git clone https://github.com/texttron/tevatron
cd tevatron
pip install --editable .
cd ..
# download torch-1.10.0+cu102-cp37-cp37m-win_amd64.whl
pip install torch-1.10.0+cu102-cp37-cp37m-win_amd64.whl
conda install -c conda-forge faiss-gpu
```

Assuming you are using Python 3.7.0, create an environment by running `conda create -n tevatron python=3.7.0`. And install Tevatron as an editable package for further development.

To install PyTorch (GPU version, CUDA 10.2, Python 3.7), download [cu102/torch-1.10.0%2Bcu102-cp37-cp37m-win_amd64.whl](https://download.pytorch.org/whl/cu102/torch-1.10.0%2Bcu102-cp37-cp37m-win_amd64.whl) from https://download.pytorch.org/whl/torch_stable.html and install it using `pip install xxx.whl`.

Tips: To verify your CUDA version, run `nvidia-smi`. It should be greater than 10.2.



