python -m pip install virtualenv
python -m virtualenv doppelgpp-env
source doppelgpp-env/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -r dust3r/requirements.txt
pip install -r dust3r/requirements_optional.txt


# 3. DUST3R (optional) relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
# cd dust3r/croco/models/curope/
# python setup.py build_ext --inplace
# cd ../../../../


# # 4. (Integration with MASt3R-SfM) Compile and install ASMK:
# pip install cython

# git clone https://github.com/jenicek/asmk
# cd asmk/cython/
# cythonize *.pyx
# cd ..
# pip install .  # or python3 setup.py build_ext --inplace
# cd ..
