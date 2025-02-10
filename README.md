# Doppelgangers++: Improved Visual Disambiguation with Geometric 3D Features

[Yuanbo Xiangli](https://kam1107.github.io/), [Ruojin Cai](https://www.cs.cornell.edu/~ruojin/), [Hanyu Chen](https://hanyuc.com/), [Jeffrey Byrne](https://www.jeffreybyrne.com/), [Noah Snavely](https://www.cs.cornell.edu/~snavely/) <br />


[[`Project Page`](https://doppelgangers25.github.io/doppelgangers_plusplus/)]

## Overview

Implementation of Doppelganger++, an enhanced pairwise image classifier that excels in visual disambiguation across diverse and challenging scenes. Doppelgangers++ seamlessly integrates with SfM pipelines such as COLMAP and MASt3R-SfM, producing correct and complete reconstructions.

<p align="center">
<img src="assets/teaser.jpg" width=100% height=100% 
class="center">
</p>




## Installation

1. Clone this repo:

```
git clone --recursive https://github.com/thuanaislab/doppelgangers-plusplus.git 
cd doppelgangers-plusplus
```

2. Create the environment and install dependencies.
```bash
conda create -n doppelgangers_pp python=3.11 cmake=3.14.0
conda activate doppelgangers_pp 
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt
pip install -r dust3r/requirements.txt
pip install -r dust3r/requirements_optional.txt
```

3. Optional, compile the cuda kernels for RoPE (as in CroCo v2):
```bash
# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd dust3r/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../
```

4. (Integration with MASt3R-SfM) Compile and install ASMK:
```bash
pip install cython

git clone https://github.com/jenicek/asmk
cd asmk/cython/
cythonize *.pyx
cd ..
pip install .  # or python3 setup.py build_ext --inplace
cd ..
```

5. (Integration with COLMAP) Download COLMAP from their [``installation page``](https://colmap.github.io/install.html), we used vesion 3.10.


## Data

Create a fata folder inside the project path: 

```bash
mkdir data
mkdir data/pairs_metadata
```
Next, download the Doppelgangers dataset (Cai et al.) and VisymScenes data, and place them under a desired direcory, e.g. ```data/```.

The final folder layout should look like:

```
|---data
    |---doppelgangers
        |---images
            |---test_set
                |---...
            |---train_set_flip
                |---...
            |---train_set_noflip
                |---...
            |---train_megadepth
                |---...
    |---visymscenes
        |---siteSTR0001
            |---...
        |---siteSTR0002
            |---...
        |---...
    |---pairs_metadata
        |---train_pairs_flip.npy
        |---train_pairs_noflip.npy
        |---train_pairs_megadepth.npy
        |---test_pairs.npy
        |---train_pairs_visym.npy
        |---test_pairs_visym.npy
    
```

### Doppelgangers Dataset

Follow the data download and preparation of [``Doppelgangers: Learning to Disambiguate Images of Similar Structures``](https://github.com/RuojinCai/doppelgangers/tree/main/data/doppelgangers_dataset). Or following the instructions:

* Training images without flip augmentation: [`train_set_noflip.tar.gz`](https://doppelgangers.cs.cornell.edu/dataset/train_set_noflip.tar.gz)
* Training images with flip augmentation: 
    * First download base images: [`train_set_flip.tar.gz`](https://doppelgangers.cs.cornell.edu/dataset/train_set_flip.tar.gz) and MegaDepth subset: [`train_megadepth.tar.gz`](https://doppelgangers.cs.cornell.edu/dataset/train_megadepth.tar.gz). 
    * They provide a Python script [`flip_augmentation.py`](https://doppelgangers.cs.cornell.edu/dataset/flip_augmentation.py) to perform the image flip augmentation on the provided base images. To use this script, please modify the configuration options at the beginning of the script and run with `python flip_augmentation.py`.
* Test set: [`test_set.tar.gz`](https://doppelgangers.cs.cornell.edu/dataset/test_set.tar.gz)
* Image pair metadata for all training and test sets: [`pairs_metadata.tar.gz`](https://doppelgangers.cs.cornell.edu/dataset/pairs_metadata.tar.gz)

Organize the data directory with:
```bash
mv data/doppelgangers/pairs_metadata/* data/pairs_metadata/
rm -rf data/doppelgangers/pairs_metadata
```

### VisymScenes Dataset
* Download chunked zip files from [``huggingface``](https://huggingface.co/datasets/doppelgangers25/VisymScenes). Recover original files with script:
```python
import os
import zipfile

def recover_folder_from_chunks(chunks_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    zip_files = [f for f in os.listdir(chunks_folder) if f.endswith(".zip")]
    zip_files.sort()

    for zip_file in zip_files:
        zip_path = os.path.join(chunks_folder, zip_file)
        print(f"Extracting {zip_path}...")
        
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(output_folder)
    
    print(f"Recovery complete. Files restored to {output_folder}")

chunks_folder = "/path/to/chunked/zipfiles"
output_folder = "data/visymscenes"

recover_folder_from_chunks(chunks_folder, output_folder)
```
* Also download `test_pairs_visym.npy` and `train_pairs_visym.npy`, put them under `data/pairs_metadata`.


## Pretrained Model
Download the checkpoints from [`huggingface`](https://huggingface.co/doppelgangers25/doppelgangers_plusplus) and put them under a desired path, e.g. ```checkpoints/```.

## Model Usage in Structure-from-Motion
The general idea is to first perform feature extraction and matching, which generates a list of image pairs.
Then use our visual disambiguation model on these image pairs to remove the ones with lower probability (i.e. more likely to be a doppelganger).
Finally, run mapper on the filter image pairs for reconstruction.

We provide example scripts of integrating our visual disambiguation model into SfM pipelins:

### Integration with COLMAP
```bash
python colmap_usage.py --colmap_exe_command colmap \
    --input_image_path /path/to/your/images \
    --output_path /path/to/output/directory \
    --threshold 0.8 \
    --pretrained checkpoints/checkpoint-dg+visym.pth \
```
### Integration with MASt3R-SfM
Download both the [`trainingfree.pth`](https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth) and [`codebook.pkl`](https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl) files following [MASt3R-SfM](https://github.com/naver/mast3r/tree/mast3r_sfm?tab=readme-ov-file#retrieval-model)'s instruction, and put them in the same directory.

```bash
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/
```

Then run:
```bash
python mast3r_sfm_usage.py \
    --input_image_path /path/to/your/images \
    --output_path /path/to/output/directory \
    --pretrained checkpoints/checkpoint-dg+visym.pth \
    --retrieval_model checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth \
    --threshold 0.8
```  


## Training
Given an image pair, we first create a symmetrized version of the pair and feed it into the frozen MASt3R model. Multi-layer features are extracted from each decoder branch, concatenated, and fed into two learnable doppelganger classification heads. Each head generates predictions supervised by cross-entropy loss. 

<p align="center">
<img src="assets/model.jpg" width=100% height=100% 
class="center">
</p>

To train the model, first download a pretrained MASt3R model, for example [`MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`](https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth):
```bash
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
```
An example training script:
```bash
python train.py \
    --train_dataset="Doppelgangers(ROOT='data', split='train', trainon=['dg', 'visym'], teston=['dg'], aug_crop=16, aug_monocular=0, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter)" \
    --test_dataset="Doppelgangers(ROOT='data', split='val', trainon=['dg'], teston=['dg'], resolution=(512,336), seed=777)" \
    --model "AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', head_type_dg='transformer', output_mode='pts3d+desc24', output_mode_dg='dg_score', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True, desc_conf_mode=('exp', 0, inf), add_dg_pred_head=True, freeze=['mask','encoder','decoder','head'])" \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='?avg_dis'), alpha=0.2) + 0.075*ConfMatchingLoss(MatchingLoss(InfoNCE(mode='proper', temperature=0.05), negatives_padding=0, blocksize=8192), alpha=10.0, confmode='mean')" \
    --test_criterion "Regr3D_ScaleShiftInv(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0) + -1.*MatchingLoss(APLoss(nq='torch', fp=torch.float16), negatives_padding=12288)" \
    --train_criterion_dg="FocalLoss()" \
    --test_criterion_dg="FocalLoss()" \
    --pretrained "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" \
    --lr 0.0001 --min_lr 1e-06 --warmup_epochs=20 --epochs=100 --batch_size 4 --accum_iter 2 \
    --save_freq 1 --keep_freq 5 --eval_freq 1 --print_freq=10 --disable_cudnn_benchmark \
    --output_dir "checkpoints/dgpp-tr[dg+visym]"
```



## Evaluation
To evaluate a model, use `test.py`, and specify the pretrained model path, e.g. `checkpoints/checkpoint-dg+visym.pth`. Change `teston` value in `test_dataset` to switch test sets.
```bash
python test.py \
    --train_dataset="Doppelgangers(ROOT='data', split='train', trainon=['dg'], teston=['dg'], aug_crop=16, aug_monocular=0, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter)" \
    --test_dataset="Doppelgangers(ROOT='data', split='val', trainon=['dg'], teston=['visym'], resolution=(512,336), seed=777)" \
    --model "AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', head_type_dg='transformer', output_mode='pts3d+desc24', output_mode_dg='dg_score', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True, desc_conf_mode=('exp', 0, inf), add_dg_pred_head=True, freeze=['mask','encoder','decoder','head'])" \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='?avg_dis'), alpha=0.2) + 0.075*ConfMatchingLoss(MatchingLoss(InfoNCE(mode='proper', temperature=0.05), negatives_padding=0, blocksize=8192), alpha=10.0, confmode='mean')" \
    --test_criterion "Regr3D_ScaleShiftInv(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0) + -1.*MatchingLoss(APLoss(nq='torch', fp=torch.float16), negatives_padding=12288)" \
    --train_criterion_dg="FocalLoss()" \
    --test_criterion_dg="FocalLoss()" \
    --pretrained "checkpoints/checkpoint-dg+visym.pth" \
    --lr 0.0001 --min_lr 1e-06 --warmup_epochs=20 --epochs=100 --batch_size 4 --accum_iter 2 \
    --save_freq 1 --keep_freq 5 --eval_freq 1 --print_freq=10 --disable_cudnn_benchmark \
    --output_dir "checkpoints/checkpoint-dg+visym"
```

## Citation
If you find our work helpful, please consider citing:
```bibtex
@misc{xiangli2024doppelgangersimprovedvisualdisambiguation,
      title={Doppelgangers++: Improved Visual Disambiguation with Geometric 3D Features}, 
      author={Yuanbo Xiangli and Ruojin Cai and Hanyu Chen and Jeffrey Byrne and Noah Snavely},
      year={2024},
      eprint={2412.05826},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.05826}, 
}
```

