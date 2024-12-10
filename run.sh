torchrun --nproc_per_node=2 --master_port=25911 train.py \
    --train_dataset="Doppelgangers(ROOT='/share/phoenix/nfs06/S9/yx642/doppelganger', split='train', trainon=['dg', 'visym'], teston=['dg'], aug_crop=16, aug_monocular=0, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter)" \
    --test_dataset="Doppelgangers(ROOT='/share/phoenix/nfs06/S9/yx642/doppelganger', split='val', trainon=['dg'], teston=['dg'], resolution=(512,336), seed=777)" \
    --model "AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', head_type_dg='transformer', output_mode='pts3d+desc24', output_mode_dg='dg_score', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True, desc_conf_mode=('exp', 0, inf), add_dg_pred_head=True, freeze=['mask','encoder','decoder','head'])" \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='?avg_dis'), alpha=0.2) + 0.075*ConfMatchingLoss(MatchingLoss(InfoNCE(mode='proper', temperature=0.05), negatives_padding=0, blocksize=8192), alpha=10.0, confmode='mean')" \
    --test_criterion "Regr3D_ScaleShiftInv(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0) + -1.*MatchingLoss(APLoss(nq='torch', fp=torch.float16), negatives_padding=12288)" \
    --train_criterion_dg="FocalLoss()" \
    --test_criterion_dg="FocalLoss()" \
    --pretrained "/share/phoenix/nfs06/S9/yx642/ckpt/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" \
    --lr 0.0001 --min_lr 1e-06 --warmup_epochs=20 --epochs=100 --batch_size 4 --accum_iter 2 \
    --save_freq 1 --keep_freq 5 --eval_freq 1 --print_freq=10 --disable_cudnn_benchmark \
    --output_dir "/share/phoenix/nfs06/S9/yx642/ckpt/test"

python test.py \
    --train_dataset="Doppelgangers(ROOT='/share/phoenix/nfs06/S9/yx642/doppelganger', split='train', trainon=['dg'], teston=['dg'], aug_crop=16, aug_monocular=0, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter)" \
    --test_dataset="Doppelgangers(ROOT='/share/phoenix/nfs06/S9/yx642/doppelganger', split='val', trainon=['dg'], teston=['visym'], resolution=(512,336), seed=777)" \
    --model "AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', head_type_dg='transformer', output_mode='pts3d+desc24', output_mode_dg='dg_score', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True, desc_conf_mode=('exp', 0, inf), add_dg_pred_head=True, freeze=['mask','encoder','decoder','head'])" \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='?avg_dis'), alpha=0.2) + 0.075*ConfMatchingLoss(MatchingLoss(InfoNCE(mode='proper', temperature=0.05), negatives_padding=0, blocksize=8192), alpha=10.0, confmode='mean')" \
    --test_criterion "Regr3D_ScaleShiftInv(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0) + -1.*MatchingLoss(APLoss(nq='torch', fp=torch.float16), negatives_padding=12288)" \
    --train_criterion_dg="FocalLoss()" \
    --test_criterion_dg="FocalLoss()" \
    --pretrained "/share/phoenix/nfs06/S9/yx642/ckpt/release_dg+visym/checkpoint-dg+visym.pth" \
    --lr 0.0001 --min_lr 1e-06 --warmup_epochs=20 --epochs=100 --batch_size 4 --accum_iter 2 \
    --save_freq 1 --keep_freq 5 --eval_freq 1 --print_freq=10 --disable_cudnn_benchmark \
    --output_dir "/share/phoenix/nfs06/S9/yx642/ckpt/release"

python colmap_usage.py --colmap_exe_command colmap \
    --input_image_path /share/phoenix/nfs04/S7/rc844/SFM3D/reconstruct/disambiguity_dataset/images/alexander_nevsky_cathedral/images \
    --output_path /share/phoenix/nfs06/S9/yx642/mapillary_probe/alexander_nevsky_cathedral/refactor \
    --threshold 0.8 \
    --pretrained /share/phoenix/nfs06/S9/yx642/ckpt/release/checkpoint-best.pth \
    --skip_feature_matching \
    --database_path /share/phoenix/nfs06/S9/yx642/mapillary_probe/alexander_nevsky_cathedral/refactor/database.db

python mast3r_sfm_usage.py \
    --input_image_path /share/phoenix/nfs04/S7/rc844/SFM3D/reconstruct/disambiguity_dataset/images/desk/images \
    --output_path /share/phoenix/nfs06/S9/yx642/mapillary_probe/desk/refactor_mast3rsfm2 \
    --pretrained /share/phoenix/nfs06/S9/yx642/ckpt/release/checkpoint-best.pth \
    --retrieval_model ../doppelganger_mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth \
    --threshold 0.8
