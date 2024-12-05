import os
import numpy as np
import argparse
import torch
import subprocess
from tqdm import tqdm
from scipy.special import softmax
from mast3r.model import AsymmetricMASt3R
from mast3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from utils.process_database import create_image_pair_list, remove_doppelgangers


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Structure from Motion disambiguation with Doppelgangers classification model.')
    parser.add_argument('--colmap_exe_command', default='colmap', type=str, help='COLMAP executable command.')
    parser.add_argument('--matching_type', default='vocab_tree_matcher', type=str, help="Feature matching type: ['vocab_tree_matcher', 'exhaustive_matcher']")
    # parser.add_argument('--skip_feature_matching', action='store_true', help="Skip COLMAP feature matching stage.")
    # parser.add_argument('--database_path', type=str, default=None, help="Path to the COLMAP database.")
    parser.add_argument('--input_image_path', type=str, required=True, help='Path to the input image dataset.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save output results.')
    parser.add_argument('--threshold', type=float, default=0.8, help='Doppelgangers threshold.')
    parser.add_argument('--pretrained', type=str, default='checkpoints/dopp-crop-focalloss_lr1e-3_warmup20/checkpoint-best.pth', help="Path to the pretrained model checkpoint.")
    parser.add_argument('--skip_mapper', action='store_true', help="Skip COLMAP mapping stage.")
    
    args = parser.parse_args()
    return args


def colmap_runner(args):
    os.makedirs('weights', exist_ok=True)
    vocab_tree_path = 'weights/vocab_tree_flickr100K_words1M.bin'

    if args.matching_type == 'vocab_tree_matcher' and not os.path.exists(vocab_tree_path):
        subprocess.run(
            ["wget", "https://demuc.de/colmap/vocab_tree_flickr100K_words1M.bin", "-P", "weights/"],
            check=True
        )

    commands = [
        [
            args.colmap_exe_command, "feature_extractor",
            "--image_path", args.input_image_path,
            "--database_path", args.database_path
        ],
        [
            args.colmap_exe_command, args.matching_type,
            "--database_path", args.database_path,
        ]
    ]

    if args.matching_type == 'vocab_tree_matcher':
        commands[1].extend(["--VocabTreeMatching.vocab_tree_path", vocab_tree_path])

    for command in commands:
        subprocess.run(command, check=True)
        

def doppelgangers_classifier(args):
    """Classify image pairs to filter Doppelgangers."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = AsymmetricMASt3R.from_pretrained(args.pretrained).to(device)
    model = AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', head_type_dg='transformer', 
                             output_mode='pts3d+desc24', output_mode_dg='dg_score', depth_mode=('exp', -np.inf, np.inf), conf_mode=('exp', 1, np.inf), 
                             enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True, desc_conf_mode=('exp', 0, np.inf), 
                             add_dg_pred_head=True, freeze=['mask','encoder','decoder','head']).from_pretrained(args.pretrained).to(device)

    pairs = np.load(f"{args.output_path}/pairs_list.npy")
    prob_list = []

    for pair in tqdm(pairs, desc="Disambiguating pairs"):
        img1, img2, *_ = pair
        img_paths = [os.path.join(args.input_image_path, img) for img in [img1, img2]]
        images = load_images(img_paths, size=512, verbose=False)
        output = inference(make_pairs(images), model, device, verbose=False)

        pred1, pred2 = output['pred1'], output['pred2']
        if isinstance(output['pred1'], list):
            pred1 = torch.stack(output['pred1'], dim=0)
        else:
            pred1 = output['pred1']
            
        if isinstance(output['pred2'], list):
            pred2 = torch.stack(output['pred2'], dim=0)
        else:
            pred2 = output['pred2']
            
        score_s1 = softmax(pred1.detach().cpu().numpy(), axis=1)
        score_s2 = softmax(pred2.detach().cpu().numpy(), axis=1)
        vote_0 = sum(score_s1[:,0] > score_s1[:,1]) + sum(score_s2[:,0] > score_s2[:,1])
        vote_1 = sum(score_s1[:,1] > score_s1[:,0]) + sum(score_s2[:,1] > score_s2[:,0])
        if vote_1 > vote_0:
            score = np.max((score_s1[:,1], score_s2[:,1]))
        elif vote_1 < vote_0:
            score = np.min((score_s1[:,1], score_s2[:,1]))
        else:
            score = np.mean((score_s1[:,1], score_s2[:,1]))
            
        prob_list.append(score)

    np.save(f"{args.output_path}/pair_probability_list_dust3r.npy", {'prob': np.array(prob_list).reshape(-1, 1)})


def main():
    args = get_args()
    os.makedirs(args.output_path, exist_ok=True)

    args.database_path = os.path.join(args.output_path, 'database.db')
    colmap_runner(args)

    pair_path = create_image_pair_list(args.database_path, args.output_path)
    doppelgangers_classifier(args)
    update_database_path = remove_doppelgangers(args.database_path, f"{args.output_path}/pair_probability_list_dust3r.npy", pair_path, args.threshold)
    
    if not args.skip_mapper:
        doppelgangers_result_path = os.path.join(args.output_path, 'sparse_doppelgangers_%.3f_pheonix'%args.threshold)    
        os.makedirs(doppelgangers_result_path, exist_ok=True)       
        subprocess.run([args.colmap_exe_command, 'mapper',
                '--database_path', update_database_path,
                '--image_path', args.input_image_path,
                '--output_path', doppelgangers_result_path,
                ])
    


if __name__ == '__main__':
    main()
