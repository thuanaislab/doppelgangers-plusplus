import pycolmap
import os
import numpy as np
import copy
import shutil
import torch
import subprocess
import fnmatch
import json
import argparse

from kapture.converter.colmap.database_extra import kapture_to_colmap
from kapture.converter.colmap.database import COLMAPDatabase

from mast3r.colmap.mapping import kapture_import_image_folder_or_list, run_mast3r_matching, run_mast3r_matching_doppelgangers
from mast3r.retrieval.processor import Retriever
from mast3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from mast3r.model import AsymmetricMASt3R

def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Structure from Motion disambiguation with Doppelgangers classification model.')
    parser.add_argument('--colmap_exe_command', default='colmap', type=str, help='COLMAP executable command.')
    parser.add_argument('--input_image_path', type=str, required=True, help='Path to the input image dataset.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save output results.')
    parser.add_argument('--threshold', type=float, default=0.8, help='Doppelgangers threshold.')
    parser.add_argument('--pretrained', type=str, default='checkpoints/dopp-crop-focalloss_lr1e-3_warmup20/checkpoint-best.pth', help="Path to the pretrained model checkpoint.")
    parser.add_argument('--retrieval_model', type=str, default=None, help="Path to the retrieval model checkpoint.")
    parser.add_argument('--skip_mapper', action='store_true', help="Skip COLMAP mapping stage.")
    args = parser.parse_args()
    return args


def colmap_run_mapper(colmap_bin, colmap_db_path, recon_path, image_root_path):
    print("running mapping")
    args = [
        'mapper',
        '--database_path',
        colmap_db_path,
        '--image_path',
        image_root_path,
        '--output_path',
        recon_path,
    ]
    args.insert(0, colmap_bin)
    colmap_process = subprocess.Popen(args)
    colmap_process.wait()

    if colmap_process.returncode != 0:
        raise ValueError(
            '\nSubprocess Error (Return code:'
            f' {colmap_process.returncode} )')


if __name__ == '__main__':
    args = get_args()
    
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    print("loading model")
    pretrained = args.pretrained
    model = AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', head_type_dg='transformer', 
                                output_mode='pts3d+desc24', output_mode_dg='dg_score', depth_mode=('exp', -np.inf, np.inf), conf_mode=('exp', 1, np.inf), 
                                enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True, desc_conf_mode=('exp', 0, np.inf), 
                                add_dg_pred_head=True, freeze=['mask','encoder','decoder','head']).from_pretrained(pretrained).to(device)
    
    shared_intrinsics = False
    scenegraph_type = "retrieval"
    winsize = 1
    win_cyclic = False
    refid = 0
    
    filelist = []
    for root, dirs, files in os.walk(args.input_image_path):
        for file in files:
            if file.endswith('jpg') or file.endswith('png') or file.endswith('jpeg') or file.endswith('JPG') or file.endswith('PNG') or file.endswith('JPEG'):
                filelist.append(os.path.join(root, file))
    # remove corrupted image
    # filelist.pop(filelist.index(f'{args.input_image_path}/0388.jpg'))
    
    imgs = load_images(filelist, size=512, verbose=False)
    image_size = 512
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        filelist = [filelist[0], filelist[0] + '_2']
    

    os.makedirs(args.output_path, exist_ok=True)
    colmap_db_path = os.path.join(args.output_path, 'colmap.db')
    if not os.path.isfile(colmap_db_path) or not os.path.exists(args.output_path + '/pairs.txt'):
        scene_graph_params = [scenegraph_type]
        if scenegraph_type in ["swin", "logwin"]:
            scene_graph_params.append(str(winsize))
        elif scenegraph_type == "oneref":
            scene_graph_params.append(str(refid))
        elif scenegraph_type == "retrieval":
            winsize = min(20, len(filelist))
            refid = min(len(filelist) - 1, 10)
            scene_graph_params.append(str(winsize))  # Na
            scene_graph_params.append(str(refid))  # k
        if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
            scene_graph_params.append('noncyclic')
        scene_graph = '-'.join(scene_graph_params)

        sim_matrix = None
        retrieval_model = args.retrieval_model
        if 'retrieval' in scenegraph_type:
            assert retrieval_model is not None
            print("start retrieval")
            retriever = Retriever(retrieval_model, backbone=model, device=device)
            with torch.no_grad():
                sim_matrix = retriever(filelist)

            # Cleanup
            del retriever
            torch.cuda.empty_cache()
            print("finish retrieval")
            

        print("make pairs")
        pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True, sim_mat=sim_matrix)
        root_path = os.path.commonpath(filelist)
        filelist_relpath = [
            os.path.relpath(filename, root_path).replace('\\', '/')
            for filename in filelist
        ]
        kdata = kapture_import_image_folder_or_list((root_path, filelist_relpath), shared_intrinsics)
        image_pairs = [
            (filelist_relpath[img1['idx']], filelist_relpath[img2['idx']])
            for img1, img2 in pairs
        ]

        os.makedirs(os.path.dirname(colmap_db_path), exist_ok=True)
        dopp_pred_path = colmap_db_path.replace('colmap.db', 'doppelgangers_propability_list.npy')
        colmap_db = COLMAPDatabase.connect(colmap_db_path)
        # try:
        kapture_to_colmap(kdata, root_path, tar_handler=None, database=colmap_db,
                        keypoints_type=None, descriptors_type=None, export_two_view_geometry=False)
        print("start processing pairs")
        colmap_image_pairs = run_mast3r_matching_doppelgangers(model, image_size, 16, device,
                                                kdata, root_path, image_pairs, colmap_db,
                                                False, 5, 1.001,
                                                False, 3,
                                                dopp_pred_path, args.threshold)
        colmap_db.close()
        # except Exception as e:
        #     print(f'Error {e}')
        #     colmap_db.close()
        #     exit(1)

        if len(colmap_image_pairs) == 0:
            raise Exception("no matches were kept")

        # colmap db is now full, run colmap
        colmap_world_to_cam = {}
        print("verify_matches")
        f = open(args.output_path + '/pairs.txt', "w")
        for image_path1, image_path2 in colmap_image_pairs:
            f.write("{} {}\n".format(image_path1, image_path2))
        f.close()
        pycolmap.verify_matches(colmap_db_path, args.output_path + '/pairs.txt')

    if not args.skip_mapper:
        reconstruction_path = os.path.join(args.output_path, "reconstruction")
        os.makedirs(reconstruction_path, exist_ok=True)
        
        try:
            colmap_run_mapper(args.colmap_exe_command, colmap_db_path, reconstruction_path, root_path)
        except:
            print("unable to reconstruct scene")
    



