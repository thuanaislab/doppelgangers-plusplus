source doppelgpp-env/bin/activate

export img_path=/media/cool/input-store1/Thuan_Workspace/segmentation-pipeline/outputs/Anew/BayField/Oct2024/Plot430/sites/Plot430/Images/images
export output_path=/media/cool/input-store1/Thuan_Workspace/segmentation-pipeline/outputs/Anew/BayField/Oct2024/Plot430

python colmap_usage.py --colmap_exe_command colmap \
    --input_image_path $img_path \
    --output_path $output_path \
    --matching_type exhaustive_matcher \
    --threshold 0.6 \
    --pretrained checkpoints/checkpoint-dg+visym.pth \

deactivate