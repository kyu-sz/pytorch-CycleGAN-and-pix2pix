python test.py --dataroot ./kitti_with_mask --name n2r_pix2pix --model test --which_model_netG unet_256\
    --which_direction BtoA --dataset_mode single --norm batch --how_many 5000 --which_epoch freezed\
    --resize_or_crop none
