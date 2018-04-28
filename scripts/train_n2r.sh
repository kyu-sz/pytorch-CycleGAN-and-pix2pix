python train.py --dataroot ../datasets/rigid --name n2r_pix2pix --model pix2pix --which_model_netG unet_256\
    --which_direction BtoA --lambda_A 100 --dataset_mode randomly_masked --no_lsgan --norm batch --pool_size 0\
    --gpu_ids 3 --batchSize 64 --continue_train