for s in 1 2 3 4 5
do
    CUDA_VISIBLE_DEVICES=0 python train.py task=cartpole_swingup seed=$s use_wandb=False  num_train_frames=1001000
done
