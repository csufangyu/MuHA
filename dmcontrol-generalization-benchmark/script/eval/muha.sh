export MUJOCO_GL=egl
domain=cartpole
task=swingup
algorithm_name=muha
CUDA_VISIBLE_DEVICES=0 python src/reseval.py --algorithm $algorithm_name \
                      --eval_episodes 100 \
                      --seed 1\
                      --eval_mode video_easy \
                      --action_repeat 2 \
                      --domain_name $domain \
                      --task_name $task

CUDA_VISIBLE_DEVICES=0 python src/reseval.py --algorithm $algorithm_name \
                      --eval_episodes 100 \
                      --seed 1 \
                      --eval_mode video_hard \
                      --action_repeat 2 \
                      --domain_name $domain \
                      --task_name $task
