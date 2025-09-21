export MUJOCO_GL=egl
domain=ball_in_cup
task=catch
algorithm_name=muha_t_freq
CUDA_VISIBLE_DEVICES=0
# CUDA_VISIBLE_DEVICES=0 python src/reseval.py --algorithm $algorithm_name \
#                       --eval_episodes 100 \
#                       --seed 1\
#                       --eval_mode video_easy \
#                       --action_repeat 2 \
#                       --domain_name $domain \
#                       --task_name $task
# python src/reseval.py --algorithm $algorithm_name \
#                       --eval_episodes 100 \
#                       --seed 2\
#                       --eval_mode video_easy \
#                       --action_repeat 2 \
#                       --domain_name $domain \
#                       --task_name $task
# python src/reseval.py --algorithm $algorithm_name \
#                       --eval_episodes 100 \
#                       --seed 3 \
#                       --eval_mode video_easy \
#                       --action_repeat 2 \
#                       --domain_name $domain \
#                       --task_name $task
# python src/reseval.py --algorithm $algorithm_name \
#                       --eval_episodes 100 \
#                       --seed 4 \
#                       --eval_mode video_easy \
#                       --action_repeat 2 \
#                       --domain_name $domain \
#                       --task_name $task
python src/reseval.py --algorithm $algorithm_name \
                      --eval_episodes 100 \
                      --seed 3 \
                      --eval_mode video_easy \
                      --action_repeat 2 \
                      --domain_name $domain \
                      --task_name $task
# python src/reseval.py --algorithm $algorithm_name \
#                       --eval_episodes 100 \
#                       --seed 1 \
#                       --eval_mode video_hard \
#                       --action_repeat 2 \
#                       --domain_name $domain \
#                       --task_name $task
# python src/reseval.py --algorithm $algorithm_name \
#                       --eval_episodes 100 \
#                       --seed 2 \
#                       --eval_mode video_hard \
#                       --action_repeat 2 \
#                       --domain_name $domain \
#                       --task_name $task
# python src/reseval.py --algorithm $algorithm_name \
#                       --eval_episodes 100 \
#                       --seed 3 \
#                       --eval_mode video_hard \
#                       --action_repeat 2 \
#                       --domain_name $domain \
#                       --task_name $task
# python src/reseval.py --algorithm $algorithm_name \
#                       --eval_episodes 100 \
#                       --seed 4 \
#                       --eval_mode video_hard \
#                       --action_repeat 2 \
#                       --domain_name $domain \
#                       --task_name $task
python src/reseval.py --algorithm $algorithm_name \
                      --eval_episodes 100 \
                      --seed 3 \
                      --eval_mode video_hard \
                      --action_repeat 2 \
                      --domain_name $domain \
                      --task_name $task
