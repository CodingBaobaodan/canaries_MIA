### Jiacheng's args

#CUDA_VISIBLE_DEVICES=0 python3 canary_if.py --select_top_num 10  --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar100_alexnet_4_50_0_0_0_0_0.0_1_0.0_loss_0.0_12.0_100 --name alexnet --save_name alexnet --net alexnet --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 30 --num_aug 30 --start 0 --end 5000

#CUDA_VISIBLE_DEVICES=1 python3 canary_if.py --select_top_num 10 --num_classes 10 --dataset cifar10 --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar10_alexnet_5_40_0_0_0_0_0.0_1_0.0_loss_0.0_1.0_100 --name alexnet --save_name alexnet --net alexnet --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 30 --num_aug 30 --start 0 --end 5000

#CUDA_VISIBLE_DEVICES=2 python3 canary_if.py --select_top_num 10  --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar100_alexnet_1_200_0_0_0_0_0.0_0_0.0_loss_0.0_0.0_100 --name alexnet --save_name alexnet --net alexnet --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 30 --num_aug 30 --start 0 --end 5000

#CUDA_VISIBLE_DEVICES=3 python3 canary_if.py --select_top_num 10 --num_classes 10 --dataset cifar10 --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar10_alexnet_1_200_0_0_0_0_0.0_0_0.0_loss_0.0_0.0_100 --name alexnet --save_name alexnet --net alexnet --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 20 --num_aug 20 --start 0 --end 5000

### Jilin's args

#  cifar100 args                                                                                                                                          
CUDA_VISIBLE_DEVICES=2 python3 canary_if.py --select_top_num 10 --num_classes 100 --dataset cifar100 --checkpoint_prefix /home/915688516/code/saved_models/ --name jilin_net --save_name jilin_net --net jilin_net --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 30 --num_aug 30 --start 0 --end 100


### Parameters explain"

# -- checkpoint_prefix: the checkpoint of the shadow model
# --num_gen: # of noises generated 
# --num_aug: ???
