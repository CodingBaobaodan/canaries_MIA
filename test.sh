### alexnet cifar100
# base
#CUDA_VISIBLE_DEVICES=1 python3 gen_canary.py --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar100_alexnet_1_200_0_0_0_0_0.0_0_0.0_loss_0.0_0.0_100 --name alexnet --save_name alexnet --net alexnet --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000
# all def
CUDA_VISIBLE_DEVICES=0 python3 gen_canary.py --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar100_alexnet_5_40_0_0_0_0_0.0_1_0.0_loss_0.0_32.5_55 --name alexnet --save_name alexnet --net alexnet --num_shadow 50 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000
# mixup
#CUDA_VISIBLE_DEVICES=1 python3 gen_canary.py --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar100_alexnet_1_200_0_0_0_0_16.0_1_0.0_loss_0.0_0.0_100 --name alexnet --save_name alexnet --net alexnet --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000
### selena
#CUDA_VISIBLE_DEVICES=0 python3 gen_canary.py --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/selena/selena_alexnet_cifar100 --name alexnet --save_name alexnet --net alexnet --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000

### alexnet cifar10
# base
#CUDA_VISIBLE_DEVICES=1 python3 gen_canary.py --dataset cifar10 --num_classes 10 --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar10_alexnet_1_200_0_0_0_0_0.0_0_0.0_loss_0.0_0.0_100 --name alexnet --save_name alexnet --net alexnet --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000
# all def
#CUDA_VISIBLE_DEVICES=2 python3 gen_canary.py  --dataset cifar10 --num_classes 10 --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar10_alexnet_5_40_0_0_0_0_0.0_1_0.0_loss_0.0_1.0_100 --name alexnet --save_name alexnet --net alexnet --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000
# mixup
#CUDA_VISIBLE_DEVICES=1 python3 gen_canary.py  --dataset cifar10 --num_classes 10 --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar10_alexnet_1_200_0_0_0_0_0.5_1_0.0_loss_0.0_0.0_100 --name alexnet --save_name alexnet --net alexnet --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000
### selena
#CUDA_VISIBLE_DEVICES=3 python3 gen_canary.py --dataset cifar10 --num_classes 10 --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/selena/selena_alexnet_cifar10 --name alexnet --save_name alexnet --net alexnet --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000


### resnet18 cifar100
# base
#CUDA_VISIBLE_DEVICES=0 python3 gen_canary.py --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar100_resnet18_1_200_0_0_0_0_0.0_0_0.0_loss_0.0_0.0_100 --name resnet18 --save_name resnet18 --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000
# all def
#CUDA_VISIBLE_DEVICES=1 python3 gen_canary.py --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar100_resnet18_4_50_0_0_0_0_0.0_1_0.0_loss_0.0_6.25_100 --name resnet18 --save_name resnet18 --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000
# mixup
#CUDA_VISIBLE_DEVICES=3 python3 gen_canary.py --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar100_resnet18_1_200_0_0_0_0_0.025_1_0.0_loss_0.0_0.0_100 --name resnet18 --save_name resnet18 --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000

### resnet18 cifar10
# base
#CUDA_VISIBLE_DEVICES=0 python3 gen_canary.py  --dataset cifar10 --num_classes 10 --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar10_resnet18_1_200_0_0_0_0_0.0_0_0.0_loss_0.0_0.0_100 --name resnet18 --save_name resnet18 --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000
# all def
#CUDA_VISIBLE_DEVICES=1 python3 gen_canary.py  --dataset cifar10 --num_classes 10 --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar10_resnet18_2_100_0_0_0_0_0.0_1_0.0_loss_0.0_3.5_100 --name resnet18 --save_name resnet18 --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000
# mixup
#CUDA_VISIBLE_DEVICES=3 python3 gen_canary.py  --dataset cifar10 --num_classes 10 --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar10_resnet18_1_200_0_0_0_0_0.0425_1_0.0_loss_0.0_0.0_100 --name resnet18 --save_name resnet18 --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000


### densenet cifar100
# base
#CUDA_VISIBLE_DEVICES=0 python3 gen_canary.py --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar100_densenet_1_200_0_0_0_0_0.0_0_0.0_loss_0.0_0.0_100 --name densenet --save_name densenet --num_shadow 50 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000
# all def
#CUDA_VISIBLE_DEVICES=0 python3 gen_canary.py --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar100_densenet_4_50_0_0_0_0_0.0_1_0.0_loss_0.0_8.5_100 --name densenet --save_name densenet --num_shadow 50 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000
# mixup
#CUDA_VISIBLE_DEVICES=2 python3 gen_canary.py --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar100_densenet_1_200_0_0_0_0_0.88_1_0.0_loss_0.0_0.0_100 --name densenet --save_name densenet --num_shadow 50 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000

### densenet cifar10
# base
#CUDA_VISIBLE_DEVICES=1 python3 gen_canary.py  --dataset cifar10 --num_classes 10 --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar10_densenet_1_200_0_0_0_0_0.0_0_0.0_loss_0.0_0.0_100 --name densenet --save_name densenet --num_shadow 50 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000
# all def
#CUDA_VISIBLE_DEVICES=0 python3 gen_canary.py  --dataset cifar10 --num_classes 10 --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar10_densenet_3_66_0_0_0_0_0.0_1_0.0_loss_0.0_5.0_100 --name densenet --save_name densenet --num_shadow 50 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000
# mixup
#CUDA_VISIBLE_DEVICES=3 python3 gen_canary.py  --dataset cifar10 --num_classes 10 --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar10_densenet_1_200_0_0_0_0_0.1375_1_0.0_loss_0.0_0.0_100 --name densenet --save_name densenet --num_shadow 50 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000


### texas
# base
#CUDA_VISIBLE_DEVICES=1 python3 gen_canary.py  --dataset texas --num_classes 100 --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/texas_texas_1_400_0_0_0_0_0.0_0_0.0_loss_0.0_0.0_100 --name texas --save_name texas --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000
# all def
#CUDA_VISIBLE_DEVICES=1 python3 gen_canary.py --dataset texas --num_classes 100 --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/texas_texas_10_40_0_0_0_0_0.0_0_0.0_loss_0.0_60.0_100 --name texas --save_name texas --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000
# mixup
#CUDA_VISIBLE_DEVICES=1 python3 gen_canary.py --dataset texas --num_classes 100 --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/texas_texas_1_400_0_0_0_0_0.1_1_0.0_loss_0.0_0.0_100 --name texas --save_name texas --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000


### purchase
# base
#CUDA_VISIBLE_DEVICES=1 python3 gen_canary.py --dataset purchase --num_classes 100 --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/purchase_purchase_1_400_0_0_0_0_0.0_0_0.0_loss_0.0_0.0_100 --name purchase --save_name purchase --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000
# all def
#CUDA_VISIBLE_DEVICES=1 python3 gen_canary.py --dataset purchase --num_classes 100 --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/purchase_purchase_4_100_0_0_0_0_0.0_0_0.0_loss_0.0_25.0_100 --name purchase --save_name purchase --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000
# mixup
#CUDA_VISIBLE_DEVICES=1 python3 gen_canary.py --dataset purchase --num_classes 100 --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/purchase_purchase_1_400_0_0_0_0_1.0_1_0.0_loss_0.0_0.0_100 --name purchase --save_name purchase --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000



#CUDA_VISIBLE_DEVICES=3 python3 gen_canary.py --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar100_alexnet_4_50_0_0_0_0_0.0_1_0.0_loss_0.0_8.0_100 --name alexnet --save_name alexnet --net alexnet --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000

#CUDA_VISIBLE_DEVICES=3 python3 gen_canary.py --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar100_alexnet_4_50_0_0_0_0_0.0_1_0.0_loss_0.0_9.0_100 --name alexnet --save_name alexnet --net alexnet --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000

#CUDA_VISIBLE_DEVICES=3 python3 gen_canary.py --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar100_alexnet_4_50_0_0_0_0_0.0_1_0.0_loss_0.0_10.0_100 --name alexnet --save_name alexnet --net alexnet --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000

#CUDA_VISIBLE_DEVICES=3 python3 gen_canary.py --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar100_alexnet_4_50_0_0_0_0_0.0_1_0.0_loss_0.0_11.0_100 --name alexnet --save_name alexnet --net alexnet --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000

#CUDA_VISIBLE_DEVICES=3 python3 gen_canary.py --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar100_alexnet_4_50_0_0_0_0_0.0_1_0.0_loss_0.0_12.0_100 --name alexnet --save_name alexnet --net alexnet --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000

#CUDA_VISIBLE_DEVICES=3 python3 gen_canary.py --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar100_alexnet_4_50_0_0_0_0_0.0_1_0.0_loss_0.0_13.0_100 --name alexnet --save_name alexnet --net alexnet --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000

#CUDA_VISIBLE_DEVICES=3 python3 gen_canary.py --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar100_alexnet_4_50_0_0_0_0_0.0_1_0.0_loss_0.0_14.0_100 --name alexnet --save_name alexnet --net alexnet --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000

#CUDA_VISIBLE_DEVICES=3 python3 gen_canary.py --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar100_alexnet_4_50_0_0_0_0_0.0_1_0.0_loss_0.0_15.0_100 --name alexnet --save_name alexnet --net alexnet --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000

#CUDA_VISIBLE_DEVICES=3 python3 gen_canary.py --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar100_alexnet_4_50_0_0_0_0_0.0_1_0.0_loss_0.0_16.0_100 --name alexnet --save_name alexnet --net alexnet --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000

#CUDA_VISIBLE_DEVICES=0 python3 gen_canary.py  --dataset cifar10 --num_classes 10 --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar10_alexnet_4_50_0_0_0_0_0.0_1_0.0_loss_0.0_0.5_100 --name alexnet --save_name alexnet --net alexnet --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000

#CUDA_VISIBLE_DEVICES=0 python3 gen_canary.py  --dataset cifar10 --num_classes 10 --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar10_alexnet_5_40_0_0_0_0_0.0_1_0.0_loss_0.0_1.0_100 --name alexnet --save_name alexnet --net alexnet --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000

#CUDA_VISIBLE_DEVICES=0 python3 gen_canary.py  --dataset cifar10 --num_classes 10 --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar10_alexnet_5_40_0_0_0_0_0.0_1_0.0_loss_0.0_1.5_100 --name alexnet --save_name alexnet --net alexnet --num_shadow 69 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000

#sleep 1h
#CUDA_VISIBLE_DEVICES=3 python3 gen_canary.py  --dataset cifar10 --num_classes 10 --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar10_alexnet_4_50_0_0_0_0_0.0_1_0.0_loss_0.0_2.0_100 --name alexnet --save_name alexnet --net alexnet --num_shadow 50 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000

#CUDA_VISIBLE_DEVICES=3 python3 gen_canary.py  --dataset cifar10 --num_classes 10 --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar10_alexnet_4_50_0_0_0_0_0.0_1_0.0_loss_0.0_2.5_100 --name alexnet --save_name alexnet --net alexnet --num_shadow 50 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000

#CUDA_VISIBLE_DEVICES=2 python3 gen_canary.py  --dataset cifar10 --num_classes 10 --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar10_alexnet_4_50_0_0_0_0_0.0_1_0.0_loss_0.0_3.0_100 --name alexnet --save_name alexnet --net alexnet --num_shadow 50 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000

#CUDA_VISIBLE_DEVICES=2 python3 gen_canary.py  --dataset cifar10 --num_classes 10 --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar10_alexnet_4_50_0_0_0_0_0.0_1_0.0_loss_0.0_3.5_100 --name alexnet --save_name alexnet --net alexnet --num_shadow 50 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000

#CUDA_VISIBLE_DEVICES=1 python3 gen_canary.py  --dataset cifar10 --num_classes 10 --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar10_alexnet_4_50_0_0_0_0_0.0_1_0.0_loss_0.0_4.0_100 --name alexnet --save_name alexnet --net alexnet --num_shadow 50 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000

#CUDA_VISIBLE_DEVICES=1 python3 gen_canary.py  --dataset cifar10 --num_classes 10 --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar10_alexnet_4_50_0_0_0_0_0.0_1_0.0_loss_0.0_4.5_100 --name alexnet --save_name alexnet --net alexnet --num_shadow 50 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000

#CUDA_VISIBLE_DEVICES=1 python3 gen_canary.py  --dataset cifar10 --num_classes 10 --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar10_alexnet_4_50_0_0_0_0_0.0_1_0.0_loss_0.0_5.0_100 --name alexnet --save_name alexnet --net alexnet --num_shadow 50 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000

#CUDA_VISIBLE_DEVICES=1 python3 gen_canary.py  --dataset cifar10 --num_classes 10 --checkpoint_prefix /home/lijiacheng/worstcase/saved_models/cifar10_alexnet_4_50_0_0_0_0_0.0_1_0.0_loss_0.0_5.5_100 --name alexnet --save_name alexnet --net alexnet --num_shadow 50 --iter 30 --stochastic_k 2 --lr 0.05 --weight_decay 0.001 --init target_img --opt adamw --in_model_loss target_logits --out_model_loss target_logits --target_logits 10 0 --aug_strategy try_random_out_class --num_gen 10 --num_aug 10 --start 0 --end 5000
