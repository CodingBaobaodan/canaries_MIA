
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import os
import argparse
import copy
# import wandb
import statistics
import random
from pynvml import *
import tqdm
from data import *
from utils import *
from data.data_prepare import dataset, part_pytorch_dataset
from utils import cal_results, calculate_loss, calibrate_logits, generate_aug_imgs, get_attack_loss, get_curr_shadow_models, get_logits, progress_bar, set_random_seed, split_shadow_models,cal_results_jilin
from models.inferencemodel import *
from pytorch_influence_functions.influence_functions.influence_functions import *
from torch.utils.data import TensorDataset, DataLoader

in_sd_diff = 0
out_sd_diff = 0
counter = 0

def calc_single_influences(X_train, y_train, test_loader, net):
    train_size = X_train.shape[0]
    test_size = test_loader.dataset.__len__()
    influence_mat = np.zeros((test_size, train_size))
    for i in tqdm(range(test_size)):
        for j in range(train_size):
            tensor_dataset = TensorDataset(torch.from_numpy(np.expand_dims(X_train[j], 0)),
                                           torch.from_numpy(np.expand_dims(y_train[j], 0)))
            train_loader = DataLoader(tensor_dataset, batch_size=1, shuffle=False,
                                      pin_memory=False, drop_last=False)
            influence, _, _, _ = calc_influence_single(net, train_loader, test_loader, i, 0, 1, 1)
            influence_mat[i, j] = influence.item()
    return influence_mat

def generate_class_dict(args):
	dataset_class_dict = [[] for _ in range(args.num_classes)]
	for i in range(len(args.aug_trainset)):
		_, tmp_class = args.aug_trainset[i]
		dataset_class_dict[tmp_class].append(i)
	return dataset_class_dict

def generate_close_imgs(args):
	canaries = []
	target_class_list = args.dataset_class_dict[args.target_img_class]
	
	if args.aug_strategy and 'same_class_imgs' in args.aug_strategy:
		# assume always use the target img
		canaries = [args.aug_trainset[args.target_img_id][0].unsqueeze(0)]
		for i in range(args.num_gen - 1):
			img_id = random.sample(target_class_list, 1)[0]
			x = args.aug_trainset[img_id][0]
			x = x.unsqueeze(0)
			
			canaries.append(x)
	elif args.aug_strategy and 'nearest_imgs' in args.aug_strategy:
		similarities = []
		target_img = args.aug_trainset[args.target_img_id][0]
		canaries = []
		for i in target_class_list:
			similarities.append(torch.abs(target_img - args.aug_trainset[i][0]).sum())
		
		top_k_indx = np.argsort(similarities)[:(args.num_gen)]
		target_class_list = np.array(target_class_list)
		final_list = target_class_list[top_k_indx]
		
		for i in final_list:
			canaries.append(args.aug_trainset[i][0].unsqueeze(0))
	
	return canaries


def initialize_poison(args):
	"""Initialize according to args.init.
	Propagate initialization in distributed settings.
	"""
	if args.aug_strategy and ('same_class_imgs' in args.aug_strategy or 'nearest_imgs' in args.aug_strategy):
		if 'dataset_class_dict' not in args:
			args.dataset_class_dict = generate_class_dict(args)
		
		fixed_target_img = generate_close_imgs(args)
		args.fixed_target_img = torch.cat(fixed_target_img, dim=0).to(args.device)
	else:
		fixed_target_img = generate_aug_imgs(args)
		args.fixed_target_img = torch.cat(fixed_target_img, dim=0).to(args.device)
	
	# ds has to be placed on the default (cpu) device, not like self.ds
	dm = torch.tensor(args.data_mean)[None, :, None, None]
	ds = torch.tensor(args.data_std)[None, :, None, None]
	if args.init == 'zero':
		init = torch.zeros(args.num_gen, *args.canary_shape)
	elif args.init == 'rand':
		init = (torch.rand(args.num_gen, *args.canary_shape) - 0.5) * 2
		init *= 1 / ds
	elif args.init == 'randn':
		init = torch.randn(args.num_gen, *args.canary_shape)
		init *= 1 / ds
	elif args.init == 'normal':
		init = torch.randn(args.num_gen, *args.canary_shape)
	elif args.init == 'target_img':
		# init = torch.zeros(args.num_gen, *args.canary_shape).to(args.device)
		# init.data[:] = copy.deepcopy(args.canary_trainset[args.target_img_id][0])
		init = copy.deepcopy(args.fixed_target_img)
		init.requires_grad = True
		return init
	else:
		raise NotImplementedError()
	
	init = init.to(args.device)
	dm = dm.to(args.device)
	ds = ds.to(args.device)
	
	if args.epsilon:
		x_diff = init.data - args.fixed_target_img.data
		x_diff.data = torch.max(torch.min(x_diff, args.epsilon /
										  ds / 255), -args.epsilon / ds / 255)
		x_diff.data = torch.max(torch.min(x_diff, (1 - dm) / ds -
										  args.fixed_target_img), -dm / ds - args.fixed_target_img)
		init.data = args.fixed_target_img.data + x_diff.data
	else:
		init = torch.max(torch.min(init, (1 - dm) / ds), -dm / ds)
	
	init.requires_grad = True
	
	return init

def choose_neg_img(args,out_model_keep):
	### we need to find one image that has the same class as the args.target_img_class and is used in the current show model training
	## find all training data that are in the same class
	same_class_index = np.arange(len(args.original_targetset.train_label))[args.original_targetset.train_label == args.original_targetset.train_label[args.target_img_id]]
	## find all same class img used in training of this shadow model
	intersection = np.intersect1d(same_class_index,np.array(out_model_keep))
	neg_index = np.random.choice(intersection,1,replace=False)
	#print (neg_index)
	neg_img, neg_img_class = args.canary_trainset[neg_index]
	#print (neg_img)
	return neg_img,neg_img_class
	
def calculate_influence_loss(x,y,shadow_models,args):
	in_models, out_models = split_shadow_models(shadow_models, args.target_img_id)
	
	### for each canary, we should use the same set of neg samples for fairness
	all_neg_img = []
	for this_shadow_model in out_models:
		neg_img,neg_img_class = choose_neg_img(args,out_model_keep=this_shadow_model.in_data)
		all_neg_img.append((neg_img,neg_img_class))
	
	### first part of the loss should be the influence score from the true image, coming from in_models
	all_pos_inf = []
	all_neg_inf = []
	for this_shadow_model in in_models:
		### prepare train / test loader. this train loder is just the true target image, and the test loader is just the crafted image
		tensor_dataset = TensorDataset(args.target_img,torch.unsqueeze(args.target_img_class,0))
		#print (len(tensor_dataset))
		#print (tensor_dataset[0])
		train_loader = DataLoader(tensor_dataset, batch_size=1, shuffle=False,
								  pin_memory=False, drop_last=False,num_workers=0)
		# support batched x case
		if (len(x.shape) == 3):
			tensor_dataset = TensorDataset(torch.unsqueeze(x, 0),torch.unsqueeze(y, 0))
			test_loader = DataLoader(tensor_dataset, batch_size=1, shuffle=False,
								  pin_memory=False, drop_last=False)
		else:
			tensor_dataset = TensorDataset(x,y)
			test_loader = DataLoader(tensor_dataset, batch_size=1, shuffle=False,
								  pin_memory=False, drop_last=False)
			
		### create a new shadow model to require grad
		for param in this_shadow_model.model.parameters():
			param.requires_grad = True
		
		#calc_influence_single return a list, which contains the influence score of all training point to one test point， training loader == 1
		inf_from_true_image = [calc_influence_single(this_shadow_model.model,this_shadow_model.train_loader,test_loader,test_id_num=i,target_influence_loader=train_loader)[0][0] for i in range(args.num_gen)]
		for param in this_shadow_model.model.parameters():
			param.requires_grad = False
		#print (f"influence of the true image{inf_from_true_image}")
		inf_from_true_image = torch.from_numpy(np.array(inf_from_true_image))
		all_pos_inf.append(inf_from_true_image)
		#print (f"influence shape of the true image{inf_from_true_image.shape}")
	all_pos_inf = torch.vstack(all_pos_inf)
	#print (all_pos_inf.shape)
	all_pos_inf = torch.sum(all_pos_inf,dim=0)/len(in_models) #add inf value vertically and average
	#print (all_pos_inf.shape)
	
	for idx,this_shadow_model in enumerate(out_models):
		### prepare train / test loader. this train loder is a random image from the same class(which is also used in train),
		# and the test loader is just the crafted image
		choose_neg_img(args,out_model_keep=this_shadow_model.in_data)
		tensor_dataset = TensorDataset(torch.unsqueeze(all_neg_img[idx][0], 0), all_neg_img[idx][1])
		train_loader = DataLoader(tensor_dataset, batch_size=1, shuffle=False,
								  pin_memory=False, drop_last=False)
		#print (len(tensor_dataset))
		#print (tensor_dataset[0])
		#for x, y in train_loader:
		#	print (x.shape,y)
		
		# support batched x case
		if (len(x.shape) == 3):
			tensor_dataset = TensorDataset(torch.unsqueeze(x, 0),torch.unsqueeze(y, 0))
			test_loader = DataLoader(tensor_dataset, batch_size=1, shuffle=False,
								  pin_memory=False, drop_last=False)
		else:
			tensor_dataset = TensorDataset(x,y)
			test_loader = DataLoader(tensor_dataset, batch_size=1, shuffle=False,
								  pin_memory=False, drop_last=False)
		
		### create a new shadow model to require grad
		for param in this_shadow_model.model.parameters():
			param.requires_grad = True
		inf_from_neg_image = [calc_influence_single(this_shadow_model.model,this_shadow_model.train_loader,test_loader,test_id_num=i,target_influence_loader=train_loader)[0][0] for i in range(args.num_gen)]
		#print (f"influence of the negative image{inf_from_neg_image}")
		for param in this_shadow_model.model.parameters():
			param.requires_grad = False
		inf_from_neg_image = torch.from_numpy(np.array(inf_from_neg_image))
		all_neg_inf.append(inf_from_neg_image)
	all_neg_inf = torch.vstack(all_neg_inf)
	all_neg_inf = torch.sum(all_neg_inf,dim=0)/len(in_models)
	
	return all_pos_inf.detach() - all_neg_inf.detach() ## this is the influence loss

def generate_canary_one_shot(shadow_models, args, return_loss=False):
	target_img_class = args.target_img_class
	
	# get loss functions
	args.in_criterion = get_attack_loss(args.in_model_loss)
	args.out_criterion = get_attack_loss(args.out_model_loss)
	
	# initialize patch
	x = initialize_poison(args) # canaries are initialise here, x is stacked so that len(x) == num_gen
	y = torch.tensor([target_img_class] * args.num_gen).to(args.device)
	
	dm = torch.tensor(args.data_mean)[None, :, None, None].to(args.device)
	ds = torch.tensor(args.data_std)[None, :, None, None].to(args.device)
	
	# initialize optimizer
	if args.opt.lower() in ['adam', 'signadam']:
		optimizer = torch.optim.Adam([x], lr=args.lr, weight_decay=args.weight_decay)
	elif args.opt.lower() in ['sgd', 'signsgd']:
		optimizer = torch.optim.SGD([x], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
	elif args.opt.lower() in ['adamw']:
		optimizer = torch.optim.AdamW([x], lr=args.lr, weight_decay=args.weight_decay)
	
	if args.scheduling:
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.iter // 2.667, args.iter // 1.6,
																				args.iter // 1.142], gamma=0.1)
	else:
		scheduler = None
	
	# generate canary
	last_loss = 100000000000
	trigger_times = 0
	# initialise the noise, and requires_grad=True to ensure that the gradients of loss 
	# are computed and can be used for optimization purposes.
	loss = torch.tensor([0.0], requires_grad=True)
	
	for step in range(args.iter):
		# choose shadow models
		curr_shadow_models = get_curr_shadow_models(shadow_models, x, args)
		
		for _ in range(args.inner_iter):
			optimizer.zero_grad()
			### we aim for minimising both in_loss and out_loss which is achieved by optimising the loss
			# in_loss is (prediction logit - target_logit[0])^2, where [0] should be a positive and large value (aiming for good prediction)
			# out_loss is (prediction logit - target_logit[1])^2, where [1] should be a relatively small value (aiming for bad prediction)
			loss, in_loss, out_loss, reg_norm = calculate_loss(x, y, curr_shadow_models, args)
			### backward
			if loss != 0:
				#compute the gradient of the loss with respect to x
				x.grad, = torch.autograd.grad(loss, [x])
			if args.opt.lower() in ['signsgd', 'signadam'] and x.grad is not None:
				x.grad.sign_()
			optimizer.step()
			
			if scheduler is not None:
				scheduler.step()
			
			# projection --> to ensure that the calculation lies within the epsilon constraints.
			with torch.no_grad():
				if args.epsilon:
					x_diff = x.data - args.fixed_target_img.data
					x_diff.data = torch.max(torch.min(x_diff, args.epsilon /
													  ds / 255), -args.epsilon / ds / 255)
					x_diff.data = torch.max(torch.min(x_diff, (1 - dm) / ds -
													  args.fixed_target_img), -dm / ds - args.fixed_target_img)
					x.data = args.fixed_target_img.data + x_diff.data
				else:
					x.data = torch.max(torch.min(x, (1 - dm) / ds), -dm / ds)
			#if 1:
			if args.print_step:
				print(f'step: {step}, ' + 'loss: %.3f, in_loss: %.3f, out_loss: %.3f, reg_loss: %.3f' % (loss, in_loss, out_loss, reg_norm))
		
		if args.stop_loss is not None and loss <= args.stop_loss:
			break
		
		if args.early_stop and loss > last_loss:
			trigger_times += 1
			
			if trigger_times >= args.patience:
				break
		else:
			trigger_times = 0
		# returns the numerical value in the loss tensor
		last_loss = loss.item()
		
	### sort the generated canary based on their influence score, choose the top ones
	#print (x.shape,y.shape)
	#inf_score = calculate_influence_loss(x,y,shadow_models,args) ## inf_score is a array, x is the perturbed image 
	#argsort_index = torch.argsort(inf_score,descending=True)
	#print (f"all inf score {inf_score}")
	### todo: maybe we need to figure out how to visualize and analyze this inf score
	#x = (x[argsort_index])[:args.select_top_num]
	#loss =(loss[argsort_index])[:args.select_top_num]

	#if return_loss:
	#	return x.detach(), loss.item(),inf_score
	#else:
	#	return x.detach(),inf_score
	
	if return_loss:
		return x.detach(), loss.item()
	else:
		return x.detach()
	

def generate_canary(shadow_models, args):
	canaries = []
	all_inf_score = []
	if args.aug_strategy is not None:
		rand_start = random.randrange(args.num_classes)
		
		for out_target_class in range(1000):  # need to be simplified later
			if args.canary_aug:
				args.target_img, args.target_img_class = args.aug_trainset[args.target_img_id]
				args.target_img = args.target_img.unsqueeze(0).to(args.device)
			
			if 'try_all_out_class' in args.aug_strategy:
				out_target_class = (rand_start + out_target_class) % args.num_classes
			elif 'try_random_out_class' in args.aug_strategy:
				out_target_class = random.randrange(args.num_classes)
			elif 'try_random_diff_class' in args.aug_strategy:
				pass
			else:
				raise NotImplementedError()
			
			if out_target_class != args.target_img_class:
				if args.print_step:
					print(f'Try class: {out_target_class}')
				
				if 'try_random_diff_class' in args.aug_strategy:
					out_target_class = []
					for _ in range(args.num_gen):
						a = random.randrange(args.num_classes)
						while a == args.target_img_class:
							a = random.randrange(args.num_classes)
						
						out_target_class.append(a)
					
					args.out_target_class = torch.tensor(out_target_class).to(args.device)
				else:
					args.out_target_class = out_target_class
				
				#x, loss,inf_score = generate_canary_one_shot(shadow_models, args, return_loss=True)
				x, loss = generate_canary_one_shot(shadow_models, args, return_loss=True)
				# canaries is the Xmal (X with added noise)
				canaries.append(x)

				#all_inf_score.append(inf_score)
				args.canary_losses[-1].append(loss)
			
			#if (len(canaries) >= args.num_aug):
			#	break
			if sum([len(canary) for canary in canaries]) >= args.num_aug:  ### wrong code? (always generate one canary)?
				break
	else:
		x, loss = generate_canary_one_shot(shadow_models, args, return_loss=True)
		canaries.append(x)
		args.canary_losses[-1].append(loss)
	
	return canaries,all_inf_score

def noise_test(canaries: list, x_id: list, trainset, num_compare, shadow_models, args):

	"""This function takes the generated canaries, detach the noise added and 
	examine the effect by adding them into other random x. Return the top num_return 
	of canaries where their noises have relatively less effect on selected random x

    Arguments:
        canaries: list of canaries
		x_id: the target image id list that macthes the canaries list
        num_compare: number of random x generated for comparision
        num_return: the top num_return # of canaries will be return
        shawdow_models: the shawdow_models

    Returns:
        selected canaries: list of tensor (selected canaries) """

	# TODO: the list of canaries has only one element, all the canaries are stacked in canaries[0]
	# TODO: change the code 

	# create a list to store the absolute change in loss when noises are added for each canary 
	data = []
	# initialise the list with dictionaries that contains the image id, canaries and it's loss change effect on random x
	for canaries, x_id in zip(canaries,x_id):
		data.append({"id": x_id, "canaries": canaries, "loss": []})

	count = 0
	# start the iteration for each input data point
	for data_curr in data:


		# get the target image with respect to the current canaries
		target_img, _ = trainset[data_curr['id']]
		target_img = target_img.unsqueeze(0).to(args.device)

		# for each canary (noise) in canaries 
		for i in range(len(data_curr['canaries'])):

			# get the noise
			noise = data_curr['canaries'][i].unsqueeze(0).to(args.device) - target_img

			# generate N # of random indices
			indices = list(range(1,data_curr['id'])) + list(range(data_curr['id'] + 1, len(args.original_targetset.train_label)))
			random_N_img = random.sample(indices, num_compare)

			# create lists to store the totla loss (in_loss + out_loss) for the images with and without noise 
			ori_loss  = []
			withnoise_loss = []
			# create lists to store the loss's standard deviation for IN and OUT model with and without noise 
			ori_sd_in = []
			ori_sd_out = []
			withnoise_sd_in = []
			withnoise_sd_out = []

			# Turn on inference context manager
			with torch.inference_mode():
				# for each random x
				for id in random_N_img:

					# get random image from the current id
					x_rnd, target_class_rnd =  trainset[id]
					x_rnd = x_rnd.unsqueeze(0).to(args.device)
					# get x_rnd_noise by adding noise to x_rnd 
					x_rnd_noise = x_rnd + noise
					
					#tmp = []
					#for i in range(args.num_aug):
					#	target_img_rnd = args.aug_trainset[id][0]
					#	target_img_rnd = target_img_rnd.unsqueeze(0)
					#	tmp.append(target_img_rnd)
					#target_img_rnd = torch.cat(tmp, dim=0).to(args.device)

					# get IN and OUT model according to the chosen random image
					in_models, out_models = split_shadow_models(shadow_models, target_img_id = id)
					in_models = [models.eval() for models in in_models]
					out_models = [models.eval() for models in out_models]
					# initialise the IN and OUT loss for each iteration for one random image
					in_loss_curr  = []
					in_loss_noise_curr = []
					out_loss_curr = []
					out_loss_noise_curr = []


					# get the y_rnd ready to compute the losses
					with torch.no_grad():
						tmp_outputs = out_models[0](x_rnd)
					y_rnd = torch.zeros(tmp_outputs.shape, device=args.device, dtype=tmp_outputs.dtype)
					y_rnd[:, target_class_rnd] += args.target_logits[0]
					y_rnd = y_rnd[:, target_class_rnd]
					y_out = torch.zeros(tmp_outputs.shape, device=args.device, dtype=tmp_outputs.dtype)
					y_out[:, target_class_rnd] += args.target_logits[1] 
					y_out = y_out[:, target_class_rnd]


					# calculate losses for IN models with and without noise
					for curr_model in in_models:
						outputs = curr_model(x_rnd)
						outputs = outputs[:, target_class_rnd]
						curr_loss = args.in_criterion(outputs, y_rnd)
						in_loss_curr.append(curr_loss.item())

						outputs_noise = curr_model(x_rnd_noise)
						outputs_noise = outputs_noise[:, target_class_rnd]
						curr_loss_noise = args.in_criterion(outputs_noise, y_rnd)
						in_loss_noise_curr.append(curr_loss_noise.item())

					# calculate losses for the OUT models with and without noise
					for curr_model in out_models:
						outputs = curr_model(x_rnd)
						outputs = outputs[:, target_class_rnd]
						curr_loss = args.out_criterion(outputs, y_out)
						out_loss_curr.append(curr_loss.item()) 

						outputs_noise = curr_model(x_rnd_noise)
						outputs_noise = outputs_noise[:, target_class_rnd]
						curr_loss_noise = args.out_criterion(outputs_noise, y_out)
						out_loss_noise_curr.append(curr_loss_noise.item()) 

					# average the IN and OUT losses and their sd
					mean_in_loss = statistics.mean(in_loss_curr)
					sd_in_loss = statistics.stdev(in_loss_curr)

					mean_out_loss = statistics.mean(out_loss_curr)
					sd_out_loss = statistics.stdev(out_loss_curr)

					mean_in_loss_noise = statistics.mean(in_loss_noise_curr)
					sd_in_loss_noise = statistics.stdev(in_loss_noise_curr)

					mean_out_loss_noise = statistics.mean(out_loss_noise_curr)
					sd_out_loss_noise = statistics.stdev(out_loss_noise_curr)
					

					# append all the original loss and sd
					ori_loss.append(mean_in_loss + mean_out_loss)
					withnoise_loss.append(mean_in_loss_noise + mean_out_loss_noise)
					ori_sd_in.append(sd_in_loss)
					ori_sd_out.append(sd_out_loss)
					withnoise_sd_in.append(sd_in_loss_noise)
					withnoise_sd_out.append(sd_out_loss_noise)

				# average all the ori_diff and withnoise_diff and calculate their differece to get a single value diff_tmp for each canary
				# this is to evaluate the effect that the noise impose on the loss separation on random x
				loss_tmp = (sum(ori_loss) / len(ori_loss)) - (sum(withnoise_loss) / len(withnoise_loss))
				# for each "count", there are a list of diff with length == num_gen
				data[count]['loss'].append(loss_tmp)

				'''
				# compare and print out the average sd variation 
				global in_sd_diff, out_sd_diff, counter
				for a, b, c ,d in zip(ori_sd_in, ori_sd_out, withnoise_sd_in, withnoise_sd_out):
					in_sd_diff += a - c
					out_sd_diff += b - d
				counter += 1
				if (counter == 5000):
					print(f"change in IN sd is {in_sd_diff / 10}, change in OUT sd is {out_sd_diff / 10}")
					with open("/home/915688516/code/canary_main/canary/output.txt", "w") as file:
						file.write(str(in_sd_diff / (5000 * num_compare)) + "\n")
						file.write(str(out_sd_diff / (5000 * num_compare)) + "\n")
						file.write(str("5000 noise generated") + "\n")
				'''
		count += 1

	return data


def main(args):

	# usewandb = not args.nowandb
	# if usewandb:
	#    wandb.init(project='canary_generation',name=args.save_name)
	#    wandb.config.update(args)
	
	#print(args)
	# set random seed
	set_random_seed(args.seed)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	args.device = device
	
	# tv_dataset = get_dataset(args) ### maybe we need to relace this
	# dataset
	transform_train = transforms.Compose([
		transforms.ToPILImage(),
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])
	args.data_mean = (0.4914, 0.4822, 0.4465)
	args.data_std = (0.2023, 0.1994, 0.2010)
	
	targetset = dataset(dataset_name=args.dataset)
	args.original_targetset = targetset ## this is not a pytorch dataset object, but one object to directly get train / test data
	tv_dataset = part_pytorch_dataset(targetset.train_data, targetset.train_label, transform=transform_train,train=True)
	args.canary_trainset = tv_dataset
	trainset = tv_dataset
	# args.canary_trainset = tv_dataset(root='./data', train=True, download=True, transform=transform_train)
	# trainset = tv_dataset(root='./data', train=True, download=True, transform=transform_train)
	
	transform_aug = transforms.Compose([
		transforms.ToPILImage(),
		transforms.RandomCrop(32, padding=4),
		transforms.Resize(32),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])
	
	args.aug_trainset = part_pytorch_dataset(targetset.train_data, targetset.train_label, transform=transform_aug,train=True)
	args.aug_testset = part_pytorch_dataset(targetset.test_data, targetset.test_label, transform=transform_aug,train=False)
	# args.aug_trainset = tv_dataset(root='./data', train=True, download=True, transform=transform_aug)
	# args.aug_testset = tv_dataset(root='./data', train=False, download=True, transform=transform_aug)

	# load shadow and target models
	shadow_models = []
	for i in range(args.num_shadow):
		#checkpoint_name = args.checkpoint_prefix + f'_{i}.pth'
		checkpoint_name = args.checkpoint_prefix + f'{i}_last.pth'
		#print (checkpoint_name)
		curr_model = InferenceModel(i, args,checkpoint_name=checkpoint_name).to(args.device)
		### create shadow model train loader
		curr_model_index = curr_model.in_data
		this_shadow_train_set =  part_pytorch_dataset(targetset.train_data[curr_model_index], targetset.train_label[curr_model_index], transform=transform_train,train=True)
		this_shadow_train_loader = DataLoader(this_shadow_train_set, batch_size=100, shuffle=False,pin_memory=False, drop_last=False,num_workers=0)
		curr_model.train_loader = this_shadow_train_loader
		shadow_models.append(curr_model)

	checkpoint_name = args.checkpoint_prefix + f'{args.num_shadow}_last.pth'
	target_model = InferenceModel(-1, args,checkpoint_name=checkpoint_name).to(args.device)
	curr_model_index = target_model.in_data
	this_shadow_train_set =  part_pytorch_dataset(targetset.train_data[curr_model_index], targetset.train_label[curr_model_index], transform=transform_train,train=True)
	this_shadow_train_loader = DataLoader(this_shadow_train_set, batch_size=100, shuffle=False,pin_memory=False, drop_last=False,num_workers=0)
	target_model.train_loader = this_shadow_train_loader

	args.target_model = target_model
	args.shadow_models = shadow_models
	
	args.img_shape = trainset[0][0].shape
	args.canary_shape = args.canary_trainset[0][0].shape
	
	args.pred_logits = []  # N x (num of shadow + 1) x num_trials x num_class (target at -1)
	args.in_out_labels = []  # N x (num of shadow + 1)
	args.canary_losses = []  # N x num_trials
	args.class_labels = []  # N
	args.img_id = []  # N
	all_inf_score = []
	all_canaries = []
	
	for i in range(args.start, args.end):
		args.target_img_id = i
		
		args.target_img, args.target_img_class = trainset[args.target_img_id] # args.target_img_class is a int value
		args.target_img = args.target_img.unsqueeze(0).to(args.device) # change from shape [3,32,32] to [1,3,32,32]

		
		args.in_out_labels.append([])
		args.canary_losses.append([])
		args.pred_logits.append([])
		
		if args.num_val:
			in_models, out_models = split_shadow_models(shadow_models, args.target_img_id)
			num_in = min(int(args.num_val / 2), len(in_models))
			num_out = args.num_val - num_in

			
			train_shadow_models = random.sample(in_models, num_in)
			train_shadow_models += random.sample(out_models, num_out)
			
			
			val_shadow_models = train_shadow_models
		else:
			train_shadow_models = shadow_models
			val_shadow_models = shadow_models
		
		if args.aug_strategy and 'baseline' in args.aug_strategy:
			curr_canaries = generate_aug_imgs(args)
		else:
			curr_canaries,all_inf_score_temp = generate_canary(train_shadow_models, args)
		

		# append all the influence score for each iteration
		all_inf_score.append(all_inf_score_temp)
		# append all the canaries for each iteration
		curr_canaries = torch.cat(curr_canaries, dim=0)
		all_canaries.append(curr_canaries)

		# get logits
		curr_canaries = curr_canaries.to(args.device) # turn the curr_canaries from a one element list to a tensor, shape of the tensor does not change
		for curr_model in val_shadow_models:
			args.pred_logits[-1].append(get_logits(curr_canaries, curr_model))
			args.in_out_labels[-1].append(int(args.target_img_id in curr_model.in_data))
		
		args.pred_logits[-1].append(get_logits(curr_canaries, target_model))
		args.in_out_labels[-1].append(int(args.target_img_id in target_model.in_data))
		
		args.img_id.append(args.target_img_id)
		args.class_labels.append(args.target_img_class)
		
		progress_bar(i, args.end - args.start)
	
	# accumulate results
	pred_logits = np.array(args.pred_logits) # shape of the pred_logits array is (N, # of shadow models + 1, num_gen, # of classes)
	in_out_labels = np.array(args.in_out_labels)
	canary_losses = np.array(args.canary_losses)
	class_labels = np.array(args.class_labels)
	img_id = np.array(args.img_id)
	
	# save predictions
	#os.makedirs(f'saved_predictions/{args.name}/', exist_ok=True)
	#np.savez(f'saved_predictions/{args.name}/{args.save_name}.npz', pred_logits=pred_logits, in_out_labels=in_out_labels, canary_losses=canary_losses, class_labels=class_labels,
	#		 img_id=img_id)
	#np.savez(f'saved_predictions/{args.name}/inf_score.npy',np.array(all_inf_score))

	'''
	os.makedirs(f'/home/915688516/code/canary_main/canary/result/{args.name}/', exist_ok=True)
	np.savez(f'/home/915688516/code/canary_main/canary/result/{args.name}/{args.save_name}.npz', pred_logits=pred_logits, in_out_labels=in_out_labels, canary_losses=canary_losses, class_labels=class_labels,
			 img_id=img_id, all_canaries = all_canaries)
	np.savez(f'/home/915688516/code/canary_main/canary/result/{args.name}/inf_score.npy',np.array(all_inf_score))
	
	### dummy calculatiton of auc and acc
	### to be simplified
	pred = np.load(f'/home/915688516/code/canary_main/canary/result/{args.name}/{args.save_name}.npz')
	
	pred_logits = pred['pred_logits']
	in_out_labels = pred['in_out_labels']
	canary_losses = pred['canary_losses']
	class_labels = pred['class_labels']
	img_id = pred['img_id']
	'''
	
	in_out_labels = np.swapaxes(in_out_labels, 0, 1).astype(bool)
	pred_logits = np.swapaxes(pred_logits, 0, 1)
	
	# calibrate_logits() calculates the scores based on the model prediction on each classes 
	# shape of scores array is: (# of shadow models + 1, N, num_gen)
	scores = calibrate_logits(pred_logits, class_labels, args.logits_strategy)
	
	
	shadow_scores = scores[:-1]
	target_scores = scores[-1:]
	shadow_in_out_labels = in_out_labels[:-1]
	target_in_out_labels = in_out_labels[-1:]


	# testing the how would the noise attached on canary affect other random X's loss separation
	some_stats = cal_results_jilin(shadow_scores, shadow_in_out_labels, target_scores, target_in_out_labels, logits_mul=args.logits_mul)

	threshold_low = some_stats['fix_threshold@0.001FPR'] 
	# get the vulnerable datapoints, meaning they are relatively easy to attack
	vulnerable_datapoint_id = (np.argwhere(some_stats['fix_final_preds'] >= threshold_low)).flatten()
	selected_canaries = []
	for i in vulnerable_datapoint_id:
		selected_canaries.append(all_canaries[i])

	num_compare = 10
	result = noise_test(canaries = selected_canaries, 
	    x_id = vulnerable_datapoint_id, 
		trainset = trainset, 
		num_compare = num_compare , 
		shadow_models = shadow_models, 
		args = args)
	result = [{"id": r['id'], "loss": r['loss'] } for r in result]
	np.savez(f'/home/915688516/code/canary_main/canary/loss_result.npy',np.array(result))
		
	
	print(f"# of dict {len(result)}")
	print(f"example of id output {result[0]['id']}")
	print(f"length of loss {len(result[0]['loss'])}")
	print(f"example of loss {result[0]['loss']}")


	'''
	some_stats = cal_results(shadow_scores, shadow_in_out_labels, target_scores, target_in_out_labels, logits_mul=args.logits_mul)
	print(some_stats)
	'''

	# if usewandb:
	#	wandb.log(some_stats)
	
	#if not args.save_preds:
	#	os.remove(f'saved_predictions/{args.name}/{args.save_name}.npz')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Gen Canary')
	parser.add_argument('--bs', default=512, type=int)
	parser.add_argument('--size', default=32, type=int)
	parser.add_argument('--canary_size', default=32, type=int)
	parser.add_argument('--name', default='test')
	parser.add_argument('--save_name', default='test')
	parser.add_argument('--num_shadow', default=None, type=int, required=True)
	parser.add_argument('--seed', default=0, type=int)
	parser.add_argument('--net', default='res18')
	parser.add_argument('--num_classes', default=100,type=int)
	parser.add_argument('--patch', default=4, type=int, help="patch for ViT")
	parser.add_argument('--dimhead', default=512, type=int)
	parser.add_argument('--convkernel', default=8, type=int, help="parameter for convmixer")
	parser.add_argument('--init', default='rand')
	parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
	parser.add_argument('--opt', default='Adam')
	parser.add_argument('--iter', default=100, type=int)
	parser.add_argument('--scheduling', action='store_true')
	parser.add_argument('--start', default=0, type=int)
	parser.add_argument('--end', default=50000, type=int)
	parser.add_argument('--in_model_loss', default='ce', type=str)
	parser.add_argument('--out_model_loss', default='ce', type=str)
	parser.add_argument('--stop_loss', default=None, type=float)
	parser.add_argument('--print_step', action='store_true')
	parser.add_argument('--out_target_class', default=None, type=int)
	parser.add_argument('--aug_strategy', default=None, nargs='+')
	parser.add_argument('--early_stop', action='store_true')
	parser.add_argument('--patience', default=3, type=int)
	parser.add_argument('--nowandb', action='store_true', help='disable wandb')
	parser.add_argument('--num_aug', default=1, type=int)
	parser.add_argument('--logits_mul', default=1, type=int)
	parser.add_argument('--logits_strategy', default='log_logits')
	parser.add_argument('--in_model_loss_weight', default=1, type=float)
	parser.add_argument('--out_model_loss_weight', default=1, type=float)
	parser.add_argument('--momentum', default=0.9, type=float)
	parser.add_argument('--weight_decay', default=0, type=float)
	parser.add_argument('--reg_lambda', default=0.001, type=float)
	parser.add_argument('--regularization', default=None)
	parser.add_argument('--stochastic_k', default=None, type=int)
	parser.add_argument('--in_stop_loss', default=None, type=float)
	parser.add_argument('--out_stop_loss', default=None, type=float)
	parser.add_argument('--nesterov', action='store_true')
	parser.add_argument('--inner_iter', default=1, type=int)
	parser.add_argument('--canary_aug', action='store_true')
	parser.add_argument('--num_val', default=None, type=int)
	parser.add_argument('--num_gen', default=1, type=int)  # number of canaries generated during opt
	parser.add_argument('--epsilon', default=1, type=float)
	parser.add_argument('--no_dataset_aug', action='store_true')
	parser.add_argument('--balance_shadow', action='store_true')
	parser.add_argument('--dataset', default='cifar100')
	parser.add_argument('--target_logits', default=None, nargs='+', type=float)
	parser.add_argument('--save_preds', action='store_true')
	parser.add_argument('--offline', action='store_true')
	parser.add_argument('--select_top_num',type=int,default=0)
	parser.add_argument('--checkpoint_prefix',default='')

	args = parser.parse_args()
	print (args)
	#seed_list = [1,2,3,4,5]
	seed_list = [1]
	for this_seed in seed_list:
		args.seed = this_seed
		main(args)
