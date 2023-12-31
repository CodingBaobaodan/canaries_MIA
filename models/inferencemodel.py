import torch

import os

import torch
import torch.nn as nn

from models import *
from models.vit import ViT
from models.convmixer import ConvMixer
from models.utils import load_model


class InferenceModel(nn.Module):
	def __init__(self, shadow_id, args, checkpoint_name):
		super().__init__()
		self.shadow_id = shadow_id
		self.args = args
		
		if self.shadow_id == -1:
			# -1 for target model
			resume_checkpoint = checkpoint_name
			#resume_checkpoint = f'saved_models/{self.args.name}/70_last.pth'
		else:
			resume_checkpoint = checkpoint_name
			#resume_checkpoint = f'saved_models/{self.args.name}/{self.shadow_id}_last.pth'
		#print (resume_checkpoint)
		assert os.path.isfile(resume_checkpoint), 'Error: no checkpoint found!'
		checkpoint = torch.load(resume_checkpoint)
		if 'model_arch' in checkpoint:
			args.net = checkpoint['model_arch']
		self.model = load_model(args)
		
		### model keys
		#keys = checkpoint['model'].keys()
		#for (name,_ ),key in zip(self.model.named_parameters(),keys):
		#	print (name,key)

		self.model.load_state_dict(checkpoint['model'])
		
		self.in_data = checkpoint['in_data']
		self.keep_bool = checkpoint['keep_bool']
		
		# no grad by default
		self.deactivate_grad()
		self.model.eval()
		
		self.is_in_model = False # False for out_model
	
	def forward(self, x):
		return self.model(x)
	
	def deactivate_grad(self):
		for param in self.model.parameters():
			param.requires_grad = False
	
	def activate_grad(self):
		for param in self.model.parameters():
			param.requires_grad = True
	
	def load_state_dict(self, checkpoint):
		self.model.load_state_dict(checkpoint)
