import argparse
import numpy as np
import torch
import json
import pprint
from PIL import Image, ImageDraw
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, RandomGrayscale, ColorJitter
import tempfile
import tqdm
import os
import collections
#import clip
import sklearn.metrics
from scipy.stats import rankdata
from torchvision.utils import save_image
import torchvision.transforms.functional as F
from frozendict import frozendict
import random
from region_prompt_generator import get_region_context_combo, get_region_cpt_combo, get_region_circle_combo 
from region_prompt_generator import get_cpt_combo, get_circle_combo, get_region_combo, get_context_combo, get_color_focus_grey_context, get_sig_color_focus_grey_context


class image2rgb(object):
	def __call__(self, image):
		return image.convert("RGB")

class CLIPDatasetLeaderboard(torch.utils.data.Dataset):
	def __init__(self, data, args, tokenizer):
		self.args = args
		self.data = data
		self.tokenizer = tokenizer
		self.id2data = {d['instance_id']: d for d in self.data}
		n_px = min(args.input_resolution)
		self.preprocess = self._transform_test((n_px, n_px))

	def url2filepath(self, url):
		if 'VG_' in url:
			return self.args.vg_dir + '/'.join(url.split('/')[-2:])
		else:
			# http://s3-us-west-2.amazonaws.com/ai2-rowanz/vcr1images/lsmdc_3023_DISTRICT_9/3023_DISTRICT_9_01.21.02.808-01.21.16.722@5.jpg
			if 'vcr1images' in self.args.vcr_dir:
				return self.args.vcr_dir + '/'.join(url.split('/')[-2:])
			else:
				return self.args.vcr_dir + '/'.join(url.split('/')[-3:])


	def _transform_test(self, n_px):
		return Compose([
			Resize(n_px, interpolation=InterpolationMode.BICUBIC),
			#CenterCrop(n_px),
			image2rgb(),
			ToTensor(),
			Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
		])


	def image_to_torch_tensor(self, image, box_images):
		trans_images = self.preprocess(image)

		rand_index = random.randrange(len(box_images))
		sig_box_image = box_images[rand_index]
		trans_box_images =  self.preprocess(sig_box_image)
		return trans_images, trans_box_images


	def __getitem__(self, idx):
		c_data = self.data[idx]
		image = Image.open(self.url2filepath(c_data['image']['url']))
		reg = c_data['region']
		if self.args.region_prompt == 1001: # Mixed Mode
			image1, box_image1   = get_region_context_combo(image, reg)
			image2, box_image2   = get_region_cpt_combo(image, reg)
			image3, box_image3   = get_region_circle_combo(image, reg)
			#image4, box_image4   = get_color_focus_grey_context(image, reg)

		elif self.args.region_prompt == 101: # Region + Context (R-CTX)
			image, box_image = get_region_context_combo(image, reg)
		elif self.args.region_prompt == 102: # Region + CPT (R-CPT)
			image, box_image = get_region_cpt_combo(image, reg)
		elif self.args.region_prompt == 103: # Region + Circle (R-CIR)
			image, box_image = get_region_circle_combo(image, reg)
		elif self.args.region_prompt == 104: # Region + Box (R-Box)
			image, box_image = get_color_focus_grey_context(image, reg)

		elif self.args.region_prompt == 0: # CTX
			image, box_image = get_context_combo(image, reg)
		elif self.args.region_prompt == 1: # REG
			image, box_image = get_region_combo(image, reg)
		elif self.args.region_prompt == 2: # CPT
			image, box_image = get_cpt_combo(image, reg)
		elif self.args.region_prompt == 3: # CIR
			image, box_image = get_circle_combo(image, reg)
		elif self.args.region_prompt == 4: # Box
			image, box_image = get_sig_color_focus_grey_context(image, reg)

		inference_raw = c_data['inference']
		if inference_raw[-1] != ".": 
			inference_raw += "."

		#caption = self.tokenizer('inference {}'.format(inference_raw)).squeeze()
		caption = self.tokenizer('{}'.format(inference_raw))
		caption_input_ids = torch.tensor(caption['input_ids'])
		caption_attention_mask = torch.tensor(caption['attention_mask'])
		#print("Caption {}".format(caption))
		cid = c_data['instance_id']

		if self.args.region_prompt == 1001:
			image1, box_image1 = self.image_to_torch_tensor(image1, box_image1)
			image2, box_image2 = self.image_to_torch_tensor(image2, box_image2)
			image3, box_image3 = self.image_to_torch_tensor(image3, box_image3)
			#image4, box_image4 = self.image_to_torch_tensor(image4, box_image4)
			return {'image':image1, 'caption_input_ids':caption_input_ids, 'caption_attention_mask': caption_attention_mask, 'id': cid, 'box_image': box_image1, 'image2':image2, 'box2_image': box_image2, 'image3':image3, 'box3_image': box_image3,}
				#'image4':image4,
				#'box4_image': box_image4}
		else:
			image, box_image = self.image_to_torch_tensor(image, box_image)
			return {'image':image, 'caption_input_ids':caption_input_ids, 'caption_attention_mask': caption_attention_mask, 'id': cid, 'box_image': box_image}

	def __len__(self):
		return len(self.data)


class CLIPDatasetImageOnlyLeaderboard(CLIPDatasetLeaderboard):
	def __init__(self, images, args, tokenizer):
		self.args = args
		# images must contain [{'url': ..., 'bboxes': [ ... ]}]
		self.images = images
		self.tokenizer = tokenizer
		n_px = min(args.input_resolution)
		self.preprocess = self._transform_test((n_px, n_px))

	def __getitem__(self, idx):
		c_data = self.images[idx]
		image = Image.open(self.url2filepath(c_data['url']))
		reg = c_data['bboxes']

		if self.args.region_prompt == 1001: # Mixed Mode
			image1, box_image1   = get_region_context_combo(image, reg)
			image2, box_image2   = get_region_cpt_combo(image, reg)
			image3, box_image3   = get_region_circle_combo(image, reg)
			#image4, box_image4   = get_color_focus_grey_context(image, reg)

		elif self.args.region_prompt == 101: # Region + Context (R-CTX)
			image, box_image = get_region_context_combo(image, reg)
		elif self.args.region_prompt == 102: # Region + CPT (R-CPT)
			image, box_image = get_region_cpt_combo(image, reg)
		elif self.args.region_prompt == 103: # Region + Circle (R-CIR)
			image, box_image = get_region_circle_combo(image, reg)
		elif self.args.region_prompt == 104: # Region + Box (R-Box)
			image, box_image = get_color_focus_grey_context(image, reg)

		elif self.args.region_prompt == 0: # CTX
			image, box_image = get_context_combo(image, reg)
		elif self.args.region_prompt == 1: # REG
			image, box_image = get_region_combo(image, reg)
		elif self.args.region_prompt == 2: # CPT
			image, box_image = get_cpt_combo(image, reg)
		elif self.args.region_prompt == 3: # CIR
			image, box_image = get_circle_combo(image, reg)
		elif self.args.region_prompt == 4: # Box
			image, box_image = get_sig_color_focus_grey_context(image, reg)

		if self.args.region_prompt == 1001:
			image1, box_image1 = self.image_to_torch_tensor(image1, box_image1)
			image2, box_image2 = self.image_to_torch_tensor(image2, box_image2)
			image3, box_image3 = self.image_to_torch_tensor(image3, box_image3)
			#image4, box_image4 = self.image_to_torch_tensor(image4, box_image4)
			return {'image':image1, 'box_image': box_image1, 
				'image2':image2, 'box2_image': box_image2, 
				'image3':image3, 'box3_image': box_image3,}
				#'image4':image4, 'box4_image': box_image4}
		else:
			image, box_image = self.image_to_torch_tensor(image, box_image)
			return {'image': image, 'box_image': box_image}

	def __len__(self):
		return len(self.images)


class CLIPDatasetCaptionOnlyLeaderboard(CLIPDatasetLeaderboard):
	def __init__(self, captions, args, tokenizer):
		self.args = args
		# images must contain [{'inference': ...}]
		self.captions = captions
		self.tokenizer = tokenizer

	def __getitem__(self, idx):
		c_data = self.captions[idx]

		inference_raw = c_data['caption']
		if inference_raw[-1] != ".": 
			inference_raw += "."

		#caption = self.tokenizer('inference {}'.format(inference_raw)).squeeze()
		caption = self.tokenizer('{}'.format(inference_raw))
		caption_input_ids = torch.tensor(caption['input_ids'])
		caption_attention_mask = torch.tensor(caption['attention_mask'])
		return {'caption_input_ids': caption_input_ids, 'caption_attention_mask':caption_attention_mask}

	def __len__(self):
		return len(self.captions)

	
class CLIPDatasetLocalizationLeaderboard(CLIPDatasetLeaderboard):
	def __init__(self, url2instances, args, tokenizer):
		self.args = args
		self.tokenizer = tokenizer
		self.url2instances = url2instances
		self.ordered_urls = list(self.url2instances.keys())
		n_px = min(args.input_resolution)
		self.preprocess = self._transform_test((n_px, n_px))

	def __getitem__(self, idx):
		c_url = self.ordered_urls[idx]
		c_instances = self.url2instances[c_url]
		unique_inferences = list(set([inst['inference'] for inst in c_instances]))
		unique_regions = list(set([tuple([frozendict(x) for x in inst['region']]) for inst in c_instances]))

		inf2idx = {inf: idx for idx, inf in enumerate(unique_inferences)}
		reg2idx = {reg: idx for idx, reg in enumerate(unique_regions)}

		# assume we are going to do image x caption similarity
		# we want the lookup for each instance id
		# we can return a list of instance ids
		# a list of rows (image idxs)
		# and a list of cols (cap idxs)
		# then, we can zip lookup later.
		instance_ids = [inst['test_id'] for inst in c_instances]
		image_idxs = [reg2idx[tuple([frozendict(x) for x in inst['region']])] for inst in c_instances]
		cap_idxs = [inf2idx[inst['inference']] for inst in c_instances]
			
		#caption = self.tokenizer(['inference {}'.format(cap) for cap in unique_inferences])
		#caption = self.tokenizer('inference {}'.format(inference_raw))
		caption = unique_inferences
		#caption = self.tokenizer(caption)
		caption_input_ids =[  torch.tensor(self.tokenizer(x)['input_ids']) for x in unique_inferences]
		#print(caption_input_ids)
		#caption_input_ids = torch.tensor(caption_input_ids)

		caption_attention_mask =[  torch.tensor(self.tokenizer(x)['attention_mask']) for x in unique_inferences]
		#caption_attention_mask = torch.tensor(caption_attention_mask)
		#caption_input_ids = torch.tensor(caption['input_ids'])
		#caption_attention_mask = torch.tensor(caption['attention_mask'])
		#caption = self.tokenizer(unique_inferences)
		
		image = Image.open(self.url2filepath(c_url))
		image1_list = []
		image2_list = []
		image3_list = []
		#image4_list = []
		box1_image_list = []
		box2_image_list = []
		box3_image_list = []
		#box4_image_list = []

		for reg in unique_regions:
			if self.args.region_prompt == 1001: # Mixed Mode
				image1, box_image1   = get_region_context_combo(image, reg)
				image2, box_image2   = get_region_cpt_combo(image, reg)
				image3, box_image3   = get_region_circle_combo(image, reg)
				#image4, box_image4   = get_color_focus_grey_context(image, reg)
				image1, box_image1   = self.image_to_torch_tensor(image1, 
										box_image1)
				image2, box_image2   = self.image_to_torch_tensor(image2, 
										box_image2)
				image3, box_image3   = self.image_to_torch_tensor(image3, 
										box_image3)
				#image4, box_image4   = self.image_to_torch_tensor(image4, 
				#						box_image4)
				image1_list.append(image1)
				image2_list.append(image2)
				image3_list.append(image3)
				#image4_list.append(image4)
				box1_image_list.append(box_image1)
				box2_image_list.append(box_image2)
				box3_image_list.append(box_image3)
				#box4_image_list.append(box_image4)


			elif self.args.region_prompt == 101: # Region + Context (R-CTX)
				image1, box_image1   = get_region_context_combo(image, reg)
				image1, box_image1   = self.image_to_torch_tensor(image1, 
										box_image1)
				image1_list.append(image1)
				box1_image_list.append(box_image1)

			elif self.args.region_prompt == 102: # Region + CPT (R-CPT)
				image1, box_image1   = get_region_cpt_combo(image, reg)
				image1, box_image1   = self.image_to_torch_tensor(image1, 
										box_image1)
				image1_list.append(image1)
				box1_image_list.append(box_image1)

			elif self.args.region_prompt == 103: # Region + Circle (R-CIR)
				image1, box_image1   = get_region_circle_combo(image, reg)
				image1, box_image1   = self.image_to_torch_tensor(image1, 
										box_image1)
				image1_list.append(image1)
				box1_image_list.append(box_image1)

			elif self.args.region_prompt == 104: # Region + Box (R-Box)
				image1, box_image1   = get_color_focus_grey_context(image, reg)
				image1, box_image1   = self.image_to_torch_tensor(image1, 
										box_image1)
				image1_list.append(image1)
				box1_image_list.append(box_image1)

			elif self.args.region_prompt == 0: # CTX
				image1, box_image1   = get_context_combo(image, reg)
				image1, box_image1   = self.image_to_torch_tensor(image1, 
										box_image1)
				image1_list.append(image1)
				box1_image_list.append(box_image1)

			elif self.args.region_prompt == 1: # REG
				image1, box_image1   = get_region_combo(image, reg)
				image1, box_image1   = self.image_to_torch_tensor(image1, 
										box_image1)
				image1_list.append(image1)
				box1_image_list.append(box_image1)

			elif self.args.region_prompt == 2: # CPT
				image1, box_image1   = get_cpt_combo(image, reg)
				image1, box_image1   = self.image_to_torch_tensor(image1, 
										box_image1)
				image1_list.append(image1)
				box1_image_list.append(box_image1)

			elif self.args.region_prompt == 3: # CiR
				image1, box_image1   = get_circle_combo(image, reg)
				image1, box_image1   = self.image_to_torch_tensor(image1, 
										box_image1)
				image1_list.append(image1)
				box1_image_list.append(box_image1)

			elif self.args.region_prompt == 4: # Box
				image1, box_image1   = get_sig_color_focus_grey_context(image, reg)
				image1, box_image1   = self.image_to_torch_tensor(image1, 
										box_image1)
				image1_list.append(image1)
				box1_image_list.append(box_image1)

			#sig_image, sig_box = self.image_to_torch_tensor(image1, box_image1)
			#image1_list.append(sig_image)
			#box_image_list.append(sig_box)

		#image1	 = torch.stack(image1_list)
		#box1_image = torch.stack(box1_image_list)

		if self.args.region_prompt == 1001:
			image1	 = torch.stack(image1_list)
			image2	 = torch.stack(image2_list)
			image3	 = torch.stack(image3_list)
			#image4	 = torch.stack(image4_list)
			box1_image = torch.stack(box1_image_list)
			box2_image = torch.stack(box2_image_list)
			box3_image = torch.stack(box3_image_list)
			#box4_image = torch.stack(box4_image_list)
			return {'caption_input_ids': caption_input_ids, 'caption_attention_mask':caption_attention_mask,'image': image1, 'box_image':box1_image,'instance_ids': instance_ids, 'image_idxs': image_idxs, 'cap_idxs': cap_idxs,'image2': image2, 'box2_image':box2_image,'image3': image3, 'box3_image':box3_image}
		else:
			image1	 = torch.stack(image1_list)
			box1_image = torch.stack(box1_image_list)
			return {'caption_input_ids': caption_input_ids, 'caption_attention_mask':caption_attention_mask, 'image': image1, 'box_image':box1_image,'instance_ids': instance_ids, 'image_idxs': image_idxs, 'cap_idxs': cap_idxs}

	def __len__(self):
		return len(self.ordered_urls)


def worker_init_fn(worker_id):
	np.random.seed(np.random.get_state()[1][0] + worker_id)


def clip_forward(model, box_image, image, text, only_features=True):
	if model.visual.image_size[0] > model.visual.image_size[1]:
		merge_image = torch.cat((box_image, image), dim=2) # b,c,h,w
	else:
		merge_image = torch.cat((box_image, image), dim=3) # b,c,h,w
	image_features = model.encode_image(merge_image)
	image_features = image_features / image_features.norm(dim=-1, keepdim=True)
	text_features = model.encode_text(text)
	text_features = text_features / text_features.norm(dim=-1, keepdim=True)

	if only_features:
		return image_features, text_features

	# cosine similarity as logits
	logit_scale = model.logit_scale.exp()
	logits_per_image = logit_scale * image_features @ text_features.t()
	logits_per_text = logits_per_image.t()

	# shape = [global_batch_size, global_batch_size]
	return logits_per_image, logits_per_text, image_features, text_features


def clip_forward_image(model, box_image, image):
	#if model.visual.image_size[0] > model.visual.image_size[1]:
	#	merge_image = torch.cat((box_image, image), dim=2) # b,c,h,w
	#else:
	#	merge_image = torch.cat((box_image, image), dim=3) # b,c,h,w
	image_features = model.get_image_features(image)
	image_features = image_features / image_features.norm(dim=-1, keepdim=True)
		
	box_image_features = model.get_image_features(box_image)
	box_image_features = box_image_features / box_image_features.norm(dim=-1, keepdim=True)
	image_features = (box_image_features + image_features) /2.0

	return image_features


def clip_forward_text(model, text_input_ids, text_mask):
	text_features = model.get_text_features(text_input_ids, text_mask)
	# normalized features
	text_features = text_features / text_features.norm(dim=-1, keepdim=True)
	return text_features


class CLIPExtractor(torch.nn.Module):
	def __init__(self, clip_model, args):
		super(CLIPExtractor, self).__init__()
		self.clip_model = clip_model
		self.args = args

	def forward(self, box_image, image, text):
		return clip_forward(self.clip_model, box_image, image, text)

	def image_forward(self, box_image, image):
		return clip_forward_image(self.clip_model, box_image, image)

	def text_forward(self, text_input_ids, text_mask):
		return clip_forward_text(self.clip_model, text_input_ids, text_mask)
