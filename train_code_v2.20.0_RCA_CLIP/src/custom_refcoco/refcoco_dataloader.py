# Load Python Libraries
from PIL import Image, ImageDraw
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip
from torchvision.transforms.functional import InterpolationMode, hflip

import open_clip
import numpy as np
import random
from randaugment import RandomAugment, RandomAugment_V2
from region_prompt_generator import get_region_context_combo, get_region_cpt_combo, get_region_circle_combo, get_negative_box
from region_prompt_generator import get_cpt_combo, get_circle_combo, get_region_combo, get_context_combo, get_sig_color_focus_grey_context
from region_prompt_generator import get_color_focus_grey_context
from region_prompt_generator import get_bbox_mask_image
from calIoU import computeIoU

class image2rgb(object):
	# Convert image to 
	def __call__(self, image):
		return image.convert("RGB")


def check_xyxy_in_area(one_xyxy, gt_xyxy_s, threshold):
    for sig_gt_xyxy in gt_xyxy_s:
        iou = cal_fall_iou(one_xyxy, sig_gt_xyxy)
        if iou > threshold:
            return True
    return False


def cal_fall_iou(boxA, boxB):
    # Check the percentage of overlap between two boxes
    # boxA: [x1, y1, x2, y2]
    # boxB: [x1, y1, x2, y2]
    # return: iou
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0]) # left
    yA = max(boxA[1], boxB[1]) # top
    xB = min(boxA[2], boxB[2]) # right
    yB = min(boxA[3], boxB[3]) # bottom

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    A_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    B_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(A_area + 0.0000000001)
    return iou   

# To Implement
class transform_train(object):
	def __init__(self, n_px, data_mean, data_std) -> None:
		self.resize = Resize((n_px, n_px), interpolation=InterpolationMode.BICUBIC) # Enforce Region or Image to Square
		#self.horizontal_flip = RandomHorizontalFlip()
		self.image2rgb = image2rgb()
		self.rand_aug  = RandomAugment_V2(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate'])
		self.to_tensor = ToTensor()
		self.norm = Normalize(data_mean, data_std)


	def __call__(self, image, mask=None):
		random_value = random.random()
		image = self.resize(image)
		if random_value > 0.5:
			image = hflip(image)
		image = self.image2rgb(image)


		if mask is not None:
			mask  = self.resize(mask)
			if random_value > 0.5:
				mask  = hflip(mask)
			mask = self.image2rgb(mask)	

			image, mask = self.rand_aug(image, mask)
			image = self.to_tensor(image)
			image = self.norm(image)
			mask  = self.to_tensor(mask) # div by 255

			return image, mask
		else:
			image = self.rand_aug(image)
			image = self.to_tensor(image)
			image = self.norm(image)
			return image

		#mask = self.to_tensor(mask)
		#mask = self.norm(mask)

		# Image and Mask are auggmented in the same way



class transform_test(object):
	def __init__(self, n_px, data_mean, data_std) -> None:
		self.resize = Resize((n_px, n_px), interpolation=InterpolationMode.BICUBIC) # Enforce Region or Image to Square
		self.image2rgb = image2rgb()
		self.to_tensor = ToTensor()
		self.norm = Normalize(data_mean, data_std)


	def __call__(self, image, mask=None):
		image = self.resize(image)
		image = self.image2rgb(image)
		image = self.to_tensor(image)
		image = self.norm(image)

		if mask is not None:
			mask  = self.resize(mask)
			mask = self.image2rgb(mask)	
			mask = self.to_tensor(mask)
			return image, mask
		else:
			return image



# Sherlock Dataloader 
class CLIPDataset(torch.utils.data.Dataset):
	def __init__(self, data, args, training=False, tokenizer=None, data_mean=(0.48145466, 0.4578275, 0.40821073), 
				data_std=(0.26862954, 0.26130258, 0.27577711), negative_box=False, gpt2_tokenizer=None, gpt2_max_len=15, prefix_len=1):
		self.negative_box = negative_box
		self.args = args
		self.data = data
		self.tokenizer = tokenizer
		self.gpt2_tokenizer = gpt2_tokenizer
		self.gpt2_max_len = gpt2_max_len
		self.prefix_len = prefix_len
		self.data_mean = data_mean
		self.data_std = data_std
		self.id2data = {d['instance_id']: d for d in self.data}
		self.training = training
		n_px = min(args.input_resolution) # 224 when input_resolution = (448, 224) or (224, 448), process region/context separately
		self.image_size = (n_px, n_px)
		self.preprocess = transform_train(n_px, data_mean, data_std) if self.training else transform_test(n_px, data_mean, data_std)

		#self.preprocess = self._transform_train(n_px) if self.training else self._transform_test(n_px)
		#self.preprocess_rect = self._transform_train_rect(self.image_size) if self.training else self._transform_test_rect(self.image_size)

		# RegionCombo (ID=101): Region + Context (R-CTX)
		if   self.args.region_prompt == 101:
			print("******************PRMPTING TYPE: R-CTX******************")
        # RegionCombo (ID=102): Region + CPT (R-CPT)
		elif self.args.region_prompt == 102:
			print("******************PRMPTING TYPE: R-CPT******************")
		# RegionCombo (ID=103): Region + Circle (R-CIR)
		elif self.args.region_prompt == 103:
			print("******************PRMPTING TYPE: R-CIR******************")
		elif self.args.region_prompt == 104:
			print("******************PRMPTING TYPE: R-Box******************")



		# Single (ID=0): Context Only (CTX)
		elif  self.args.region_prompt == 0:
			print("******************PRMPTING TYPE: CTX******************")
		# Single (ID=1): Region Only (REG)
		elif  self.args.region_prompt == 1:
			print("******************PRMPTING TYPE: REG******************")
		# Single (ID=2): CPT Only (CPT)
		elif  self.args.region_prompt == 2:
			print("******************PRMPTING TYPE: CPT******************")
		# Single (ID=3): Cricle Only (CIR)
		elif  self.args.region_prompt == 3:
			print("******************PRMPTING TYPE: CIR******************")
		# Single (ID=4): Box Only (Box Hilight)
		elif  self.args.region_prompt == 4:
			print("******************PRMPTING TYPE: Box******************")
        
		elif  self.args.region_prompt == 1001: # Mixed Mode, randomly select R-CTX, R-CPT, R-CIR during training
			print("******************PRMPTING TYPE: R-CTX, R-CPT, R-CIR******************")

	def generate_paragraph(self, blip2_caption, blip2_box_caption, grit_caption, bboxes, caption_type, num_sentences=3):
		if caption_type == 0:
			return ""
		elif caption_type == 1:
			return ""
		elif caption_type == 2:
			return ""
		elif caption_type == 3:
			return ""
		elif caption_type == 10:
			paragraph = []
			# Add BLIP2 Image Caption
			paragraph.append(blip2_caption)

			# Add BLIP2 Box Caption
			for one_box in blip2_box_caption:
				one_box_caption = one_box['caption']
				paragraph.append(one_box_caption)

			# Add GRIT Caption
			gt_xyxy_s = []
			for sig in bboxes:
				sig_gt_xyxy = [ sig['left'], sig['top'], 
                           sig['left'] + sig['width'], 
                           sig['top'] + sig['height'] ]
				gt_xyxy_s.append(sig_gt_xyxy)

			for another_box in grit_caption:
				another_box_caption = another_box['caption']
				another_box_xyxy = another_box['xyxy_bbox']
				if check_xyxy_in_area(another_box_xyxy, gt_xyxy_s, 0.1):
					paragraph.append(another_box_caption)

			if len(paragraph) > num_sentences:
				paragraph = paragraph[:num_sentences]
			else:
				# It is possible that the number of sentences is less than num_sentences
				# In this case, we repeat the sentences
				# May be repeat several times
				# For example, if num_sentences = 3, and the number of sentences is 2
				# Then we repeat the sentences 2 times, plus the remainings
				paragraph = paragraph * (num_sentences // len(paragraph)) # Repeat the sentences
				paragraph = paragraph + paragraph[:num_sentences - len(paragraph)] # Repeat the sentences

		return paragraph


	def pad_tokens(self, gpt2_tokens):
		padding = self.gpt2_max_len - gpt2_tokens.shape[0]
		if padding > 0:
			gpt2_tokens = torch.cat((gpt2_tokens, torch.zeros(padding, dtype=torch.int64) - 1))
		else:
			gpt2_tokens = gpt2_tokens[:self.gpt2_max_len]

		mask = gpt2_tokens.ge(0) # mask is zero for padding tokens
		gpt2_tokens[~mask] = 0  # replace padding tokens with 0 in indices
		mask = mask.float()
		mask = torch.cat((torch.ones(self.prefix_len), mask), dim=0) # Add prefix_len to mask
		return gpt2_tokens, mask



	def gen_yolo_box_image_with_image(self, o_image, yolo_box, region_prompt):
		if len(yolo_box) > 0:
			yolo_box_tensor = torch.tensor(yolo_box)
			#yolo_box_coord = [ {'left': int(x[0]), 'top': int(x[1]), 'width': int(x[2]), 'height': int(x[3])} for x in yolo_box ]
			full_image_list = []
			box_image_list  = []
			for x in yolo_box:
				sig_box = [{'left': int(x[0]), 'top': int(x[1]), 'width': int(x[2]), 'height': int(x[3])}]
				if region_prompt == 101:
					yolo_full_image, yolo_bboxes_images = get_region_context_combo(o_image, sig_box)
				elif region_prompt == 102:
					yolo_full_image, yolo_bboxes_images = get_region_cpt_combo(o_image, sig_box)
				elif region_prompt == 103:
					yolo_full_image, yolo_bboxes_images = get_region_circle_combo(o_image, sig_box)
				full_image_list.append(yolo_full_image)
				box_image_list.append(yolo_bboxes_images[0])

			yolo_bboxes_images = [self.image_to_sig_tensor_v2(x) for x in box_image_list]
			yolo_bboxes_images = torch.stack(yolo_bboxes_images, dim=0)			

			yolo_full_images = [self.image_to_sig_tensor_v2(x) for x in full_image_list]
			yolo_full_images = torch.stack(yolo_full_images, dim=0)			
			#yolo_full_image = self.image_to_sig_tensor_v2(yolo_full_image)
			return yolo_bboxes_images, yolo_box_tensor, yolo_full_images		

		else:
			return 0, 0

	def gen_yolo_box_image(self, o_image, yolo_box, region_prompt):
		if len(yolo_box) > 0:
			yolo_box_tensor = torch.tensor(yolo_box)
			yolo_box_coord = [ {'left': int(x[0]), 'top': int(x[1]), 'width': int(x[2]), 'height': int(x[3])} for x in yolo_box ]
			if region_prompt == 101:
				_, yolo_bboxes_images = get_region_context_combo(o_image, yolo_box_coord)
			elif region_prompt == 102:
				_, yolo_bboxes_images = get_region_cpt_combo(o_image, yolo_box_coord)
			elif region_prompt == 103:
				_, yolo_bboxes_images = get_region_circle_combo(o_image, yolo_box_coord)

			#if region_prompt == 1001:
			#	rand_value = np.random.rand()
			#	if rand_value > 2/3.0:
			#		_, yolo_bboxes_images = get_region_context_combo(o_image, yolo_box_coord)
			#	elif rand_value > 1/3.0: 
			#		_, yolo_bboxes_images = get_region_cpt_combo(o_image, yolo_box_coord)
			#	else:
			#		_, yolo_bboxes_images = get_region_circle_combo(o_image, yolo_box_coord)

			yolo_bboxes_images = [self.image_to_sig_tensor_v2(x) for x in yolo_bboxes_images]
			yolo_bboxes_images = torch.stack(yolo_bboxes_images, dim=0)			
			#yolo_full_image = self.image_to_sig_tensor_v2(yolo_full_image)
			return yolo_bboxes_images, yolo_box_tensor		

		else:
			return 0, 0

	def gen_yolo_box_image_test(self, o_image, yolo_box, region_prompt):
		if len(yolo_box) > 0:
			yolo_box_tensor = torch.tensor(yolo_box)
			yolo_box_coord = [ {'left': int(x[0]), 'top': int(x[1]), 'width': int(x[2]), 'height': int(x[3])} for x in yolo_box ]
			_, yolo_bboxes_images = get_region_context_combo(o_image, yolo_box_coord)
			if region_prompt == 1001:
				_, yolo_bboxes_images = get_region_context_combo(o_image, yolo_box_coord)
				_, yolo_bboxes_images_2 = get_region_cpt_combo(o_image, yolo_box_coord)
				_, yolo_bboxes_images_3 = get_region_circle_combo(o_image, yolo_box_coord)

			yolo_bboxes_images = [self.image_to_sig_tensor_v2(x) for x in yolo_bboxes_images]
			yolo_bboxes_images = torch.stack(yolo_bboxes_images, dim=0)			

			yolo_bboxes_images_2 = [self.image_to_sig_tensor_v2(x) for x in yolo_bboxes_images_2]
			yolo_bboxes_images_2 = torch.stack(yolo_bboxes_images_2, dim=0)			

			yolo_bboxes_images_3 = [self.image_to_sig_tensor_v2(x) for x in yolo_bboxes_images_3]
			yolo_bboxes_images_3 = torch.stack(yolo_bboxes_images_3, dim=0)			
			return yolo_bboxes_images, yolo_bboxes_images_2, yolo_bboxes_images_3, yolo_box_tensor				

		else:
			return 0, 0


	def __getitem__(self, idx):
		c_data = self.data[idx]
		cid = c_data['instance_id'] # Instance ID
        # Load Image
		o_image = Image.open(self.url2filepath(c_data['inputs']['image']['url']))

		# YoloV8 Box
		yolo_box = c_data["yolo_box"]
		gt_bbox = c_data['inputs']['bboxes']
		gt_bbox = torch.tensor([gt_bbox[0]['left'], gt_bbox[0]['top'], gt_bbox[0]['width'], gt_bbox[0]['height'] ])

		if not self.training:
			# Testing, return all yolob boxes
			#proposal_yolo_bboxes_images, proposal_yolo_bboxes_images_2, proposal_yolo_bboxes_images_3, yolo_box_tensor = self.gen_yolo_box_image_test(
			#	o_image, yolo_box, self.args.region_prompt
			#)
			proposal_yolo_bboxes_images, yolo_box_tensor, full_images = self.gen_yolo_box_image_with_image(
				o_image, yolo_box, self.args.region_prompt
			)
		else:
			proposal_yolo_bboxes_images, yolo_box_tensor, full_images = self.gen_yolo_box_image_with_image(
				o_image, yolo_box, self.args.region_prompt
			)
			# Select One Negative from yolo box
			neg_list = []
			for ii, sig_yolo_box in enumerate(yolo_box_tensor):
				iou = computeIoU(sig_yolo_box, gt_bbox)
				if iou < 0.5:
					#print("find negatives")
					neg_list.append(ii)
			if len(neg_list) >=1:
				pick_one = np.random.choice(neg_list, size=1, replace=False)[0]
				#print("ZZZZZZZ {}".format(pick_one))
				proposal_yolo_bboxes_images = proposal_yolo_bboxes_images[pick_one]
				full_images = full_images[pick_one]
			else:
				#print("NO Negatives")
				neg_auto_box = []
				width, height = o_image.size
				while len(neg_auto_box) < 1:
					n_x = np.random.randint(0, width-50)
					n_y = np.random.randint(0, height-50)
					n_w = np.random.randint(0, width-n_x)
					n_h = np.random.randint(0, height-n_y)
					iou = computeIoU([n_x, n_y, n_w, n_h], gt_bbox)
					if iou < 0.5:
						neg_auto_box.append([n_x, n_y, n_w, n_h])
				proposal_yolo_bboxes_images, _,full_image = self.gen_yolo_box_image_with_image(
					o_image, neg_auto_box, self.args.region_prompt
				)
				proposal_yolo_bboxes_images = proposal_yolo_bboxes_images[0]
				full_images = full_images[0]
				

		# Tokenizing "Clue" and "Inference"
		clue_raw = c_data['inputs']['clue']
		inference_raw = c_data['targets']['inference']
		if clue_raw[-1] != ".":
			clue_raw = clue_raw + "."
		if inference_raw[-1] != ".":
			inference_raw = inference_raw + "."

		clue      = self.tokenizer("clue {}".format(clue_raw)).squeeze()
		inference = self.tokenizer("inference {}".format(inference_raw)).squeeze()



		# RegionCombo (ID=101): Region + Context (R-CTX)
		if   self.args.region_prompt == 101:
			image, bboxes_images = get_region_context_combo(o_image, c_data['inputs']['bboxes'])
        # RegionCombo (ID=102): Region + CPT (R-CPT)
		elif self.args.region_prompt == 102:
			image, bboxes_images = get_region_cpt_combo(o_image, c_data['inputs']['bboxes'])
		# RegionCombo (ID=103): Region + Circle (R-CIR)
		elif self.args.region_prompt == 103:
			image, bboxes_images = get_region_circle_combo(o_image, c_data['inputs']['bboxes'])
		elif self.args.region_prompt == 104:
			image, bboxes_images = get_color_focus_grey_context(o_image, c_data['inputs']['bboxes'])


		# Single (ID=0): Context Only (CTX)
		elif  self.args.region_prompt == 0:
			image, bboxes_images = get_context_combo(o_image, c_data['inputs']['bboxes'])
		# Single (ID=1): Region Only (REG)
		elif  self.args.region_prompt == 1:
			image, bboxes_images = get_region_combo(o_image, c_data['inputs']['bboxes'])
		# Single (ID=2): CPT Only (CPT)
		elif  self.args.region_prompt == 2:
			image, bboxes_images = get_cpt_combo(o_image, c_data['inputs']['bboxes'])
		# Single (ID=3): Cricle Only (CIR)
		elif  self.args.region_prompt == 3:
			image, bboxes_images = get_circle_combo(o_image, c_data['inputs']['bboxes'])
		# Single (ID=4): Box Only (Box Hilight)
		elif  self.args.region_prompt == 4:
			image, bboxes_images = get_sig_color_focus_grey_context(o_image, c_data['inputs']['bboxes'])
		
        
		elif  self.args.region_prompt == 1001: # Mixed Mode, randomly select R-CTX, R-CPT, R-CIR during training
			if self.training:
				rand_value = np.random.rand()
				if rand_value > 2/3.0:
					image, bboxes_images = get_region_context_combo(o_image, c_data['inputs']['bboxes'])
				elif rand_value > 1/3.0: 
					image, bboxes_images = get_region_cpt_combo(o_image, c_data['inputs']['bboxes'])
				else:
					image, bboxes_images = get_region_circle_combo(o_image, c_data['inputs']['bboxes'])
				#else:
				#	image, bboxes_images = get_color_focus_grey_context(o_image, c_data['inputs']['bboxes'])

			else:

				im1, bb_1 = get_region_context_combo(o_image, c_data['inputs']['bboxes'])
				im2, bb_2 = get_region_cpt_combo(o_image, c_data['inputs']['bboxes'])
				im3, bb_3 = get_region_circle_combo(o_image, c_data['inputs']['bboxes'])
				#im4, bb_4 = get_color_focus_grey_context(o_image, c_data['inputs']['bboxes'])
					
				im1, bb_1 = self.image_to_torch_tensor_v2(im1, bb_1)
				im2, bb_2 = self.image_to_torch_tensor_v2(im2, bb_2)
				im3, bb_3 = self.image_to_torch_tensor_v2(im3, bb_3)
				#im4, bb_4 = self.image_to_torch_tensor_v2(im4, bb_4)
				
				return {'image':im1, 'clue': clue, 'inference': inference,'id': cid, 'box':bb_1, 
	    				'image2':im2, 'box2':bb_2,'image3':im3, 'box3':bb_3,
						"proposal_yolo_bboxes_images": proposal_yolo_bboxes_images,
						"proposal_yolo_bboxes_images_2": proposal_yolo_bboxes_images_2,
						"proposal_yolo_bboxes_images_3": proposal_yolo_bboxes_images_3,
						'yolo_box':yolo_box_tensor,
						'gt_bbox':gt_bbox
						}
					#'image4':im4, 'box4':bb_4
					#							}

		if self.training:
			image, bbox_image = self.image_to_torch_tensor_v2(image, bboxes_images)
			return {'image':image, 'clue': clue, 'inference': inference,'id': cid, 'box':bbox_image, 
				'proposal_yolo_bboxes_images':proposal_yolo_bboxes_images,
				'full_images':full_images,
	   				}
		else:
			image, bbox_image = self.image_to_torch_tensor_v2(image, bboxes_images)
			return {'image':image, 'clue': clue, 'inference': inference,'id': cid, 'box':bbox_image, 
				'proposal_yolo_bboxes_images':proposal_yolo_bboxes_images,
				'yolo_box':yolo_box_tensor,
				'gt_bbox':gt_bbox,
				'full_images':full_images,
	  				}



	def get(self, cid):
		return self.id2data[cid]


	def __len__(self):
		return len(self.data)


	def url2filepath(self, url):
		# Convert url to filepath, customized for Sherlock dataset
		if 'VG_' in url:
			return self.args.vg_dir + '/'.join(url.split('/')[-2:])
		if 'refexp' in url:
			return url
		else:
			# http://s3-us-west-2.amazonaws.com/ai2-rowanz/vcr1images/lsmdc_3023_DISTRICT_9/3023_DISTRICT_9_01.21.02.808-01.21.16.722@5.jpg
			if 'vcr1images' in self.args.vcr_dir:
				return self.args.vcr_dir + '/'.join(url.split('/')[-2:])
			else:
				return self.args.vcr_dir + '/'.join(url.split('/')[-3:])
			
	def _transform_train(self, n_px):
		return Compose([
			Resize((n_px, n_px), interpolation=InterpolationMode.BICUBIC), # Enforce Region or Image to Square
			#RandomHorizontalFlip(),
			image2rgb(),
			#RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
			RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize']),
			ToTensor(),
			Normalize(self.data_mean, self.data_std),
		])


	def _transform_test(self, n_px):
		return Compose([
			Resize((n_px, n_px), interpolation=InterpolationMode.BICUBIC), # Enforce Region or Image to Square
			image2rgb(),
			ToTensor(),
			Normalize(self.data_mean, self.data_std),
		])


	def _transform_train_rect(self, image_size):
		return Compose([
			Resize(image_size, interpolation=InterpolationMode.BICUBIC), # Enforce Region or Image to Square
			RandomHorizontalFlip(),
			image2rgb(),
			RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
			ToTensor(),
			Normalize(self.data_mean, self.data_std),
		])


	def _transform_test_rect(self, image_size):
		return Compose([
			Resize(image_size, interpolation=InterpolationMode.BICUBIC), # Enforce Region or Image to Square
			image2rgb(),
			ToTensor(),
			Normalize(self.data_mean, self.data_std),
		])


	def image_to_torch_tensor(self, image, bboxes_image=None):
		# randomly select one box during training;
        # use specified one box during testing
		image = self.preprocess(image)
		if bboxes_image is not None:
			rand_i = random.randrange(len(bboxes_image))
			bbox_image = bboxes_image[rand_i]
			bbox_image=self.preprocess(bbox_image)
		else:
			bbox_image = None # no bbox_image, if so take cares of returns for get_item function
		return image, bbox_image


	def image_to_sig_tensor(self, image):
		# randomly select one box during training;
        # use specified one box during testing
		image = self.preprocess_rect(image)
		return image
	
	#def image_to_torch_tensor_v2(self, image, bboxes_image, mask_image):
	#	# randomly select one box during training;
	#       # use specified one box during testing
	#	image, mask_image = self.preprocess(image, mask_image)
	#	if bboxes_image is not None:
	#		rand_i = random.randrange(len(bboxes_image))
	#		bbox_image = bboxes_image[rand_i]
	#		bbox_image =self.preprocess(bbox_image)
	#	else:
	#		bbox_image = None # no bbox_image, if so take cares of returns for get_item function
	#	return image, bbox_image, mask_image
	
	def image_to_sig_tensor_v2(self, image):
		# randomly select one box during training;
        # use specified one box during testing
		image = self.preprocess(image)
		return image

	def image_to_torch_tensor_v2(self, image, bboxes_image):
		# randomly select one box during training;
	       # use specified one box during testing
		image = self.preprocess(image)
		if bboxes_image is not None:
			rand_i = random.randrange(len(bboxes_image))
			bbox_image = bboxes_image[rand_i]
			bbox_image =self.preprocess(bbox_image)
		else:
			bbox_image = None # no bbox_image, if so take cares of returns for get_item function
		return image, bbox_image
