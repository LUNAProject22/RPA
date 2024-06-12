'''
Prediction script for leaderboard
'''
import argparse
import numpy as np
import torch
import json
import pprint
from PIL import Image, ImageDraw
import tempfile
import tqdm
import os
import collections
import open_clip
import sklearn.metrics
from scipy.stats import rankdata
import sys
import clip_inference_iterator

import pprint
from frozendict import frozendict
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def worker_init_fn(worker_id):
	np.random.seed(np.random.get_state()[1][0] + worker_id)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--adapter_rate', type=float, default=0.25)
	parser.add_argument('--Adapt_TxtEncoder', type=lambda x: (str(x).lower() == 'true'), default=False)
	parser.add_argument('--Adapt_VisEncoder', type=lambda x: (str(x).lower() == 'true'), default=False)
	parser.add_argument('--region_prompt', type=int, default=0, 
                        help='1001=Mixed Mode: R-CTX, R-CPT, R-CIR'
                        '101=R-CTX'
                        '102=R-CPT'
                        '103=R-CIR'
                        '104=R-Box'
                        '0=Context  (CTX)'
                        '1=Region   (REG)'
                        '2=Colorful (CPT)'
                        '3=Circle   (CIR)'
                        '4=Box      (Box)'
                        )
	parser.add_argument('--AdaType', type=int, default=0, 
                        help='0=MLP-A, ATTEN-A, MAP-A,'
                        '1=MLP-A'
                        '2=ATTEN-A'
                        '3=MAP-A'
                        '4=MLP-A, MAP-A'
                        '5=MLP-A, ATTEN-A'
                        '6=ATTEN-A, MAP-A'
                        )
	parser.add_argument('--instances')
	parser.add_argument('--load_model_from')
	parser.add_argument('--output_npy')
	parser.add_argument('--task',
		default='retrieval',
		choices=['retrieval',
		'localization',
		'comparison'])

	parser.add_argument('--clip_model',
			default='ViT-B-16',
			choices=['ViT-B-16-672x336', 'ViT-B-16-224x112', 'ViT-B-16-224x448', 'ViT-B-16-448x224', 'ViT-H-14-448x224', 'ViT-H-14-672x336','ViT-L-14-336x672','ViT-L-14-672x336', 'ViT-L-14-336x504', 'ViT-L-14-336', 'ViT-B-32', 'ViT-B-16', 'ViT-B-16-cust-384', 'RN50x16', 'RN50x64', 'xlm-roberta-large-ViT-H-14'])
	parser.add_argument(
			 "--pretrained",
			default='',
			type=str,
			help="Use a pretrained CLIP model weights with the specified tag or file path.",)
	parser.add_argument('--batch_size',
			default=256,
			type=int,
			help='batch size. due to numerical precision annoyance, keep at 256 for exact replication.')

	parser.add_argument(
		'--vcr_dir',
		default='/net/nfs2.mosaic/jackh/vcr_images/vcr1images/',
		help='directory with all of the VCR image data, contains, e.g., movieclips_Lethal_Weapon')

	parser.add_argument(
		'--vg_dir',
		default='/net/nfs2.mosaic/jackh/extract_butd_image_features_sherlock/',
		help='directory with visual genome data, contains VG_100K and VG_100K_2')

	parser.add_argument('--workers_dataloader',
			type=int, default=4)

	args = parser.parse_args()

	if args.vcr_dir[-1] != '/':
		args.vcr_dir += '/'
	if args.vg_dir[-1] != '/':
		args.vg_dir += '/'

	if os.path.exists(args.output_npy):
		print('{} already exists! continuing'.format(args.output_npy))
		quit()

	return args


def comparison_main(model, args, tokenizer):
	with open(args.instances) as f:
		instances = json.load(f)
	print('{} comparison instances'.format(len(instances)))
	for inst in instances:
		inst['instance_id'] = inst['test_id']

	dset = torch.utils.data.DataLoader(
		clip_inference_iterator.CLIPDatasetLeaderboard(instances, args, tokenizer),
		batch_size=args.batch_size, num_workers=args.workers_dataloader, shuffle=False, worker_init_fn=worker_init_fn)

	all_im_embs, all_txt_embs = [], []
	with torch.no_grad():
		bar = tqdm.tqdm(dset, total=len(dset))
		for batch in bar:
			images, captions = batch['image'].to(args.device), batch['caption'].to(args.device)
			box_images = batch['box_image'].to(args.device)
			image_features, text_features = model(box_images, images, captions)
			if args.region_prompt == 1001:
				images2= batch['image2'].to(args.device)
				box2_images = batch['box2_image'].to(args.device)
				image2_features, text2_features = model(box2_images, images2, captions)
				images3= batch['image3'].to(args.device)
				box3_images = batch['box3_image'].to(args.device)
				image3_features, text3_features = model(box3_images, images3, captions)
				#images4= batch['image4'].to(args.device)
				#box4_images = batch['box4_image'].to(args.device)
				#image4_features, text4_features = model(box4_images, images4, captions)
				image_features = (image_features + image2_features + image3_features ) / 3.0

			all_im_embs.append(image_features)
			all_txt_embs.append(text_features)

	all_im_embs = torch.cat(all_im_embs).cpu()
	all_txt_embs = torch.cat(all_txt_embs).cpu()
	# this is *not* the most efficient way to do this but,
	# nonetheless, for reproducability, this is how this was computed
	# using the other setup.
	im2text_sim = -sklearn.metrics.pairwise_distances(all_im_embs,
													  all_txt_embs,
													  metric='cosine',
													  n_jobs=args.workers_dataloader) + 1
	sims = np.diag(im2text_sim).tolist()
	result = {}
	for inst, sim in zip(instances, sims):
		result[inst['test_id']] = sim

	sorted_scores = np.array(
		[result[k] for k in sorted(result.keys())]).astype(np.float32)
	np.save(args.output_npy, sorted_scores)

	print('writing {} to {}'.format(len(result), args.output_npy))

	
def retrieval_main(model, args, tokenizer):

	with open(args.instances) as f:
		instances = json.load(f)
	print('{} retrieval instances'.format(len(instances)))

	all_images, all_inferences = set(), set()
	for d in tqdm.tqdm(instances):
		cur_image = frozendict({'url': d['image']['url'], 'bboxes': tuple([frozendict(r) for r in d['region']])})
		cur_inference = frozendict({'caption': d['inference']})
		all_images.add(cur_image)
		all_inferences.add(cur_inference)

	all_images = list(all_images)
	im2idx = {k: idx for idx, k in enumerate(all_images)}
	all_inferences = list(all_inferences)
	inf2idx = {k: idx for idx, k in enumerate(all_inferences)}

	image_dataloader = torch.utils.data.DataLoader(
		clip_inference_iterator.CLIPDatasetImageOnlyLeaderboard(all_images, args, tokenizer),
		batch_size=args.batch_size, num_workers=args.workers_dataloader, shuffle=False, worker_init_fn=worker_init_fn)

	inference_dataloader = torch.utils.data.DataLoader(
		clip_inference_iterator.CLIPDatasetCaptionOnlyLeaderboard(all_inferences, args, tokenizer),
		batch_size=args.batch_size, num_workers=args.workers_dataloader, shuffle=False, worker_init_fn=worker_init_fn)

	all_im_embs, all_txt_embs = [], []
	with torch.no_grad():
		bar = tqdm.tqdm(image_dataloader, total=len(image_dataloader))
		for batch in bar:
			box_images = batch['box_image'].to(args.device)
			images = batch['image'].to(args.device)
			im_embs = model.image_forward(box_images, images)
			if args.region_prompt == 1001:
				box2_images = batch['box2_image'].to(args.device)
				images2 = batch['image2'].to(args.device)
				im2_embs = model.image_forward(box2_images, images2)

				box3_images = batch['box3_image'].to(args.device)
				images3 = batch['image3'].to(args.device)
				im3_embs = model.image_forward(box3_images, images3)

				#box4_images = batch['box4_image'].to(args.device)
				#images4 = batch['image4'].to(args.device)
				#im4_embs = model.image_forward(box4_images, images4)

				im_embs =  (im_embs + im2_embs + im3_embs) / 3.0

			all_im_embs.append(im_embs.cpu())
		bar = tqdm.tqdm(inference_dataloader, total=len(inference_dataloader))
		for batch in bar:
			captions = batch['caption'].to(args.device)
			txt_embs = model.text_forward(captions)
			all_txt_embs.append(txt_embs)

	all_im_embs = torch.cat(all_im_embs).cpu()
	all_txt_embs = torch.cat(all_txt_embs).cpu()
	im2text_sim = -sklearn.metrics.pairwise_distances(all_im_embs,
													  all_txt_embs,
													  metric='cosine',
													  n_jobs=args.workers_dataloader) + 1
	result = {}
	for d in tqdm.tqdm(instances):
		cur_image = frozendict({'url': d['image']['url'], 'bboxes': tuple([frozendict(r) for r in d['region']])})
		cur_inference = frozendict({'caption': d['inference']})
		im_idx, inf_idx = im2idx[cur_image], inf2idx[cur_inference]
		result[d['test_id']] = im2text_sim[im_idx, inf_idx]

	print('writing {} to {}'.format(len(result), args.output_npy))

	sorted_scores = np.array(
		[result[k] for k in sorted(result.keys())]).astype(np.float32)
	np.save(args.output_npy, sorted_scores)


def localization_main(model, args, tokenizer):
	with open(args.instances) as f:
		instances = json.load(f)
	print('{} localization instances'.format(len(instances)))

	# for the paper, we floatified the model for localization. it
	# doesn't matter much, but for reproducability...  while all the
	# result differences are very small, somewhat surprisingly,
	# float16 seems to be less deterministic. Possibly should consider
	# floatifying everything; that's how training was done after all.
	model.float()
	url2instances = collections.defaultdict(list)
	all_images, all_inferences = set(), set()
	for d in tqdm.tqdm(instances):
		url2instances[d['image']['url']].append(d)

	localization_dataloader = torch.utils.data.DataLoader(
		clip_inference_iterator.CLIPDatasetLocalizationLeaderboard(url2instances, args, tokenizer),
		batch_size=1, num_workers=args.workers_dataloader, shuffle=False, worker_init_fn=worker_init_fn)

	result = {}
	#x = torch.cuda.FloatTensor(256, 1024, 6000)
	x = torch.cuda.FloatTensor(256, 4024, 9000)
	with torch.no_grad():
		bar = tqdm.tqdm(localization_dataloader, total=len(localization_dataloader))
		for batch in bar:
			im_embs = model.image_forward(
				batch['box_image'].to(args.device).squeeze(),
				batch['image'].to(args.device).squeeze(0))
			if args.region_prompt == 1001:
				im2_embs = model.image_forward(
					batch['box2_image'].to(args.device).squeeze(),
					batch['image2'].to(args.device).squeeze(0))
				im3_embs = model.image_forward(
					batch['box3_image'].to(args.device).squeeze(),
					batch['image3'].to(args.device).squeeze(0))
				#im4_embs = model.image_forward(
				#	batch['box4_image'].to(args.device).squeeze(),
				#	batch['image4'].to(args.device).squeeze(0))

				im_embs = (im_embs + im2_embs + im3_embs) / 3.0
			txt_embs = model.text_forward(batch['caption'].to(args.device).squeeze(0))
			im2txt = im_embs @ torch.transpose(txt_embs, 0, 1)
			im2txt = im2txt.cpu().numpy()
			for inst_id, im_idx, txt_idx in zip(batch['instance_ids'],
							batch['image_idxs'],
							batch['cap_idxs']):
				result[inst_id[0]] = im2txt[im_idx.item(), txt_idx.item()]

	print('writing {} to {}'.format(len(result), args.output_npy))
	sorted_scores = np.array(
		[result[k] for k in sorted(result.keys())]).astype(np.float32)
	np.save(args.output_npy, sorted_scores)


def main():
	args = parse_args()
	np.random.seed(1)

	args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
	#model, preprocess = clip.load(args.clip_model, jit=False)
	model, _, _ = open_clip.create_model_and_transforms(args.clip_model, 
                jit=False, pretrained=args.pretrained,
                Adapt_TxtEncoder=args.Adapt_TxtEncoder,
                Adapt_VisEncoder=args.Adapt_VisEncoder,
                adapter_rate=args.adapter_rate,
                ada_type=args.AdaType)
	tokenizer   = open_clip.get_tokenizer(args.clip_model)
	# Compile the model for faster training
	#model = torch.compile(model)
	if args.load_model_from != 'None':
		print('Getting model weights from {}'.format(args.load_model_from))
		state = torch.load(args.load_model_from, map_location=args.device)
		state_dict = state['state_dict']

		state_dict = {k.replace('_orig_mod.', '') : v for k, v in state_dict.items()}
		state_dict = {k.replace('module.clip_model.', '') : v for k, v in state_dict.items()}
		state_dict = {k.replace('clip_model.', '') : v for k, v in state_dict.items()}
		state_dict = {k.replace('module.', '') : v for k, v in state_dict.items()}
		model.load_state_dict(state_dict, strict=False)

	try:
		args.input_resolution = model.visual.image_size
	except:
		args.input_resolution = model.input_resolution

	model = clip_inference_iterator.CLIPExtractor(model, args)
	model.to(args.device)
	#model.eval()

	if args.task == 'retrieval':
		retrieval_main(model, args, tokenizer)
	elif args.task == 'localization':
		localization_main(model, args, tokenizer)
	elif args.task == 'comparison':
		comparison_main(model, args, tokenizer)


if __name__ == '__main__':
	main()
