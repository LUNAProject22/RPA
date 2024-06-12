import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import time
from timm.utils import ModelEmaV2
try:
	import wandb
except ImportError:
	wandb = None

from open_clip import ClipLoss_RefCOCO, get_cast_dtype, ClipLoss_RefCOCO_v2
from training.distributed import is_master
from training.zero_shot import zero_shot_eval
from training.precision import get_autocast


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def unwrap_model(model):
	if hasattr(model, 'module'):
		return model.module
	else:
		return model


def train_one_epoch(model, model_ema, data, loader, data_sampler, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
	device = torch.device(args.device)
	autocast = get_autocast(args.precision)
	cast_dtype = get_cast_dtype(args.precision)
	#x = torch.cuda.FloatTensor(256, 4024, 4000) bJabalumodel.train()
	loss = ClipLoss_RefCOCO(
		local_loss=args.local_loss,
		gather_with_grad=args.gather_with_grad,
		cache_labels=True,
		rank=args.rank,
		world_size=args.world_size,
		use_horovod=args.horovod)

	if data_sampler is not None:
		data_sampler.set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
	dataloader = loader
	num_batches_per_epoch = len(dataloader)
	sample_digits = math.ceil(math.log(len(data) + 1, 10))

	loss_m = AverageMeter()
	batch_time_m = AverageMeter()
	data_time_m = AverageMeter()
	end = time.time()
	for i, batch in enumerate(dataloader):
		step = num_batches_per_epoch * epoch + i
		
		if not args.skip_scheduler:
			scheduler(step)

		images = batch['image']
		images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
		boxes = batch['box']
		boxes = boxes.to(device=device, dtype=cast_dtype, non_blocking=True)
		clue_texts = batch['clue']
		clue_texts = clue_texts.to(device=device, non_blocking=True)
		inference_texts = batch['inference']
		inference_texts = inference_texts.to(device=device, non_blocking=True)

		#neg_yolo_box = batch['neg_yolo_box']
		#neg_yolo_box = neg_yolo_box.to(device=device, dtype=cast_dtype, non_blocking=True)
		neg_yolo_box_image = batch['neg_yolo_box_image']
		neg_yolo_box_image = neg_yolo_box_image.to(device=device, dtype=cast_dtype, non_blocking=True)


		data_time_m.update(time.time() - end)
		optimizer.zero_grad()

		with autocast():
			logit_scale = model.module.logit_scale.exp()
			if args.input_resolution[0] > args.input_resolution[1]:
				merge_image = torch.cat((boxes, images), dim=2) # b, c, 2*h, w
				#print(neg_yolo_box_image.shape)
				b, n, c, h, w = neg_yolo_box_image.shape
				neg_merge_image = torch.cat((neg_yolo_box_image, images.unsqueeze(1).repeat(1, n , 1,1,1)), dim=3) # b, n, c, 2*h, w
				neg_merge_image = neg_merge_image.view(b*n, c, neg_merge_image.shape[3], neg_merge_image.shape[4])
			else:
				merge_image = torch.cat((boxes, images), dim=3) # b, c, h, 2*w
				b, n, c, h, w = neg_yolo_box_image.shape
				neg_merge_image = torch.cat((neg_yolo_box_image, images.unsqueeze(1).repeat(1, n, 1,1,1)), dim=4) # b, n, c, h, 2*w
				#print(neg_merge_image.shape)
				neg_merge_image = neg_merge_image.view(b*n, c, neg_merge_image.shape[3], neg_merge_image.shape[4])

			image_features = model.module.encode_image(merge_image)
			image_features = F.normalize(image_features, dim=-1)
	
			neg_image_features = model.module.encode_image(neg_merge_image)
			neg_image_features = F.normalize(neg_image_features, dim=-1)

			#clue_text_features = model.module.encode_text(clue_texts)
			#clue_text_features = F.normalize(clue_text_features, dim=-1)
			inference_text_features = model.module.encode_text(inference_texts)
			inference_text_features = F.normalize(inference_text_features, dim=-1)

			#clue_loss = loss(image_features, clue_text_features, logit_scale)
			## Use RefCOCO_v2 loss
			##neg_image_features = neg_image_features.view(b, n, neg_image_features.shape[1])
			infe_loss = loss(image_features, neg_image_features, inference_text_features, logit_scale)
			#total_loss = (clue_loss + infe_loss) / 2.0
			total_loss = infe_loss

		if scaler is not None:
			scaler.scale(total_loss).backward()
			if args.horovod:
				optimizer.synchronize()
				scaler.unscale_(optimizer)
				if args.grad_clip_norm is not None:
					torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
				with optimizer.skip_synchronize():
					scaler.step(optimizer)
			else:
				if args.grad_clip_norm is not None:
					scaler.unscale_(optimizer)
					torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
				scaler.step(optimizer)
			scaler.update()
		else:
			total_loss.backward()
			if args.grad_clip_norm is not None:
				torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
			optimizer.step()

		if model_ema is not None:
			model_ema.update(model)
			#print("Using EMA...")

		# Note: we clamp to 4.6052 = ln(100), as in the original paper.
		with torch.no_grad():
			unwrap_model(model).logit_scale.clamp_(0, math.log(100))

		batch_time_m.update(time.time() - end)
		end = time.time()
		batch_count = i + 1
		if is_master(args) and (i % 1 == 0 or batch_count == num_batches_per_epoch):
			batch_size = len(images)
			num_samples = i * args.batch_size * args.world_size + batch_size * args.world_size
			samples_per_epoch = len(data)
			percent_complete = 100.0 * batch_count / num_batches_per_epoch

			# NOTE loss is coarsely sampled, just master node and per log update
			loss_m.update(total_loss.item(), batch_size)
			logit_scale_scalar = logit_scale.item()
			logging.info(
				f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
				f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
				f"Data (t): {data_time_m.avg:.3f} "
				f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s "
				f"LR: {optimizer.param_groups[0]['lr']:.9f} "
				f"Logit Scale: {logit_scale_scalar:.3f}"
			)

			# Save train loss / etc. Using non avg meter values as loggers have their own smoothing
			log_data = {
				"loss": loss_m.val,
				"data_time": data_time_m.val,
				"batch_time": batch_time_m.val,
				"samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
				"scale":  logit_scale_scalar,
				"lr": optimizer.param_groups[0]["lr"]
			}
			for name, val in log_data.items():
				name = "train/" + name
				if tb_writer is not None:
					tb_writer.add_scalar(name, val, step)
				if args.wandb:
					assert wandb is not None, 'Please install wandb.'
					wandb.log({name: val, 'step': step})

			# resetting batch / data time meters per log window
			batch_time_m.reset()
			data_time_m.reset()
	# end for
	if is_master(args):
		wandb.log({"train/epoch_loss": loss_m.avg, "epoch":epoch })

def evaluate_refcoco(model, data, loader, data_sampler, epoch, args, tb_writer=None, use_ema=False, name="val"):
	metrics = {}
	#if not is_master(args):
	#	x = torch.cuda.FloatTensor(624, 4024, 4000)
	#	return metrics
	#x = torch.cuda.FloatTensor(624, 4024, 4000)
	device = torch.device(args.device)
	model.eval()

	num_batches_per_epoch = len(loader)
	autocast = get_autocast(args.precision)
	cast_dtype = get_cast_dtype(args.precision)

	dataloader = loader
	num_samples = 0
	samples_per_val = len(data)

	# FIXME this does not scale past small eval datasets
	# all_image_features @ all_text_features will blow up memory and compute very quickly
	cumulative_loss = 0.0
	all_image_features, all_text_features = [], []
	find_match = []
	correct_match = []
	with torch.no_grad():
		for i, batch in enumerate(dataloader):
			images = batch['image']
			images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
			boxes = batch['box']
			boxes = boxes.to(device=device, dtype=cast_dtype, non_blocking=True)
			clue_texts = batch['clue']
			clue_texts = clue_texts.to(device=device, non_blocking=True)
			inference_texts = batch['inference']
			inference_texts = inference_texts.to(device=device, non_blocking=True)

			yolo_box = batch['yolo_box']
			yolo_box_image = batch['yolo_box_image']
			yolo_box_image = yolo_box_image.to(device=device, dtype=cast_dtype, non_blocking=True)
			gt_bbox = batch['gt_bbox']
			#print(gt_bbox.shape)
			#print(yolo_box_image.shape)
			if use_ema:
				logit_scale = model.logit_scale.exp()
			else:
				logit_scale = model.module.logit_scale.exp()

			if is_master(args) and (i % 1) == 0:
				print("{}/{}".format(i, len(dataloader)))

			yolo_box = yolo_box.squeeze(0)
			gt_bbox = gt_bbox.squeeze()
			correct_pred = 0
			if len(yolo_box) > 0:
				#print("yolo_box: ", yolo_box)
				for sig_yolo_box in yolo_box:
					iou_score = computeIoU(gt_bbox, sig_yolo_box)
					if iou_score > 0.5:
						correct_pred = 1
			find_match.append(correct_pred)

			# Concate Candiate Box and Image
			# yolo_box_image, 1xNx3xHxW
			# images, 1x3xHxW
			yolo_box_image = yolo_box_image.squeeze(0) # Nx3xHxW
			N, _, _, _ = yolo_box_image.shape
			images = images.repeat(N, 1, 1, 1) # Nx3xHxW

			if args.input_resolution[0] > args.input_resolution[1]:
				merge_image = torch.cat((yolo_box_image, images), dim=2)
			else:
				merge_image = torch.cat((yolo_box_image, images), dim=3)

			image_features = model.module.encode_image(merge_image)
			image_features = F.normalize(image_features, dim=-1)
			text_features = model.module.encode_text(inference_texts)
			text_features = F.normalize(text_features, dim=-1)

			# Find the best match
			sim = image_features @ text_features.T
			# Sim is Nx1
			# Find the max value index in Sim
			_, max_idx = torch.max(sim, dim=0)
			max_idx = max_idx.item()
			retrive_match = 0
			iou_score = computeIoU(gt_bbox, yolo_box[max_idx])
			if iou_score > 0.5:
				retrive_match = 1
			correct_match.append(retrive_match)
			#print("max_idx: ", max_idx)
								


	if is_master(args):
		total_correct = sum(find_match)
		total_amount = len(find_match)
		accuracy = total_correct * 1.0 / total_amount
		metrics['{}/recall_by_mattn'.format(name)] = accuracy
		print("Recall Accuracy {}, Total Recall {}, Total {}: ".format( accuracy, total_correct, total_amount))
		wandb.log({"{}/recall_by_mattn".format(name): accuracy, 'epoch': epoch})

		total_correct = sum(correct_match)
		total_amount = len(correct_match)
		accuracy = total_correct * 1.0 / total_amount
		print("Ground Accuracy@0.5 {}, Total Retrive {}, Total {}: ".format(accuracy, total_correct, total_amount))
		wandb.log({"{}/ground_by_RGPs".format(name): accuracy, 'epoch': epoch})

		metrics['{}/ground_by_RGPs'.format(name)] = accuracy
	return metrics
			#	# Concate Candiate Box and Image

			#	# Extract Vision Feature

			#	# Extract Text Feature

			#	# Find match

			#	# Measure with gt
			#	with autocast():
			#		if args.input_resolution[0] > args.input_resolution[1]:
			#			merge_image = torch.cat((boxes, images), dim=2) # b, c, 2*h, w
			#		else:
			#			merge_image = torch.cat((boxes, images), dim=3) # b, c, h, 2*w
			#		image_features = model.module.encode_image(merge_image)
			#		image_features = F.normalize(image_features, dim=-1)
			#		text_features = model.module.encode_text(inference_texts)
			#		#text_features = model.module.encode_text(clue_texts)
			#		text_features = F.normalize(text_features, dim=-1)

			#		# features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
			#		# however, system RAM is easily exceeded and compute time becomes problematic
			#		all_image_features.append(image_features.cpu())
			#		all_text_features.append(text_features.cpu())
			#		logit_scale = logit_scale.mean()
			#		logits_per_image = logit_scale * image_features @ text_features.t()
			#		logits_per_text = logits_per_image.t()

			#		batch_size = images.shape[0]
			#		labels = torch.arange(batch_size, device=device).long()
			#		total_loss = (
			#			F.cross_entropy(logits_per_image, labels) +
			#			F.cross_entropy(logits_per_text, labels)
			#		) / 2

			#	cumulative_loss += total_loss * batch_size
			#	num_samples += batch_size
			#	if is_master(args) and (i % 1) == 0:
			#		logging.info(
			#			f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
			#			f"Loss: {cumulative_loss / num_samples:.6f}\t")
			#time.sleep(1)
			#val_metrics = get_metrics(
			#	image_features=torch.cat(all_image_features),
			#	text_features=torch.cat(all_text_features),
			#	logit_scale=logit_scale.cpu(),
			#)
			#print(val_metrics)
			#loss = cumulative_loss / num_samples
			#metrics.update(
			#	{**val_metrics, "val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
			#)

	#if not metrics:
	#	return metrics

	#logging.info(
	#	f"Eval Epoch: {epoch} "
	#	+ "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
	#)

	#if args.save_logs:
	#	for name, val in metrics.items():
	#		if tb_writer is not None:
	#			tb_writer.add_scalar(f"val/{name}", val, epoch)

	#	with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
	#		f.write(json.dumps(metrics))
	#		f.write("\n")

	#if args.wandb:
	#	assert wandb is not None, 'Please install wandb.'
	#	for name, val in metrics.items():
	#		wandb.log({f"val/{name}": val, 'epoch': epoch})

	#return metrics

def evaluate(model, data, loader, data_sampler, epoch, args, tb_writer=None, use_ema=False):
	metrics = {}
	if not is_master(args):
		return metrics
	device = torch.device(args.device)
	model.eval()

	num_batches_per_epoch = len(loader)

	autocast = get_autocast(args.precision)
	cast_dtype = get_cast_dtype(args.precision)

	if (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
		dataloader = loader
		num_samples = 0
		samples_per_val = len(data)

		# FIXME this does not scale past small eval datasets
		# all_image_features @ all_text_features will blow up memory and compute very quickly
		cumulative_loss = 0.0
		all_image_features, all_text_features = [], []
		with torch.no_grad():
			for i, batch in enumerate(dataloader):
				images = batch['image']
				images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
				boxes = batch['box']
				boxes = boxes.to(device=device, dtype=cast_dtype, non_blocking=True)
				clue_texts = batch['clue']
				clue_texts = clue_texts.to(device=device, non_blocking=True)
				inference_texts = batch['inference']
				inference_texts = inference_texts.to(device=device, non_blocking=True)

				if use_ema:
					logit_scale = model.logit_scale.exp()
				else:
					logit_scale = model.module.logit_scale.exp()


				with autocast():
					if args.input_resolution[0] > args.input_resolution[1]:
						merge_image = torch.cat((boxes, images), dim=2) # b, c, 2*h, w
					else:
						merge_image = torch.cat((boxes, images), dim=3) # b, c, h, 2*w
					image_features = model.module.encode_image(merge_image)
					image_features = F.normalize(image_features, dim=-1)
					text_features = model.module.encode_text(inference_texts)
					#text_features = model.module.encode_text(clue_texts)
					text_features = F.normalize(text_features, dim=-1)

					# features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
					# however, system RAM is easily exceeded and compute time becomes problematic
					all_image_features.append(image_features.cpu())
					all_text_features.append(text_features.cpu())
					logit_scale = logit_scale.mean()
					logits_per_image = logit_scale * image_features @ text_features.t()
					logits_per_text = logits_per_image.t()

					batch_size = images.shape[0]
					labels = torch.arange(batch_size, device=device).long()
					total_loss = (
						F.cross_entropy(logits_per_image, labels) +
						F.cross_entropy(logits_per_text, labels)
					) / 2

				cumulative_loss += total_loss * batch_size
				num_samples += batch_size
				if is_master(args) and (i % 1) == 0:
					logging.info(
						f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
						f"Loss: {cumulative_loss / num_samples:.6f}\t")
			time.sleep(1)
			val_metrics = get_metrics(
				image_features=torch.cat(all_image_features),
				text_features=torch.cat(all_text_features),
				logit_scale=logit_scale.cpu(),
			)
			print(val_metrics)
			loss = cumulative_loss / num_samples
			metrics.update(
				{**val_metrics, "val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
			)

	if not metrics:
		return metrics

	logging.info(
		f"Eval Epoch: {epoch} "
		+ "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
	)

	if args.save_logs:
		for name, val in metrics.items():
			if tb_writer is not None:
				tb_writer.add_scalar(f"val/{name}", val, epoch)

		with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
			f.write(json.dumps(metrics))
			f.write("\n")

	if args.wandb:
		assert wandb is not None, 'Please install wandb.'
		for name, val in metrics.items():
			wandb.log({f"val/{name}": val, 'epoch': epoch})

	return metrics


def get_metrics(image_features, text_features, logit_scale):
	metrics = {}
	logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
	logits_per_text = logits_per_image.t().detach().cpu()

	logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
	ground_truth = torch.arange(len(text_features)).view(-1, 1)

	for name, logit in logits.items():
		ranking = torch.argsort(logit, descending=True)
		preds = torch.where(ranking == ground_truth)[1]
		preds = preds.detach().cpu().numpy()
		metrics[f"{name}_mean_rank"] = preds.mean() + 1
		metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
		for k in [1, 5, 10]:
			metrics[f"{name}_R@{k}"] = np.mean(preds < k)

	return metrics

	# IoU function
def computeIoU(box1, box2):
	# each box is of [x1, y1, w, h]
	inter_x1 = max(box1[0], box2[0])
	inter_y1 = max(box1[1], box2[1])
	inter_x2 = min(box1[0]+box1[2]-1, box2[0]+box2[2]-1)
	inter_y2 = min(box1[1]+box1[3]-1, box2[1]+box2[3]-1)

	if inter_x1 < inter_x2 and inter_y1 < inter_y2:
		inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
	else:
		inter = 0
	union = box1[2]*box1[3] + box2[2]*box2[3] - inter
	return float(inter)/union