import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

from fvcore.nn.activation_count import activation_count
from fvcore.nn.flop_count import flop_count
from fvcore.nn import FlopCountAnalysis, flop_count_table

from calIoU import computeIoU

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from training.distributed import is_master
from training.zero_shot import zero_shot_eval
from training.precision import get_autocast

def merge_image_func(boxes, images, args):
    if args.input_resolution[0] > args.input_resolution[1]:
        merge_image = torch.cat((boxes, images), dim=2) # b, c, 2*h, w
    else:
        merge_image = torch.cat((boxes, images), dim=3) # b, c, h, 2*w
    return merge_image

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

def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }

def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(model, data, data_loader, data_sampler, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)


    model.train()
    if args.distill:
        dist_model.eval()

    if data_sampler is not None:
        data_sampler.set_epoch(epoch)

    num_batches_per_epoch = len(data_loader)
    sample_digits = math.ceil(math.log(len(data) + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_texts_2, accum_features = [], [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(data_loader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler.step()
            #scheduler(step)

        # Vision
        images = batch['image']
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        boxes = batch['box']
        boxes = boxes.to(device=device, dtype=input_dtype, non_blocking=True)
        merge_image = merge_image_func(boxes, images, args)

        proposal_yolo_bboxes_images = batch['proposal_yolo_bboxes_images']
        proposal_yolo_bboxes_images = proposal_yolo_bboxes_images.to(
            device=device,
            dtype=input_dtype,
            non_blocking=True,
        )


        full_images = batch['full_images']
        full_images = full_images.to(device=device, dtype=input_dtype, non_blocking=True)
        neg_image = merge_image_func(proposal_yolo_bboxes_images,
                full_images, args,
        )
        neg_im_feat = model.module.encode_image(neg_image, normalize=True)

        # Text
        inference_texts = batch['inference']
        inference_texts = inference_texts.to(device=device, non_blocking=True)


        data_time_m.update(time.time() - end)
        optimizer.zero_grad()
        losses = {}
        if args.accum_freq == 1:
            with autocast():
                im_feat =  model.module.encode_image(merge_image, normalize=True)
                inference_feat = model.module.encode_text(inference_texts, normalize=True)

                logit_scale = model.module.logit_scale.exp()
                losses_2   = loss(im_feat, inference_feat, neg_im_feat,logit_scale, output_dict=True)

                total_loss = sum(losses_2.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    im_feat =  model.module.encode_image(merge_image, normalize=True)
                    clue_feat = model.module.encode_text(clue_texts, normalize=True)
                    inference_feat = model.module.encode_text(inference_texts, normalize=True)
                    if "image_features" not in accum_features:
                        accum_features["image_features"] = []
                    else:
                        accum_features["image_features"].append(im_feat)

                    if "text_features" not in accum_features:
                        accum_features["text_features"] = []
                    else:
                        accum_features["text_features"].append(clue_feat)

                    if "text_features_2" not in accum_features:
                        accum_features["text_features_2"] = []
                    else:
                        accum_features["text_features_2"].append(inference_feat)
                    #model_out = model(merge_image, clue_texts)
                    #model_out.pop("logit_scale")
                    #for key, val in model_out.items():
                    #    if key in accum_features:
                    #        accum_features[key].append(val)
                    #    else:
                    #        accum_features[key] = [val]

                accum_images.append(merge_image)
                accum_texts.append(clue_texts)
                accum_texts_2.append(inference_texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                merge_image = accum_images[j]
                clue_texts = accum_texts[j]
                inference_texts = accum_texts_2[j]
                logit_scale = model.module.logit_scale.exp()
                with autocast():
                    im_feat =  model.module.encode_image(merge_image, normalize=True)
                    clue_feat = model.module.encode_text(clue_texts, normalize=True)
                    inference_feat = model.module.encode_text(inference_texts, normalize=True)
                    model_out = {"image_features" : im_feat, "text_features" : clue_feat, "text_features_2" : inference_feat}
                    #model_out = model(merge_image, clue_texts)
                    #logit_scale = model_out.pop("logit_scale")
                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] +  [model_out[key]] + accumulated[j + 1:])
                    #losses = loss(**inputs, logit_scale=logit_scale, output_dict=True)
                    losses_1 = loss(inputs["image_features"], inputs['text_features'], logit_scale=logit_scale, output_dict=True)
                    losses_2 = loss(inputs["image_features"], inputs['text_features_2'], logit_scale=logit_scale, output_dict=True)
                    del inputs
                    total_loss = (sum(losses_1.values()) + sum(losses_2.values())) / 2.0
                    losses["loss"] = total_loss
                backward(total_loss, scaler)

        if scaler is not None:
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
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = (i_accum * args.batch_size * args.world_size + batch_size * args.world_size) * args.accum_freq
            #num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = len(data)
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:.9f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})

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
        wandb.log({"train/epoch_loss": losses_m['loss'].avg, "epoch":epoch })



def evaluate_refcoco(model, data, loader, data_sampler, epoch, args, tb_writer=None, use_ema=False, name="val"):
	metrics = {}
	#if not is_master(args):
	#	x = torch.cuda.FloatTensor(624, 4024, 4000)
	#	return metrics
	#x = torch.cuda.FloatTensor(624, 4024, 4000)
	x = torch.cuda.FloatTensor(324, 4024, 4000)
	device = torch.device(args.device)
	model.eval()

	num_batches_per_epoch = len(loader)
	autocast = get_autocast(args.precision)
	cast_dtype = get_input_dtype(args.precision)

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
			#boxes = batch['box']
			#boxes = boxes.to(device=device, dtype=cast_dtype, non_blocking=True)
			inference_texts = batch['inference']
			inference_texts = inference_texts.to(device=device, non_blocking=True)

			full_images = batch['full_images']
			full_images = full_images.to(device=device, dtype=cast_dtype, non_blocking=True)

			yolo_box = batch['yolo_box']
			yolo_box_image = batch['proposal_yolo_bboxes_images']
			yolo_box_image = yolo_box_image.to(device=device, dtype=cast_dtype, non_blocking=True)
			#yolo_box_image_2 = batch['proposal_yolo_bboxes_images_2']
			#yolo_box_image_2 = yolo_box_image_2.to(device=device, dtype=cast_dtype, non_blocking=True)
			#yolo_box_image_3 = batch['proposal_yolo_bboxes_images_3']
			#yolo_box_image_3 = yolo_box_image_3.to(device=device, dtype=cast_dtype, non_blocking=True)
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
			full_images = full_images.squeeze(0)
			#yolo_box_image_2 = yolo_box_image_2.squeeze(0) # Nx3xHxW
			#yolo_box_image_3 = yolo_box_image_3.squeeze(0) # Nx3xHxW
			N, _, _, _ = yolo_box_image.shape
			#images = images.repeat(N, 1, 1, 1) # Nx3xHxW

			if args.input_resolution[0] > args.input_resolution[1]:
				merge_image = torch.cat((yolo_box_image, full_images), dim=2)
				#merge_image_2 = torch.cat((yolo_box_image_2, images), dim=2)
				#merge_image_3 = torch.cat((yolo_box_image_3, images), dim=2)
			else:
				merge_image = torch.cat((yolo_box_image, full_images), dim=3)
				#merge_image_2 = torch.cat((yolo_box_image_2, images), dim=3)
				#merge_image_3 = torch.cat((yolo_box_image_3, images), dim=3)

			image_features = model.module.encode_image(merge_image)
			image_features = F.normalize(image_features, dim=-1)
			#image_features_2 = model.module.encode_image(merge_image_2)
			#image_features_2 = F.normalize(image_features_2, dim=-1)
			#image_features_3 = model.module.encode_image(merge_image_3)
			#image_features_3 = F.normalize(image_features_3, dim=-1)
			#image_features = (image_features_2 + image_features_3 + image_features) / 3.0
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
		metrics['{}/recall_by_yolov8'.format(name)] = accuracy
		print("Recall Accuracy {}, Total Recall {}, Total {}: ".format( accuracy, total_correct, total_amount))
		wandb.log({"{}/recall_by_yolov8".format(name): accuracy, 'epoch': epoch})

		total_correct = sum(correct_match)
		total_amount = len(correct_match)
		accuracy = total_correct * 1.0 / total_amount
		print("Ground Accuracy@0.5 {}, Total Retrive {}, Total {}: ".format(accuracy, total_correct, total_amount))
		wandb.log({"{}/ground_by_RPA".format(name): accuracy, 'epoch': epoch})

		metrics['{}/ground_by_RPA'.format(name)] = accuracy
	return metrics





def get_clip_metrics(image_features, text_features, logit_scale):
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


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)


def cal_flops_evaluate(model, data, data_loader, data_sampler, epoch, args, tb_writer=None):
    metrics = {}
    #if not is_master(args):
    #    return metrics
    device = torch.device(args.device)
    model.eval()

    #zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    #metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        num_samples = 0
        samples_per_val = len(data)

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                images = batch['image']
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                boxes = batch['box']
                boxes = boxes.to(device=device, dtype=input_dtype, non_blocking=True)
                merge_image = merge_image_func(boxes, images, args)

                inference_texts = batch['inference']
                inference_texts = inference_texts.to(device=device, non_blocking=True)

                with autocast():
                    flops = FlopCountAnalysis(model, (merge_image, inference_texts))
                    print(flop_count_table(flops))

    return 
