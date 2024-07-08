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
        #if args.input_resolution[0] > args.input_resolution[1]:
        #    merge_image = torch.cat((boxes, images), dim=2) # b, c, 2*h, w
        #else:
        #    merge_image = torch.cat((boxes, images), dim=3) # b, c, h, 2*w
        
        # Negative Image
        if args.use_negative_box:
            #neg_image = batch['neg_image']
            negative_box = batch['negative_box']
            #neg_image = neg_image.to(device=device, dtype=input_dtype, non_blocking=True)
            negative_box = negative_box.to(device=device, dtype=input_dtype, non_blocking=True)
            neg_merge_image = merge_image_func(negative_box, images, args)
            #if args.input_resolution[0] > args.input_resolution[1]:
            #    neg_merge_image = torch.cat((negative_box, images), dim=2)
            #else:
            #    neg_merge_image = torch.cat((negative_box, images), dim=3)
            #neg_im_feat = model.module.encode_image(neg_merge_image, normalize=True)

        # Text
        clue_texts = batch['clue']
        clue_texts = clue_texts.to(device=device, non_blocking=True)
        inference_texts = batch['inference']
        inference_texts = inference_texts.to(device=device, non_blocking=True)
        #if np.random.rand() > 0.5:
        #    caption_text = clue_texts
        #else:
        #    caption_text = inference_texts


        data_time_m.update(time.time() - end)
        optimizer.zero_grad()
        losses = {}
        if args.accum_freq == 1:
            with autocast():
                #model_out = model(merge_image, clue_texts)
                im_feat =  model.module.encode_image(merge_image, normalize=True)
                clue_feat = model.module.encode_text(clue_texts, normalize=True)
                inference_feat = model.module.encode_text(inference_texts, normalize=True)
                #caption_feat = model.module.encode_text(caption_text, normalize=True)

                logit_scale = model.module.logit_scale.exp()
                if args.use_negative_box:
                    #losses_1   = loss(im_feat, clue_feat, neg_im_feat, logit_scale, output_dict=True)
                    losses_2   = loss(im_feat, inference_feat, neg_im_feat, logit_scale, output_dict=True)
                else:
                    #losses_1   = loss(im_feat, caption_feat, logit_scale, output_dict=True)
                    #losses_1   = loss(im_feat, clue_feat, logit_scale, output_dict=True)
                    losses_2   = loss(im_feat, inference_feat, logit_scale, output_dict=True)
                    #losses_3   = loss(clue_feat, inference_feat, logit_scale, output_dict=True)


                total_loss = sum(losses_2.values())
                #total_loss = (sum(losses_1.values()) + sum(losses_2.values())) / 2.0
                #total_loss = (sum(losses_1.values()) + sum(losses_2.values()) + sum(losses_3.values())) / 3.0
                #total_loss = sum(losses_2.values())
                #total_loss = sum(losses_1.values())
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


def evaluate_sep(model, data, data_loader, data_sampler, epoch, args, tb_writer=None):
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
        cumulative_loss_img_clue = 0.0
        cumulative_loss_img_inf = 0.0
        cumulative_loss_inf_clue = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        all_clue_features = []
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                images = batch['image']
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                boxes = batch['box']
                boxes = boxes.to(device=device, dtype=input_dtype, non_blocking=True)
                merge_image = merge_image_func(boxes, images, args)
                #if args.input_resolution[0] > args.input_resolution[1]:
                #    merge_image = torch.cat((boxes, images), dim=2) # b, c, 2*h, w
                #else:
                #    merge_image = torch.cat((boxes, images), dim=3) # b, c, h, 2*w

                if args.region_prompt == 1001:
                    #print("XXXXXXXXXXXXXXXXXXXXXXXHEHRHEHE")
                    images2 = batch['image2']
                    images2 = images2.to(device=device, dtype=input_dtype, non_blocking=True)
                    boxes2 = batch['box2']
                    boxes2 = boxes2.to(device=device, dtype=input_dtype, non_blocking=True)
                    merge_image2 = merge_image_func(boxes2, images2, args)

                    #if args.input_resolution[0] > args.input_resolution[1]:
                    #    merge_image2 = torch.cat((boxes2, images2), dim=2) # b, c, 2*h, w
                    #else:
                    #    merge_image2 = torch.cat((boxes2, images2), dim=3) # b, c, h, 2*w

                    images3 = batch['image3']
                    images3 = images3.to(device=device, dtype=input_dtype, non_blocking=True)
                    boxes3 = batch['box3']
                    boxes3 = boxes3.to(device=device, dtype=input_dtype, non_blocking=True)
                    merge_image3 = merge_image_func(boxes3, images3, args)
                    #if args.input_resolution[0] > args.input_resolution[1]:
                    #    merge_image3 = torch.cat((boxes3, images3), dim=2)
                    #else:
                    #    merge_image3 = torch.cat((boxes3, images3), dim=3)
                    #images4 = batch['image4']
                    #images4 = images4.to(device=device, dtype=input_dtype, non_blocking=True)
                    #boxes4 = batch['box4']
                    #boxes4 = boxes4.to(device=device, dtype=input_dtype, non_blocking=True)
                    #merge_image4 = merge_image_func(boxes4, images4, args)

                clue_texts = batch['clue']
                clue_texts = clue_texts.to(device=device, non_blocking=True)
                inference_texts = batch['inference']
                inference_texts = inference_texts.to(device=device, non_blocking=True)

                with autocast():
                    if args.region_prompt != 1001:
                        im_feat =  model.module.encode_image(merge_image, normalize=True)
                    if args.region_prompt == 1001:
                        im_feat1 =  model.module.encode_image(merge_image, normalize=True)
                        im_feat2 =  model.module.encode_image(merge_image2, normalize=True)
                        im_feat3 =  model.module.encode_image(merge_image3, normalize=True)
                        #im_feat4 =  model.module.encode_image(merge_image4, normalize=True)
                        im_feat = (im_feat1 + im_feat2 + im_feat3) / 3.0
                        #im_feat = F.normalize(im_feat, dim=-1) # After Norm
                    clue_feat = model.module.encode_text(clue_texts, normalize=True)
                    inference_feat = model.module.encode_text(inference_texts, normalize=True)
                    logit_scale = model.module.logit_scale.exp()

                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(im_feat.cpu())
                    all_text_features.append(inference_feat.cpu())
                    all_clue_features.append(clue_feat.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * im_feat @ inference_feat.t()
                    logits_per_text = logits_per_image.t()

                    logits_per_image_img_clue = logit_scale * im_feat @ clue_feat.t()
                    logits_per_text_img_clue = logits_per_image_img_clue.t()

                    logits_per_image_inf_clue = logit_scale * inference_feat @ clue_feat.t()
                    logits_per_text_inf_clue = logits_per_image_inf_clue.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2
                    total_loss_img_clue = (
                        F.cross_entropy(logits_per_image_img_clue, labels) +
                        F.cross_entropy(logits_per_text_img_clue, labels)
                    ) / 2
                    total_loss_inf_clue = (
                        F.cross_entropy(logits_per_image_inf_clue, labels) +
                        F.cross_entropy(logits_per_text_inf_clue, labels)
                    ) / 2

                    #gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                cumulative_loss_inf_clue += total_loss_inf_clue * batch_size
                cumulative_loss_img_clue += total_loss_img_clue * batch_size

                num_samples += batch_size
                if is_master(args) and (i % 1) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t"
                        f"Inf-Clue Loss: {cumulative_loss_inf_clue / num_samples:.6f}\t"
                        f"Clue-Img: {cumulative_loss_img_clue / num_samples:.6f}\t")

                    #if gen_loss is not None:
                    #    cumulative_gen_loss += gen_loss * batch_size
                    #    logging.info(
                    #        f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")



            M_image_features=torch.cat(all_image_features)
            M_clue_features=torch.cat(all_clue_features)
            M_inference_features=torch.cat(all_text_features)
            labels = torch.arange(M_image_features.shape[0], device=device).long().cpu()

            logits_per_image = logit_scale.cpu() * M_image_features @ M_clue_features.t()
            logits_per_text = logits_per_image.t()
            imag_clue_loss = ( 
                            F.cross_entropy(logits_per_image, labels) +
                            F.cross_entropy(logits_per_text, labels)
                            ) / 2 

            logits_per_image = logit_scale.cpu() * M_image_features @ M_inference_features.t()
            logits_per_text = logits_per_image.t()
            imag_inference_loss = ( 
                    F.cross_entropy(logits_per_image, labels) +
                    F.cross_entropy(logits_per_text, labels)
                    ) / 2 

            logits_per_image = logit_scale.cpu() * M_inference_features @ M_clue_features.t()
            logits_per_text = logits_per_image.t()
            inference_clue_loss = ( 
                    F.cross_entropy(logits_per_image, labels) +
                    F.cross_entropy(logits_per_text, labels)
                    ) / 2 

            print("GLOBAL-loss: image-clue {}, image-infe {}， clue-infe {}".format(imag_clue_loss, imag_inference_loss, inference_clue_loss))


            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            #if gen_loss is not None:
            #    gen_loss = cumulative_gen_loss / num_samples
            #    metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs and is_master(args):
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics

def evaluate(model, data, data_loader, data_sampler, epoch, args, tb_writer=None):
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
                #if args.input_resolution[0] > args.input_resolution[1]:
                #    merge_image = torch.cat((boxes, images), dim=2) # b, c, 2*h, w
                #else:
                #    merge_image = torch.cat((boxes, images), dim=3) # b, c, h, 2*w

                if args.region_prompt == 1001:
                    #print("XXXXXXXXXXXXXXXXXXXXXXXHEHRHEHE")
                    images2 = batch['image2']
                    images2 = images2.to(device=device, dtype=input_dtype, non_blocking=True)
                    boxes2 = batch['box2']
                    boxes2 = boxes2.to(device=device, dtype=input_dtype, non_blocking=True)
                    merge_image2 = merge_image_func(boxes2, images2, args)

                    #if args.input_resolution[0] > args.input_resolution[1]:
                    #    merge_image2 = torch.cat((boxes2, images2), dim=2) # b, c, 2*h, w
                    #else:
                    #    merge_image2 = torch.cat((boxes2, images2), dim=3) # b, c, h, 2*w

                    images3 = batch['image3']
                    images3 = images3.to(device=device, dtype=input_dtype, non_blocking=True)
                    boxes3 = batch['box3']
                    boxes3 = boxes3.to(device=device, dtype=input_dtype, non_blocking=True)
                    merge_image3 = merge_image_func(boxes3, images3, args)
                    #if args.input_resolution[0] > args.input_resolution[1]:
                    #    merge_image3 = torch.cat((boxes3, images3), dim=2)
                    #else:
                    #    merge_image3 = torch.cat((boxes3, images3), dim=3)
                    #images4 = batch['image4']
                    #images4 = images4.to(device=device, dtype=input_dtype, non_blocking=True)
                    #boxes4 = batch['box4']
                    #boxes4 = boxes4.to(device=device, dtype=input_dtype, non_blocking=True)
                    #merge_image4 = merge_image_func(boxes4, images4, args)

                clue_texts = batch['clue']
                clue_texts = clue_texts.to(device=device, non_blocking=True)
                inference_texts = batch['inference']
                inference_texts = inference_texts.to(device=device, non_blocking=True)

                with autocast():
                    if args.region_prompt != 1001:
                        im_feat =  model.module.encode_image(merge_image, normalize=True)
                    if args.region_prompt == 1001:
                        im_feat1 =  model.module.encode_image(merge_image, normalize=True)
                        im_feat2 =  model.module.encode_image(merge_image2, normalize=True)
                        im_feat3 =  model.module.encode_image(merge_image3, normalize=True)
                        #im_feat4 =  model.module.encode_image(merge_image4, normalize=True)
                        im_feat = (im_feat1 + im_feat2 + im_feat3) / 3.0
                        #im_feat = F.normalize(im_feat, dim=-1) # After Norm
                    #clue_feat = model.module.encode_text(clue_texts, normalize=True)
                    inference_feat = model.module.encode_text(inference_texts, normalize=True)
                    logit_scale = model.module.logit_scale.exp()

                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(im_feat.cpu())
                    all_text_features.append(inference_feat.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * im_feat @ inference_feat.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    #gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 1) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    #if gen_loss is not None:
                    #    cumulative_gen_loss += gen_loss * batch_size
                    #    logging.info(
                    #        f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            #if gen_loss is not None:
            #    gen_loss = cumulative_gen_loss / num_samples
            #    metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs and is_master(args):
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

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
