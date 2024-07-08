import glob
import logging
import os
import re
import subprocess
import sys
import random
from datetime import datetime
import json

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
#from training.data import get_data
from training.distributed import is_master, init_distributed_device, broadcast_object
from training.logger import setup_logging
from training.scheduler import cosine_lr, const_lr, const_lr_cooldown
from training.file_utils import pt_load, check_exists, start_sync_process, remote_sync

from dhpr_dataloader import CLIPDataset

from dhpr_params import parse_args
from train_rp_adapter_dhpr import train_one_epoch, evaluate, cal_flops_evaluate, evaluate_sep

LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote : bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def main(args):
    args = parse_args(args)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    # get the name of the experiments
    tp = args.name
    print("args.Adapt_VisEncoder", args.Adapt_VisEncoder)
    print("args.Adapt_TxtEncoder", args.Adapt_TxtEncoder)
    uadapter = ""
    if args.Adapt_TxtEncoder:
        uadapter += "Txt"
    if args.Adapt_VisEncoder:
        uadapter += "Vis"
    if uadapter != "":
        uadapter += "Adapter"
        if args.AdaType == 0:
            uadapter += "-MAA"
        elif args.AdaType == 1:
            uadapter += "-MLP"
        elif args.AdaType == 2:
            uadapter += "-ATT"
        elif args.AdaType == 3:
            uadapter += "-MAP"
        elif args.AdaType == 4:
            uadapter += "-MM"
        elif args.AdaType == 5:
            uadapter += "-MLPAT"
        elif args.AdaType == 6:
            uadapter += "-MAPAT"
        elif args.AdaType == 7:
            uadapter += "-MLPAT-MSHA"
        elif args.AdaType == 8:
            uadapter += "-MLP-MSHA"
        uadapter += f"-{args.adapter_rate}"
    
    if args.region_prompt == 1001:
        region_promt = "Mix"
    elif args.region_prompt == 101:
        region_promt = "R-CTX"
    elif args.region_prompt == 102:
        region_promt = "R-CPT"
    elif args.region_prompt == 103:
        region_promt = "R-CIR"
    elif args.region_prompt == 0:
        region_promt = "CTX"
    elif args.region_prompt == 1:
        region_promt = "REG"
    elif args.region_prompt == 2:
        region_promt = "CPT"
    elif args.region_prompt == 3:
        region_promt = "CIR"

    args.name = '-'.join([
			f"Gx{args.world_size}",
			f"RPA-V220",
			f"{region_promt}",
			f"{uadapter}",
			f"{args.model}",
			f"LR{args.lr}",
			f"B{args.batch_size}",
			f"P{args.precision}",
			f"{args.pretrained}",
			f"E{args.epochs}",
			f"{tp}",
			datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
		])


    resume_latest = args.resume == 'latest'
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest:
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
            if args.save_most_recent:
                print('Error. Cannot use save-most-recent with remote_sync and resume latest.')
                return -1
            if args.remote_sync_protocol != 's3':
                print('Error. Sync protocol not supported when using resume latest.')
                return -1
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    if args.copy_codebase:
        copy_codebase(args)

    # start the sync proces if remote-sync is not None
    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        # first make sure it works
        result = remote_sync(
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('remote sync successful.')
        else:
            logging.info('Error: remote sync failed. Exiting.')
            return -1
        # if all looks good, start a process to do this every args.remote_sync_frequency seconds
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        remote_sync_process.start()

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    dist_model = None
    args.distill = args.distill_model is not None and args.distill_pretrained is not None
    if args.distill:
        #FIXME: support distillation with grad accum.
        assert args.accum_freq == 1
        #FIXME: support distillation with coca.
        assert 'coca' not in args.model.lower()

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]
    random_seed(args.seed, 0)
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        pretrained_image=args.pretrained_image,
        image_mean=args.image_mean,
        image_std=args.image_std,
        aug_cfg=args.aug_cfg,
        output_dict=True,
        Adapt_TxtEncoder=args.Adapt_TxtEncoder,
        Adapt_VisEncoder=args.Adapt_VisEncoder,
        adapter_rate=args.adapter_rate,
        ada_type=args.AdaType,
    )


    args.input_resolution = model.visual.image_size

    if args.distill:
        # FIXME: currenlty assumes the model your distilling from has the same tokenizer & transforms.
        dist_model, _, _ = create_model_and_transforms(
            args.distill_model, 
            args.distill_pretrained,
            device=device,
            precision=args.precision,
            output_dict=True,
        )
    if args.use_bnb_linear is not None:
        print('=> using a layer from bitsandbytes.\n'
              '   this is an experimental feature which requires two extra pip installs\n'
              '   pip install bitsandbytes triton'
              '   please make sure to use triton 2.0.0')
        import bitsandbytes as bnb
        from open_clip.utils import replace_linear
        print(f'=> replacing linear layers with {args.use_bnb_linear}')
        linear_replacement_cls = getattr(bnb.nn.triton_based_modules, args.use_bnb_linear)
        replace_linear(model, linear_replacement_cls)
        model = model.to(device)

    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)
    if args.lock_text:
        model.lock_text_tower(
            unlocked_layers=args.lock_text_unlocked_layers,
            freeze_layer_norm=args.lock_text_freeze_layer_norm)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
    
        if args.distill:
            dist_model = torch.nn.parallel.DistributedDataParallel(dist_model, device_ids=[device], **ddp_args)

    # create optimizer and scaler
    optimizer = None
    scaler = None

    if args.train_data or args.dataset_type == "synthetic":
        assert not args.trace, 'Cannot train with traced model'

        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)
        #if args.Adapt_TxtEncoder or args.Adapt_VisEncoder:
        if not args.full_tune:
            adapt_params = []
            pass_params  = []
            ## freeze some parameters Option-2
            for name, param in model.named_parameters():
                if 'Adapter' in name: 
                    print("Adapt Tuning: ", name)
                    adapt_params.append([name, param])

                # Vision Tower output layer               
                elif '.ln_post' in name or '.visual.proj'in name or "logit_scale" in name:
                    print("Adapt Tuning: ", name)
                    pass_params.append([name, param])

                # Text Tower output layer
                elif ".text_projection" in name or ".ln_final" in name: 
                    print("Adapt Tuning: ", name)
                    pass_params.append([name, param])
                elif ".class_embedding" in name or ".positional_embedding" in name: 
                    print("Adapt Tuning: ", name)
                    pass_params.append([name, param])
                elif ".token_embedding" in name or ".conv1" in name: 
                    print("Adapt Tuning: ", name)
                    pass_params.append([name, param])

            adapt_params_gain_or_bias = [p for n, p in adapt_params if exclude(n, p)]
            adapt_params_rest = [p for n, p in adapt_params if include(n, p)]

            pass_params_gain_or_bias = [p for n, p in pass_params if exclude(n, p)]
            pass_params_rest = [p for n, p in pass_params if include(n, p)]

            total_params = adapt_params_gain_or_bias + adapt_params_rest + pass_params_gain_or_bias + pass_params_rest

            num_param = sum(p.numel() for p in total_params)
            num_total_param = sum(p.numel() for p in model.parameters())
            logging.info('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))
            optimizer = optim.AdamW(
			    [
			    	{"params": adapt_params_rest, "weight_decay": args.wd},
                    {"params": adapt_params_gain_or_bias, "weight_decay":0},
                    {"params": pass_params_rest, "weight_decay": args.wd, "lr": 1e-5},			    
                    {"params": pass_params_gain_or_bias, "weight_decay":0, "lr": 1e-5},
                ],
			    lr=args.lr,
			    betas=(args.beta1, args.beta2),
			    eps=args.eps,
		        )
            print(optimizer)
        else:
            named_parameters = list(model.named_parameters())
            gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
            rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

            optimizer = optim.AdamW(
                [
                    {"params": gain_or_bias_params, "weight_decay": 0.},
                    {"params": rest_params, "weight_decay": args.wd},
                ],
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
            )

            num_param = sum(p.numel() for p in gain_or_bias_params + rest_params if p.requires_grad)
            num_total_param = sum(p.numel() for p in model.parameters())
            logging.info('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))


        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        scaler = GradScaler() if args.precision == "amp" else None

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            if next(iter(sd.items()))[0].startswith('_orig_mod'):
                sd = {k[len('_orig_mod.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

	# initialize datasets
    with open(args.train_data) as f:
        train_f = json.load(f)
    with open(args.val_data) as f:
        val_f = json.load(f)
    tokenizer = get_tokenizer(args.model)
    data_mean = model.module.visual.image_mean
    data_std = model.module.visual.image_std
    print("image mean {}, std {}".format(data_mean, data_std))
    train_data = CLIPDataset(train_f, args,training=True, 
        tokenizer=tokenizer, data_mean=data_mean, data_std=data_std,
        negative_box=args.use_negative_box)
    val_data = CLIPDataset(val_f, args,training=False, 
        tokenizer=tokenizer, data_mean=data_mean, data_std=data_std)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    val_sampler = None
    print("Training {}, Validating {}".format(len(train_data), len(val_data)))
    train_loader = torch.utils.data.DataLoader(train_data, 
        batch_size=args.batch_size, num_workers=args.workers, 
        shuffle=(train_data is None),pin_memory=False,
        sampler=train_sampler, drop_last=False)
    
    val_loader = torch.utils.data.DataLoader(val_data, 
        batch_size=args.val_batch_size,num_workers=args.workers,
        pin_memory=False, sampler=val_sampler, drop_last=False)

    # create scheduler if train
    scheduler = None
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0, last_epoch=-1)
    #scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = len(train_data)
        if args.val_data is not None:
            args.val_sz = len(val_data)
        # you will have to configure this for your project!
        wandb.init(
            project="RCA-DHPR", 
            entity="hzhang57",
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if args.resume == "latest" else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    if args.torchcompile:
        logging.info('Compiling model...')
        model = torch.compile(model)

    if args.evaluate:
        # Evaluate.
        evaluate_sep(model, val_data, val_loader, val_sampler, start_epoch, args, writer)
        #cal_flops_evaluate(model, val_data, val_loader, val_sampler, start_epoch, args, writer)
        return

    loss = create_loss(args)
    # Compile the model for faster training
    #model = torch.compile(model)
    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        
        train_one_epoch(model, train_data, train_loader, train_sampler, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=writer)
        completed_epoch = epoch + 1

        evaluate(model, val_data, val_loader, val_sampler, completed_epoch, args, writer)

        # Saving checkpoints.
        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)

            if args.save_most_recent:
                # try not to corrupt the latest checkpoint if save fails
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)

    if args.wandb and is_master(args):
        wandb.finish()

    # run a final sync.
    if remote_sync_process is not None:
        logging.info('Final remote sync.')
        remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('Final remote sync successful.')
        else:
            logging.info('Final remote sync failed.')
    

def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main(sys.argv[1:])
