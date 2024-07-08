import numpy as np
import torch
import json
import argparse
import open_clip
import random
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip
from torchvision.transforms.functional import InterpolationMode, hflip

from region_prompt_generator import get_region_context_combo,get_region_cpt_combo, get_region_circle_combo

device="cuda"


def get_clip_metrics(image_features, text_features, logit_scale=100):
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



def load_model(args):
    print("1. Create Model {}".format(args.model_name)) 
    model, _, _ = open_clip.create_model_and_transforms(
        args.model_name,
        jit=False, pretrained="openai",
        Adapt_TxtEncoder=args.Adapt_TxtEncoder,
        Adapt_VisEncoder=args.Adapt_VisEncoder,
        adapter_rate=args.adapter_rate,
        ada_type=args.AdaType)
    
    tokenizer   = open_clip.get_tokenizer(args.model_name)

    print("2. Load Checkpoint {}".format(args.model_path))
    state = torch.load(args.model_path, map_location=device)
    state_dict = state['state_dict']
    state_dict = {k.replace('_orig_mod.', '') : v for k, v in state_dict.items()}
    state_dict = {k.replace('module.clip_model.', '') : v for k, v in state_dict.items()}
    state_dict = {k.replace('module.', '') : v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    #print("DEBUG before {}".format(model.state_dict()['transformer.resblocks.11.mlp.c_proj.weight']))
    #print("DEBUG before {}".format(model.state_dict()['transformer.resblocks.4.MLP_Adapter.D_fc1.weight']))
    #model.load_state_dict(state_dict, strict=True)
    #print("DEBUG after {}".format(model.state_dict()['transformer.resblocks.11.mlp.c_proj.weight']))
    #print("DEBUG After {}".format(model.state_dict()['transformer.resblocks.4.MLP_Adapter.D_fc1.weight']))
    #print([name for name, _ in model.named_modules()])

    try:
        input_resolution = model.visual.image_size
    except:
        input_resolution = model.input_resolution

    return model, tokenizer, input_resolution


def get_id_to_bboxes_inference(annot_json):
    with open(annot_json, 'r') as f:
        annots = json.load(f)
    out_dict = {}
    for one_line in annots:
        img_id   = one_line['instance_id']
        img_name = one_line['inputs']['image']["url"]
        txt      = one_line['targets']['inference']
        bboxes   = one_line['inputs']['bboxes']

        out_dict[img_id] ={
            "image": img_name,
            "txt": txt,
            "bboxes": bboxes
        }
    return out_dict


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

class image2rgb(object):
    # Convert image to 
    def __call__(self, image):
        return image.convert("RGB")


def image_to_torch_tensor_v2(im, bb, n_px=336,
        data_mean=(0.48145466, 0.4578275, 0.40821073), 
        data_std=(0.26862954, 0.26130258, 0.27577711)
    ):
    process = transform_test(n_px, data_mean, data_std)
    im = process(im)
    rand_i = random.randrange(len(bb))
    bb = bb[rand_i]
    bb = process(bb)
    return im, bb


def get_img_feat(one_id, model, image_folder, id_dict):
    image_path = image_folder + id_dict[one_id]['image']
    bboxes = id_dict[one_id]['bboxes']

    # Read Image
    o_image = Image.open(image_path)
    im1, bb1 = get_region_context_combo(o_image, bboxes)
    im2, bb2 = get_region_cpt_combo(o_image, bboxes)
    im3, bb3 = get_region_circle_combo(o_image, bboxes)

    im1, bb1 = image_to_torch_tensor_v2(im1, bb1) # C, H, W
    im2, bb2 = image_to_torch_tensor_v2(im2, bb2) # C, H, W
    im3, bb3 = image_to_torch_tensor_v2(im3, bb3) # C, H, W
    im1 = im1.unsqueeze(0).to("cuda")
    im2 = im2.unsqueeze(0).to("cuda")
    im3 = im3.unsqueeze(0).to("cuda")
    bb1 = bb1.unsqueeze(0).to("cuda")
    bb2 = bb2.unsqueeze(0).to("cuda")
    bb3 = bb3.unsqueeze(0).to("cuda")

    im1 = merge_image(bb1, im1, model)
    im2 = merge_image(bb2, im2, model)
    im3 = merge_image(bb3, im3, model)
    #print("DEBUG im1 {}".format(im1.shape))

    im1_feat = model.encode_image(im1)
    im1_feat = im1_feat / im1_feat.norm(dim=-1, keepdim=True)
    im2_feat = model.encode_image(im2)
    im2_feat = im1_feat / im2_feat.norm(dim=-1, keepdim=True)
    im3_feat = model.encode_image(im3)
    im3_feat = im3_feat / im3_feat.norm(dim=-1, keepdim=True)
    #print("DEBUG im1 {}".format(im1_feat.shape))
    #img_feat = model.encode_image(one_id)
    im_feat = (im1_feat + im2_feat + im3_feat) / 3.0
    #im_feat = (im1_feat + im3_feat) / 2.0
    #im_feat = im1_feat
    return im_feat.detach()

def get_txt_feat(one_id, model, tokenizer, id_dict):
    txt = id_dict[one_id]['txt']
    if txt[-1] != ".":
        txt = txt + "."
    txt = "inference {}".format(txt)
    #txt = "{}".format(txt)
    txt_tokens = tokenizer(txt).to(device)
    txt_feat = model.encode_text(txt_tokens)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
    return txt_feat.detach()

def merge_image(bb, im, model):
    if model.visual.image_size[0] > model.visual.image_size[1]:
        combo = torch.cat((bb, im), dim=2) # b,c,h,w        
    else:
        combo = torch.cat((bb, im), dim=3) # b,c,h,w
    return combo

def get_descending_rank(similarity, i):
    sorted_indices = torch.argsort(similarity, descending=True)
    rank_of_i = (sorted_indices == i).nonzero().item()
    print(rank_of_i)
    
    return rank_of_i


def main():
    # araparse to get the arguments
    parser = argparse.ArgumentParser(description='DHPR retrieval')
    parser.add_argument('--model_name', type=str, 
                        default='ViT-L-14-672x336',
                        help='model_name')
    parser.add_argument('--Adapt_TxtEncoder', 
            type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--Adapt_VisEncoder', 
            type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--AdaType', type=int, default=0, 
                        help='0=MLP-A, ATTEN-A, MAP-A,'
                        '1=MLP-A'
                        '2=ATTEN-A'
                        '3=MAP-A'
                        '4=MLP-A, MAP-A'
                        '5=MLP-A, ATTEN-A'
                        '6=ATTEN-A, MAP-A'
                        '7=MLPAT-MSHA'
                        '8=MLP-MSHA'
                        )
    parser.add_argument('--adapter_rate', type=float, default=0.25)
    #parser.add_argument('--model_path', type=str, default='./logs/Gx2-RPA-V220-Mix-TxtVisAdapter-MAA-0.25-ViT-L-14-672x336-LR2e-05-B200-Pamp_bf16-openai-E10-CM-2024_07_05-16_25_09/checkpoints/epoch_10.pt')
    #parser.add_argument('--model_path', type=str, default='./logs/Gx2-RPA-V220-Mix-TxtVisAdapter-MAA-0.25-ViT-L-14-672x336-LR2e-05-B200-Pamp_bf16-openai-E15-CM-2024_07_07-22_10_11/checkpoints/epoch_15.pt')
    parser.add_argument('--model_path', type=str, default='./logs/Gx2-RPA-V220-Mix-TxtVisAdapter-MAA-0.25-ViT-L-14-672x336-LR2e-05-B200-Pamp_bf16-openai-E20-CM-2024_07_08-12_21_26/checkpoints/epoch_20.pt')
    #parser.add_argument('--model_path', type=str, default='./logs/Gx2-RPA-V220-Mix-TxtVisAdapter-MAA-0.25-ViT-L-14-672x336-LR2e-05-B200-Pamp_bf16-openai-E25-CM-2024_07_08-15_45_21/checkpoints/epoch_25.pt')

    parser.add_argument("--test_pair_json", type=str, 
        default='./DHPR_data/annotation/sample_key_pair_test.json')
        #default='./DHPR_data/annotation/sample_key_pair_val.json')
    parser.add_argument("--test_annot_json", type=str, 
        default='./DHPR_data/annotation/anno_test_GEN_NEAT.json')
        #default='./DHPR_data/annotation/anno_val_GEN_NEAT.json')
    parser.add_argument("--image_folder", type=str, 
        default='./DHPR_data/dhpr_image/')

    args = parser.parse_args()


    # 1. Get Model
    model, tokenizer, input_resolution = load_model(args)
    model.to(device)
    model.eval()

    # 0. Annotat
    id_dict = get_id_to_bboxes_inference(args.test_annot_json)
    #print(id_dict)

    vis_all = []
    txt_all = []   
    id_all  = []

    for ii, one_id in enumerate(id_dict):
        print("Processing {} {}".format(ii, one_id))

        img_feat = get_img_feat(one_id, model, 
                        args.image_folder, id_dict)
        txt_feat = get_txt_feat(one_id, model, 
                        tokenizer, id_dict)

        vis_all.append(img_feat)
        txt_all.append(txt_feat)
        id_all.append(one_id)

    vis_all = torch.cat(vis_all, dim=0)
    txt_all = torch.cat(txt_all, dim=0)
    metrics_all = get_clip_metrics(vis_all, txt_all)
    print("Vis {}, Txt {}".format(vis_all.shape, txt_all.shape))
    print("ALL 1000 Metrics {}".format(metrics_all))


    ## Japan test style
    with open(args.test_pair_json, 'r') as f:
        test_pair_dict = json.load(f)
    vis_to_txt_rank_list = []
    txt_to_vis_rank_list = []
    for ii, one_id in enumerate(test_pair_dict.keys()):
        print("Processing {} {}".format(ii, one_id))
        compare_ids = test_pair_dict[one_id]

        #one_index = id_all.index(one_id)
        #one_vis_feat = vis_all[one_index]
        compare_vis_feat = vis_all[np.array([id_all.index(x) for x in compare_ids])]
        one_index_compare = compare_ids.index(one_id)
        one_vis_feat = compare_vis_feat[one_index_compare]

        #one_txt_feat = txt_all[one_index]
        compare_txt_feat = txt_all[np.array([id_all.index(x) for x in compare_ids])]
        one_txt_feat = compare_txt_feat[one_index_compare]

        print("vis query {}, compare {}".format(one_vis_feat.shape, compare_vis_feat.shape))
        print("txt query {}, compare {}".format(one_txt_feat.shape, compare_txt_feat.shape))

        one_vis_to_compare_txt = one_vis_feat @ compare_txt_feat.T
        one_txt_to_compare_vis = one_txt_feat @ compare_vis_feat.T

        v2t_rank = get_descending_rank(one_vis_to_compare_txt.cpu(), one_index_compare)
        t2v_rank = get_descending_rank(one_txt_to_compare_vis.cpu(), one_index_compare)

        vis_to_txt_rank_list.append(v2t_rank)
        txt_to_vis_rank_list.append(t2v_rank)

        print("Vis to Txt Rank {}".format(v2t_rank))
        print("Txt to Vis Rank {}".format(t2v_rank))

    print("Vis to Txt Rank Mean {}".format(np.mean(vis_to_txt_rank_list) +1)) # rank start from 1 (0->1)
    print("Txt to Vis Rank Mean {}".format(np.mean(txt_to_vis_rank_list) +1))

    total = len(vis_to_txt_rank_list)
    top1_match = np.sum(np.array(vis_to_txt_rank_list) == 0)
    top5_match = np.sum(np.array(vis_to_txt_rank_list) < 5)
    print("Vis to Txt Recall@1 {}".format(top1_match / total))
    #print("Vis to Txt Recall@5 {}".format(top5_match / total))

    top1_match = np.sum(np.array(txt_to_vis_rank_list) == 0)
    top5_match = np.sum(np.array(txt_to_vis_rank_list) < 5)
    print("Txt to Vis Recall@1 {}".format(top1_match / total))
    print



    print("print process finished")

if __name__ == "__main__":
    main()
