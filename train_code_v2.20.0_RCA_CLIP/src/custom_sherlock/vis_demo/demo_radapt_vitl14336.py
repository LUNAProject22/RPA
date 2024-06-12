from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
from PIL import Image, ImageDraw, ImageFont

import open_clip
import os
import urllib.request
import zipfile
import json
from jupyter_innotater import *
import numpy as np
import tqdm
import random
from pathlib import Path
from open_clip import create_model_and_transforms, get_tokenizer

def write_text_on_white_image(width, height, text):
    # Specify the color of the blanket image
    color = (255, 255, 255)  # white

    # Create a new image of the specified size and color
    blanket_image = Image.new("RGB", (width, height), color)
    # Create a drawing object
    draw = ImageDraw.Draw(blanket_image)

    # Choose a font

    #font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 60, encoding="unic")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 60, encoding="unic")
    #font = ImageFont.load_default(size=36)

    # Choose the text to write
    #text = "Hello, World!"

    # Determine the size of the text
    text_width, text_height = draw.textsize(text, font)

    # Determine where to place the text
    x = (blanket_image.width - text_width) // 2
    y = (blanket_image.height - text_height) // 2

    # Write the text on the image
    draw.text((x, y), text, font=font, fill=(0, 0, 0))
    return blanket_image

### wrapper modeling code
def clip_forward_image(model, image):
    image_features = model.encode_image(image)

    # normalized features                                                                                                                                                                                                                                                         
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features


def clip_forward_text(model, text):
    text_features = model.encode_text(text)
    # normalized features                                                                                                                                                                                                                                                         
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features

# extract the caption features
class CLIPDatasetCaption(torch.utils.data.Dataset):
    def __init__(self, captions, prefix='', tokenizer=None):                                                                                                                                                                                                                               
        self.captions = captions
        self.prefix = prefix
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        #caption = clip.tokenize(self.prefix + '{}'.format(self.captions[idx]), truncate=True).squeeze()
        caption = self.tokenizer(self.prefix + '{}'.format(self.captions[idx])).squeeze()
        return {'caption': caption}

    def __len__(self):
        return len(self.captions)


# functions that highlight bboxes wrapped in a dataloader
class ImageHighlightBboxDataset(torch.utils.data.Dataset):
    def __init__(self, images, resolution):
        '''
        images:
            a list of [{'filepath': str, 'region': [ {'left': int, 'top': int, 'width': int, 'height':int } ]} ]
        resolution:
            the int resolution from CLIP
        '''
        self.images = images
        self.preprocess = self._transform_test(resolution)
    
    @staticmethod
    def highlight_region(image, bboxes):
        image = image.convert('RGBA')
        overlay = Image.new('RGBA', image.size, '#00000000')
        draw = ImageDraw.Draw(overlay, 'RGBA')
        for bbox in bboxes:
            x = bbox['left']
            y = bbox['top']
            draw.rectangle([(x, y), (x+bbox['width'], y+bbox['height'])],
                    fill='#ff05cd3c', outline='#05ff37ff', width=3)
                    #        outline='#05ff37ff', width=3)
        
        image = Image.alpha_composite(image, overlay)
        return image

    @staticmethod
    def highlight_box(image, bboxes):
        image = image.convert('RGBA')
        width_ratio = int(image.width / 100)
        line_width = min(int(2 * width_ratio), 5)
        overlay = Image.new('RGBA', image.size, '#00000000')
        draw = ImageDraw.Draw(overlay, 'RGBA')
        for bbox in bboxes:
            x = bbox['left']
            y = bbox['top']
            draw.rectangle([(x, y), (x+bbox['width'], y+bbox['height'])],
                    outline='red', width=5)
                    #        outline='#05ff37ff', width=3)
        
        image = Image.alpha_composite(image, overlay)
        return image
    @staticmethod
    def highlight_region_circle(image, bboxes):
        image = image.convert('RGBA')
        overlay = Image.new('RGBA', image.size, '#00000000')
        width_ratio = image.width / 100
        line_width = min(int(2 * width_ratio), 5)
        draw = ImageDraw.Draw(image, 'RGBA')
        for bbox in bboxes:
            x = bbox['left']
            y = bbox['top']
            draw.ellipse([x, y, x+bbox['width'], y+bbox['height']], outline='red', width=line_width)
            #draw.rectangle([(x, y), (x+bbox['width'], y+bbox['height'])],
            #                outline='#05ff37ff', width=3)
        
        image = Image.alpha_composite(image, overlay)
        return image


    @staticmethod
    def _transform_test(n_px):
        return Compose([
            Resize((n_px, n_px), interpolation=Image.BICUBIC),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    
    @staticmethod
    def image_to_torch_tensor(image, preprocess_fn):
        width, height = image.size
        image = preprocess_fn(image)
        return image
    
    def __getitem__(self, idx):
        c_data = self.images[idx]
        image = Image.open(c_data)
        image = self.hide_region(image, c_data['bboxes'])
        image = self.image_to_torch_tensor(image, self.preprocess)
        return {'image': image}

    def __len__(self):
        return len(self.images)



#model2sherlock_urlandpath = {
#    'ViT-L-14-672x336':'./rgp_models/GPUx4-Combo-tri-Contrast-v2.7-region_context-ViT-L-14-672x336-LR_1e-05-B_64-P_amp-Pre_openai-E_10-_NO-ClueInf-2023_02_09-10_15_42/checkpoints/epoch_10.pth'
#}

model2sherlock_urlandpath = {
    #'ViT-L-14-672x336':'Gx4-v214-RAug10RGPs+CPT+CiP-AdaType7-TextVisLoc-ViT-L-14-672x336-LR_5e-05-B_200-P_amp_bf16-Pre_openai-E_20-MR_0.25-_NG-2023_05_13-00_22_05/checkpoints/epoch_20.pt'
    'ViT-L-14-672x336': '../logs/Gx2-RPA-V220-Mix-TxtVisAdapter-MAA-0.25-ViT-L-14-672x336-LR2e-05-B200-Pamp_bf16-openai-E10-DCNIPS-2023_08_25-15_18_37/checkpoints/epoch_10.pt'
}
 ## modify these parameters if you want!
np.random.seed(1)
clip_model = 'ViT-L-14-672x336' # you can change this to whichever in the above
batch_size_caption_features, workers_caption_features = 128, 4
#candidate_limit = 1000 # can be "None"
candidate_limit = None
data_directory = 'sherlock_demo_directory'
image_path = 'coop_photo_small.jpg'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(os.getcwd())
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

####---------------------------------------------------------------------------------------------------------------------
 # load up the model + pretrained weights
local_model_path = model2sherlock_urlandpath[clip_model]

#model, preprocess = clip.load(clip_model.replace('-multitask',''), jit=False)
model, _, _ = create_model_and_transforms(
    clip_model,
    pretrained='openai',
    precision='amp',
    device=device,
    Adapt_TxtEncoder=True,
    Adapt_VisEncoder=True,
    adapter_rate=0.25,
    ada_type=0,
) 
try:
    input_resolution = model.visual.image_size
except:
    input_resolution = model.visual.image_size

print('Getting model weights from {}'.format(local_model_path))

state = torch.load(local_model_path, map_location=device)
#print(state['state_dict'].keys())
state['state_dict'] = {k.replace('_orig_mod.module.', '') : v for k, v in state['state_dict'].items()}
model.load_state_dict(state['state_dict'])
model.to(device)
tokenizer = get_tokenizer(clip_model)

####---------------------------------------------------------------------------------------------------------------------
 # download the validation clues/inferences to serve as candidates
data_url = 'https://storage.googleapis.com/ai2-mosaic-public/projects/sherlock/data/sherlock_val_with_split_idxs_v1_1.json.zip'
local_data_path = data_directory + '/sherlock_val_with_split_idxs_v1_1.json.zip'
if not os.path.exists(local_data_path):
    urllib.request.urlretrieve(data_url, local_data_path)
        
archive = zipfile.ZipFile(local_data_path, 'r')
clues, inferences = [], []
with archive.open('sherlock_val_with_split_idxs_v1_1.json',mode='r') as f:
    val = json.load(f)

all_clues = list(set([v['inputs']['clue'] for v in val]))
all_inferences = list(set([v['targets']['inference'] for v in val]))
np.random.shuffle(all_clues)
np.random.shuffle(all_inferences)
if candidate_limit:
    all_clues = all_clues[:candidate_limit]
    all_inferences = all_inferences[:candidate_limit]
print('loaded {} clues and {} inferences'.format(len(all_clues), len(all_inferences)))

####---------------------------------------------------------------------------------------------------------------------
clue_prefix = 'clue '
inference_prefix = 'inference '

clue_iterator = torch.utils.data.DataLoader(
        CLIPDatasetCaption(all_clues, clue_prefix, tokenizer=tokenizer),
        batch_size=batch_size_caption_features, num_workers=workers_caption_features, shuffle=False)

inference_iterator = torch.utils.data.DataLoader(
        CLIPDatasetCaption(all_inferences, inference_prefix, tokenizer=tokenizer),
        batch_size=batch_size_caption_features, num_workers=workers_caption_features, shuffle=False)

all_clue_feats, all_inference_feats = [], []
with torch.no_grad():
    for c in tqdm.tqdm(clue_iterator):
        all_clue_feats.append(clip_forward_text(model, c['caption'].to(device)).cpu().numpy())
with torch.no_grad():
    for c in tqdm.tqdm(inference_iterator):
        all_inference_feats.append(clip_forward_text(model, c['caption'].to(device)).cpu().numpy())
        
all_clue_feats = np.vstack(all_clue_feats)
all_inference_feats = np.vstack(all_inference_feats)
clue_idx2clue = dict(enumerate(all_clues))
inf_idx2inf = dict(enumerate(all_inferences))
print('feature shape: {}, mapping length: {}'.format(all_clue_feats.shape, len(clue_idx2clue)))
####---------------------------------------------------------------------------------------------------------------------

 ## get a bounding box


def url2filepath(url):
    vg_dir='sherlock/images/'
    vcr_dir='sherlock/images/'
    if 'VG_' in url:
        return vg_dir + '/'.join(url.split('/')[-2:])
    else:
        # http://s3-us-west-2.amazonaws.com/ai2-rowanz/vcr1images/lsmdc_3023_DISTRICT_9/3023_DISTRICT_9_01.21.02.808-01.21.16.722@5.jpg
        if 'vcr1images' in vcr_dir:
            return vcr_dir + '/'.join(url.split('/')[-2:])
        else:
            #print(self.args.vcr_dir + '/'.join(url.split('/')[-3:]))
            return vcr_dir + '/'.join(url.split('/')[-3:])

for ii, sig_v in enumerate(val):
    print("Process: {}/{}".format(ii, len(val)))
    #if ii > 1: 
    #    continue
    image_url = sig_v['inputs']['image']['url']
    print(image_url)
    image_path = url2filepath(image_url)
    image_name = Path(image_path).name
    bboxes = sig_v['inputs']['bboxes']
    rand_i = random.randrange(len(bboxes))
    box = bboxes[0]
    gt_clue = sig_v['inputs']['clue']
    gt_inf  = sig_v['targets']['inference']


    targets = np.array([[box['left'], box['top'], box['width'], box['height']]]) # Initialise bounding boxes as x,y = 0,0, width,height = 0,0
    image = Image.open(image_path)
    x1 = box['left']
    y1 = box['top']
    x2 = x1 + box['width']
    y2 = y1 + box['height']
    box_image = image.crop((x1, y1, x2, y2))    

    n_px = min(input_resolution[0], input_resolution[1])
    transform = ImageHighlightBboxDataset._transform_test(n_px)
    image_as_tensor = ImageHighlightBboxDataset.image_to_torch_tensor(image, transform).unsqueeze(0)
    box_as_tensor = ImageHighlightBboxDataset.image_to_torch_tensor(box_image, transform).unsqueeze(0)
    merge_image = torch.cat((box_as_tensor, image_as_tensor), dim=2)


    # RGPs + CPT
    image_with_highlight = ImageHighlightBboxDataset.highlight_region(
        image, [{'left': targets[0, 0], 'top': targets[0, 1], 'width': targets[0, 2], 'height': targets[0, 3]}])
    cpt_tensor = ImageHighlightBboxDataset.image_to_torch_tensor(image_with_highlight, transform).unsqueeze(0)
    merge_image_2 = torch.cat((box_as_tensor, cpt_tensor), dim=2)
    # RGPs + CiP
    image_with_highlight_circle = ImageHighlightBboxDataset.highlight_region_circle(
        image, [{'left': targets[0, 0], 'top': targets[0, 1], 'width': targets[0, 2], 'height': targets[0, 3]}])
    image_with_highlight_box = ImageHighlightBboxDataset.highlight_box(
        image, [{'left': targets[0, 0], 'top': targets[0, 1], 'width': targets[0, 2], 'height': targets[0, 3]}])
    circle_tensor = ImageHighlightBboxDataset.image_to_torch_tensor(image_with_highlight_circle, transform).unsqueeze(0)
    merge_image_3 = torch.cat((box_as_tensor, circle_tensor), dim=2)

    #image_with_highlight.show()
    #transform = ImageHighlightBboxDataset._transform_test(input_resolution)
    #image_as_tensor = ImageHighlightBboxDataset.image_to_torch_tensor(image_with_highlight, transform).unsqueeze(0)
    x = torch.cuda.FloatTensor(256, 4024, 9000)
    with torch.no_grad(): 
        image_feature_vector = clip_forward_image(model, merge_image.to(device)).squeeze().cpu().numpy()
        image_feature_vector2 = clip_forward_image(model, merge_image_2.to(device)).squeeze().cpu().numpy()
        image_feature_vector3 = clip_forward_image(model, merge_image_3.to(device)).squeeze().cpu().numpy()
        image_feature_vector = (image_feature_vector + image_feature_vector2 + image_feature_vector3 ) / 3.0
    image_feature_vector.shape

    clue_vis = -image_feature_vector @ all_clue_feats.transpose()
    infe_vis = -image_feature_vector @ all_inference_feats.transpose()
    top_clue_idxs = np.argsort(clue_vis)[:5]
    top_inf_idxs = np.argsort(infe_vis)[:5] 
    print("GT-CLUE and GT-INFERENCE")
    str1 = "GT-CLUE: {}".format(gt_clue)
    print(str1)
    str2 = "GT-INFERENCE: {}".format(gt_inf)
    print(str2)
    print("**"*6)
    str3 = ""
    #str3 ='Top clues (NOTE: THESE ARE UNRELIABLE EXCEPT FOR THE MULTITASK MODEL!):\n'
    #for c_idx in top_clue_idxs:
    #    str3 += "{}:{:.3f}".format(clue_idx2clue[c_idx], -clue_vis[c_idx]) + "\n"
    str3 += 'Top inferences:' + "\n"
    for c_idx in top_inf_idxs:
        if gt_inf.strip() == inf_idx2inf[c_idx].strip():
            str3 += "FIND!!! {}:{:.3f}".format(inf_idx2inf[c_idx], -infe_vis[c_idx]) + "\n"
        else:
            str3 += "{}:{:.3f}".format(inf_idx2inf[c_idx], -infe_vis[c_idx]) + "\n"

    all_text = str1 + "\n" + str2 + "\n" + str3
    width, height = 3600, 2000
    text_image = write_text_on_white_image(width, height, all_text)

    # create a new image with the combined width and the maximum height
    new_image = Image.new('RGB', (width*2, height))

    # paste the first image onto the new image
    #new_image.paste(image_with_highlight_box.resize((width, height)), (0, 0))
    new_image.paste(image_with_highlight_circle.resize((width, height)), (0, 0))

    # paste the second image next to the first image
    new_image.paste(text_image, (width, 0))

    if "FIND!!!" in str3:
        # Save the merged image
        new_image.save("temp/{}.jpg".format(image_name))

    ## create a new image with the combined width and the maximum height
    #new_image = Image.new('RGB', (width*3, height))

    ## paste the first image onto the new image
    #new_image.paste(box_image.resize((width, height)), (0, 0))
    #new_image.paste(image.resize((width, height)), (width, 0))

    ## paste the second image next to the first image
    #new_image.paste(text_image, (2*width, 0))

    ## Save the merged image
    #new_image.save("temp/{}.jpg".format(image_name))
