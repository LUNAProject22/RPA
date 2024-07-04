import os
import json
import numpy as np

from region_prompt_generator import get_region_context_combo
from region_prompt_generator import get_region_cpt_combo
from region_prompt_generator import get_region_circle_combo

from PIL import Image, ImageDraw

def main():
    sherlock_folder = "./sherlock/"
    #neat_json = sherlock_folder + "sherlock_train_v1_1_NEAT.json"
    #out_json =  "dataset/train/img_text_pair.json"
    #out_folder = "dataset/train/imgs/"
    neat_json = sherlock_folder + "sherlock_val_with_split_idxs_v1_1_NEAT.json"
    out_json =  "dataset/val/img_text_pair.json"
    out_folder = "dataset/val/imgs/"

    with open(neat_json, "r") as f:
        annots = json.load(f)

    
    out = []
    for i, one_annot in enumerate(annots):
        print("Processing {}/{}".format(i, len(annots)))
        inputs = one_annot["inputs"]
        image  = inputs["image"]["url"]
        bboxes = inputs["bboxes"]
        clue   = inputs["clue"]
        inference = one_annot["targets"]["inference"]

        subs  = image.split("/")
        img_path = "{}/{}".format(subs[-2], subs[-1])
        if "VG_" not in img_path:
            img_path = "vcr1images/" + img_path

        img_path = sherlock_folder + "images/" + img_path
        #print(img_path)
        pil_image = Image.open(img_path)

        img = "{}".format(subs[-1])
        img_box = img.split(".")[0] + "_box.jpg"
        caption = inference
        
        rand_value = np.random.rand()
        if rand_value > 2/3.0:
            pil_img, pil_box = get_region_circle_combo(pil_image, bboxes)
        elif rand_value > 1 / 3.0:
            pil_img, pil_box = get_region_cpt_combo(pil_image, bboxes)
        else:
            pil_img, pil_box = get_region_context_combo(pil_image, bboxes)

        #print(out_folder + img)
        pil_img.convert('RGB').save(out_folder + img)
        pil_box[0].convert('RGB').save(out_folder + img_box)

        new = {
                "img": img,
                "caption": caption,
                "clue":  clue,
                "img_box": img_box
        }

        out.append(new)
    
    # save
    with open(out_json, "w") as f:
        json.dump(out, f, indent=4)


    print("Process Finished")


if __name__ == "__main__":
    main()
