import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

def hide_region_circle(image, bboxes):

    #highlight mode
	draw = ImageDraw.Draw(image)
	width_ratio = image.width / 100
	line_width = min(int(2 * width_ratio), 5)


	for bbox in bboxes:
		x0 = bbox['left']
		y0 = bbox['top']
		x1 = x0+bbox['width']
		y1 = y0+bbox['height']
       	# highlight mode
		draw.ellipse([x0, y0, x1, y1], outline='red', width=line_width)
		#draw.rectangle([(x, y), (x+bbox['width'], y+bbox['height'])],
        #                   fill='#ff05cd3c', outline='#05ff37ff', width=3)
	    #image = Image.alpha_composite(image, overlay)
		#overlay.close()

	width, height = image.size
	if width >= height:
		im1 = {'height': height, 'width': height, 'left': 0, 'top': 0}
		im2 = {'height': height, 'width': height, 'left': width-height, 'top': 0}
	else:
		im1 = {'height': width, 'width': width, 'left': 0, 'top': 0}
		im2 = {'height': width, 'width': width, 'left': 0, 'top': height-width}
	regions = [image.crop((bbox['left'], bbox['top'], bbox['left'] + bbox['width'], bbox['top'] + bbox['height'])) for bbox in [im1, im2]]

	return regions[0], [ regions[1] ]



def hide_region_rgps_cpt_circle(o_image, bboxes):
	bboxes_image = []
	for bbox in bboxes:
		x = bbox['left']
		y = bbox['top']
		x2 = x+bbox['width']
		y2 = y+bbox['height']
		bbox_image = o_image.crop((x, y, x2, y2))
		bboxes_image.append(bbox_image)

	image= o_image.convert('RGBA')
    #highlight mode
	overlay = Image.new('RGBA', image.size, '#00000000')
	draw = ImageDraw.Draw(overlay, 'RGBA')
	width_ratio = image.width / 100
	line_width = min(int(2 * width_ratio), 5)
	for bbox in bboxes:
		x0 = bbox['left']
		y0 = bbox['top']
		x1 = x0+bbox['width']
		y1 = y0+bbox['height']

    	# highlight mode
		draw.ellipse([x0, y0, x1, y1], 
					fill='#ff05cd3c',
					outline='red', width=line_width)
		#draw.rectangle([(x, y), (x+bbox['width'], y+bbox['height'])],
        #                   fill='#ff05cd3c', outline='#05ff37ff', width=3)
	image = Image.alpha_composite(image, overlay)
	overlay.close()

	return image, bboxes_image


def hide_region_cpt_circle( o_image, bboxes):
	
	image= o_image.convert('RGBA')
    #highlight mode
	overlay = Image.new('RGBA', image.size, '#00000000')
	draw = ImageDraw.Draw(overlay, 'RGBA')
	width_ratio = image.width / 100
	line_width = min(int(2 * width_ratio), 5)
	for bbox in bboxes:
		x0 = bbox['left']
		y0 = bbox['top']
		x1 = x0+bbox['width']
		y1 = y0+bbox['height']

    	# highlight mode
		draw.ellipse([x0, y0, x1, y1], 
				fill='#ff05cd3c',
				outline='red', width=line_width)
		#draw.rectangle([(x, y), (x+bbox['width'], y+bbox['height'])],
  	      #                   fill='#ff05cd3c', outline='#05ff37ff', width=3)
	image = Image.alpha_composite(image, overlay)
	overlay.close()

	width, height = image.size
	if width >= height:
		im1 = {'height': height, 'width': height, 'left': 0, 'top': 0}
		im2 = {'height': height, 'width': height, 'left': width-height, 'top': 0}
	else:
		im1 = {'height': width, 'width': width, 'left': 0, 'top': 0}
		im2 = {'height': width, 'width': width, 'left': 0, 'top': height-width}
	regions = [image.crop((bbox['left'], bbox['top'], bbox['left'] + bbox['width'], bbox['top'] + bbox['height'])) for bbox in [im1, im2]]

	return regions[0], [ regions[1] ]

def hide_region_anchery_target(image, bboxes, num=4):

    #highlight mode
	draw = ImageDraw.Draw(image)
	width_ratio = image.width / 100
	line_width = min(int(2 * width_ratio), 5)


	for bbox in bboxes:
		print("bounding box: ", bbox)
		x0 = bbox['left']
		y0 = bbox['top']
		w = bbox['width']
		h = bbox['height']
		x1 = x0+bbox['width']
		y1 = y0+bbox['height']
		x0s = np.arange(x0, x0+int(w/2), int(int(w/2) / num))
		y0s = np.arange(y0, y0+int(h/2), int(int(h/2) / num))
		ws  = np.arange(w, 0, -1* int(w/num))
		hs  = np.arange(h, 0, -1* int(h/num))
		for x0, y0, w, h in zip(x0s, y0s, ws, hs):
			print("x0, y0, w, h: ", x0, y0, w, h)
			draw.ellipse([x0, y0, x0+w, y0+h], outline='red', width=line_width)

	width, height = image.size
	if width >= height:
		im1 = {'height': height, 'width': height, 'left': 0, 'top': 0}
		im2 = {'height': height, 'width': height, 'left': width-height, 'top': 0}
	else:
		im1 = {'height': width, 'width': width, 'left': 0, 'top': 0}
		im2 = {'height': width, 'width': width, 'left': 0, 'top': height-width}
	regions = [image.crop((bbox['left'], bbox['top'], bbox['left'] + bbox['width'], bbox['top'] + bbox['height'])) for bbox in [im1, im2]]
	return regions[0], [ regions[1] ]


def hide_region_cpt_circle(o_image, bboxes):

	image= o_image.convert('RGBA')
    #highlight mode
	overlay = Image.new('RGBA', image.size, '#00000000')
	draw = ImageDraw.Draw(overlay, 'RGBA')
	width_ratio = image.width / 100
	line_width = min(int(2 * width_ratio), 5)
	for bbox in bboxes:
		x0 = bbox['left']
		y0 = bbox['top']
		x1 = x0+bbox['width']
		y1 = y0+bbox['height']

    	# highlight mode
		draw.ellipse([x0, y0, x1, y1], 
				outline='red', width=line_width)
		#draw.rectangle([(x, y), (x+bbox['width'], y+bbox['height'])],
 	      #                   fill='#ff05cd3c', outline='#05ff37ff', width=3)
	image = Image.alpha_composite(image, overlay)
	overlay.close()

	image2= o_image.convert('RGBA')
    #highlight mode
	overlay2 = Image.new('RGBA', image2.size, '#00000000')
	draw2 = ImageDraw.Draw(overlay2, 'RGBA')
	width_ratio = image2.width / 100
	line_width = min(int(2 * width_ratio), 5)
	for bbox in bboxes:
		x0 = bbox['left']
		y0 = bbox['top']
		x1 = x0+bbox['width']
		y1 = y0+bbox['height']

		draw2.rectangle([(x0, y0), (x1, y1)],
 	                         fill='#ff05cd3c', outline='#05ff37ff', width=3)
	image2 = Image.alpha_composite(image2, overlay2)
	overlay2.close()


	return image, [image2]

# Get Negative Box
def get_mean_size(bboxes):
	area_s = []
	for bbox in bboxes:
		x = bbox['left']
		y = bbox['top']
		w = bbox['width']
		h = bbox['height']
		area = w*h
		area_s.append(area)
	mean_len = np.sqrt(np.mean(area_s))
	return mean_len

def get_negative_box(o_image, bboxes):
	# Crop regions from image
	negative_box = None
	width, height = o_image.size
	mean_len = get_mean_size(bboxes)
	for i in range(20):
		n_x = np.random.randint(0, width-mean_len)
		n_y = np.random.randint(0, height-mean_len)
		n_w = np.random.randint(mean_len//2, mean_len)
		n_h = np.random.randint(mean_len//2, mean_len)

		n_w = min(n_w, width-n_x)
		n_h = min(n_h, height-n_y)
		n_w = max(n_w, 1)
		n_h = max(n_h, 1)

		neg_flag = 1
		for bbox in bboxes:
			x = bbox['left'] 
			y = bbox['top']
			w = bbox['width']
			h = bbox['height']

			iou = computeIoU([n_x, n_y, n_w, n_h], [x, y, w, h])
			if iou > 0.5:
				neg_flag = neg_flag * 0
		if neg_flag == 1:
			negative_box = [n_x, n_y, n_w, n_h]
			break
	if negative_box == None:
		negative_box_img = o_image.crop((0, 0, 5, 5))
		neg_img = random_cpt_circel(o_image, [[0, 0, 5, 5]])
	else:
		negative_box_img = o_image.crop((negative_box[0], negative_box[1], negative_box[0]+negative_box[2], negative_box[1]+negative_box[3]))
		neg_img = random_cpt_circel(o_image, [negative_box])
		#bbox_image = o_image.crop((x, y, x2, y2))
		#bboxes_image.append(bbox_image)

	return neg_img, negative_box_img


def random_cpt_circel(o_image, bboxes):
	image= o_image.convert('RGBA') #highlight mode
	overlay = Image.new('RGBA', image.size, '#00000000')
	draw = ImageDraw.Draw(overlay, 'RGBA')
	width_ratio = image.width / 100
	line_width = min(int(2 * width_ratio), 5)
	for bbox in bboxes:
		x0 = bbox[0]
		y0 = bbox[1]
		x1 = bbox[2] + bbox[0]
		y1 = bbox[3] + bbox[1]
		if np.random.rand() >0.5:
			print('circle')
			draw.ellipse([x0, y0, x1, y1], outline='red', width=line_width)
		else:
			print('rect')
			draw.rectangle([(x0, y0), (x1, y1)],
                              fill='#ff05cd3c', outline='#05ff37ff', width=3)

	image = Image.alpha_composite(image, overlay)
	overlay.close()

	return image


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


def get_color_focus_grey_context(o_image, bboxes):
	bboxes_image = []
	for bbox in bboxes:
		x = bbox['left']
		y = bbox['top']
		x2 = x+bbox['width']
		y2 = y+bbox['height']
		bbox_image = o_image.crop((x, y, x2, y2))
		bboxes_image.append(bbox_image)

	# Create Grey Image
	image = o_image.convert("L")
	image = image.convert("RGB") 
	draw = ImageDraw.Draw(image)
	width_ratio = image.width / 100
	line_width = min(int(2 * width_ratio), 5)
	# Create Focus Region
	for bbox in bboxes:
		x0 = bbox['left']
		y0 = bbox['top']
		x1 = x0+bbox['width']
		y1 = y0+bbox['height']
		image.paste(o_image.crop((x0, y0, x1, y1)), (x0, y0))
		draw.rectangle([(x0, y0), (x1, y1)],
                            outline='red', width=line_width)

	return image, bboxes_image


def get_hp_region_lp_context(o_image, bboxes):
	lp_context = o_image.filter(ImageFilter.GaussianBlur(radius=5))
	hp_context = o_image.filter(ImageFilter.EDGE_ENHANCE_MORE)


	bboxes_image = []
	for bbox in bboxes:
		x = bbox['left']
		y = bbox['top']
		x2 = x+bbox['width']
		y2 = y+bbox['height']
		bbox_image = hp_context.crop((x, y, x2, y2))
		bboxes_image.append(bbox_image)


	width_ratio = o_image.width / 100
	line_width = min(int(2 * width_ratio), 5)

	draw = ImageDraw.Draw(lp_context)
	for bbox in bboxes:
		x0 = bbox['left']
		y0 = bbox['top']
		x1 = x0+bbox['width']
		y1 = y0+bbox['height']
		lp_context.paste(hp_context.crop((x0, y0, x1, y1)), (x0, y0))
		draw.rectangle([(x0, y0), (x1, y1)],
                            outline='red', width=line_width)
	return lp_context, bboxes_image


def write_sentence_on_image(o_image, sentence="", max_word_per_line=5):
	width, height = o_image.size
	word_list = sentence.split()
	len_word  = len(word_list)
	lines = len_word // max_word_per_line + 1
	text_height_bound = height // (lines + 1)
	image = o_image.copy()
	draw = ImageDraw.Draw(image)
	for i in range(lines):
		current_line = word_list[i*max_word_per_line:(i+1)*max_word_per_line]
		current_line = " ".join(current_line)
		# write current line
		# get the size of current line
		font_size =  width // (len(current_line) + 1)
		font_size = min(font_size, text_height_bound)
		#font = ImageFont.truetype("arial.ttf", font_size)
		font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
		text_width, text_height = font.getsize(current_line)
		# get a drawing context
		# draw text
		draw.text((0, i* text_height), current_line, font=font, fill=(255, 0, 0))
	return image



from textwrap import wrap
def get_y_and_heights(text_wrapped, dimensions, margin, font):
    """Get the first vertical coordinate at which to draw text and the height of each line of text"""
    # https://stackoverflow.com/a/46220683/9263761
    ascent, descent = font.getmetrics()

    # Calculate the height needed to draw each line of text (including its bottom margin)
    line_heights = [
        font.getmask(text_line).getbbox()[3] + descent + margin
        for text_line in text_wrapped
    ]
    # The last line doesn't have a bottom margin
    line_heights[-1] -= margin

    # Total height needed
    height_text = sum(line_heights)

    # Calculate the Y coordinate at which to draw the first line of text
    y = (dimensions[1] - height_text) // 2

    # Return the first Y coordinate and a list with the height of each line
    return (y, line_heights)


def write_sentence_on_image_v2(o_image, text="", CHAR_LIMIT=12, V_MARGIN=2, font=None):
	width, height = o_image.size
	fontsize = 1
	font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", fontsize)
	example_text = "A" * CHAR_LIMIT
	while font.getsize(example_text)[0] < 0.8*o_image.size[0]:
    	# iterate until the text size is just larger than the criteria
		fontsize += 1
		font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", fontsize)
	# Wrap the `text` string into a list of `CHAR_LIMIT`-character strings
	text_lines = wrap(text, CHAR_LIMIT)
	# Get the first vertical coordinate at which to draw text and the height of each line of text
	y, line_heights = get_y_and_heights(
	    text_lines,
   		 (width, height),
    	V_MARGIN,
    	font,
	)

	image = o_image.copy()
	draw_interface = ImageDraw.Draw(image)
	# Draw each line of text
	for i, line in enumerate(text_lines):
		# Calculate the horizontally-centered position at which to draw this line
		line_width = font.getmask(line).getbbox()[2]
		x = ((width - line_width) // 2)

	    # Draw this line
		draw_interface.text((x, y), line, font=font, fill=(255, 0, 0))

	    # Move on to the height at which the next line should be drawn at
		y += line_heights[i]
	return image
	

def get_typograhp(clue="", image_size=(224, 224)):
	# Generate Blanket Image
	blanket_image = Image.new("RGB", image_size, (255, 255, 255))

	if clue != "":
		#print(clue, type(clue))
		#print(isinstance(clue, list))
		if isinstance(clue, list):
			ii = np.random.randint(len(clue))
			sig_clue = clue[ii]
		else:
			sig_clue = clue
			#print(sig_clue)
		blanket_image = write_text_on_image(blanket_image, sig_clue)
	return blanket_image


def write_text_on_image(o_image, text="", CHAR_LIMIT=12, V_MARGIN=2, font=None):
	width, height = o_image.size
	fontsize = 1
	font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", fontsize)
	example_text = "A" * CHAR_LIMIT
	while font.getsize(example_text)[0] < 0.98*o_image.size[0]:
    	# iterate until the text size is just larger than the criteria
		fontsize += 1
		font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", fontsize)
	# Wrap the `text` string into a list of `CHAR_LIMIT`-character strings
	text_lines = wrap(text, CHAR_LIMIT)
	# Get the first vertical coordinate at which to draw text and the height of each line of text
	if text_lines == 0:
		print('no text')
	y, line_heights = get_y_and_heights(
	    text_lines,
   		 (width, height),
    	V_MARGIN,
    	font,
	)

	image = o_image.copy()
	draw_interface = ImageDraw.Draw(image)
	# Draw each line of text
	for i, line in enumerate(text_lines):
		# Calculate the horizontally-centered position at which to draw this line
		line_width = font.getmask(line).getbbox()[2]
		x = ((width - line_width) // 2)

	    # Draw this line
		draw_interface.text((x, y), line, font=font, fill="red")

	    # Move on to the height at which the next line should be drawn at
		y += line_heights[i]
	return image


def get_y_and_heights(text_wrapped, dimensions, margin, font):
    """Get the first vertical coordinate at which to draw text and the height of each line of text"""
    # https://stackoverflow.com/a/46220683/9263761
    ascent, descent = font.getmetrics()

    # Calculate the height needed to draw each line of text (including its bottom margin)
    line_heights = [
        font.getmask(text_line).getbbox()[3] + descent + margin
        for text_line in text_wrapped
    ]
    #print(line_heights, text_wrapped)
    # The last line doesn't have a bottom margin
    if len(line_heights) < 1:
        print("empty {}".format(line_heights))
        line_heights.append(margin)
    line_heights[-1] -= margin

    # Total height needed
    height_text = sum(line_heights)

    # Calculate the Y coordinate at which to draw the first line of text
    y = (dimensions[1] - height_text) // 2

    # Return the first Y coordinate and a list with the height of each line
    return (y, line_heights)




# Test Same Augmentation
from randaugment import RandomAugment_V2
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip
from torchvision.transforms.functional import InterpolationMode, hflip
import random

class transform_train(object):
	def __init__(self, n_px, data_mean, data_std) -> None:
		self.resize = Resize((n_px, n_px), interpolation=InterpolationMode.BICUBIC) # Enforce Region or Image to Square
		#self.horizontal_flip = RandomHorizontalFlip()
		self.image2rgb = image2rgb()
		self.rand_aug  = RandomAugment_V2(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate'])
		self.to_tensor = ToTensor()
		self.norm = Normalize(data_mean, data_std)


	def __call__(self, image, mask):
		image = self.resize(image)
		mask  = self.resize(mask)

		if random.random() > 0.5:
			image = hflip(image)
			mask  = hflip(mask)

		image = self.image2rgb(image)
		mask = self.image2rgb(mask)		
		
		print(type(image), type(mask), type(self.rand_aug))
		image, mask = self.rand_aug(image, mask)
		mask = mask / 255.0

		#image = self.to_tensor(image)
		#image = self.norm(image)
		#mask = self.to_tensor(mask)
		#mask = self.norm(mask)

		# Image and Mask are auggmented in the same way
		return image, mask


class image2rgb(object):
	# Convert image to 
	def __call__(self, image):
		return image.convert("RGB")


def get_bbox_mask_image(image_size, bboxes):
	blanket_image = Image.new("RGB", image_size, (0, 0, 0))
	draw = ImageDraw.Draw(blanket_image)
	for bbox in bboxes:
		x0 = bbox['left']
		y0 = bbox['top']
		x1 = x0+bbox['width']
		y1 = y0+bbox['height']
		draw.rectangle([(x0, y0), (x1, y1)], fill=(255, 255, 255))

	return blanket_image

if __name__ == '__main__':
    bboxes = [{'left': 0, 'top': 0, 'width': 100, 'height': 100}, {'left': 100, 'top': 100, 'width': 200, 'height': 400}, {'left': 500, 'top': 500, 'width': 300, 'height': 300}]
    img = Image.open('fig1.jpg')
    width, height = img.size
    image_size = (width, height)
    mask_image = get_bbox_mask_image(image_size, bboxes)

    #img = write_sentence_on_image_v2(img, "hello world, hello world, hello world, hello world, hello world, hello world")
    #img = write_text_on_image(img, "Attention")
    process = transform_train(512, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    im1, im2 = process(img, mask_image)
    print(type(im1), type(im2))
    #print(im1, im2)
    masked_image = im2 * im1
    print(masked_image)
    masked_image = Image.fromarray(np.uint8(masked_image)).convert('RGB')

    #bboxes = [{'left': 0, 'top': 0, 'width': 100, 'height': 100}, {'left': 100, 'top': 100, 'width': 200, 'height': 400}, {'left': 500, 'top': 500, 'width': 300, 'height': 300}]
    #img, boxes = get_hp_region_lp_context(img, bboxes)
    #boxes[2].save('box_out.png')
    masked_image.save('test_out.png')