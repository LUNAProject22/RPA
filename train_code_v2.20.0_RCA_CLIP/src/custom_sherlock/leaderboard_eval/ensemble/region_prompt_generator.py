from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np
from textwrap import wrap
import random
from torchvision.transforms import Resize

# RegionCombo (ID=101): Region + Context (R-CTX)
def get_region_context_combo(o_image, bboxes):
	# Crop regions from image
	bboxes_image = []
	for ii, bbox in enumerate(bboxes):
		x = bbox['left']
		y = bbox['top']
		x2 = x+bbox['width']
		y2 = y+bbox['height']
		bbox_image = o_image.crop((x, y, x2, y2))
		bboxes_image.append(bbox_image)

	return o_image, bboxes_image


# RegionCombo (ID=102): Region + CPT (R-CPT)
def get_region_cpt_combo(o_image, bboxes):
	# Crop regions from image
	bboxes_image = []
	for ii, bbox in enumerate(bboxes):
		x = bbox['left']
		y = bbox['top']
		x2 = x+bbox['width']
		y2 = y+bbox['height']
		bbox_image = o_image.crop((x, y, x2, y2))
		bboxes_image.append(bbox_image)

	# Draw Colorful Prompt on image
	image= o_image.convert('RGBA') #highlight mode
	overlay = Image.new('RGBA', image.size, '#00000000')
	draw = ImageDraw.Draw(overlay, 'RGBA')

	#fill_color = gen_fill_random_color(fix)
	#line_color = gen_line_random_color(fix)
	#line_width = gen_random_line_width(fix)

	for bbox in bboxes:
		x = bbox['left']
		y = bbox['top']
		draw.rectangle([(x, y), (x+bbox['width'], y+bbox['height'])],
                              fill='#ff05cd3c', outline='#05ff37ff', width=3)
                              #fill=fill_color, outline=line_color, width=line_width)
	image = Image.alpha_composite(image, overlay)
	overlay.close()

	return image, bboxes_image


# RegionCombo (ID=103): Region + Circle (R-CIR)
def get_region_circle_combo(o_image, bboxes):
	# Crop regions from image
	bboxes_image = []
	for ii, bbox in enumerate(bboxes):
		x = bbox['left']
		y = bbox['top']
		x2 = x+bbox['width']
		y2 = y+bbox['height']
		bbox_image = o_image.crop((x, y, x2, y2))
		bboxes_image.append(bbox_image)

	# Draw Red Cricle on image
	width_ratio = o_image.width / 100
	line_width = min(int(2 * width_ratio), 5)

	image= o_image.convert('RGBA') #highlight mode
	overlay = Image.new('RGBA', image.size, '#00000000')
	draw = ImageDraw.Draw(overlay, 'RGBA')
	for bbox in bboxes:
		x = bbox['left']
		y = bbox['top']
		draw.ellipse([x, y, x+bbox['width'], y+bbox['height']], outline='red', width=line_width)
	image = Image.alpha_composite(image, overlay)
	overlay.close()

	return image, bboxes_image


# RegionCombo (ID-104): color focus, gray context
def get_color_focus_grey_context(o_image, bboxes):
	bboxes_image = []
	for ii, bbox in enumerate(bboxes):
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


# Single (ID=0): Context Only
def get_context_combo(o_image, bboxes):
	width, height = o_image.size
	if width >= height:
		im1 = {'height': height, 'width': height, 'left': 0, 'top': 0}
		im2 = {'height': height, 'width': height, 'left': width-height, 'top': 0}
	else:
		im1 = {'height': width, 'width': width, 'left': 0, 'top': 0}
		im2 = {'height': width, 'width': width, 'left': 0, 'top': height-width}
	regions = [o_image.crop((bbox['left'], bbox['top'], bbox['left'] + bbox['width'], bbox['top'] + bbox['height'])) for bbox in [im1, im2]]
	return regions[0], [ regions[1] ]	


# Single (ID=1): Region Only
def get_region_combo(o_image, bboxes):
	bboxes_image = []
	for bbox in bboxes:
		x = bbox['left']
		y = bbox['top']
		x2 = x+bbox['width']
		y2 = y+bbox['height']
		bbox_image = o_image.crop((x, y, x2, y2))
		bboxes_image.append(bbox_image)

	return bboxes_image[0], bboxes_image


# Single (ID=2): CPT only (CPT)
def get_cpt_combo(o_image, bboxes):
	# Draw Colorful Prompt on image
	image= o_image.convert('RGBA') #highlight mode
	overlay = Image.new('RGBA', image.size, '#00000000')
	draw = ImageDraw.Draw(overlay, 'RGBA')
	for bbox in bboxes:
		x = bbox['left']
		y = bbox['top']
		draw.rectangle([(x, y), (x+bbox['width'], y+bbox['height'])],
                              fill='#ff05cd3c', outline='#05ff37ff', width=3)
	image = Image.alpha_composite(image, overlay)
	overlay.close()

	# Left/Right crop
	width, height = image.size
	if width >= height:
		im1 = {'height': height, 'width': height, 'left': 0, 'top': 0}
		im2 = {'height': height, 'width': height, 'left': width-height, 'top': 0}
	else:
		im1 = {'height': width, 'width': width, 'left': 0, 'top': 0}
		im2 = {'height': width, 'width': width, 'left': 0, 'top': height-width}
	regions = [image.crop((bbox['left'], bbox['top'], bbox['left'] + bbox['width'], bbox['top'] + bbox['height'])) for bbox in [im1, im2]]

	return regions[0], [ regions[1] ]


# Single (ID=3): Cricle only (CIR)
def get_circle_combo(o_image, bboxes):
	# Draw Red Circle Prompt on image
	image= o_image.convert('RGBA') #highlight mode
	overlay = Image.new('RGBA', image.size, '#00000000')
	draw = ImageDraw.Draw(overlay, 'RGBA')
	width_ratio = image.width / 100
	line_width = min(int(2 * width_ratio), 5)
	for bbox in bboxes:
		x0 = bbox['left']
		y0 = bbox['top']
		x1 = x0+bbox['width']
		y1 = y0+bbox['height']
		draw.ellipse([x0, y0, x1, y1], outline='red', width=line_width)
	image = Image.alpha_composite(image, overlay)
	overlay.close()

	# Left/Right crop
	width, height = image.size
	if width >= height:
		im1 = {'height': height, 'width': height, 'left': 0, 'top': 0}
		im2 = {'height': height, 'width': height, 'left': width-height, 'top': 0}
	else:
		im1 = {'height': width, 'width': width, 'left': 0, 'top': 0}
		im2 = {'height': width, 'width': width, 'left': 0, 'top': height-width}
	regions = [image.crop((bbox['left'], bbox['top'], bbox['left'] + bbox['width'], bbox['top'] + bbox['height'])) for bbox in [im1, im2]]

	return regions[0], [ regions[1] ]


# Single (ID=4): color focus, gray context
def get_sig_color_focus_grey_context(o_image, bboxes):

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

	# Left/Right crop
	width, height = image.size
	if width >= height:
		im1 = {'height': height, 'width': height, 'left': 0, 'top': 0}
		im2 = {'height': height, 'width': height, 'left': width-height, 'top': 0}
	else:
		im1 = {'height': width, 'width': width, 'left': 0, 'top': 0}
		im2 = {'height': width, 'width': width, 'left': 0, 'top': height-width}
	regions = [image.crop((bbox['left'], bbox['top'], bbox['left'] + bbox['width'], bbox['top'] + bbox['height'])) for bbox in [im1, im2]]

	return regions[0], [ regions[1] ]


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
	mean_len = int(get_mean_size(bboxes))
	mean_len = min(mean_len, width-20, height-20)
	n_x = np.random.randint(0, width-mean_len)
	n_y = np.random.randint(0, height-mean_len)
	n_w = np.random.randint(mean_len//3, mean_len)
	n_h = np.random.randint(mean_len//3, mean_len)
	negative_box = [n_x, n_y, n_w, n_h]
	#print(mean_len, width, height)
	#for i in range(20):
	#	n_x = np.random.randint(0, width-mean_len)
	#	n_y = np.random.randint(0, height-mean_len)
	#	n_w = np.random.randint(mean_len//3, mean_len)
	#	n_h = np.random.randint(mean_len//3, mean_len)

	#	n_w = min(n_w, width-n_x)
	#	n_h = min(n_h, height-n_y)
	#	n_w = max(n_w, 1)
	#	n_h = max(n_h, 1)

	#	neg_flag = 1
	#	for bbox in bboxes:
	#		x = bbox['left'] 
	#		y = bbox['top']
	#		w = bbox['width']
	#		h = bbox['height']

	#		iou = computeIoU([n_x, n_y, n_w, n_h], [x, y, w, h])
	#		if iou > 0.5:
	#			neg_flag = neg_flag * 0
	#	if neg_flag == 1:
	#		negative_box = [n_x, n_y, n_w, n_h]
	#		break
	if negative_box == None:
		negative_box_img = o_image.crop((0, 0, 5, 5))
		#neg_img = random_cpt_circel(o_image, [[0, 0, 5, 5]])
	else:
		negative_box_img = o_image.crop((negative_box[0], negative_box[1], negative_box[0]+negative_box[2], negative_box[1]+negative_box[3]))
		#neg_img = random_cpt_circel(o_image, [negative_box])
		#bbox_image = o_image.crop((x, y, x2, y2))
		#bboxes_image.append(bbox_image)

	return negative_box_img


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
			#print('circle')
			draw.ellipse([x0, y0, x1, y1], outline='red', width=line_width)
		else:
			#print('rect')
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


def gen_fill_random_color(fix=True):
	if fix:
		color = '#ff05cd3c'
	else:
		color_list = ['#ff05cd', '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF'] # purple, red, green, blue, yellow, Magenta, cyan
		alph_list  = ['3c', '2c', '1c', '0c']
		# generate color with alpha
		color = random.choice(color_list)
		alph = random.choice(alph_list)
		color = color + alph
	return color


def gen_line_random_color(fix=True):
	if fix:
		color = '#05ff37ff'
	else:
		color_list = ['#05ff37', '#ff05cd', '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF'] # purple, red, green, blue, yellow, Magenta, cyan
		# generate color with alpha
		color = random.choice(color_list)
		alph = 'ff'
		color = color + alph
	return color


def gen_random_line_width(fix=True):
	if fix:
		line_width = 5
	else:
		line_width_list = [2, 3, 4, 5, 6, 7, 8]
		line_width = random.choice(line_width_list)
	return line_width


def resize_image(image, size):
	# Resize image to size
	return Resize(size, interpolation=Image.BICUBIC)(image)

# Gen box mask image
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


# Gen Blanket Image
def get_typograhp(clue="", image_size=(224, 224)):
	# Generate Blanket Image
	image_mean = [0.48145466, 0.4578275, 0.40821073]
	image_mean = [int(x * 255) for x in image_mean]
	blanket_image = Image.new("RGB", image_size, tuple(image_mean))

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