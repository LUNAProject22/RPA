from PIL import Image, ImageDraw, ImageFont

def add_repeated_text_watermark(input_image_path, output_image_path, text, bboxes, font_path, font_size, angle, word_gap, repeat_interval, image_size=(336, 336)):
    # Load the input image
    image = Image.open(input_image_path)
    image = image.resize(image_size)
    # convert image to RGBA
    image = image.convert('RGBA')
    txt_image = Image.new('RGBA', image.size, (255,255,255,0))
    txt_image_2 = Image.new('RGBA', image.size, (255,255,255,0))
    # Create a drawing context
    draw = ImageDraw.Draw(txt_image)

    # Load the font
    font = ImageFont.truetype(font_path, font_size)

    # Calculate the width and height of the watermark text
    watermark_width, watermark_height = draw.textsize(text, font=font)

    # Calculate the number of times to repeat the watermark in both directions
    image_width, image_height = image.size
    num_repeats_x = image_width // (watermark_width + word_gap) + 1
    num_repeats_y = image_height // (watermark_height + word_gap) + 1

    # Draw the repeated text watermark on the image
    for i in range(num_repeats_x):
        for j in range(num_repeats_y):
            x_position = i * (watermark_width + word_gap)
            y_position = j * (watermark_height + word_gap)
            draw.text((x_position, y_position), text, fill=(255, 0, 0, 185), font=font, angle=angle)

    for one_box in bboxes:
        x0, y0, x1, y1 = one_box
        txt_image_2.paste(txt_image.crop((x0, y0, x1, y1)), (x0, y0))


    image = Image.alpha_composite(image, txt_image_2).convert("RGB")
    # Save the output image
    image.save(output_image_path)

if __name__ == "__main__":
    input_image_path = "input_image.jpg"  # Replace with the path to your input image
    output_image_path = "output_image.jpg"  # Replace with the desired path for the output image
    text = "2 "  # Replace with the desired watermark text
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Replace with the path to your font file (.ttf or .otf)
    font_size = 16  # Replace with the desired font size
    angle = 45  # Replace with the desired rotation angle for the watermark (in degrees)
    word_gap = 0  # Replace with the desired gap between words in the watermark
    repeat_interval = 100  # Replace with the desired interval between repeated watermarks
    alpha = 0.5

    #bboxes = [[0, 0, 100, 100], [100, 100, 200, 200]]
    # bboxes to center
    bboxes = [[500, 500, 800, 900], [100, 100, 200, 200]]

    add_repeated_text_watermark(input_image_path, output_image_path, text, bboxes, font_path, font_size, angle, word_gap, repeat_interval)
