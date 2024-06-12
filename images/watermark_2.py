from PIL import Image, ImageDraw, ImageFont, ImageOps
import math

def add_repeated_text_watermark(input_image_path, output_image_path, text, font_path, font_size, angle, word_gap, repeat_interval):
    # Load the input image
    image = Image.open(input_image_path)

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Load the font
    font = ImageFont.truetype(font_path, font_size)

    # Calculate the size of the watermark text
    watermark_width, watermark_height = draw.textsize(text, font=font)

    # Calculate the number of times to repeat the watermark in both directions
    image_width, image_height = image.size
    num_repeats_x = image_width // (watermark_width + word_gap) + 1
    num_repeats_y = image_height // (watermark_height + word_gap) + 1

    # Create a rotated version of the watermark text
    rotated_watermark = Image.new("RGBA", (watermark_width, watermark_height), (255, 255, 255, 0))
    draw_rotated = ImageDraw.Draw(rotated_watermark)
    draw_rotated.text((0, 0), text, fill=(255, 255, 255), font=font)

    # Calculate the bounding box of the rotated watermark
    rotated_bbox = rotated_watermark.getbbox()

    # Calculate the rotated watermark's offset to keep it centered
    watermark_offset_x = (watermark_width - rotated_bbox[2]) // 2
    watermark_offset_y = (watermark_height - rotated_bbox[3]) // 2

    # Calculate the maximum dimension to accommodate the rotated watermark
    max_dimension = int(math.sqrt((rotated_bbox[2] - rotated_bbox[0])**2 + (rotated_bbox[3] - rotated_bbox[1])**2))

    # Create a larger canvas for the rotated watermark
    rotated_watermark_canvas = Image.new("RGBA", (max_dimension, max_dimension), (255, 255, 255, 0))
    rotated_watermark_canvas.paste(rotated_watermark, (watermark_offset_x, watermark_offset_y))

    # Rotate the watermark and paste it on the original image
    for i in range(num_repeats_x):
        for j in range(num_repeats_y):
            x_position = i * (watermark_width + word_gap)
            y_position = j * (watermark_height + word_gap)
            rotated_image = rotated_watermark_canvas.rotate(angle, expand=True)
            image.paste(rotated_image, (x_position, y_position), rotated_image)

    # Save the output image
    image.save(output_image_path)


if __name__ == "__main__":
    input_image_path = "input_image.jpg"  # Replace with the path to your input image
    output_image_path = "output_image.jpg"  # Replace with the desired path for the output image
    text = "Your Watermark Text" * 2 # Replace with the desired watermark text
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Replace with the path to your font file (.ttf or .otf)
    font_size = 45  # Replace with the desired font size
    angle = 45  # Replace with the desired rotation angle for the watermark (in degrees)
    word_gap = 5  # Replace with the desired gap between words in the watermark
    repeat_interval = 10  # Replace with the desired interval between repeated watermarks

    add_repeated_text_watermark(input_image_path, output_image_path, text, font_path, font_size, angle, word_gap, repeat_interval)
