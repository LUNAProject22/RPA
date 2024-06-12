from PIL import Image, ImageDraw, ImageFont

def add_watermark(image_path, output_path, text, box_position, box_size, font_size, word_gap, transparency, color, font_path):
    # Open the image
    image = Image.open(image_path)

    # Get the size of the image
    width, height = image.size

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Load the font
    font = ImageFont.truetype(font_path, font_size)

    # Set text color with alpha channel (transparency)
    text_color = color + (int(255 * transparency),)

    # Split the text into words
    words = text.split()

    # Initialize variables for text wrapping
    wrapped_lines = []
    current_line = ""
    current_line_width = 0

    for word in words:
        word_width, word_height = draw.textsize(word, font=font)

        if current_line_width + word_width + word_gap <= box_size[0]:
            # Word fits on the current line
            current_line += word + " "
            current_line_width += word_width + word_gap
        else:
            # Word needs to go to the next line
            wrapped_lines.append((current_line, current_line_width))
            current_line = word + " "
            current_line_width = word_width + word_gap

    # Add the last line to the wrapped lines
    wrapped_lines.append((current_line, current_line_width))

    # Calculate the total height of the wrapped text
    total_height = len(wrapped_lines) * (word_height + word_gap) - word_gap

    # Calculate the vertical offset to center the text within the bounding box
    vertical_offset = (box_size[1] - total_height) // 2

    # Draw the wrapped text within the bounding box
    for line, line_width in wrapped_lines:
        x_offset = (box_size[0] - line_width) // 2
        draw.text((box_position[0] + x_offset, box_position[1] + vertical_offset), line, fill=text_color, font=font)
        vertical_offset += word_height + word_gap

    # Save the watermarked image
    image.save(output_path)

if __name__ == "__main__":
    image_path = "input_image.jpg"  # Replace with the path to your image
    output_path = "output_image.jpg"  # Replace with the desired output path
    text = "Your Watermark Text"
    box_position = (100, 100)  # Top-left corner of the bounding box
    box_size = (300, 200)  # Width and height of the bounding box
    font_size = 36
    word_gap = 5
    transparency = 0.5  # 0 (fully transparent) to 1 (fully opaque)
    color = (255, 255, 255)  # RGB color of the text
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

    add_watermark(image_path, output_path, text, box_position, box_size, font_size, word_gap, transparency, color, font_path)
