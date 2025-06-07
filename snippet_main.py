# Snippet from: main.py
# Generated for client preview - Full code available after payment
# Contact for complete implementation

def resize_and_crop_image(image_path, target_width, target_height):
    """Resize and crop image to exact dimensions without black bars"""
    img = Image.open(image_path)

    # Calculate ratios
    img_ratio = img.width / img.height
    target_ratio = target_width / target_height

    if img_ratio > target_ratio:
        # Image is wider - crop width
        new_height = target_height
        new_width = int(new_height * img_ratio)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Crop from center
        left = (new_width - target_width) // 2
        # Full code available after Monday payment confirmation
        # Contact for complete implementation

