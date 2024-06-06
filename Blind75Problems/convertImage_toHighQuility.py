from PIL import Image, ImageEnhance, ImageFilter

# Load the PNG image
image_path = r'D:/Master/images/effect of time constatn.png'  # Replace with your image path
img = Image.open(image_path)

# Example 1: Enhance image sharpness
sharpness_factor = 2.0  # Adjust the factor as needed
enhancer_sharpness = ImageEnhance.Sharpness(img)
img_sharp = enhancer_sharpness.enhance(sharpness_factor)

# Example 2: Enhance image contrast
contrast_factor = 1.5  # Adjust the factor as needed
enhancer_contrast = ImageEnhance.Contrast(img_sharp)
img_contrast = enhancer_contrast.enhance(contrast_factor)

# Example 3: Enhance image brightness
brightness_factor = 1.2  # Adjust the factor as needed
enhancer_brightness = ImageEnhance.Brightness(img_contrast)
img_brightness = enhancer_brightness.enhance(brightness_factor)

# Example 4: Apply a smoothing filter (to reduce noise)
img_smooth = img_brightness.filter(ImageFilter.SMOOTH)

# Save the enhanced image as PNG with 300 DPI
output_path = 'D:/Master/images/enhanced_image.png'  # Replace with desired output path
img_smooth.save(output_path, dpi=(300, 300))

