import os
import requests
import typing
import io
import IPython
from PIL import Image as PILImage
from vertexai.preview.vision_models import ImageGenerationModel, Image

def upscale_and_save_image(image_path, upscale_factor):
    try:
        # Read the raw bytes of the image
        with open(image_path, 'rb') as f:
            img_data = f.read()

        # Instead of using PIL directly, load the image using the Vertex AI Image class
        try:
            vertex_image = Image(image_bytes=img_data)
        except Exception as e:
            print(f"Error loading image into Vertex Image format: {e}")
            return None

        IPython.display.display(PILImage.open(io.BytesIO(img_data)))
        print("Original Image loaded successfully.")

        # Initialize the model
        model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")

        # Upscale the image
        try:
            upscaled_vertex_image = model.upscale_image(
                image=vertex_image, 
                upscale_factor=upscale_factor
            )
        except Exception as e:
            print(f"Error upscaling image: {e}")
            return None

        # The upscaled image is a Vertex Image. Convert it to PIL for saving locally.
        # According to the Vertex AI doc, you can obtain bytes from the upscaled image:
        upscaled_pil_image = typing.cast(PILImage.Image, upscaled_vertex_image._pil_image)

        # Save the upscaled image
        upscaled_filename = os.path.join(output_folder, f"upscaled_{os.path.basename(image_path)}")
        upscaled_pil_image.save(upscaled_filename)
        print(f"Upscaled image saved to: {upscaled_filename}")

        return upscaled_pil_image

    except FileNotFoundError:
        print(f"File not found: {image_path}")
        return None

    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return None

# Define paths
image_folder = "input"  # Replace with the actual path
output_folder = "upscaled_imagesV2"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define upscale factor
upscale_factor = "x4"

# Get all image paths from the folder
image_paths = [
    os.path.join(image_folder, filename) 
    for filename in os.listdir(image_folder) 
    if os.path.isfile(os.path.join(image_folder, filename))
]

# Process each image in the folder
for image_path in image_paths:
    # Check if it's an image file
    if image_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        upscaled_image = upscale_and_save_image(image_path, upscale_factor)
        if upscaled_image:
            print(f"Image '{image_path}' upscaled and saved to {output_folder}")
