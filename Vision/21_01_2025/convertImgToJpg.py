import os
from PIL import Image, UnidentifiedImageError

def convert_images_to_jpg(directory):
    if not os.path.exists(directory):
        print(f"The directory '{directory}' does not exist.")
        return

    converted_dir = os.path.join(directory, 'AlreadyConverted')
    if not os.path.exists(converted_dir):
        os.makedirs(converted_dir)

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if not os.path.isfile(file_path):
            continue

        if filename.lower().endswith(('.png', '.jpeg', '.bmp', '.gif', '.tiff')):
            try:
                with Image.open(file_path) as img:
                    new_filename = os.path.splitext(filename)[0] + '.jpg'
                    new_img_path = os.path.join(directory, new_filename)
                    img.convert('RGB').save(new_img_path, 'JPEG')

                old_img_path = os.path.join(converted_dir, filename)
                os.rename(file_path, old_img_path)

                print(f"Converted '{filename}' to '{new_filename}'")

            except UnidentifiedImageError:
                print(f"Skipping '{filename}': not a valid image.")
            except Exception as e:
                print(f"Error processing '{filename}': {e}")

if __name__ == "__main__":
    images_directory = 'Images'
    convert_images_to_jpg(images_directory)
