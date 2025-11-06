from pathlib import Path

import numpy as np
from PIL import Image

MODULE_PATH = Path(__file__).parents[2]


def combine_covers(image_paths, output_path):
    sorted_im = sorted(image_paths, key=lambda x: x.stat().st_mtime, reverse=False)
    imgs = [Image.open(i) for i in sorted_im]

    # Resize images to a 420px width while maintaining aspect ratio
    all_imgs = [i.resize((420, int(420 * i.height / i.width))) for i in imgs]

    # Paginate images to a 5x5 grid
    for page in range(0, len(all_imgs) // 25 + 1):
        print(f"Combining images {page * 25} to {(page + 1) * 25}...")
        img_files = all_imgs[page * 25 : (page + 1) * 25]
        # get the maximum height of the images
        height = int(np.median([img.height for img in img_files]))

        # Create a new image with a white background
        new_im = Image.new("RGB", (420 * 5, height * 5), (255, 255, 255))

        # Paste images into the new image
        for i, img in enumerate(img_files):
            x = (i % 5) * 420
            # Adjust the height to center the images vertically
            if img.height < height:
                img = img.resize((img.width, height))
            else:
                img = img.crop((0, 0, img.width, height))
            y = (i // 5) * height
            new_im.paste(img, (x, y))

        # Save the combined image
        print(f"Saving combined image {page}...")
        new_im.save(output_path / f"combined_covers_{page}.jpg")


if __name__ == "__main__":
    finished_path = MODULE_PATH / "data" / "covers" / "finished"
    list_im = list((finished_path).rglob("*.jpg"))
    list_im += list((finished_path).rglob("*.png"))
    list_im += list((finished_path).rglob("*.jpeg"))
    combine_covers(list_im, MODULE_PATH / "data" / "covers")
