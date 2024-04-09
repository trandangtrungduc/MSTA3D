import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageOps
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def crop_image_each_scene(folder_path):
    for img_path in folder_path:
        image = Image.open(img_path)
        image.load()
        imageSize = image.size
        invert_img = image.convert("RGB")
        invert_img = ImageOps.invert(invert_img)
        bbox = invert_img.getbbox()
        cropped_img = image.crop(bbox)
        print(f"Original Size: {imageSize}. New size: {bbox}.")
        cropped_img.save(img_path)

def merge_image(image_path, result_path, scene):
    # print(image_path)
    spp_h = mpimg.imread(image_path[1])
    spp_l = mpimg.imread(image_path[2])
    origin_pc = mpimg.imread(image_path[5])

    fig, axs = plt.subplots(1, 3, figsize=(20, 10))
    axs[0].imshow(origin_pc)
    axs[0].set_title("Input", fontdict = {'fontweight': 'bold'})
    axs[1].imshow(spp_h)
    axs[1].set_title("High-scale Superpoint", fontdict = {'fontweight': 'bold'})
    axs[2].imshow(spp_l)
    axs[2].set_title("Low-scale Superpoint", fontdict = {'fontweight': 'bold'})
    for ax in axs:
        ax.axis("off")
    plt.subplots_adjust(wspace=0.02)
    print(f"{result_path}{scene}.png")
    plt.savefig(f"{result_path}{scene}.png", bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    plt.close(fig)

if __name__ == "__main__":
    subset = "val"
    output_path = f"output/{subset}/images/"
    scene_list = sorted(os.listdir(output_path))
    result_path = f"output/{subset}/input_spp/"

    if not os.path.exists(result_path):
            os.mkdir(result_path)

    for scene in scene_list:
        image_path = glob.glob(f"{output_path}{scene}/*.png")
        crop_image_each_scene(image_path)
        merge_image(image_path, result_path, scene)
        print(f"Successfully merged {scene}.")
