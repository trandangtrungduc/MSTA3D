import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageOps

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

def merge_image_visualization(folder_path, subset):
    if subset == "val":
        instance_gt = mpimg.imread(folder_path[0])
        instance_pred = mpimg.imread(folder_path[1])
        origin_pc = mpimg.imread(folder_path[2])
        fig, axs = plt.subplots(1, 3, figsize=(20, 10))
        axs[0].imshow(origin_pc)
        axs[0].set_title("Input")
        axs[1].imshow(instance_gt)
        axs[1].set_title("Ground Truth")
        axs[2].imshow(instance_pred)
        axs[2].set_title("Prediction")
        for ax in axs:
            ax.axis("off")
        plt.subplots_adjust(wspace=0.02)
        plt.savefig(f"{folder_path[0][:31]}/{folder_path[0][18:30]}.png", bbox_inches='tight', pad_inches=0)
    elif subset == "test":
        instance_pred = mpimg.imread(folder_path[0])
        origin_pc = mpimg.imread(folder_path[1])
        fig, axs = plt.subplots(1, 2, figsize=(15, 8))
        axs[0].imshow(origin_pc)
        axs[0].set_title("Input")
        axs[1].imshow(instance_pred)
        axs[1].set_title("Prediction")
        for ax in axs:
            ax.axis("off")
        plt.subplots_adjust(wspace=0.02)
        plt.savefig(f"{folder_path[0][:31]}/{folder_path[0][18:31]}.png", bbox_inches='tight', pad_inches=0)
    else:
        raise ValueError("Unknown subset")
    # plt.show()
    plt.close(fig)

if __name__ == "__main__":
    subset = "val" # test
    image_path = f"output/{subset}/images/"
    scene_list = glob.glob(f"{image_path}*")
    scene_list = sorted(list(map(lambda x: x[-12:], scene_list)))
    for scene in scene_list:
        scene_path = f"{scene}"
        folder_path = glob.glob(f"{image_path}{scene_path}/{scene_path}_*.png")
        crop_image_each_scene(folder_path)
        merge_image_visualization(folder_path, subset)
        print(f"Successfully merged {scene_path}.")