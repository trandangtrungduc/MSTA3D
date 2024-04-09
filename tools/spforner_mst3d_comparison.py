import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def merge_image(spformer_path, my_path, result_path, scene):
    spformer = mpimg.imread(spformer_path[2])
    my_model = mpimg.imread(my_path[2])
    ground_truth = mpimg.imread(my_path[1])
    origin_pc = mpimg.imread(my_path[3])

    fig, axs = plt.subplots(1, 4, figsize=(20, 10))
    axs[0].imshow(origin_pc)
    axs[0].set_title("Input", fontdict = {'fontweight': 'bold'})
    axs[1].imshow(ground_truth)
    axs[1].set_title("Ground Truth", fontdict = {'fontweight': 'bold'})
    axs[2].imshow(spformer)
    axs[2].set_title("SPFormer", fontdict = {'fontweight': 'bold'})
    axs[3].imshow(my_model)
    axs[3].set_title("MST3D", fontdict = {'fontweight': 'bold'})
    for ax in axs:
        ax.axis("off")
    plt.subplots_adjust(wspace=0.02)
    plt.savefig(f"{result_path}{scene}.png", bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    plt.close(fig)

if __name__ == "__main__":
    subset = "val"
    spformer_output_path = f"Project/SPFormer/output/{subset}/images/"
    my_output_path = f"output/{subset}/images/"
    scene_list = sorted(os.listdir(my_output_path))
    result_path = f"output/{subset}/spformer_mst3d_comparison/"

    if not os.path.exists(result_path):
            os.mkdir(result_path)

    for scene in scene_list:
        spformer_path = glob.glob(f"{spformer_output_path}{scene}/*.png")
        my_path = glob.glob(f"{my_output_path}{scene}/*.png")
        merge_image(spformer_path, my_path, result_path, scene)
        print(f"Successfully merged {scene}.")
