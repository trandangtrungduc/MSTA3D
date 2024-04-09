from subprocess import call

files = open("data/scannetv2/scannetv2_val.txt", "r")
files = sorted(files.readlines())

COMMAND = "python"
SUBSET_ARG = "--subset"
SUBSET = "val"
VISUALIZATION_TOOL = "tools/visualize.py"
PREDICTION_PATH_ARG = "--prediction_path"
ROOM_NAME_ARG = "--room_name"
TASK_ARG = "--task"
OUTPUT_PATH = f"output/{SUBSET}/"
TASKS =  ["instance_gt"] # "origin_pc", "instance_pred"

for task in TASKS:
    SAVE_PATH = f"output/{SUBSET}/{task}/"
    for room in files:
        room = room[:-1]
        call([COMMAND, VISUALIZATION_TOOL, SUBSET_ARG, SUBSET, PREDICTION_PATH_ARG, OUTPUT_PATH, ROOM_NAME_ARG, room, TASK_ARG, task])