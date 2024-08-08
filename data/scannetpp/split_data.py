import os
import shutil
from tqdm import tqdm

splits = ['train', 'val'] # , 'test'

for split in splits:
    original_path = "/mnt/1A3EAE503EAE24AB/Duc/3D_Data/Scannetpp/data/"
    print('processing', split)
    f_name = '/mnt/1A3EAE503EAE24AB/Duc/3D_Data/Scannetpp/splits/nvs_sem_{}.txt'.format(split)
    f = open(f_name, 'r')
    scans = f.readlines()
    os.makedirs(split, exist_ok=True)
    progress_bar = tqdm(total=(len(scans)))
    for scan_name in scans:
        scan = scan_name.strip()
        src = original_path+'{}_ply/scans/mesh_aligned_0.05.ply'.format(scan)
        dest = '{}_ply/{}_mesh_aligned_0.05.ply'.format(split, scan)
        shutil.copyfile(src, dest)
        progress_bar.update()
    progress_bar.close()
print('done')