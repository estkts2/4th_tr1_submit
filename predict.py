from glob import glob
import sys
import json
from pathlib import Path
import random
from collections import OrderedDict as Dict
import shutil

import pandas as pd
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from tqdm.auto import tqdm

TEAM_ID   = 'est_kts2'
BASE      = '/aichallenge'
#DATA_ROOT = '/dataset/4th_track1/'

SAVE_PATH = f'./t1_res_{TEAM_ID}.json'
SAVE_PATH2 = f'{BASE}/t1_res_{TEAM_ID}.json'

no_exception = True

config_file     = f'{BASE}/weights/f0f1f2f3f41f42_sgd_0.00075_warm_1000/vfnet_r2_101_fpn_mdconv_c3-c5_mstrain_2x_coco.py'
checkpoint_file = f'{BASE}/weights/f0f1f2f3f41f42_sgd_0.00075_warm_1000/epoch_14.pth'

def to_frame(img_path, infer_result, conf_th = 0.8):
    bboxes = np.vstack(infer_result)
    if len(bboxes) == 0:
        return Dict(file_name=Path(img_path).name, box=[])
    
    
    bboxes = bboxes[conf_th <= bboxes[:,4]]
    bboxes = bboxes[bboxes[:,4].argsort()][-2:,:]
    
    def to_box(bbox):
        box  = np.round(bbox[:4]).astype(np.uint).tolist()
        conf = bbox[4]
        return Dict(position=box, confidence_score=str(conf))
        
    boxes = [to_box(bbox) for bbox in bboxes]
    return Dict(file_name=Path(img_path).name, box=boxes)

       
def main():
    data_root = Path(sys.argv[1])
    
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    print('init success')
    
    fps = 1
    stride = int(round(15/fps))
    sample_imgs = sorted(glob(f'{data_root}/*/*.jpg'))[::stride]
    print(f'len(sample):{len(sample_imgs)}')
    
    sample_results = []
    
    for idx, img in enumerate(tqdm(sample_imgs)):
        result = inference_detector(model, img, 0)
        sample_results.append(result)
        
    df = pd.DataFrame({'image_path':sample_imgs, 'results':sample_results})
    df = df.sort_values('image_path') 
    df['name'] = df['image_path'].str.split('/', expand=True).values[:,-1]
    results_dict = df[['name', 'results']].set_index('name')['results'].to_dict()
    del df
    
    videos = sorted(glob(f'{data_root}/*'))
    
    frame_results = []
    for video in videos:
        frames = glob(f'{video}/*.jpg')
        prev_result = Dict(file_name='dummy', box=[])
        for f in frames:
            f = Path(f).name
            if f in results_dict:
                prev_result = to_frame(f, results_dict[f])
                frame_results.append(prev_result)
            else:
                t = Dict(file_name=f, box=prev_result['box'])
                frame_results.append(t)
                
    anno = Dict(annotations=frame_results)
         
    try:
        with open(SAVE_PATH, 'w') as f:
            json.dump(anno, f)
        print('success ' if no_exception else 'fail!!', SAVE_PATH)
    except: 
        print('save fail', SAVE_PATH)
        
    try:
        with open(SAVE_PATH2, 'w') as f:
            json.dump(anno, f)
        print('success ' if no_exception else 'fail!!', SAVE_PATH2)
    except:
        print('save fail', SAVE_PATH2)
    
    
if __name__ == '__main__':
    main()