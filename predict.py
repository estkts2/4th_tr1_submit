from glob import glob
import sys
import json
from pathlib import Path
import random
from collections import OrderedDict as Dict
import shutil

import pandas as pd
import numpy as np
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from tqdm.auto import tqdm

FPS = 5
TEAM_ID   = 'U0000000217'
BASE      = '/aichallenge'
TOP_N = 5

MIN_SIZE=30 # 규정은 32px 이지만 안전을 위해 30으로 설정

SAVE_PATH = f'./t1_res_{TEAM_ID}.json'
SAVE_PATH2 = f'{BASE}/t1_res_{TEAM_ID}.json'

no_exception = True

config_file     = f'{BASE}/weights/v4/config.py'
checkpoint_file = f'{BASE}/weights/v4/epoch_27.pth'

def to_frame(img_path, infer_result, conf_th = 0.0):
    
    def nms(r):
        return r[0]
    
    bboxes = nms(infer_result)
    if len(bboxes) == 0:
        return Dict(file_name=Path(img_path).name, box=[])
    
    
    # 필터링 하는 코드 
    bboxes = bboxes[conf_th <= bboxes[:,4]]
    bboxes = bboxes[bboxes[:,4].argsort()][-TOP_N:,:]
    #min_filter
    bboxes_idx = []
    for i, box in enumerate(bboxes):
        x1, y1, x2, y2, c = box
        if MIN_SIZE <= (x2-x1+1) and MIN_SIZE <= (y2-y1+1):
            bboxes_idx.append(i)
    bboxes = bboxes[bboxes_idx]
       
    
    # 형식 변환하는 코드
    def to_box(bbox):
        box  = np.round(bbox[:4]).astype(np.uint).tolist()
        conf = bbox[4]
        return Dict(position=box, confidence_score=str(conf))
        
    boxes = [to_box(bbox) for bbox in bboxes[::-1]] # 혹시나 몰라서 conf가 높은 것을 앞에 적어 줌
    return Dict(file_name=Path(img_path).name, box=boxes)

       
def main():
    data_root = Path(sys.argv[1])
    
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    print('init success')
    print('config_file:', config_file)
    print('checkpoint_file:', checkpoint_file)
    
    stride = int(round(15/FPS))
    total_imgs = sorted(glob(f'{data_root}/*/*.jpg'))
    sample_imgs = total_imgs[::stride]
    print(f'len(total):{len(total_imgs)}')
    print(f'len(sample):{len(sample_imgs)}')
    print(f'top_{TOP_N}, {FPS}fps, min_size:{MIN_SIZE}')
    
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
        frames = sorted(glob(f'{video}/*.jpg'))
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