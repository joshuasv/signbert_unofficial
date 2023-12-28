
import os
import glob
import time
from concurrent.futures import ProcessPoolExecutor, wait
from functools import partial

import numpy as np
from tqdm import tqdm
# from mmpose.apis import MMPoseInferencer
from mmpose.apis import inference_topdown, init_model

from utils import create_or_clean_directory
from IPython import embed


def extract_kps(in_dpath, out_dpath):
    video_id = in_dpath.split('/')[-1]
    split = in_dpath.split('/')[-2]
    out_dpath = os.path.join(out_dpath, split)
    out_fpath = os.path.join(out_dpath, f'{video_id}.npy')
    
    # Ensure split directory exists
    if not os.path.isdir(out_dpath):
        os.makedirs(out_dpath)

    # Grab video frames
    video_frames = sorted(glob.glob(os.path.join(in_dpath, '*.png')))

    # Extract skeleton data
    keypoints = []
    scores = []
    for frame in video_frames:
        result = extract_kps.inference(img=frame)
        keypoints.append(result[0].pred_instances.keypoints[0])
        scores.append(result[0].pred_instances.keypoint_scores[0])
    keypoints = np.stack(keypoints)
    scores = np.stack(scores)

    # Merge keypoints and scores
    skeleton_data = np.dstack((keypoints, scores)).astype(np.float32)

    # Save to disk
    np.save(out_fpath, skeleton_data)

def init_worker(function):
    model_cfg = '/home/gts/projects/jsoutelo/SignBERT+/scripts/rtmpose-l_8xb64-270e_coco-wholebody-256x192.py'
    ckpt = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-256x192-6f206314_20230124.pth'
    device = 'cpu'
    model = init_model(model_cfg, ckpt, device)
    function.inference = partial(inference_topdown, model=model)

def main():
    dataset_dpath = '/home/temporal2/jsoutelo/datasets/PHOENIX-2014-T/features/fullFrame-210x260px/'
    skeleton_dpath = '/home/gts/projects/jsoutelo/SignBERT+/datasets/RWTH-PHOENIX-WeatherT/features/skeleton-fullFrame-210x260px/rtmpose-l_8xb64-270e_coco-wholebody-256x192'

    # Ensure clean output directory
    create_or_clean_directory(skeleton_dpath)

    # Grab all videos
    videos_dpath = glob.glob(os.path.join(dataset_dpath, '*', '*'))

    start = time.time()
    with ProcessPoolExecutor(max_workers=os.cpu_count(), initializer=init_worker, initargs=(extract_kps,)) as executor:
        futures = []
        with tqdm(total=len(videos_dpath)) as progress:
            for dpath in videos_dpath:
                future = executor.submit(partial(extract_kps, out_dpath=skeleton_dpath), dpath)
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)
            print('All tasks submited. Waiting until done.')
            wait(futures)
    elapsed = time.time() - start

    print('Done! Took:', elapsed, 'seconds')

if __name__ == '__main__':

    main()

    
    # for dpath in video_dpaths:
    #     model_cfg = '/home/gts/projects/jsoutelo/SignBERT+/scripts/rtmpose-l_8xb64-270e_coco-wholebody-256x192.py'
    #     ckpt = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-256x192-6f206314_20230124.pth'
    #     device = 'cpu'
    #     embed(); exit()
    #     model = init_model(model_cfg, ckpt, device)
    #     r = inference_topdown()

        # batch_results = inference_topdown(model, )




       