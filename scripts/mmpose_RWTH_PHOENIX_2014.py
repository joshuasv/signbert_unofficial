
import os
import glob
import time
from concurrent.futures import ProcessPoolExecutor, wait
from functools import partial

import numpy as np
from tqdm import tqdm
# from mmpose.apis import MMPoseInferencer
from mmpose.apis import inference_topdown, init_model

from IPython import embed; from sys import exit


def extract_kps(fpath):
    # out_fpath = os.path.join(fpath, 'keypoints-score.npy')
    out_fpath = fpath.replace('.png', '.npy')
    result = extract_kps.inference(img=fpath)
    
    # results = [result for result in result_generator]
    # keypoints = []
    # scores = []
    # for r in results:
    #     keypoints.append(r['predictions'][0][0]['keypoints'])
    #     scores.append(r['predictions'][0][0]['keypoint_scores'])
    # keypoints = np.stack(keypoints)
    # scores = np.stack(scores)
    keypoints = result[0].pred_instances.keypoints
    scores = result[0].pred_instances.keypoint_scores
    tosave = np.dstack((keypoints, scores))
    tosave = np.squeeze(tosave, axis=0).astype(np.float32)
    np.save(out_fpath, tosave)

def init_worker(function):
    # inferencer = MMPoseInferencer('rtmpose-l_8xb64-270e_coco-wholebody-256x192', device='cpu')
    # inferencer.inferencer.show_progress = False
    model_cfg = '/home/gts/projects/jsoutelo/SignBERT+/scripts/rtmpose-l_8xb64-270e_coco-wholebody-256x192.py'
    ckpt = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-256x192-6f206314_20230124.pth'
    device = 'cpu'
    model = init_model(model_cfg, ckpt, device)
    function.inference = partial(inference_topdown, model=model)


def main():
    dataset_dpath = '/home/gts/projects/jsoutelo/SignBERT+/datasets/RWTH-PHOENIX-Weather/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px'
    frames_fpath = glob.glob(os.path.join(dataset_dpath, '*', '*', '*', '*.png'))

    # init_worker(extract_kps)
    # for frame in tqdm(frames_fpath):
    #     extract_kps(frame)

    start = time.time()
    with ProcessPoolExecutor(max_workers=os.cpu_count(), initializer=init_worker, initargs=(extract_kps,)) as executor:
        futures = []
        with tqdm(total=len(frames_fpath)) as progress:
            for dpath in frames_fpath:
                future = executor.submit(extract_kps, dpath)
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




       