import os
import time
import glob
import argparse
import threading
import multiprocessing as mp
from functools import partial
from multiprocessing import Lock
from concurrent.futures import ProcessPoolExecutor, as_completed, wait

import cv2
import torch
import numpy as np
import pandas as pd
import yt_dlp as youtube_dl
from tqdm import tqdm
from mmpose.apis.inferencers import MMPoseInferencer


from utils import read_json, create_or_clean_directory
from IPython import embed


def video_to_frames(video_path, size=None):
    """
    video_path -> str, path to video.
    size -> (int, int), width, height.
    """
    # with LOCK:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            if size:
                frame = cv2.resize(frame, size)
            frames.append(frame)
        else:
            break
    cap.release()

    return frames

def extract_kps(video_fpath, out_dpath):
    out_fpath = os.path.join(out_dpath, os.path.split(video_fpath)[-1].replace('mp4', 'npy'))
    frames = video_to_frames(video_fpath)
    # Extract skeleton data
    res_gen = extract_kps.inference(inputs=frames)
    keypoints = []
    scores = []
    for res in res_gen:
        keypoints.append(res['predictions'][0][0]['keypoints'])
        scores.append(res['predictions'][0][0]['keypoint_scores'])
    keypoints = np.array(keypoints)
    scores = np.array(scores)
    skeleton_data = np.dstack((keypoints, scores)).astype(np.float32)

    # Save to disk
    np.save(out_fpath, skeleton_data)

def init_worker(function):
    worker_id = os.getpid()
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(worker_id % num_gpus)
    inferencer = MMPoseInferencer(pose2d='rtmpose-l_8xb64-270e_coco-wholebody-256x192', pose2d_weights='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-256x192-6f206314_20230124.pth')
    function.inference = inferencer

def mp_extract_skeleton(dataset_dpath, out_dpath):
    videos = glob.glob(os.path.join(dataset_dpath, '*.mp4'))
    num_gpus = torch.cuda.device_count()
    start = time.time()
    with ProcessPoolExecutor(max_workers=num_gpus, initializer=init_worker, initargs=(extract_kps,)) as executor:
        futures = []
        with tqdm(total=len(videos)) as progress:
            for v in videos:
                future = executor.submit(partial(extract_kps, out_dpath=out_dpath), v)
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)
            print('All tasks submited. Waiting until done.')
            wait(futures)
    elapsed = time.time() - start
    print('Done! Took:', elapsed, 'seconds')

def main():
    dataset_dpath = '/home/gts/projects/jsoutelo/SignBERT+/datasets/WLASL/start_kit/videos'
    skeleton_dpath = '/home/gts/projects/jsoutelo/SignBERT+/datasets/WLASL/start_kit/skeleton-data/rtmpose-l_8xb64-270e_coco-wholebody-256x192'
    create_or_clean_directory(skeleton_dpath)
    mp_extract_skeleton(dataset_dpath, skeleton_dpath)

if __name__ == '__main__': 
    mp.set_start_method('spawn', force=True)
    
    main()