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


REQUEST_INTERVAL = 1
# FILE_LOCK = Lock()
# LOCK = threading.Lock()

class FakeLogger(object):
    def debug(self, msg): pass

    def warning(self, msg): pass

    def error(self, msg): pass

def download_video(url, out_dpath, missing_videos_fpath):
    # Options for youtube_dl
    ydl_opts = {
        'format': 'best',
        'outtmpl': os.path.join(out_dpath, '%(id)s.%(ext)s'),  # Save file as VIDEO_ID.EXTENSION
        'quiet': True,  # Run youtube_dl in quiet mode
        'logger': FakeLogger(),
        'nopart': True,
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
        except youtube_dl.utils.DownloadError as e:
            # Handle the case where the video is not accessible
            video_id = url.split('watch?v=')[-1].split('&')[0]
            log_unavailable_video(video_id, missing_videos_fpath)
            print(f"Video with ID {video_id} is not available. Logged to file.")

def log_unavailable_video(video_id, missing_videos_fpath):
    # with FILE_LOCK:
    with open(missing_videos_fpath, 'a') as file:
        file.write(video_id + '\n')

def download_videos_with_delay(url, missing_videos_fpath):
    url, out_dpath = url
    download_video(url, out_dpath, missing_videos_fpath)
    time.sleep(REQUEST_INTERVAL)  # Wait for the specified interval

def mp_download_videos(dataset_dpath, out_dpath, missing_videos_fpath, checked_videos=[]):
    test_data = read_json(os.path.join(dataset_dpath, 'MSASL_test.json'))
    train_data = read_json(os.path.join(dataset_dpath, 'MSASL_train.json'))
    val_data = read_json(os.path.join(dataset_dpath, 'MSASL_val.json'))

    test_dpath = os.path.join(out_dpath, 'test')
    train_dpath = os.path.join(out_dpath, 'train')
    val_dpath = os.path.join(out_dpath, 'val')
    os.makedirs(test_dpath, exist_ok=True)
    os.makedirs(train_dpath, exist_ok=True)
    os.makedirs(val_dpath, exist_ok=True)
    
    urls = [(d['url'], test_dpath) for d in test_data]
    urls.extend([(d['url'], train_dpath) for d in train_data])
    urls.extend([(d['url'], val_dpath) for d in val_data])
    urls = set(urls)
    # Clean urls if already downloaded
    if checked_videos:
        urls = [url for url in urls if not is_video_downloaded(url, checked_videos)]

    start = time.time()
    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = []
        with tqdm(total=len(urls)) as progress:
            for url in urls:
                future = executor.submit(partial(download_videos_with_delay, missing_videos_fpath=missing_videos_fpath), url)
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)
            
            print('All tasks submited. Waiting until done.')
            wait(futures)
    elapsed = time.time() - start

    print('Done! Took:', elapsed, 'seconds')

def get_downloaded_videos(dpath):
    # Grab downloaded videos
    video_id_list = [os.path.split(video)[-1].split('.mp4')[0] for video in glob.glob(os.path.join(dpath, '*', '*.mp4'))]
    # Grab videos from missing file
    with open(os.path.join(dpath, 'missing.txt'), 'r') as fid: 
        lines = fid.readlines()
    video_id_list.extend([l.rstrip() for l in lines])

    return video_id_list

def is_video_downloaded(url, checked_videos):
    video_id = url[0].split('watch?v=')[-1].split('&')[0]

    return video_id in checked_videos

def video_to_frames(video_path, size=None):
    """
    video_path -> str, path to video.
    size -> (int, int), width, height.
    """
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
    
# def extract_frame_as_video(src_video_path, start_frame, end_frame, video_size):
#     frames = video_to_frames(src_video_path, size=video_size)

#     return frames[start_frame: end_frame]

# def format_data_to_process(train, val, test):
#     train_data, train_dpath = train
#     val_data, val_dpath = val
#     test_data, test_dpath = test
#     data_to_process = []
#     for idx, data in enumerate(train_data):
#         data_to_process.append({
#             'id': idx,
#             'video_id': data['url'].split('watch?v=')[-1].split('&')[0],
#             'start_frame': data['start'],
#             'end_frame': data['end'],
#             'label': data['label'],
#             'signer_id': data['signer_id'],
#             'split': 'train',
#             'skeleton_fpath': os.path.join(train_dpath, f'{idx:0>{5}}.npy'),
#             'width': int(data['width']),
#             'height': int(data['height']),
#         })
#     for idx, data in enumerate(val_data, start=idx + 1):
#         data_to_process.append({
#             'id': idx,
#             'video_id': data['url'].split('watch?v=')[-1].split('&')[0],
#             'start_frame': data['start'],
#             'end_frame': data['end'],
#             'label': data['label'],
#             'signer_id': data['signer_id'],
#             'split': 'val',
#             'skeleton_fpath': os.path.join(val_dpath, f'{idx:0>{5}}.npy'),
#             'width': int(data['width']),
#             'height': int(data['height']),
#         })
#     for idx, data in enumerate(test_data, start=idx + 1):
#         data_to_process.append({
#             'id': idx,
#             'video_id': data['url'].split('watch?v=')[-1].split('&')[0],
#             'start_frame': data['start'],
#             'end_frame': data['end'],
#             'label': data['label'],
#             'signer_id': data['signer_id'],
#             'split': 'test',
#             'skeleton_fpath': os.path.join(test_dpath, f'{idx:0>{5}}.npy'),
#             'width': int(data['width']),
#             'height': int(data['height']),
#         })

#     return data_to_process

def clean_data(data, missing_fpath):
    # Open missing videos
    with open(missing_fpath, 'r') as fid:
        missing = fid.readlines()
    missing = [video_id.rstrip() for video_id in missing]
    clean_data = [d for d in data if d['video_id'] not in missing]

    return clean_data

def extract_kps(video_fpath, out_dpath):
    video_id = os.path.split(video_fpath)[-1].replace('.mp4', '')
    split = os.path.split(os.path.split(video_fpath)[0])[-1]
    out_fpath = os.path.join(out_dpath, split, f'{video_id}.npy')
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
    raw_videos_dpath = os.path.join(dataset_dpath, 'raw_videos')
    os.makedirs(os.path.join(out_dpath, 'test'), exist_ok=True)
    os.makedirs(os.path.join(out_dpath, 'train'), exist_ok=True)
    os.makedirs(os.path.join(out_dpath, 'val'), exist_ok=True)
    videos = glob.glob(os.path.join(raw_videos_dpath, '*', '*.mp4'))
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

def single_gpu_extract_skeleton(dataset_dpath, out_dpath):
    test_data = read_json(os.path.join(dataset_dpath, 'MSASL_test.json'))
    train_data = read_json(os.path.join(dataset_dpath, 'MSASL_train.json'))
    val_data = read_json(os.path.join(dataset_dpath, 'MSASL_val.json'))
    raw_videos_dpath = os.path.join(dataset_dpath, 'raw_videos')
    data_csv_fpath = os.path.join(out_dpath, 'data.csv')
    test_dpath = os.path.join(out_dpath, 'test')
    train_dpath = os.path.join(out_dpath, 'train')
    val_dpath = os.path.join(out_dpath, 'val')
    os.makedirs(test_dpath, exist_ok=True)
    os.makedirs(train_dpath, exist_ok=True)
    os.makedirs(val_dpath, exist_ok=True)
    data = format_data_to_process(
        train=(train_data, train_dpath),
        val=(val_data, val_dpath),
        test=(test_data, test_dpath),
    )
    # Remove those data points without video
    data = clean_data(data, missing_fpath=os.path.join(raw_videos_dpath, 'missing.txt'))

    inferencer = MMPoseInferencer(
        pose2d='rtmpose-l_8xb64-270e_coco-wholebody-256x192', 
        pose2d_weights='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-256x192-6f206314_20230124.pth'
    )

    start = time.time()
    for d in tqdm(data):
        video_fpath = os.path.join(raw_videos_dpath, d['split'], f'{d["video_id"]}.mp4')
        video_size = (d['width'], d['height'])
        start_frame = d['start_frame']
        end_frame = d['end_frame']
        try:
            frames = extract_frame_as_video(video_fpath, start_frame, end_frame, video_size)
        except Exception as e:
            print(f'Error on video {video_fpath}: {str(e)}')
            
        # Extract skeleton data
        try:
            res_gen = inferencer(inputs=frames)
        except Exception as e:
            print(f'Error inference on video {video_fpath}')

        keypoints = []
        scores = []
        for res in res_gen:
            keypoints.append(res['predictions'][0][0]['keypoints'])
            scores.append(res['predictions'][0][0]['keypoint_scores'])
        keypoints = np.array(keypoints)
        scores = np.array(scores)
        skeleton_data = np.dstack((keypoints, scores)).astype(np.float32)

        # Save to disk
        np.save(d['skeleton_fpath'], skeleton_data)

    df = pd.DataFrame(data)
    df.to_csv(data_csv_fpath)
    print('Saved {data_csv_fpath}')
    elapsed = time.time() - start
    print('Done! Took:', elapsed, 'seconds')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_raw', action='store_true', help='download raw videos from YouTube')
    parser.add_argument('--extract_skeleton', action='store_true', help='preprocess the dataset')
    args = parser.parse_args()

    dataset_dpath = '/home/gts/projects/jsoutelo/SignBERT+/datasets/MS-ASL'
    raw_videos_dpath = '/home/gts/projects/jsoutelo/SignBERT+/datasets/MS-ASL/raw_videos'
    missing_videos_fpath = '/home/gts/projects/jsoutelo/SignBERT+/datasets/MS-ASL/raw_videos/missing.txt'
    skeleton_dpath = '/home/gts/projects/jsoutelo/SignBERT+/datasets/MS-ASL/skeleton-data/rtmpose-l_8xb64-270e_coco-wholebody-256x192'

    if args.download_raw:
        response = create_or_clean_directory(raw_videos_dpath)
        if not response:
            checked_videos = list_checked_videos = get_downloaded_videos(raw_videos_dpath)
        else:
            checked_videos = []
        mp_download_videos(dataset_dpath, raw_videos_dpath, missing_videos_fpath, checked_videos)
    elif args.extract_skeleton:
        response = create_or_clean_directory(skeleton_dpath)
        # single_gpu_extract_skeleton(dataset_dpath, skeleton_dpath)
        mp_extract_skeleton(dataset_dpath, skeleton_dpath)
    else:
        print('No argument provided'); exit()

if __name__ == '__main__': 
    mp.set_start_method('spawn', force=True)
    
    main()