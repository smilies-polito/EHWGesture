import argparse
import os
import numpy as np
import aedat
import tonic
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

def build_path(mod, *args):
    paths=[]
    subj_folder=args[0]
    if mod=='event':
        side_folder=subj_folder+'_'+args[1]
        task_folder = args[2][:-2] if ('NOSE' not in args[2]  and 'TR' not in args[2])  else args[2][:-1]
        filename=f'dvSave_{args[2]}.aedat4'
        paths.append(os.path.join(ROOT_EVENTS, subj_folder, side_folder, task_folder, filename))
    if mod == 'output':
        side_folder = subj_folder + '_' + args[1][0]
        task_folder = 'Prova_' + args[2]
        camera_event = 'event_' + args[2]
        paths.append(os.path.join(OUT_DIR, side_folder, task_folder, camera_event))
    return paths

def parse_crop(s):
    try:
        parts = [int(x) for x in s.split(',')]
        if len(parts) != 4:
            raise ValueError("Exactly 4 integers required.")
        return parts
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid crop coordinate format: {e}")

def retrieve_triggers_timestamp(path_event):
    try:
        decoder = aedat.Decoder(path_event)
    except Exception as e:
        print(f"file {path_event} not found, skipping")
        return []
    
    triggers_list = []
    for packet in decoder:
        if "triggers" in packet:
            if packet["triggers"]["source"] == 1:
                triggers_list.append(packet["triggers"]["t"][0])
    
    return triggers_list

def crop_event(image, crop_coords, output_resolution):
    left, top, right, bottom = crop_coords
    h, w = image.shape[1:3]
    left = max(0, left)
    top = max(0, top)
    right = min(w, right)
    bottom = min(h, bottom)
    cropped = image[:, top:bottom, left:right]
    if cropped.size == 0:
        raise ValueError("Check crop coordinates.")

    ch, cw = cropped.shape[1:3]
    scale = output_resolution / float(min(ch, cw))
    new_w = int(round((right - left) * scale))
    new_h = int(round((bottom - top) * scale))
    resized = cv2.resize(cropped.transpose(1,2,0), (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized

def accumulate_events(path_event, path_to_save, event_crop_region, output_resolution_event, output_ds):
    trigger_times = retrieve_triggers_timestamp(path_event)
    
    if len(trigger_times) < 2:
        return

    events = tonic.io.read_aedat4(path_event)
    frames_to_process = [i for i in range(len(trigger_times)) if i % output_ds == 0]
    
    if not frames_to_process:
        return

    os.makedirs(path_to_save, exist_ok=True)
    
    sensor_size = (320, 240, 2)
    transform = tonic.transforms.Compose([
        tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToImage(sensor_size=sensor_size)
    ])
    
    max_t = events['t'].max()
    
    for idx, i in enumerate(frames_to_process):
        start_time = trigger_times[i]
        if i == len(trigger_times) - 1:
            end_time = start_time + 33333
        else:
            end_time = trigger_times[i + 1]
        
        if end_time > max_t:
            break
        
        mask = (events['t'] >= start_time) & (events['t'] < end_time)
        event_window = events[mask]
        frame = crop_event(transform(event_window), event_crop_region, output_resolution_event)
        
        to_kill = (frame[..., 0] != 0) & (frame[..., 1] != 0)
        frame[to_kill] = [0, 0]
        np.save(os.path.join(path_to_save, f'{idx}.npy'), frame)

def extract_process_event_frames(subject, side, task, root_events, out_dir, 
                                  event_crop_region, output_resolution_event, output_ds):
    """Worker function for parallel processing"""
    global ROOT_EVENTS, OUT_DIR, EVENT_CROP_REGION, OUTPUT_RESOLUTION_EVENT, OUTPUT_DS
    ROOT_EVENTS = root_events
    OUT_DIR = out_dir
    EVENT_CROP_REGION = event_crop_region
    OUTPUT_RESOLUTION_EVENT = output_resolution_event
    OUTPUT_DS = output_ds
    
    subj = f"X{subject:02}"
    path_event = build_path('event', subj, side, task)[0]
    
    if not os.path.exists(path_event):
        return f"Skipped {task} for subject {subject}, side {side} (file not found)"
    
    output_paths = build_path('output', subj, side, task)
    output_path_event = output_paths[0]
    
    accumulate_events(path_event, output_path_event, event_crop_region, 
                      output_resolution_event, output_ds)
    
    return f"Completed {task} for subject {subject}, side {side}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate event frames for gesture classification model training with cropping and downsampling"
    )
    parser.add_argument("--src", type=str, default="D:/EHWGesture",
                        help="Source dataset folder containing the videos.")
    parser.add_argument("--dest", type=str, default="dataset_preprocessed",
                        help="Destination folder to save the processed videos.")
    parser.add_argument("--event_crop", type=parse_crop, default="100,50,250,200",
                        help="Crop coordinates for sub videos as comma-separated ints: left,top,right,bottom")
    parser.add_argument("--down_sample_factor", type=float, default=2,
                        help="Desired downsample to reduce framerate.")
    parser.add_argument("--output_resolution_event", type=int, default=150,
                        help="Output resolution for event (shortest side in pixels).")
    parser.add_argument("--workers", type=int, default=16,
                        help="Number of parallel workers (default: number of CPU cores)")

    args = parser.parse_args()

    ROOT_EVENTS = os.path.join(args.src, 'DataEvent')
    OUT_DIR = args.dest
    EVENT_CROP_REGION = args.event_crop
    OUTPUT_RESOLUTION_EVENT = args.output_resolution_event
    OUTPUT_DS = args.down_sample_factor

    tasks = []
    for id in range(1, 26):
        for side in ['LEFT', 'RIGHT']:
            for task in ['FTS1', 'FTS2', 'FTN1', 'FTN2', 'FTF1', 'FTF2', 'OCS1', 'OCS2', 
                         'OCN1', 'OCN2', 'OCF1', 'OCF2', 'TR1', 'TR2', 'NOSE1', 'NOSE2', 
                         'PSS1', 'PSS2', 'PSN1', 'PSN2', 'PSF1', 'PSF2']:
                tasks.append((id, side, task))
    
    worker_func = partial(extract_process_event_frames, 
                          root_events=ROOT_EVENTS,
                          out_dir=OUT_DIR,
                          event_crop_region=EVENT_CROP_REGION,
                          output_resolution_event=OUTPUT_RESOLUTION_EVENT,
                          output_ds=OUTPUT_DS)
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker_func, subj, side, task): (subj, side, task) 
                   for subj, side, task in tasks}
        
        with tqdm(total=len(tasks)) as pbar:
            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)
    
    print("All processing complete!")