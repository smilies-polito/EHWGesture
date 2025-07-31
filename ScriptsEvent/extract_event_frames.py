import argparse
import os
import numpy as np
import aedat
import tonic
import cv2
import pandas as pd

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
    trigger_cnt = 0
    triggers_list = []
    trigger_old = 0

    for packet in decoder:
       if "triggers" in packet:
            # print("{} trigger events".format(len(packet["triggers"])))
            # print("Trigger at time:{} is a {}".format( (packet["triggers"]["t"]), (packet["triggers"]["source"]) ) )
            if (packet["triggers"]["source"] == 1):
                flag_sync = 1
                t_octo = np.int64(packet["triggers"]["t"])
                triggers_list.append((packet["triggers"]["t"])[0])
                trigger_cnt += 1
                delta_trigger = np.subtract(packet["triggers"]["t"][0], trigger_old)
                # print ("Delta trigger: ", delta_trigger)
                trigger_old = packet["triggers"]["t"][0]

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
        # breakpoint()
        resized = cv2.resize(cropped.transpose(1,2,0), (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized


def accumulate_events(path_event, path_to_save):

    trigger_times=retrieve_triggers_timestamp(path_event)

    # Load events from the .aedat4 file using tonic
    events = tonic.io.read_aedat4(path_event)
    toskip_mask=np.zeros(len(trigger_times))
    #specify frames to skip
    for f in range(len(trigger_times)):
        if f%OUTPUT_DS!=0:
            toskip_mask[f]=1

    if len(trigger_times) < 2:
        #print("Not enough triggers to accumulate between them.")
        return
    processed=0
    # Process events between each triggers
    for i in range(0,len(trigger_times)):
        if toskip_mask[i] == 1: #this frame is downsampled
            continue
        start_time = trigger_times[i]
        if i==len(trigger_times)-1:
            end_time=start_time+33333 #assuming 30 fps also for accumulation after last trigger
        else:
            end_time = trigger_times[i + 1]

        # Select events within this trigger window
        if end_time>max(events['t']):
            #print("LAST IS TOO SHORT")
            break
        event_window = events[(events['t'] >= start_time) & (events['t'] < end_time)]

        # Convert events to frames at 30 FPS
        frame_interval = end_time-start_time  # 30 FPS → 33.33 ms per frame
        sensor_size=(320, 240, 2)

        transform =  tonic.transforms.Compose([
        tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToImage(sensor_size=sensor_size)])
        frame=crop_event(transform(event_window), EVENT_CROP_REGION, OUTPUT_RESOLUTION_EVENT)

        # remove background noise
        to_kill = (frame[..., 0] != 0) & (frame[..., 1] != 0)
        # Set those pixels to zero in both channels
        frame[to_kill] = [0, 0]
        #print(f"Trigger {i + 1}: Generated  a frame between {start_time} and {end_time} µs")
        os.makedirs(path_to_save, exist_ok=True)
        np.save(os.path.join(path_to_save, f'{processed}.npy'), frame)
        processed+=1
    return






def extract_process_event_frames(subject, side, task):
    subj =  f"X{subject:02}"
    path_event=build_path('event', subj, side, task)[0]

    #create output_path for saving preprocessed frames
    output_paths=build_path('output', subj, side, task)
    output_path_event=output_paths[0]

    accumulate_events(path_event, output_path_event)

    return(f"Task {task} of subject {subject}, side {side} was completed!")




if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate event frames for gesture classification model training with cropping and downsampling"
    )
    parser.add_argument("--src", type=str, default="F:\EHWGesture",
                        help="Source dataset folder containing the videos.")
    parser.add_argument("--dest", type=str, default="dataset_processed2",
                        help="Destination folder to save the processed videos.")
    parser.add_argument("--event_crop", type=parse_crop, default="100,50,250,200",
                        help="Crop coordinates for sub videos as comma-separated ints: left,top,right,bottom")
    parser.add_argument("--down_sample_factor", type=float, default=2,
                        help="Desired downsample to reduce framerate.")
    parser.add_argument("--output_resolution_event", type=int, default=150,
                        help="Output resolution for event (shortest side in pixels).")

    args = parser.parse_args()

    global ROOT_EVENTS, OUT_DIR, EVENT_CROP_REGION, OUTPUT_RESOLUTION_EVENT, OUTPUT_DS

    ROOT_EVENTS = os.path.join(args.src, 'DataEvent')
    OUT_DIR = args.dest
    EVENT_CROP_REGION = args.event_crop
    OUTPUT_RESOLUTION_EVENT = args.output_resolution_event
    OUTPUT_DS = args.down_sample_factor
    for id in range(1,26):
        for side in ['LEFT', 'RIGHT']:
            for task in ['FTS1', 'FTS2', 'FTN1', 'FTN2', 'FTF1', 'FTF2', 'OCS1', 'OCS2', 'OCN1', 'OCN2', 'OCF1', 'OCF2',
                         'TR1', 'TR2', 'NOSE1', 'NOSE2', 'PSS1', 'PSS2', 'PSN1', 'PSN2', 'PSF1','PSF2']:
                extract_process_event_frames(id, side, task)
        print(f"Completed for {id} and {side}")

