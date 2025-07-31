import argparse
import os
import random
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import dv_processing as dv
import aedat
import cv2
import tonic
import matplotlib.pyplot as plt
import numpy as np

heatmap=np.zeros((240,320))

def retrieve_triggers_timestamp(path_event):

    decoder = aedat.Decoder(path_event)
    trigger_cnt = 0
    triggers_list = []
    trigger_old = 0

    for packet in decoder:
       if "triggers" in packet:
            trigger_cnt += 1
            # print("{} trigger events".format(len(packet["triggers"])))
            # print("Trigger at time:{} is a {}".format( (packet["triggers"]["t"]), (packet["triggers"]["source"]) ) )
            if (packet["triggers"]["source"] == 1):
                flag_sync = 1
                t_octo = np.int64(packet["triggers"]["t"])
                triggers_list.append((packet["triggers"]["t"])[0])

                delta_trigger = np.subtract(packet["triggers"]["t"][0], trigger_old)
                # print ("Delta trigger: ", delta_trigger)
                trigger_old = packet["triggers"]["t"][0]

    return triggers_list

def process_events(path_event):
    sensor_size = (320, 240, 2)
    local_heatmap = np.zeros((240, 320))
    trigger_times = retrieve_triggers_timestamp(path_event)

    # Load events from the .aedat4 file using tonic
    events = tonic.io.read_aedat4(path_event)
    transform = tonic.transforms.Denoise(filter_time=10000)
    frames = transform(events)
    for f in frames:
        local_heatmap[f[2]][f[1]]+=1
    return local_heatmap


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Identify ROI in event data."
    )
    parser.add_argument("--src", type=str, default="EHWGesture",
                        help="Source dataset folder containing the videos.")
    args = parser.parse_args()

    root_video_dir = os.path.join(args.src, 'DataEvent')
    video_files = glob.glob(os.path.join(root_video_dir, "**", "*.aedat4"), recursive=True)

    def get_output_path(video_path):
        rel_path = os.path.normpath(video_path)
        if ':' in rel_path:
            rel_path = rel_path.split(':', 1)[-1].lstrip(os.sep)
        base, _ = os.path.splitext(rel_path)
        return os.path.join("traj_output", base + ".txt")

    video_files = [vf for vf in video_files if not os.path.exists(get_output_path(vf))]

    random.shuffle(video_files)

    total_files = len(video_files)
    i=0
    if total_files == 0:
        print("No videos to process (or all outputs already exist).")
    else:
        print(f"Processing {total_files} videos...")
        # for vf in video_files:
        #     print(vf)
        #     process_events(vf)
        #
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(process_events, vf): vf for vf in video_files}

            for future in as_completed(futures):
                video_file = futures[future]
                try:
                    result = future.result()
                    heatmap+=result
                    i+=1
                    print(f'{i} videos completed so far')
                except Exception as exc:
                    print(f"Video {video_file} generated an exception: {exc}")

    print("Processing complete.")
    plt.imshow(heatmap)
    os.makedirs(os.path.join('ScriptsEvents', 'assets'), exist_ok=True)
    plt.savefig(os.path.join('ScriptsEVents', 'assets', 'heatmap_events.png'), dpi=300)