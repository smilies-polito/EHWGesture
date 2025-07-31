import argparse
import os
import re
from itertools import count

import numpy as np
import matplotlib.pyplot as plt
import dv_processing as dv
import aedat
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import json


def extract_info_events(path):
    """Extract subject ID (XNN), body side (LEFT/RIGHT), and label (FT, NOSE, etc.) from the given path."""
    match = re.search(r"[/\\](X\d+\w*)[/\\](X\d+\w*)_(LEFT|RIGHT)[/\\]([\w]+)$", path)
    if match:
        subject = match.group(1)  # Extracts XNN (e.g., X01N)
        body_side = match.group(3)  # Extracts body side (LEFT or RIGHT)
        label = match.group(4)  # Extracts label (e.g., FT or NOSE)
        return subject, body_side, label
    return None, None, None


def process_aedat4(file_path):
    """Process AEDAT4 file to retrieve frames and count events."""
    try:
        reader = dv.AedatFile(file_path)
        frame_list = []
        frame_count = 0

        for event_frame in reader["frames"]:
            frame_list.append(event_frame.image)
            frame_count += 1

        return frame_list, frame_count

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, 0

def find_synchro_triggers(path):
    decoder = aedat.Decoder(path)
    frame_cnt = 0
    trigger_cnt=0
    t_octo= 0
    frame_end = 0
    delta = 0
    flag_sync = 0
    frames_begin_list = []
    triggers_list = []
    delta_frame = 0
    trigger_old = 0
    delta_trigger = 0

    for packet in decoder:
        #print(packet["stream_id"], end=": ")
        if "events" in packet:
            continue
            #print("{} polarity events".format(len(packet["events"])))
        elif "frame" in packet:
            frame_cnt += 1
          #  print("---------------------------------------------------")
          #  print("Timestamp:{} of frame: " .format(packet["frame"]["t"]), frame_cnt)
          #  print("Exposure begins at:{} and ends at:{} " .format(packet["frame"]["exposure_begin_t"], packet["frame"]["exposure_end_t"] ) )
            delta_frame = np.subtract(packet["frame"]["exposure_end_t"],packet["frame"]["exposure_begin_t"])
          #  print("Delta frame: ", delta_frame)
            if(flag_sync):
                frame_end = np.int64(packet["frame"]["exposure_end_t"])
                delta = np.subtract(frame_end,t_octo)
               # print("Sfasamento= ", delta)
                frames_begin_list.append(packet["frame"]["exposure_begin_t"])
            #print("---------------------------------------------------")

        elif "triggers" in packet:
            #print("{} trigger events".format(len(packet["triggers"])))
            #print("Trigger at time:{} is a {}".format( (packet["triggers"]["t"]), (packet["triggers"]["source"]) ) )
            if(packet["triggers"]["source"] == 1 ):
                flag_sync = 1
                t_octo =np.int64( packet["triggers"]["t"])
                triggers_list.append((packet["triggers"]["t"])[0])
                trigger_cnt += 1
                delta_trigger= np.subtract(packet["triggers"]["t"][0], trigger_old)
                #print ("Delta trigger: ", delta_trigger)
                trigger_old = packet["triggers"]["t"][0]

    print(f'Number of synchronization trigger received by event camera: {trigger_cnt}')

    x= 1715260000000000 #removing the offset
    y= 1000000          #converting from microsec to second
    triggers_list_new = [(value - x )/y for value in triggers_list]
    frames_begin_list_new = [(value - x )/y for value in frames_begin_list]

    # plt.vlines(triggers_list_new[:40],ymin=0,ymax=1, label='Triggers rising edge', color='red')
    # plt.vlines(frames_begin_list_new[:40],ymin=0, ymax=2, label='Frames begin', color='blue')
    # plt.legend()
    # plt.show()
    return trigger_cnt

def traverse_and_process_events(root_dir, out_dir):
    """Recursively traverse directories and process AEDAT4 files."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir,"synchronization_triggers_event.csv"), "w") as file_writer:
        file_writer.write('Subject,Side,Task,Triggers\n')
        for root, dirs, files in os.walk(root_dir):
            aedat_files = [f for f in files if f.endswith(".aedat4")]
            for event_recording in aedat_files:
                sbj_id, side, label=extract_info_events(root)
                label=event_recording.split("_")[1].split(".")[0]
                sync_triggers=find_synchro_triggers(os.path.join(root, event_recording))
                file_writer.write(f'{sbj_id},{side},{label},{sync_triggers}\n')
                print(f'{sbj_id},{side},{label},{sync_triggers}')




if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Extract and save the number of synchronization triggers from all event aedat recordings."
    )
    parser.add_argument("--src", type=str, default="EHWGesture",
                        help="Source dataset folder.")
    parser.add_argument("--dest", type=str, default="output",
                        help="Destination folder to save the info about synchronization triggers.")

    args = parser.parse_args()
    root_directory_events = os.path.join(args.src, 'DataEvent')  # Adjust path
    traverse_and_process_events(root_directory_events, args.dest)
