import argparse

import dv_processing as dv
import aedat 
import numpy as np
import matplotlib.pyplot as plt


def extract_aedatfile_info(file_path):

    decoder = aedat.Decoder(file_path)
    print("Dizionario:", decoder.id_to_stream())

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
            print("---------------------------------------------------")
            print("Timestamp:{} of frame: " .format(packet["frame"]["t"]), frame_cnt)
            print("Exposure begins at:{} and ends at:{} " .format(packet["frame"]["exposure_begin_t"], packet["frame"]["exposure_end_t"] ) )
            delta_frame = np.subtract(packet["frame"]["exposure_end_t"],packet["frame"]["exposure_begin_t"])
            print("Delta frame: ", delta_frame)
            if(flag_sync):
                frame_end = np.int64(packet["frame"]["exposure_end_t"])
                delta = np.subtract(frame_end,t_octo)
                print("Sfasamento= ", delta)
                frames_begin_list.append(packet["frame"]["exposure_begin_t"])
            print("---------------------------------------------------")

        elif "triggers" in packet:
            trigger_cnt += 1
            #print("{} trigger events".format(len(packet["triggers"])))
            print("Trigger at time:{} is a {}".format( (packet["triggers"]["t"]), (packet["triggers"]["source"]) ) )
            if(packet["triggers"]["source"] == 1 ):
                flag_sync = 1
                t_octo =np.int64( packet["triggers"]["t"])
                triggers_list.append((packet["triggers"]["t"])[0])

                delta_trigger= np.subtract(packet["triggers"]["t"][0], trigger_old)
                print ("Delta trigger: ", delta_trigger)
                trigger_old = packet["triggers"]["t"][0]

    print(trigger_cnt)
    #print(triggers_list)
    #print(frames_begin_list)

    x= 1715260000000000 #removing the offset
    y= 1000000          #converting from microsec to second
    triggers_list_new = [(value - x )/y for value in triggers_list]
    frames_begin_list_new = [(value - x )/y for value in frames_begin_list]

    plt.vlines(triggers_list_new[:40],ymin=0,ymax=1, label='Triggers rising edge', color='red')
    plt.vlines(frames_begin_list_new[:40],ymin=0, ymax=2, label='Frames begin', color='blue')
    plt.legend()
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Visualize synchronization triggers from event aedat recordings."
    )
    parser.add_argument("--file_path", type=str, default="EHWGesture\DataEvent\X01\X01_LEFT\FT\\dvSave_FTF1.aedat4",
                        help="Path and name of aedat path to open.")
    args = parser.parse_args()
    extract_aedatfile_info(args.file_path)

