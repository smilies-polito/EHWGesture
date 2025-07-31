import argparse
from doctest import master

from scipy import signal
from os.path import basename
from scipy.interpolate import interp1d
import cv2
import matplotlib
import numpy as np
import mediapipe as mp
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pyk4a import PyK4APlayback
from TriggerDetection.class_metrics import FingerTapping, NoseTapping, OpenClose, PronoSupination
from ScriptsMOCAP.TrackingData import FT, OC, NOSE, PS
from ScriptsMOCAP.groundtruth_for_triggering import parse_marker_mapping, expected_cadence_from_filename
matplotlib.use('TkAgg')
# Constants
center_x, center_y = 1920 // 2, 1080 // 2
PROCESSOR_CLASSES = {'FT': FingerTapping, 'NOSE': NoseTapping, 'OC': OpenClose,
                     'PS': PronoSupination}  # TODO: make these actually make sense, ora sono accrocchi di distanze uscite dal culo
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=5, min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)
hands_utils=mp.solutions.hands
FPS_MOCAP=120
FPS_KIN=30
import pandas as pd
PROCESSOR_CLASSES = {'FT': FingerTapping, 'NOSE': NoseTapping, 'OC': OpenClose, 'PS': PronoSupination} #TODO: make these actually make sense, ora sono accrocchi di distanze uscite dal culo

def load_master_kin_data(file_path):
    # Open the mp4 file with VideoCapture
    cap = cv2.VideoCapture(file_path)
    task = basename(file_path).split('_')[1].split('.')[0]
    processor = PROCESSOR_CLASSES[task[:-1] if 'NOSE' in task else task[:-2]]()
    ref_kin = []

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Convert the frame from BGR to BGRA and then to RGB for mediapipe processing
            color_image = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2RGB)
            results = mp_hands.process(rgb_image)

            if results.multi_hand_landmarks:
                closest_hand = min(
                    results.multi_hand_landmarks,
                    key=lambda hand: np.sqrt(
                        ((min(lm.x for lm in hand.landmark) + max(lm.x for lm in hand.landmark)) / 2 *
                         color_image.shape[1] - center_x) ** 2 +
                        ((min(lm.y for lm in hand.landmark) + max(lm.y for lm in hand.landmark)) / 2 *
                         color_image.shape[0] - center_y) ** 2
                    )
                )
                reference_signal = processor.compute_metrics(closest_hand, color_image.shape)
                ref_kin.append(reference_signal)
        except Exception as e:
            print(f"Error processing frame: {e}")
            break

    cap.release()
    return np.array(ref_kin)

def load_mocap_trace(file_path):
    base_name = os.path.basename(file_path)
    if 'LEFT' in file_path:
        handedness = 'LEFT'
    else:
        handedness = 'RIGHT'
    task = None
    for m in ["FT", "OC", "NOSE", "PS", "TR"]:
        if m in base_name.upper():
            task = m
            break
    if not task:
        raise ValueError("No known metric found in the filename. Expecting one of: FT, OC, NOSE, PS, TR.")

    # For tasks other than TR, get expected cadence in frames.
    exp_cadence = None
    if task not in ["TR", "NOSE"]:
        exp_cadence = expected_cadence_from_filename(base_name)
    else:
        exp_cadence = FPS_MOCAP / 2

    # Read the whole file without skipping header rows.
    try:
        full_df = pd.read_csv(file_path, header=None)
    except:
        print(f"File {file_path} is missing, moving to the next..")
        return
    header_rows = 6
    header_df = full_df.iloc[:header_rows]
    # Data starts after header rows + one extra row (adjust as needed).
    data_df = pd.read_csv(file_path, skiprows=header_rows + 1, header=None)
    # Rename first two columns
    data_df.rename(columns={data_df.columns[0]: "Frame", data_df.columns[1]: "Time"}, inplace=True)
    data_df["Frame"] = pd.to_numeric(data_df["Frame"], errors="coerce")
    data_df["Time"] = pd.to_numeric(data_df["Time"], errors="coerce")
    data_df = data_df.dropna(subset=["Frame", "Time"]).reset_index(drop=True)

    # Use the header from data_df for marker mapping.
    marker_mapping = parse_marker_mapping(header_df)
    # Reassign new column names: first two are Frame and Time, then three columns per marker in the order of marker_mapping keys.
    new_col_names = ['Frame', 'Time']
    for marker in marker_mapping:
        new_col_names.extend([f"{marker}_X", f"{marker}_Y", f"{marker}_Z"])
    data_df.columns = new_col_names

    # Reorganize the DataFrame to a dictionary where each marker name maps to its (X, Y, Z) as numpy arrays.
    tracking_data = {}
    tracking_data['Frame'] = data_df['Frame']
    tracking_data['Time'] = data_df['Time']
    for marker in marker_mapping:
        tracking_data[marker] = data_df[[f"{marker}_X", f"{marker}_Y", f"{marker}_Z"]].values

    # Create and run the task based on filename.
    if task == 'FT':
        obj = FT(tracking_data, base_name.split(".")[0], exp_cadence, handedness)
    elif task == 'OC':
        obj = OC(tracking_data, base_name.split(".")[0], exp_cadence, handedness)
    elif task == 'NOSE':
        # For NOSE, you might still want to use cadence if desired.
        obj = NOSE(tracking_data, base_name.split(".")[0], exp_cadence if exp_cadence else 60, handedness)
    elif task == 'PS':
        obj = PS(tracking_data, base_name.split(".")[0], exp_cadence, handedness)

    obj.compute_ref_signal()
    return obj


def compare_reference_trajectories(obj_mocap, ref_kin):
    # Create time vectors
    t_kin_orig = np.arange(0, (len(ref_kin))*(1/FPS_KIN), 1 / FPS_KIN)
    t_kin_upsampled = np.arange(0, 20.033, 1 / FPS_MOCAP)
    plots=False

    # Upsample Kinect data
    interp_func = interp1d(t_kin_orig, ref_kin, kind='linear', fill_value="extrapolate")
    ref_kin_upsampled = interp_func(t_kin_upsampled)
    if plots:
        # Normalize and plot
        plt.figure(figsize=(10, 4))
        plt.plot(obj_mocap.tracking_data['Time'], obj_mocap.ref_signal / max(obj_mocap.ref_signal), label="Mocap (120 Hz)")
        plt.plot(t_kin_upsampled, ref_kin_upsampled / abs(max(ref_kin_upsampled)), label="Kinect (120 Hz)", linestyle='dashed')
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.title("Two Signals with Same Sampling Rate (120 Hz)")
        plt.show()

    # Compute lag using cross-correlation
    t0 = 2
    ref_mocap_win = obj_mocap.ref_signal[:t0 * FPS_MOCAP] / max(obj_mocap.ref_signal[:t0 * FPS_MOCAP])
    ref_mocap_time = obj_mocap.tracking_data['Time'][:t0 * FPS_MOCAP]
    ref_kin_win = ref_kin_upsampled[:t0 * FPS_MOCAP] / abs(max(ref_kin_upsampled[:t0 * FPS_MOCAP]))

    correlation = signal.correlate(ref_mocap_win - np.mean(ref_mocap_win),
                                   ref_kin_win - np.mean(ref_kin_win), mode="full")
    lags = signal.correlation_lags(len(ref_mocap_win), len(ref_kin_win), mode="full")
    positive_lags = lags[lags >= 0]
    positive_correlation = correlation[lags >= 0]
    lag = positive_lags[np.argmax(abs(positive_correlation))]
    print(lag)

    # Apply lag correction to full signal
    ref_mocap_time_shifted = obj_mocap.tracking_data['Time'][lag:] - obj_mocap.tracking_data['Time'].iloc[lag]
    ref_mocap_signal_shifted = obj_mocap.ref_signal[lag:]
    ref_kin_upsampled_shifted = ref_kin_upsampled[:len(ref_mocap_signal_shifted)]

    # Plot after realignment for full signal
    if plots:
        plt.figure(figsize=(10, 4))
        norm_mocap=ref_mocap_signal_shifted / max(ref_mocap_signal_shifted)
        plt.plot(ref_mocap_time_shifted, norm_mocap-np.mean(norm_mocap), label="Mocap (120 Hz)")
        norm_kin=ref_kin_upsampled_shifted / abs(max(ref_kin_upsampled_shifted))
        plt.plot(t_kin_upsampled[:len(ref_mocap_signal_shifted)],
                 norm_kin-np.mean(norm_kin), label="Kinect (120 Hz)", linestyle='dashed')
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.title("After Realignment Removing Lag - Full Signal")
        plt.show()

    return lag

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compute residual offset between mocap and kinects."
    )
    parser.add_argument("--src", type=str, default="EHWGesture",
                        help="Source dataset folder.")
    parser.add_argument("--dest", type=str, default="output",
                        help="Destination folder to save the processed lags.")

    args = parser.parse_args()
    src_kinect = os.path.join(args.src, 'DataKinects')
    src_mocap = os.path.join(args.src, 'DataMOCAP')
    sbj_list = subj_list = [name for name in os.listdir(src_mocap)
                            if os.path.isdir(os.path.join(src_mocap, name))]
    for sbj in sbj_list:
        mocap_path=os.path.join(src_mocap, sbj)
        for side in ['Left', 'Right']:
            lags = pd.DataFrame(columns=['Task', 'Lag (at 120 fps)'])
            mocap_path_side = os.path.join(mocap_path, side)
            master_kin_path_side = os.path.join(src_kinect, sbj, side, 'rgb')
            out_path_side = os.path.join(args.dest, sbj +'_L' if side == 'Left' else sbj + '_R')

            for task in ['OC', 'NOSE', 'PS', 'FT']:
                if task=='NOSE':
                    try:
                        mocap_path_task = os.path.join(mocap_path_side, task + '1.csv')
                        master_kin_path_task = os.path.join(master_kin_path_side,'Prova_'+task+'1', 'master_' + task + "1.mp4")
                        obj_mocap = load_mocap_trace(mocap_path_task)
                        ref_kin = load_master_kin_data(master_kin_path_task)
                        lag = compare_reference_trajectories(obj_mocap, -ref_kin)
                        lags.loc[len(lags)] = [task + '1', lag]
                    except Exception as e:
                        print(f"Error:{e} while processing {mocap_path}, moving to next...")

                    try:
                        mocap_path_task = os.path.join(mocap_path_side, task + '2.csv')
                        master_kin_path_task = os.path.join(master_kin_path_side,'Prova_'+task+'2', 'master_' + task + "2.mp4")
                        obj_mocap = load_mocap_trace(mocap_path_task)
                        ref_kin = load_master_kin_data(master_kin_path_task)
                        lag = compare_reference_trajectories(obj_mocap, -ref_kin)
                        lags.loc[len(lags)] = [task + '2', lag]
                    except Exception as e:
                        print(f"Error:{e} while processing {mocap_path}, moving to next...")
                else:
                    for cad in ['S1', 'S2', 'N1', 'N2', 'F1', 'F2']:

                        try:
                            mocap_path_task = os.path.join(mocap_path_side,  task + cad + '.csv')
                            master_kin_path_task = os.path.join(master_kin_path_side, 'Prova_'+task+cad, 'master_' + task + cad + ".mp4")
                            obj_mocap = load_mocap_trace(mocap_path_task)
                            ref_kin = load_master_kin_data(master_kin_path_task)
                            lag = compare_reference_trajectories(obj_mocap, ref_kin)
                        except Exception as e:
                            print(f"Error:{e} while processing {mocap_path_task}, moving to next...")
                        lags.loc[len(lags)]=[task+cad, lag]
            filename=sbj+'_L' if side=='Left' else sbj+'_R'
            os.makedirs(out_path_side, exist_ok=True)
            lags.to_csv(os.path.join(out_path_side, f'{filename}_lags.csv'))

