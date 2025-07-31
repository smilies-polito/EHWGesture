import argparse

import cv2
import numpy as np
import mediapipe as mp
import os
import matplotlib.pyplot as plt
import pandas as pd
from numba.np.new_arraymath import return_false
from scipy.signal import find_peaks
from pyk4a import PyK4APlayback
from class_metrics import FingerTapping, NoseTapping, OpenClose, PronoSupination

# Constants
center_x, center_y = 1920 // 2, 1080 // 2
PROCESSOR_CLASSES = {'FT': FingerTapping, 'NOSE': NoseTapping, 'OC': OpenClose, 'PS': PronoSupination}
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=5, min_detection_confidence=0.5, min_tracking_confidence=0.5)


def extract_metrics(folder_path):
    # Determine the correct processor based on the folder name
    folder_key = next((x for x in PROCESSOR_CLASSES if x in folder_path.split('/')[-1]), None)
    processor = PROCESSOR_CLASSES[folder_key]()
    all_metrics = []

    # Loop through files in the folder and process only MP4 files.
    for filename in os.listdir(folder_path):
        if filename.endswith('.mp4'):
            print(f"Processing {filename}...")
            video_path = os.path.join(folder_path, filename)
            cap = cv2.VideoCapture(video_path)
            metrics_per_video = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # End of video

                # frame is in BGR format; convert it to RGB for mediapipe
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = mp_hands.process(rgb_image)

                if results.multi_hand_landmarks:
                    # Ensure that center_x and center_y are defined in your context.
                    closest_hand = min(
                        results.multi_hand_landmarks,
                        key=lambda hand: np.sqrt(
                            ((min(lm.x for lm in hand.landmark) + max(lm.x for lm in hand.landmark)) / 2 * frame.shape[
                                1] - center_x) ** 2 +
                            ((min(lm.y for lm in hand.landmark) + max(lm.y for lm in hand.landmark)) / 2 * frame.shape[
                                0] - center_y) ** 2
                        )
                    )
                    metric = processor.compute_metrics(closest_hand, frame.shape)
                    metrics_per_video.append(metric)

            cap.release()
            all_metrics.append(metrics_per_video)

    return all_metrics, processor


def aggregate_and_plot(all_metrics, processor, smoothing_window=5):
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    for vid_idx, metrics in enumerate(all_metrics):
        if metrics:
            axs[0].plot(range(len(metrics)), metrics, label=f'Modality {vid_idx + 1}')
    axs[0].set_xlabel('Frame')
    axs[0].set_ylabel('Metric')
    axs[0].legend()
    axs[0].set_title('Video Metrics')

    # Aggregate master and sub videos
    aggregated = processor.aggregate_metrics([all_metrics[0], all_metrics[1]])
    smoothed = np.convolve(aggregated, np.ones(smoothing_window)/smoothing_window, mode='same') # 1D conv for smoothing, TODO: avoid magic number for win length
    axs[1].plot(range(len(smoothed)), smoothed, label='Aggregated Metrics')
    axs[1].set_xlabel('Frame')
    axs[1].set_ylabel('Aggregated Metric')
    axs[1].legend()
    axs[1].set_title('Aggregated Metrics')

    # Detect events (local minima)
    minima_indices, _ = find_peaks(-np.array(smoothed))
    for idx in minima_indices:
        axs[1].axvline(x=idx, color='red', linestyle='--')

    plt.tight_layout()
    plt.show()
    
    return minima_indices


def display_results(folder_path, minima_indices):
    # Open a capture for each MP4 file in the folder
    caps = [cv2.VideoCapture(os.path.join(folder_path, f))
            for f in os.listdir(folder_path) if f.endswith('.mp4')]

    while True:
        for cap_id, cap in enumerate(caps):
            try:
                ret, frame = cap.read()
                if not ret:
                    # End display if any video file is done reading frames
                    return

                # Resize and convert the frame (BGR to BGRA)
                resized_frame = cv2.resize(frame, (800, 450))
                rgb_image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2BGRA)

                # Get current frame index to compare with minima_indices
                frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if any(abs(frame_index - idx) <= 1 for idx in minima_indices):
                    cv2.putText(rgb_image, 'EVENT', (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
                                3, (0, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow(f'RGB Data {cap_id}', rgb_image)
                if cv2.waitKey(5) != -1:
                    return
            except Exception as e:
                print(f"Error displaying frame: {e}")
                return

    cv2.destroyAllWindows()


def compare_with_mocap(minima_indices, triggers_mocap, lag_trial):
    triggers_mocap=triggers_mocap[triggers_mocap['Frame'] > lag_trial]

    #reposition triggers removing lag
    triggers_mocap['Frame']-=lag_trial
    triggers_mocap['Time']-=(1/120)*lag_trial
    triggers_mocap=triggers_mocap[triggers_mocap['Time']<=20] #i limit to the part of the recordings in common
    triggers_mocap_frm=triggers_mocap['Frame'].to_numpy()
    triggers_mocap_time= triggers_mocap['Time'].to_numpy()
    minima_indices_astime=minima_indices*0.033

    correct_detections=0
    spurious_detections=0
    spurious_detections_cnt=0
    mae=[]
    for i in range(0,triggers_mocap_time.shape[0]):
        #check if minima fell in a valid position and how many to detect correct detections and spurious detection. Compute mae between closest detection and mocap
        if i!=triggers_mocap_time.shape[0]-1:
            epsilon = (triggers_mocap_time[i + 1] - triggers_mocap_time[i]) / 2
            positions = np.where((minima_indices_astime >= triggers_mocap_time[i]-epsilon) & (minima_indices_astime < (triggers_mocap_time[i+1]- epsilon)))[0]
        else:
            positions = np.where((minima_indices_astime >= triggers_mocap_time[i] - epsilon) & (
                        minima_indices_astime < 20))[0]
        if len(positions)==1:
            #detected!
            correct_detections+=1
            mae.append(abs(triggers_mocap_time[i]-minima_indices_astime[positions[0]]))
        elif len(positions)>1:
            #detected but with spurious detections
            spurious_detections += 1
            spurious_detections_cnt += len(positions) - 1
            correct_detections += 1
            mae.append(abs(triggers_mocap_time[i]-minima_indices_astime[positions[0]]))

    return mae, correct_detections, spurious_detections, spurious_detections_cnt, triggers_mocap_time.shape[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Performs trigger detection on RGB.")
    parser.add_argument("--input_folder", default='EHWGesture', type=str,
                        help="Path to root of the EHWGesture dataset.")
    parser.add_argument("--output_folder", default='output', type=str,
                        help="Path to output folder to save color frames.")
    parser.add_argument("--smoothing_window", default=5, type=str,
                        help="Smoothing window size for 1DConv")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    smoothing_window = args.smoothing_window

    categories_ft = ['FTS', 'FTN', 'FTF']
    categories_oc = ['OCS', 'OCN', 'OCF']
    categories_ps = ['PSS', 'PSN', 'PSF']
    categories_nose = ['NOSE']

    all_categories = categories_ft + categories_oc + categories_ps + categories_nose

    correct_detections = {key: [] for key in all_categories}
    ground_truths = {key: [] for key in all_categories}
    spurious_detections = {key: [] for key in all_categories}
    spurious_detections_cnt = {key: [] for key in all_categories}
    mae = {key: [] for key in all_categories}


    for sbj in ['X01', 'X05', 'X08', 'X17', 'X25']: #loop only on test subjects

        for side in ['Left', 'Right']:
            sbj_side=sbj + '_L' if side == 'Left' else sbj + '_R'
            lags = pd.read_csv(os.path.join(args.input_folder, 'DataKinects', sbj, side, sbj_side + '_lags.csv'))
            for task in ['PS', 'FT', 'NOSE', 'OC']:
                if task == 'NOSE':
                    try:
                        path_task = os.path.join(input_folder, 'DataKinects', sbj, side, 'rgb',
                                                            'Prova_' + task + '1')
                        all_metrics, processor = extract_metrics(path_task)
                        print(f'{path_task} is being processed')
                        # load information about opto-kinect lag
                        lag_trial = lags[lags['Task'] == 'NOSE1']['Lag (at 120 fps)'].values[0]
                        # load triggers obtained from opto
                        triggers_mocap = pd.read_csv(os.path.join(input_folder, 'Annotations', 'GestureTriggers', sbj, side, 'NOSE1', 'NOSE1_triggers.csv'))
                        triggers_mocap = triggers_mocap[triggers_mocap['Gesture'] == 'outward']

                        minima_indices = aggregate_and_plot(all_metrics, processor,
                                                            smoothing_window=smoothing_window)

                        mae_inner, correct_detections_inner, spurious_det_inner, spurious_det_count_inner, gt = compare_with_mocap(minima_indices, triggers_mocap,lag_trial)
                        mae['NOSE'].append(mae_inner)
                        correct_detections['NOSE'].append(correct_detections_inner)
                        spurious_detections['NOSE'].append(spurious_det_inner)
                        spurious_detections_cnt['NOSE'].append(spurious_det_count_inner)
                        ground_truths['NOSE'].append(gt)

                    except Exception as e:
                        print(f"Error:{e} while processing {path_task}, moving to next...")

                    try:
                        path_task = os.path.join(input_folder, 'DataKinects', sbj, side, 'rgb',
                                                            'Prova_' + task + '2')
                        print(f'{path_task} is being processed')
                        all_metrics, processor = extract_metrics(path_task)
                        minima_indices = aggregate_and_plot(all_metrics, processor,
                                                            smoothing_window=smoothing_window)
                        # load information about residual opto-kinect lag
                        lag_trial = lags[lags['Task'] == 'NOSE2']['Lag (at 120 fps)'].values[0]

                        # load triggers obtained from opto
                        triggers_mocap = pd.read_csv(os.path.join(input_folder, 'Annotations', 'GestureTriggers', sbj,
                                         side, 'NOSE2', 'NOSE2_triggers.csv'))

                        triggers_mocap=triggers_mocap[triggers_mocap['Gesture']=='outward']
                        mae_inner, correct_detections_inner, spurious_det_inner, spurious_det_count_inner, gt = compare_with_mocap(minima_indices, triggers_mocap,lag_trial)
                        mae['NOSE'].append(mae_inner)
                        correct_detections['NOSE'].append(correct_detections_inner)
                        spurious_detections['NOSE'].append(spurious_det_inner)
                        spurious_detections_cnt['NOSE'].append(spurious_det_count_inner)
                        ground_truths['NOSE'].append(gt)
                    except Exception as e:
                        print(f"Error:{e} while processing {path_task}, moving to next...")
                else:
                    for cad in ['S1', 'S2', 'N1', 'N2', 'F1', 'F2']:
                        try:
                            path_task = os.path.join(input_folder, 'DataKinects', sbj, side, 'rgb',
                                                     'Prova_' + task + cad)
                            print(f'{path_task} is being processed')
                            all_metrics, processor = extract_metrics(path_task)

                            # load information about opto-kinect lag
                            lag_trial = lags[lags['Task'] == task+cad]['Lag (at 120 fps)'].values[0]
                            # load triggers obtained from opto
                            triggers_mocap = pd.read_csv(
                                os.path.join(input_folder, 'Annotations', 'GestureTriggers', sbj,
                                             side, task+cad, task+cad+'_triggers.csv'))
                            if task=='OC':
                                triggers_mocap = triggers_mocap[triggers_mocap['Gesture'] == 'closed']
                            if task=='PS':
                                triggers_mocap = triggers_mocap[triggers_mocap['Gesture'] == 'forehand'] if side=='Left' else triggers_mocap[triggers_mocap['Gesture'] == 'palm']

                            minima_indices = aggregate_and_plot(all_metrics, processor,
                                                                smoothing_window=smoothing_window)
                            #display_results(path_task, minima_indices)
                            mae_inner, correct_detections_inner, spurious_det_inner, spurious_det_count_inner, gt = compare_with_mocap(
                                minima_indices, triggers_mocap, lag_trial)
                            mae[task+cad[0]].append(mae_inner)
                            correct_detections[task+cad[0]].append(correct_detections_inner)
                            spurious_detections[task+cad[0]].append(spurious_det_inner)
                            spurious_detections_cnt[task+cad[0]].append(spurious_det_count_inner)
                            ground_truths[task+cad[0]].append(gt)
                        except Exception as e:
                            print(f"Error:{e} while processing {path_task}, moving to next...")

    import json

    # Group all dictionaries into one
    data_to_save = {
        "correct_detections": correct_detections,
        'ground_truths':ground_truths,
        "spurious_detections": spurious_detections,
        "spurious_detections_cnt": spurious_detections_cnt,
        "mae": mae
    }

    # Save metrics as JSON file
    with open(os.path.join(output_folder, f'metrics_sw{smoothing_window}.json'), 'w') as f:
        json.dump(data_to_save, f, indent=4)