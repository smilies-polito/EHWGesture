import os
from doctest import master
import pandas as pd
import cv2
import argparse
import numpy as np
from pyk4a import PyK4APlayback

def first_non_empty_capture_kinect(playback_kin):
    # skip emtpy captures on master
    capture_k1 = playback_kin.get_next_capture()
    while capture_k1 is None:
        print("Skipping this master frame because color capture is empty")
        capture_k1 = playback_kin.get_next_capture()
    return capture_k1


def extract_sync_frames_kinects(path_kin1, path_kin2, synchr_frames, output_base_folder, depth_output_base_folder, input_folder):

    relative_path_kin1 = os.path.splitext(os.path.relpath(path_kin1, start=input_folder))[0]
    relative_path_kin2 = os.path.splitext(os.path.relpath(path_kin2, start=input_folder))[0]
    color_output_folder_kin1 = os.path.join(output_base_folder, relative_path_kin1)
    color_output_folder_kin2 = os.path.join(output_base_folder, relative_path_kin2)
    os.makedirs(color_output_folder_kin1, exist_ok=True)
    os.makedirs(color_output_folder_kin2, exist_ok=True)

    depth_output_path_kin1 = os.path.join(depth_output_base_folder, relative_path_kin1 + "_depth.mp4")
    depth_output_path_kin2 = os.path.join(depth_output_base_folder, relative_path_kin2 + "_depth.mp4")
    os.makedirs(os.path.dirname(depth_output_path_kin1), exist_ok=True)

    depth_writer_kin1 = None
    depth_writer_kin2 = None
    try:
        playback_k1 = PyK4APlayback(path_kin1)
        playback_k1.open()

        playback_k2 = PyK4APlayback(path_kin2)
        playback_k2.open()
    except Exception as e:
        return f"ERROR opening {path_kin1} or {path_kin2}: {e}"

    try:
        i=0
        for sync_fs in synchr_frames.iterrows():
            capture_k1=first_non_empty_capture_kinect(playback_k1)
            capture_k2=first_non_empty_capture_kinect(playback_k2)
            k1_found=False
            k2_found=False
            while not (k1_found and k2_found):
                #skip captures with no color stream
                if not k1_found:
                    while capture_k1.color is None:
                        #print("Color image missing from master, automatically go to next")
                        capture_k1 = first_non_empty_capture_kinect(playback_k1)
                if not k2_found:
                    while capture_k2.color is None:
                        #print("Color image missing from master, automatically go to next")
                        capture_k2 = first_non_empty_capture_kinect(playback_k2)

                if capture_k1.color_timestamp_usec == sync_fs[1]['Kinect1']:
                    k1_found=True
                else:
                    capture_k1 = first_non_empty_capture_kinect(playback_k1)

                if capture_k2.color_timestamp_usec == sync_fs[1]['Kinect2']:
                    k2_found = True
                else:
                    capture_k2 = first_non_empty_capture_kinect(playback_k2)

            #print("Found the sync pair!")
            #verify that not only RGB is ok but also depth...
            if capture_k1.depth is None or capture_k2.depth is None:
                i += 1
                continue
            try:
                if depth_writer_kin1 is None:
                    height, width = capture_k1.depth.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = 30
                    depth_writer_kin1 = cv2.VideoWriter(depth_output_path_kin1, fourcc, fps, (width, height), isColor=False)

                if depth_writer_kin2 is None:
                    height, width = capture_k1.depth.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = 30
                    depth_writer_kin2 = cv2.VideoWriter(depth_output_path_kin2, fourcc, fps, (width, height), isColor=False)

                write_frames(capture_k1, depth_writer_kin1, depth_output_path_kin1, color_output_folder_kin1, i)
                write_frames(capture_k2, depth_writer_kin2, depth_output_path_kin2, color_output_folder_kin2, i)
            except Exception as e:
                print(f"Error processing frames pair {i} from {path_kin1} and {path_kin1}: {e}")
                #print("Since there was an error processing one of the two frames in the pair, this pair is discarded")
                i+=1
                continue

            i+=1

    except Exception as e:
        print(f"ERROR processing {path_kin1} or {path_kin2}: {e}")
    finally:
        playback_k1.close()
        playback_k2.close()
        depth_writer_kin1.release() if depth_writer_kin2 is not None else None
        print(f"Processed {i} frames from {path_kin1} and {path_kin2}")


def write_frames(capture, depth_writer, depth_output_path, color_output_folder, frame_count):
    color_image = cv2.imdecode(capture.color, cv2.IMREAD_COLOR)
    frame_path = os.path.join(color_output_folder, f"{frame_count:04d}.jpg")
    cv2.imwrite(frame_path, color_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    depth_image = capture.depth

    depth_norm = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype('uint8')
    depth_writer.write(depth_uint8)

def extract_frames(input_file, output_base_folder, depth_output_base_folder, input_folder):
    playback = PyK4APlayback(input_file)
    try:
        playback.open()
    except Exception as e:
        print(f"Error opening video {input_file}: {e}")
        return

    relative_path = os.path.splitext(os.path.relpath(input_file, start=input_folder))[0]
    color_output_folder = os.path.join(output_base_folder, relative_path)
    os.makedirs(color_output_folder, exist_ok=True)
    
    depth_output_path = os.path.join(depth_output_base_folder, relative_path + "_depth.mp4")
    os.makedirs(os.path.dirname(depth_output_path), exist_ok=True)

    depth_writer = None
    frame_count = 0

    while True:
        try:
            capture = playback.get_next_capture()
            
            if capture.color is not None and capture.depth is not None:
                color_image = cv2.imdecode(capture.color, cv2.IMREAD_COLOR)
                frame_path = os.path.join(color_output_folder, f"{frame_count:04d}.jpg")
                cv2.imwrite(frame_path, color_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                depth_image = capture.depth 

                depth_norm = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
                depth_uint8 = depth_norm.astype('uint8')

                if depth_writer is None:
                    height, width = depth_uint8.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = 30 
                    depth_writer = cv2.VideoWriter(depth_output_path, fourcc, fps, (width, height), isColor=False)
                
                depth_writer.write(depth_uint8)
            else:
                print(f"Depth or color in {input_file} is missing for {frame_count} so I am skipping both")
            frame_count += 1
        except EOFError:
            break

    playback.close()
    if depth_writer is not None:
        depth_writer.release()
    print(f"Processed {frame_count} frames from {input_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract color frames and depth video from videos.")
    parser.add_argument("--input_folder", default='H:\Registrazioni_IEIIT_Event', type=str, help="Path to input folder containing videos.")
    parser.add_argument("--output_folder", default='F:\\TRAIN_SET_EHWGESTURE\\rgb_data', type=str, help="Path to output folder to save color frames.")
    parser.add_argument("--depth_output_folder", default='F:\\TRAIN_SET_EHWGESTURE\\depth_anonimized', type=str, help="Path to output folder to save depth videos.")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    depth_output_folder = args.depth_output_folder

    args = parser.parse_args()
    for sbj in ['X17', 'X19', 'X20']:
        for side in ['Left', 'Right']:
            for task in ['FT', 'OC', 'NOSE', 'PS', 'TR']:
                if task == 'NOSE' or task=='TR':
                    try:
                        master_kin_path_task = os.path.join(input_folder, sbj + '_L' if side == 'Left' else sbj + '_R', 'Prova_' + task + '1',
                                                            'master_' + task + "1.mkv")
                        sub_kin_path_task = os.path.join(input_folder, sbj + '_L' if side == 'Left' else sbj + '_R', 'Prova_' + task + '1',
                                                            'sub2_' + task + "1.mkv")
                        path_sync_frames = os.path.join(input_folder, sbj + '_L' if side == 'Left' else sbj + '_R', 'Prova_' + task + '1',
                                                            f'{task+"1"}_sync_frames.csv')
                        out_path_dept = os.path.join(depth_output_folder, sbj, side, task + "1")
                        sync_frames = pd.read_csv(path_sync_frames)
                        #extract_frames(master_kin_path_task, out_path_frames, depth_output_folder, input_folder)
                        #extract_frames(sub_kin_path_task, out_path_frames, depth_output_folder, input_folder)
                        extract_sync_frames_kinects(master_kin_path_task, sub_kin_path_task, sync_frames, output_folder,depth_output_folder, input_folder)
                    except Exception as e:
                        print(f"Error:{e} while processing {master_kin_path_task}, moving to next...")

                    try:
                        master_kin_path_task = os.path.join(input_folder, sbj + '_L' if side == 'Left' else sbj + '_R',
                                                            'Prova_' + task + '2',
                                                            'master_' + task + "2.mkv")
                        sub_kin_path_task = os.path.join(input_folder, sbj + '_L' if side == 'Left' else sbj + '_R',
                                                         'Prova_' + task + '2',
                                                         'sub2_' + task + "2.mkv")
                        path_sync_frames = os.path.join(input_folder, sbj + '_L' if side == 'Left' else sbj + '_R',
                                                        'Prova_' + task + '2',
                                                        f'{task + "2"}_sync_frames.csv')
                        sync_frames = pd.read_csv(path_sync_frames)
                        out_path_dept = os.path.join(depth_output_folder, sbj, side, task + "2")
                        #extract_frames(master_kin_path_task, out_path_frames, depth_output_folder, input_folder)
                        #extract_frames(sub_kin_path_task, out_path_frames, depth_output_folder, input_folder)
                        extract_sync_frames_kinects(master_kin_path_task, sub_kin_path_task, sync_frames, output_folder,depth_output_folder, input_folder)
                    except Exception as e:
                        print(f"Error:{e} while processing {master_kin_path_task}, moving to next...")
                else:
                    for cad in ['S1', 'S2', 'N1', 'N2', 'F1', 'F2']:
                        try:
                            master_kin_path_task = os.path.join(input_folder,
                                                                sbj + '_L' if side == 'Left' else sbj + '_R',
                                                                'Prova_' + task + cad,
                                                                'master_' + task + cad + ".mkv")
                            sub_kin_path_task = os.path.join(input_folder,
                                                                sbj + '_L' if side == 'Left' else sbj + '_R',
                                                                'Prova_' + task + cad,
                                                                'sub2_' + task + cad + ".mkv")
                            out_path_dept = os.path.join(depth_output_folder, sbj, side, task + cad)
                            path_sync_frames = os.path.join(input_folder, sbj + '_L' if side == 'Left' else sbj + '_R',
                                                            'Prova_' + task + cad,
                                                            f'{task + cad}_sync_frames.csv')
                            sync_frames = pd.read_csv(path_sync_frames)
                            #extract_frames(master_kin_path_task, out_path_frames, depth_output_folder, input_folder)
                            #extract_frames(sub_kin_path_task, out_path_frames, depth_output_folder, input_folder)
                            extract_sync_frames_kinects(master_kin_path_task, sub_kin_path_task, sync_frames,
                                                        output_folder, depth_output_folder, input_folder)
                        except Exception as e:
                            print(f"Error:{e} while processing {master_kin_path_task}, moving to next...")



    # for root, _, files in os.walk(input_folder):
    #     for file in files:
    #         if file.endswith(".mkv") or file.endswith(".mp4"):
    #             input_file = os.path.join(root, file)
    #             print(f"Processing file: {input_file}")
    #             extract_frames(input_file, output_folder, depth_output_folder, input_folder)
