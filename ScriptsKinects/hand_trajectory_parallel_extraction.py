import os
import random
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import mediapipe as mp
from pyk4a import PyK4APlayback, ImageFormat

def convert_to_bgra_if_required(color_format, color_image):
    if color_format == ImageFormat.COLOR_MJPG:
        color_image = cv2.imdecode(color_image, cv2.IMREAD_COLOR)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2BGRA)
    elif color_format == ImageFormat.COLOR_NV12:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_NV12)
    elif color_format == ImageFormat.COLOR_YUY2:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_YUY2)
    return color_image

def process_video(video_path):
    rel_path = os.path.normpath(video_path)
    # Remove drive letter if any (Windows <3):
    if ':' in rel_path:
        rel_path = rel_path.split(':', 1)[-1].lstrip(os.sep)
    base, _ = os.path.splitext(rel_path)
    output_path = os.path.join("output", base + ".txt")
    
    if os.path.exists(output_path):
        return f"SKIPPED: {video_path}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    # left right top bottom
    detection_lines = []

    try:
        playback = PyK4APlayback(video_path)
        playback.open()
    except Exception as e:
        return f"ERROR opening {video_path}: {e}"

    try:
        while True:
            for _ in range(4): # take one frame every four to avoid processing for one million years
                capture = playback.get_next_capture()

            if capture is None:
                break

            if capture.color is not None:
                color_image = convert_to_bgra_if_required(
                    playback.configuration.get("color_format"), capture.color)
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                h, w, _ = color_image.shape
                results = hands.process(color_image)

                if results.multi_hand_landmarks: # we write position of all detected hands - we will filter by main hand position later
                    for hand_landmarks in results.multi_hand_landmarks:
                        xs = [lm.x * w for lm in hand_landmarks.landmark]
                        ys = [lm.y * h for lm in hand_landmarks.landmark]
                        left = int(min(xs))
                        right = int(max(xs))
                        top = int(min(ys))
                        bottom = int(max(ys))
                        detection_lines.append(f"{left} {right} {top} {bottom}")
    except EOFError:
        pass
    except Exception as e:
        return f"ERROR processing {video_path}: {e}"
    finally:
        playback.close()
        hands.close()

    try:
        with open(output_path, "w") as f:
            for line in detection_lines:
                f.write(line + "\n")
    except Exception as e:
        return f"ERROR writing {output_path}: {e}"

    return f"PROCESSED: {video_path}"

if __name__ == '__main__':
    root_video_dir = "F:/EHWdataset"  # change this to your dataset folder
    video_files = glob.glob(os.path.join(root_video_dir, "**", "*.mkv"), recursive=True)

    def get_output_path(video_path):
        rel_path = os.path.normpath(video_path)
        if ':' in rel_path:
            rel_path = rel_path.split(':', 1)[-1].lstrip(os.sep)
        base, _ = os.path.splitext(rel_path)
        return os.path.join("traj_output", base + ".txt")

    video_files = [vf for vf in video_files if not os.path.exists(get_output_path(vf))]

    random.shuffle(video_files) # shuffle so we can say its a "Monte carlo estimate" if the processing dies halfway in :D

    total_files = len(video_files)
    if total_files == 0:
        print("No videos to process (or all outputs already exist).")
    else:
        print(f"Processing {total_files} videos...")

        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_video, vf): vf for vf in video_files}

            for future in as_completed(futures):
                video_file = futures[future]
                try:
                    result = future.result()
                    print(result)
                except Exception as exc:
                    print(f"Video {video_file} generated an exception: {exc}")

    print("Processing complete.")
