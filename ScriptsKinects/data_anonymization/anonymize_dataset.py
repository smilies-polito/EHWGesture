import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor
import cv2
from pyk4a import PyK4APlayback
import numpy as np
import os
import tempfile
import shutil
import atexit
from ultralytics import YOLO
import mediapipe as mp
import argparse



def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    xi1, yi1, xi2, yi2 = max(x1, x1g), max(y1, y1g), min(x2, x2g), min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area else 0


width, height = 1920, 1080
model_face = YOLO('data_anonymization/xinet_videoanony.pt')
model_hand = YOLO('data_anonymization/xinet_pose_hand_xs.pt') 
# predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")
predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-base-plus")
def anonymize_video(input_file, output_base_folder, iou_threshold=0.1):
    global model_face, model_hand, predictor, width, height
    playback = PyK4APlayback(input_file)
    try:
        playback.open()
    except Exception as e:
        print(f"Error processing video: {e}")
        return

    temp_dir = tempfile.mkdtemp()
    def delete_temp_dir():
        shutil.rmtree(temp_dir)
    atexit.register(delete_temp_dir)

    
    output_file = os.path.join(output_base_folder, input_file)
    # breakpoint()
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping...")
        return
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    frame_count = 0
    while True:
        try:
            capture = playback.get_next_capture()
            if capture.color is not None:
                color_image = cv2.imdecode(capture.color, cv2.IMREAD_COLOR)
                frame_path = os.path.join(temp_dir, f"{frame_count:04d}.jpg")
                cv2.imwrite(frame_path, color_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                frame_count += 1
        except EOFError:
            break

    playback.close()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, 30.0, (width, height))
    frame_files = sorted([os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.jpg')])

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        color_image = cv2.imread(frame_files[0])
        results = model_face.predict(color_image[:, :, :3], conf=0.5, verbose=False)
        face_boxes = [[int(x) for x in box.xyxy[0]] for result in results for box in result.boxes]
        padded_face_boxes = [[max(0, x1 - 40), max(0, y1 - 40), min(width, x2 + 40), min(height, y2 + 40)] for x1, y1, x2, y2 in face_boxes]

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(color_image_rgb)
        # results = model_hand.predict(color_image[:, :, :3], conf=0.5, verbose=False)
        hand_boxes = []
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                x_min = min([landmark.x for landmark in hand_landmarks.landmark])
                y_min = min([landmark.y for landmark in hand_landmarks.landmark])
                x_max = max([landmark.x for landmark in hand_landmarks.landmark])
                y_max = max([landmark.y for landmark in hand_landmarks.landmark])
                hand_boxes.append([int(x_min * width), int(y_min * height), int(x_max * width), int(y_max * height)])
        hands.close()

        use_sam = any(iou(hb, fb) > iou_threshold for hb in hand_boxes for fb in padded_face_boxes)

        
        if use_sam:
            state = predictor.init_state(temp_dir)
            print("Using SAM to anonymize video...")
            for id_b, box in enumerate(face_boxes):
                predictor.add_new_points_or_box(state, frame_idx=0, obj_id=id_b, box=box)
            for id_b, box in enumerate(hand_boxes):
                predictor.add_new_points_or_box(state, frame_idx=0, obj_id=200 + id_b, box=box)
        
            kernel_expand = np.ones((20, 20), np.uint8)
            kernel_morph = np.ones((5, 5), np.uint8)

            for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                base_frame = cv2.imread(os.path.join(temp_dir, f"{frame_idx:04d}.jpg"))
                masks_np = masks.cpu().numpy().clip(0, 1).reshape(-1, 1080, 1920, 1)

                face_masks = masks_np[:len(face_boxes)]
                hand_masks = masks_np[len(face_boxes):]

                mask_image = cv2.dilate(np.sum(face_masks, axis=0), kernel_expand, iterations=1).clip(0, 1).reshape(1080, 1920, 1)
                mask_image -= np.sum(hand_masks, axis=0)

                blurred_frame = cv2.medianBlur(base_frame, 31) 
                mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, kernel_morph)
                mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_OPEN, kernel_morph)
                base_frame[mask_image > 0.5] = blurred_frame[mask_image > 0.5]

                out.write(base_frame)

        else:
            print("Using fixed bb to process video")
            for frame_idx, frame_file in enumerate(frame_files):
                base_frame = cv2.imread(frame_file)

                for x1, y1, x2, y2 in padded_face_boxes:
                    blurred_region = cv2.medianBlur(base_frame[y1:y2, x1:x2], 31)
                    base_frame[y1:y2, x1:x2] = blurred_region

                out.write(base_frame)
    
    out.release()
    print(f"Video saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anonymize videos in a given folder.")
    parser.add_argument("--input_folder", default='example_dataset', type=str, help="Path to the input folder containing videos.")
    parser.add_argument("--output_folder", default='out_anonimized', type=str, help="Path to the output folder to save anonymized videos.")
    parser.add_argument("--fraction", type=str, default="0.0:1.0", help="Fraction of videos to process in the format 'start:end', e.g., '0:0.5' for the first half.")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    start_fraction, end_fraction = map(float, args.fraction.split(':'))

    all_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".mkv") or file.endswith(".mp4"):
                all_files.append(os.path.join(root, file))

    num_files = len(all_files)
    start_index = int(start_fraction * num_files)
    end_index = int(end_fraction * num_files)

    print(f"Processing {end_index - start_index} out of {num_files} files (from {start_index} to {end_index}).")

    for i in range(start_index, end_index):
        input_file = all_files[i]
        print(f"Processing file: {input_file}")
        anonymize_video(input_file, output_folder)
