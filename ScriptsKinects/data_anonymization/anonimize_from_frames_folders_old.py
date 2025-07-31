import os
import cv2
import torch
import numpy as np
import argparse
import mediapipe as mp
from ultralytics import YOLO
from sam2.sam2_video_predictor import SAM2VideoPredictor

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    xi1, yi1, xi2, yi2 = max(x1, x1g), max(y1, y1g), min(x2, x2g), min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - y1g)
    union_area = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - inter_area
    return inter_area / union_area if union_area else 0

width, height = 1920, 1080
model_face = YOLO('xinet_videoanony.pt')
model_hand = YOLO('xinet_pose_hand_xs.pt')
predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-base-plus")

def detect_hands(image, width, height, detection_confidence=0.5, max_num_hands=2):
    mp_hands = mp.solutions.hands
    hand_boxes = []
    with mp_hands.Hands(static_image_mode=True, max_num_hands=max_num_hands, 
                        min_detection_confidence=detection_confidence) as hands:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(image_rgb)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                x_min = min([landmark.x for landmark in hand_landmarks.landmark])
                y_min = min([landmark.y for landmark in hand_landmarks.landmark])
                x_max = max([landmark.x for landmark in hand_landmarks.landmark])
                y_max = max([landmark.y for landmark in hand_landmarks.landmark])
                hand_boxes.append([int(x_min * width), int(y_min * height),
                                   int(x_max * width), int(y_max * height)])
    return hand_boxes

def anonymize_with_sam(frame_files, face_boxes, hand_boxes, input_folder, video_writer):

    state = predictor.init_state(input_folder)
    print("Using SAM to anonymize video...")
    
    # Add face and hand boxes as objects for SAM
    for id_b, box in enumerate(face_boxes):
        predictor.add_new_points_or_box(state, frame_idx=0, obj_id=id_b, box=box)
    for id_b, box in enumerate(hand_boxes):
        predictor.add_new_points_or_box(state, frame_idx=0, obj_id=200 + id_b, box=box)

    for frame_idx, _, masks in predictor.propagate_in_video(state):
        base_frame = cv2.imread(frame_files[frame_idx])
        masks_np = masks.cpu().numpy().clip(0, 1).reshape(-1, height, width, 1)
        face_masks = np.sum(masks_np[:len(face_boxes)], axis=0).clip(0, 1)
        hand_masks = np.sum(masks_np[len(face_boxes):], axis=0).clip(0, 1)
        face_masks = cv2.dilate(face_masks, np.ones((20, 20), np.uint8), iterations=1).clip(0, 1).reshape(height, width, 1)
        mask_image = (face_masks - hand_masks).squeeze()

        blurred_frame = cv2.medianBlur(base_frame, 31)
        base_frame[mask_image > 0.5] = blurred_frame[mask_image > 0.5]

        video_writer.write(base_frame)

def anonymize_with_fixed_bbox(frame_files, padded_face_boxes, video_writer):
    print("Using fixed bounding boxes to anonymize video")
    for frame_file in frame_files:
        base_frame = cv2.imread(frame_file)
        for x1, y1, x2, y2 in padded_face_boxes:
            blurred_region = cv2.medianBlur(base_frame[y1:y2, x1:x2], 31)
            base_frame[y1:y2, x1:x2] = blurred_region
        video_writer.write(base_frame)

def anonymize_frames(input_folder, output_folder, force_sam=False, iou_threshold=0.1):
    frame_files = sorted([os.path.join(input_folder, f) 
                          for f in os.listdir(input_folder) if f.endswith('.jpg')])
    if not frame_files:
        print(f"No frames found in {input_folder}, skipping...")
        return

    output_file = os.path.join(os.path.dirname(output_folder), os.path.basename(input_folder) + ".mp4")
    print(f"Anonymizing frames in {input_folder} to {output_file}")
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping...")
        return
    else:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_file, fourcc, 30.0, (width, height))

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # detect faces in the first frame of the video
        color_image = cv2.imread(frame_files[0])
        threshold = 0.5
        face_boxes = []
        while threshold > 0:
            results = model_face.predict(color_image[:, :, :3], conf=threshold, verbose=False)
            face_boxes = [[int(x) for x in box.xyxy[0]] 
                  for result in results for box in result.boxes]
            if face_boxes:
                break
            print(f"Warning: No face boxes detected at threshold {threshold}. Lowering threshold...")
            threshold -= 0.1
        if not face_boxes:
            print("Warning: No face boxes detected even at the lowest threshold.")
        # Pad face boxes to avoid issues when moving the face
        padded_face_boxes = [[max(0, x1 - 40), max(0, y1 - 40), 
                              min(width, x2 + 40), min(height, y2 + 40)]
                             for x1, y1, x2, y2 in face_boxes]

        hand_boxes = detect_hands(color_image, width, height)

        # Decide which anonymization method to use based on IOU threshold, for NOSE always use SAM
        use_sam = (any(iou(hb, fb) > iou_threshold
                      for hb in hand_boxes for fb in padded_face_boxes)) or ('NOSE' in input_folder) or (force_sam)

        if use_sam:
            anonymize_with_sam(frame_files, face_boxes, hand_boxes, input_folder, video_writer)
        else:
            anonymize_with_fixed_bbox(frame_files, padded_face_boxes, video_writer)

    video_writer.release()
    print(f"Video saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anonymize extracted video frames.")
    parser.add_argument("--input_folder", default='example_dataset_frames\X05N_R\Prova_PSN1', type=str, 
                        help="Path to folder containing extracted frames. Each subfolder is processed separately.")
    parser.add_argument("--output_folder", default='out_anonimized', type=str, 
                        help="Path to output folder for anonymized videos.")

    parser.add_argument("--force_sam", default=False, type=bool,
                        help="Force using sam for segmenting faces")

    args = parser.parse_args()

    for root, dirs, _ in os.walk(args.input_folder):
        for d in dirs:
            input_subfolder = os.path.join(root, d)
            output_subfolder = os.path.join(args.output_folder, os.path.relpath(input_subfolder, start=args.input_folder))

            anonymize_frames(input_subfolder, output_subfolder, args.force_sam)
