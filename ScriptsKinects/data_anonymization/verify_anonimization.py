import cv2
import os
import glob
import numpy as np
from ultralytics import YOLO
import argparse

model_face1 = YOLO('xinet_videoanony.pt')
haar_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# cv2_detector = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
# dlib_detector = dlib.get_frontal_face_detector()


PIXEL_DIFF_THRESHOLD = 60  # min MSE between original and anonymized face crops

def detect_faces_yolo(image, model, conf_threshold=0.5):
    results = model.predict(image[:, :, :3], conf=conf_threshold, verbose=False)
    return [[int(x) for x in box.xyxy[0]] for result in results for box in result.boxes]


def detect_faces_dlib(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = dlib_detector(gray)
    return [[face.left(), face.top(), face.right(), face.bottom()] for face in faces]

def detect_faces_haar(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = haar_detector.detectMultiScale(gray, 1.3, 5)
    return [[x, y, x + w, y + h] for x, y, w, h in faces]

def is_anonymized(original, anonymized, face_boxes):
    mse = 0
    for x1, y1, x2, y2 in face_boxes:
        original_crop = original[y1:y2, x1:x2]
        anonymized_crop = anonymized[y1:y2, x1:x2]

        if original_crop.shape != anonymized_crop.shape:
            return False, mse

        mse = np.mean((original_crop.astype(np.float32) - anonymized_crop.astype(np.float32)) ** 2)

        if mse < PIXEL_DIFF_THRESHOLD:
            return False, mse

    return True, mse


def verify_anonymization(original_frames_root, anonymized_videos_root, failed_output_root):
    os.makedirs(failed_output_root, exist_ok=True)

    anonymized_videos = glob.glob(os.path.join(anonymized_videos_root, "**", "*.mp4"), recursive=True) + \
                        glob.glob(os.path.join(anonymized_videos_root, "**", "*.mkv"), recursive=True)

    for video_path in anonymized_videos:
        relative_path = os.path.relpath(video_path, anonymized_videos_root)
        original_frames_folder = os.path.join(original_frames_root, os.path.dirname(relative_path))

        if not os.path.exists(original_frames_folder):
            print(f"Warning: No matching original frames found for {video_path}")
            continue

        cap = cv2.VideoCapture(video_path)
        frame_files = sorted(glob.glob(os.path.join(original_frames_folder, "**", "*.jpg"), recursive=True))

        if not frame_files:
            print(f"Warning: No frames found in {original_frames_folder}")
            continue

        frame_idx = 0
        anonimized = True
        avg_mse = 0
        min_mse = 9999
        while cap.isOpened():
            ret, anonymized_frame = cap.read()
            if not ret or frame_idx >= len(frame_files):
                break

            original_frame = cv2.imread(frame_files[frame_idx])

            face_boxes_yolo = detect_faces_yolo(original_frame, model_face1)
            face_boxes = face_boxes_yolo
            # face_boxes_dlib = detect_faces_dlib(original_frame)
            # face_boxes = face_boxes_yolo + face_boxes_dlib
            anonimization_succesful, mse = is_anonymized(original_frame, anonymized_frame, face_boxes)
            avg_mse += mse
            min_mse = min(min_mse, mse) if mse > 0 else min_mse
            if face_boxes and not anonimization_succesful:
                failed_path = os.path.join(failed_output_root, os.path.basename(video_path))
                #print(f"Failed verification for frame {frame_idx} in {video_path}. Saving failed faces to {failed_path}")
                os.makedirs(failed_path, exist_ok=True)
                anonimized = False

                for i, (x1, y1, x2, y2) in enumerate(face_boxes):
                    crop = anonymized_frame[y1:y2, x1:x2]
                    cv2.imwrite(os.path.join(failed_path, f"{frame_idx:04d}_fail_{i}.jpg"), crop)

            frame_idx += 1

        avg_mse /= frame_idx
        cap.release()
        print(f"Verification completed for: {video_path}")
        if anonimized:
            print(f"✅ SUCCESS: All faces were anonymized. avg MSE: {mse:.2f}, min MSE: {min_mse:.2f}")
        else:
            print(f"⚠️ WARNING: Some faces were not anonymized. avg MSE: {mse:.2f}, min MSE: {min_mse:.2f}")

    print("Verification process finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify video anonymization by checking face modification.")
    parser.add_argument("--original_frames", default='example_dataset_frames', type=str, help="Path to the folder containing original frames.")
    parser.add_argument("--anonymized_videos", default='out_anonimized', type=str, help="Path to the folder containing anonymized videos.")
    parser.add_argument("--failed_output", default="failed_verification", type=str, help="Folder to save faces that were not anonymized.")

    args = parser.parse_args()
    verify_anonymization(args.original_frames, args.anonymized_videos, args.failed_output)
