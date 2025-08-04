import os
import glob
import argparse
import cv2
import numpy as np
from tqdm import tqdm

def parse_crop(s):
    try:
        parts = [int(x) for x in s.split(',')]
        if len(parts) != 4:
            raise ValueError("Exactly 4 integers required.")
        return parts
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid crop coordinate format: {e}")

def crop_and_resize(image, crop_coords, output_resolution):
    left, top, right, bottom = crop_coords
    h, w = image.shape[:2]
    left = max(0, left)
    top = max(0, top)
    right = min(w, right)
    bottom = min(h, bottom)
    cropped = image[top:bottom, left:right]
    if cropped.size == 0:
        raise ValueError("Check crop coordinates.")

    ch, cw = cropped.shape[:2]
    scale = output_resolution / float(min(ch, cw))
    new_w = int(round((right-left) * scale))
    new_h = int(round((bottom-top) * scale))
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized

def process_video(video_path, dest_folder, crop_coords, output_fps, output_resolution, is_depth=False):
    norm_path = os.path.normpath(video_path)
    rel_dir = os.path.dirname(norm_path)
    base_name = os.path.splitext(os.path.basename(norm_path))[0]
    out_dir = os.path.join(dest_folder, rel_dir, base_name)

    parts = rel_dir.split(os.sep)
    for i, part in enumerate(parts):
        if part.lower() == "left" or part.lower=="right":  
            parts[i - 1] += f"_{part[0].upper()}"  
            parts.pop(i) 
            break

    # Reconstruct the output directory
    breakpoint()
    out_dir = os.path.join("dest_folder", *parts, base_name, "rgb")

    breakpoint()
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'rgb' if not is_depth else 'depth'), exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR opening {video_path}")
        return
    
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(round(input_fps / output_fps)))
    
    frame_index = 0
    output_index = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_index % frame_interval == 0:
            try:
                processed = crop_and_resize(frame, crop_coords, output_resolution)
                
                if is_depth:
                    processed = processed / 63.0  # Normalize depth to 0-255
                    processed = np.clip(processed, 0, 255).astype(np.uint8)
                    filename = os.path.join(out_dir, 'depth', f"{output_index}_depth.jpg")
                else:
                    filename = os.path.join(out_dir, 'rgb', f"{output_index}.jpg")
                
                cv2.imwrite(filename, processed)
                output_index += 1
            except Exception as e:
                print(f"Error processing frame {frame_index} in {video_path}: {e}")
        
        frame_index += 1
    
    cap.release()

def main():
    parser = argparse.ArgumentParser(description="Process MP4 videos by cropping, resizing, and saving RGB and depth frames.")
    parser.add_argument("--src", type=str, required=True, help="Base folder containing subject videos.")
    parser.add_argument("--dest", type=str, default="dataset_processed", help="Destination folder.")
    parser.add_argument("--master_crop", type=parse_crop, default="400,200,1200,600", help="Crop coords for master videos.")
    parser.add_argument("--sub_crop", type=parse_crop, default="400,250,1200,650", help="Crop coords for sub videos.")
    parser.add_argument("--output_fps", type=float, default=12, help="Output FPS.")
    parser.add_argument("--output_resolution", type=int, default=112, help="Output resolution (shortest side in pixels).")
    
    args = parser.parse_args()
    subjects = [d for d in os.listdir(args.src) if os.path.isdir(os.path.join(args.src, d))]
    
    for subject in tqdm(subjects, desc="Processing subjects"):
        for side in ["left", "right"]:
            for gesture in os.listdir(os.path.join(args.src, subject, side, "rgb")):
                rgb_path = os.path.join(args.src, subject, side, "rgb", gesture)
                depth_path = os.path.join(args.src, subject, side, "depth", gesture)
                
                for rgb_file in glob.glob(os.path.join(rgb_path, "*.mp4")):
                    crop_coords = args.master_crop if "master" in rgb_file.lower() else args.sub_crop
                    process_video(rgb_file, args.dest, crop_coords, args.output_fps, args.output_resolution, is_depth=False)
                
                for depth_file in glob.glob(os.path.join(depth_path, "*.mp4")):
                    crop_coords = args.master_crop if "master" in depth_file.lower() else args.sub_crop
                    process_video(depth_file, args.dest, crop_coords, args.output_fps, args.output_resolution, is_depth=True)
    
    print("Processing complete.")

if __name__ == '__main__':
    main()
