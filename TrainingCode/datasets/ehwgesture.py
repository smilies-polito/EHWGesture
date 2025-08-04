import torch
import torch.utils.data as data
from PIL import Image
import os
import glob
import re
import numpy as np
import random

def extract_frame_number(filename):
    match = re.search(r'(\d+)', os.path.basename(filename))
    return int(match.group(0)) if match else float('inf')

def frame_loader(path, mode='rgb'):

    if mode=='event':
        matrix=np.load(path)
        array_rgb = np.concatenate([matrix, matrix[:, :, 1:2]], axis=-1)
        array_rgb = (255 * (array_rgb - array_rgb.min()) / (array_rgb.max() - array_rgb.min())).astype(np.uint8)
        return Image.fromarray(array_rgb, mode="RGB")
    else:
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                if mode == 'rgb':
                    return img.convert('RGB')
                elif mode == 'depth':
                    img = img.convert('L')
                    img = np.array(img, dtype=np.uint8)
                    img = np.expand_dims(img, axis=-1) 
                    return Image.fromarray(np.repeat(img, 3, axis=-1))

def get_class_from_path(path):
    match = re.search(r'(NOSE|FTF|FTN|FTS|OCF|OCN|OCS|PSF|PSN|PSS|TR)', path)
    return match.group(0) if match else None

def get_subject_from_path(path):
    match = re.search(r'([A-Z0-9]+)_(L|R)', path)
    return match.group(1) if match else None

def find_folders(base_path):
    subfolders = glob.glob(os.path.join(base_path, '*', '*'))
    labels = [get_class_from_path(f) for f in subfolders] 
    subjects = [get_subject_from_path(f) for f in subfolders]
    return subfolders, labels, subjects

class EHWGestureDataset(data.Dataset):
    def __init__(self, base_path, subset='train', input_type=['rgb', 'depth'], camera=['master', 'sub2'], spatial_transform=None, temporal_transform=None, target_transform=None, sample_duration=16, time_downsample=2, random_mask_fraction=0.0, val_test_subjects = {'X01', 'X05', 'X08', 'X17'}):
        assert subset in ['train', 'val', 'test']
        
        self.base_path = base_path
        self.input_type = input_type
        self.camera = camera
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.sample_duration = sample_duration
        self.stride = sample_duration*time_downsample
        self.subset = subset
        self.time_downsample = time_downsample
        self.random_mask_fraction = random_mask_fraction

        print("Training on input types:")
        for cam, mode in zip(camera, input_type):
            print(f"Camera: {cam}, Input type: {mode}")
        
        video_paths, labels, subjects = find_folders(base_path)
        self.class_to_idx = {label: idx for idx, label in enumerate(set(labels))}

        train_indices = [i for i, subj in enumerate(subjects) if subj not in val_test_subjects]
        val_test_indices = [i for i, subj in enumerate(subjects) if subj in val_test_subjects]
        
        if subset == 'train':
            self.video_paths = [video_paths[i] for i in train_indices]
            self.labels = [labels[i] for i in train_indices]
        else:
            print(f"Validation on subjects: {val_test_subjects}")
            self.video_paths = [video_paths[i] for i in val_test_indices]
            self.labels = [labels[i] for i in val_test_indices]
        
        for cam, mod in zip(self.camera, self.input_type):
            print(f"Camera: {cam}, Input type: {mod}")
            self.samples = self._generate_samples(cam, mod)
            print(f"{subset} set: {len(self.samples)} samples")
    
    def _generate_samples(self, camera, input_type):
        samples = []
        for video_path, label in zip(self.video_paths, self.labels):
            frame_files = self.load_video_frames(video_path, camera_id=camera, input_type=input_type)  
            num_frames = len(frame_files)
            for start in range(0, num_frames - self.sample_duration*self.time_downsample + 1, self.stride):
                samples.append((video_path, start, label))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def load_video_frames(self, video_path, camera_id, input_type):
        subfolders = [f for f in glob.glob(os.path.join(video_path, '*', '*')) if os.path.isdir(f)] +  [f for f in glob.glob(os.path.join(video_path, '*',)) if os.path.isdir(f) and 'event' in f]

        if camera_id == 'event':
            modality_folder = next((f for f in subfolders if 'event' in f), None)
        elif camera_id == 'master':
            modality_folder = next((f for f in subfolders if input_type in f and ('master' in f or 'sub1' in f)), None)
        elif camera_id == 'sub2':
            modality_folder = next((f for f in subfolders if input_type in f and 'sub2' in f), None)
        else:
            raise ValueError(f"Unknown input_type: {input_type}")
        
        if modality_folder is None:
            raise FileNotFoundError(f"Could not find the required subfolder for camera_id: {camera_id} and input_type: {input_type}")
        frame_files = sorted(glob.glob(os.path.join(modality_folder, '*.jpg')) + glob.glob(os.path.join(modality_folder, '*.npy')), key=extract_frame_number)
        
        return frame_files
    
    def __getitem__(self, index):
        video_path, start_idx, label = self.samples[index]
        label_idx = self.class_to_idx[label]
        
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()

        frame_mod_lists = []
        num_modalities = len(self.input_type)
        num_masked = max(1, int(self.random_mask_fraction * num_modalities)) if self.random_mask_fraction > 0 and self.subset == 'train' else 0
        masked_indices = random.sample(range(num_modalities), num_masked) if num_masked > 0 else []
        
        for i, (cam, mod) in enumerate(zip(self.camera, self.input_type)):
            frame_files = self.load_video_frames(video_path, camera_id=cam, input_type=mod)
            frame_indices = [min(i, len(frame_files) - 1) for i in range(start_idx, start_idx + self.sample_duration*self.time_downsample, self.time_downsample)]
            
            clip = [frame_loader(frame_files[i], mode=mod) for i in frame_indices]

            if self.spatial_transform is not None:
                clip = [self.spatial_transform(img) for img in clip]
            
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
            
            if self.subset == 'train' and i in masked_indices:
                clip = torch.zeros_like(clip)  # Mask the entire clip only in training
            
            frame_mod_lists.append(clip)
        
        return frame_mod_lists, label_idx