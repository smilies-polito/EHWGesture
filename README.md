# EHWGesture - A dataset for multimodal understanding of clinical gestures

Hand gesture understanding is essential for several applications in human-computer interaction, including automatic clinical assessment of hand dexterity. While deep learning has advanced static gesture recognition, dynamic gesture understanding remains challenging due to complex spatiotemporal variations. Moreover, existing datasets often lack multimodal and multi-view diversity, precise ground-truth tracking, and an action quality component embedded within gestures. This paper introduces EHWGesture, a multimodal video dataset for gesture understanding featuring five clinically relevant gestures. It includes over 1,100 recordings (∼6 hours), captured from 25 healthy subjects using two high-resolution RGB-Depth cameras and an event camera. A motion capture system provides precise ground-truth hand landmark tracking, and all devices are spatially calibrated and synchronized to ensure cross-modal alignment. Moreover, to embed an action quality task within gesture understanding, collected recordings are organized in classes of execution speed that mirror clinical evaluations of hand dexterity. Baseline experiments highlight the dataset’s potential for gesture classification, gesture trigger detection, and action quality assessment. Thus, EHWGesture can serve as a comprehensive benchmark for advancing multimodal clinical gesture understanding.

The EHWGesture is publicly available for download here under CCBY4 license: [EHWGesture-download here](https://drive.cloud.polito.it/index.php/s/5cFBs6HFtrK7PXf). 

## Release Notes
EHWGesture v1.0:
- First release is online! MOCAP data for test subjects are already available, while those for the remaining subjects are currently under processing and will be soon available for all subjects. 

## Dataset content and structure 

The dataset contains separate folders for each modality, annotations, global calibration among all sensors and additional metadata. The repo has the following structure 

## Contents
The repository contains main scripts for reproducing the experiment in 

### ScriptsEvent
- `aedat_analizer_script.py`: Analyze AEDAT files.
- `extract_event_frames.py`: Extract frames from event data.
- `find_sync_triggers_event.py`: Find synchronization triggers in event data.
- `hand_trajectory_parallel_extraction_event.py`: Extract hand trajectories in parallel from event data.
- `visualize_npy_event.py`: Visualize `.npy` event data.

### ScriptsKinects
- `hand_trajectory_parallel_extraction.py`: Extract hand trajectories in parallel from Kinect data - useful to identify cropping windows
- `hand_trajetory_plotting.py`: Plot hand positions from raw data.

### ScriptsMOCAP
- `groundtruth_for_triggering.py`: Generate ground truth for triggering.
- `temporal_reallignment.py`: Perform temporal realignment of MOCAP data.
- `TrackingData.py`: Handle tracking data.

### TrainingCode
- `exp_num_frames.sh`: Paper experiment for window length.
- `exp_time_downsample.sh`: Paper experiment for time downsampling.
- `main.py`: Main script for training.

### TriggerDetection
- Scripts for the event localization task
