# EHWGesture
Example scripts for EHWGesture dataset


## Contents

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