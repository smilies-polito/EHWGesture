import os
import re
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from scipy.spatial import ConvexHull, distance
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

FPS = 120

def compute_distance(marker1, marker2):
    """Compute Euclidean distance between two markers using SciPy's distance.euclidean."""
    return np.array([distance.euclidean(p1, p2) for p1, p2 in zip(marker1, marker2)])

def moving_average_filter(signal, N=4):
    return np.convolve(signal, np.ones(N)/N, mode='same')


def find_combined_extrema(signal, global_distance, prominence=0.01):
    """
    Find combined indices of local minima and maxima in a 1D signal.
    Enforces a minimum separation (in samples) between consecutive extrema.
    This method computes the absolute derivative and then refines the extremum type.
    Returns:
        indices: array of indices (sorted)
        types: an array of strings ('min' or 'max') corresponding to each index.
    """
    # Compute the first derivative using central differences.
    derivative = np.gradient(signal)
    abs_deriv = np.abs(derivative)

    # Find candidate indices in the absolute derivative.
    candidate_idx, _ = find_peaks(abs_deriv, distance=global_distance, prominence=prominence)

    combined = []
    for idx in candidate_idx:
        # Skip edges.
        if idx == 0 or idx == len(signal) - 1:
            continue
        # Determine if this candidate is a minimum or maximum.
        if signal[idx] < signal[idx - 1] and signal[idx] < signal[idx + 1]:
            typ = 'min'
        elif signal[idx] > signal[idx - 1] and signal[idx] > signal[idx + 1]:
            typ = 'max'
        else:
            continue
        combined.append((idx, typ))

    # Sort by index.
    combined.sort(key=lambda x: x[0])
    if combined:
        indices = np.array([c[0] for c in combined])
        types = np.array([c[1] for c in combined])
    else:
        indices, types = np.array([]), np.array([])
    return indices, types

def compute_convex_hull_volume(points):
    """Compute convex hull volume given an array of 3D points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return np.nan

class Task(ABC):

    def __init__(self, tracking_data, task_name, handedness):
        self.tracking_data = tracking_data
        self.task_name = task_name
        self.ref_signal = None
        self.triggers = None
        self.handedness = handedness

    @abstractmethod
    def compute_ref_signal(self):
        pass

    @abstractmethod
    def compute_triggers(self, cadence):
        pass

    @abstractmethod
    def plot_reference(self):
        pass

    @abstractmethod
    def save_triggers(self, outpath):
        pass

# ----- FT Task: Finger Tapping (Thumb-Index) -----
class FT(Task):

    def __init__(self, tracking_data, task_name, handedness, cadence):
        self.__check_coordinates(tracking_data)
        super().__init__(tracking_data, task_name, handedness)
        self.cadence = cadence

    def __check_coordinates(self, tracking_data):
        try:
            self.thumb = tracking_data["Hand:Thumb"]
            self.index = tracking_data["Hand:Index"]
            # We also retrieve some extra markers if needed for other tasks.
            self.wrist = (tracking_data["Hand:Wrist_in"]+tracking_data["Hand:Wrist_out"])/2
            self.palm = tracking_data["Hand:Palm"]
        except KeyError as e:
            raise ValueError(f"Expected marker {e} not found in header mapping.")

    def compute_ref_signal(self):
        # Compute distance between thumb and index
        dist = compute_distance(np.array(self.thumb), np.array(self.index))
        self.ref_signal = moving_average_filter(dist)

    def compute_triggers(self, cadence):
        # Use find_peaks with a minimum separation of cadence/2 frames.
        self.triggers = find_peaks(-self.ref_signal, distance=cadence/2, height=-np.mean(self.ref_signal))[0]

    def plot_reference(self):

        plt.figure(figsize=(10, 6))
        plt.plot(self.tracking_data['Time'], self.ref_signal, label="Thumb-Index Distance")
        plt.plot(self.tracking_data['Time'].iloc[self.triggers], self.ref_signal[self.triggers],
                 'rv', markersize=8, label="Tapping triggers")
        plt.xlabel("Time")
        plt.ylabel("Distance")
        plt.title(f"{self.task_name} Trace with Detected Tapping Events")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def save_triggers(self, outpath):
        os.makedirs(outpath, exist_ok=True)
        to_save = pd.DataFrame({
            'Frame': self.tracking_data['Frame'].iloc[self.triggers],
            'Time': self.tracking_data['Time'].iloc[self.triggers],
            'Gesture': 'tapping'
        })
        to_save.to_csv(os.path.join(outpath, f'{self.task_name}_triggers.csv'), index=False)

# ----- OC Task: Open/Close (Middle-Wrist in/out) -----
class OC(Task):

    def __init__(self, tracking_data, task_name, cadence, handedness):
        self.__check_coordinates(tracking_data)
        super().__init__(tracking_data, task_name, handedness)
        self.cadence = cadence

    def __check_coordinates(self, tracking_data):
        try:
            self.middle = tracking_data["Hand:Middle"]
            self.pinkie = tracking_data["Hand:Pinkie"]
            self.thumb = tracking_data["Hand:Thumb"]
            self.wrist = (tracking_data["Hand:Wrist_in"] + tracking_data["Hand:Wrist_out"]) / 2
            self.palm = tracking_data["Hand:Palm"]
        except KeyError as e:
            raise ValueError(f"Expected marker {e} not found in header mapping.")

    def compute_triggers_with_grd(self, cadence):
        # For PS, extract both minima (palm orientation) and maxima (forehand orientation)
        # Compute gradient using np.gradient (central differences)
        grad = np.gradient(self.ref_signal)
        N = len(grad)

        # Define threshold: mean absolute value of gradient.
        threshold = np.mean(np.abs(grad))

        # Set minimum distance in samples.
        min_distance = int(cadence)/2
        zero_tol = 0.0005
        # Find candidate peaks in the gradient.
        # For maximum candidates: we want points where grad is a local maximum above threshold.
        candidate_max, _ = find_peaks(grad, height=threshold, distance=min_distance)
        # For minimum candidates: find peaks on -grad.
        candidate_min, _ = find_peaks(-grad, height=threshold, distance=min_distance)
        plot_debug = True
        if plot_debug:
            plt.figure(figsize=(12, 5))
            plt.plot(grad, label="Gradient")
            plt.plot(candidate_max, grad[candidate_max], "bo", label="Candidate Max")
            plt.plot(candidate_min, grad[candidate_min], "go", label="Candidate Min")
            plt.axhline(zero_tol, color='r', linestyle='--', label=f"Zero tol (+{zero_tol})")
            plt.axhline(-zero_tol, color='r', linestyle='--', label=f"Zero tol (-{zero_tol})")
            plt.xlabel("Frame")
            plt.ylabel("Gradient Value")
            plt.title("Candidate Extrema on Gradient (Pre-Moving)")
            plt.legend()
            plt.tight_layout()
            plt.show()

        # For each candidate, find the first index where the gradient crosses zero (with tolerance).

        def find_trigger(candidate_indices, is_max):
            triggers = []
            for idx in candidate_indices:
                # For a maximum candidate, we expect the gradient to eventually fall below +zero_tol.
                # For a minimum candidate, we expect the gradient to eventually rise above -zero_tol.
                # We check for a sign change between consecutive samples relative to the tolerance.
                for j in range(idx, N - 1):
                    if is_max:
                        # If the gradient at j is above zero_tol and at j+1 it is below or equal to zero_tol,
                        # we consider that a zero crossing.
                        if grad[j] > zero_tol and grad[j + 1] <= zero_tol:
                            triggers.append(j + 1)
                            break
                    else:
                        if grad[j] < -zero_tol and grad[j + 1] >= -zero_tol:
                            triggers.append(j + 1)
                            break
            return np.array(triggers, dtype=int)

        triggers_max = find_trigger(candidate_max, True)
        triggers_min = find_trigger(candidate_min, False)

        if plot_debug:
            plt.figure(figsize=(12, 5))
            plt.plot(grad, label="Gradient")
            plt.plot(triggers_max, grad[triggers_max], "bo", label="Final Max Trigger")
            plt.plot(triggers_min, grad[triggers_min], "go", label="Final Min Trigger")
            plt.axhline(zero_tol, color='r', linestyle='--', label=f"Zero tol (+{zero_tol})")
            plt.axhline(-zero_tol, color='r', linestyle='--', label=f"Zero tol (-{zero_tol})")
            plt.xlabel("Frame")
            plt.ylabel("Gradient Value")
            plt.title("Final Trigger Positions on Gradient (Post-Moving)")
            plt.legend()
            plt.tight_layout()
            plt.show()

        self.triggers = {'max': triggers_max, 'min': triggers_min}

    def compute_ref_signal(self):
        # Use y component of middle
        self.ref_signal = moving_average_filter(np.array(self.middle)[:, 1], N=3)

    def compute_triggers(self, cadence):
        # For OC, both minima and maxima are relevant (closed = min, open = max)
        minima, maxima = find_peaks(-self.ref_signal, distance=cadence, height=-np.mean(self.ref_signal))[0], find_peaks(self.ref_signal, distance=cadence, height=np.mean(self.ref_signal))[0]
        # For simplicity, we combine both arrays (in practice, you might want to distinguish them)
        self.triggers = {'min': minima, 'max': maxima}

    def plot_reference(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.tracking_data['Time'], self.ref_signal, label="Middle-Wrist Distance")
        if self.triggers:
            for typ, idx_arr in self.triggers.items():
                marker = 'rv' if typ=='min' else 'r^'
                label = "Closed Hand" if typ=='min' else "Open Hand"
                plt.plot(self.tracking_data['Time'].iloc[idx_arr], self.ref_signal[idx_arr],
                         marker, markersize=8, label=label)
        plt.xlabel("Time")
        plt.ylabel("Distance")
        plt.title(f"{self.task_name} Trace with Detected Open/Close Events")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def save_triggers(self, outpath):
        out_list = []
        for typ, idx_arr in self.triggers.items():
            for idx in idx_arr:
                gesture = "closed" if typ=='min' else "open"
                out_list.append({
                    'Frame': self.tracking_data['Frame'].iloc[idx],
                    'Time': self.tracking_data['Time'].iloc[idx],
                    'Gesture': gesture
                })
        to_save = pd.DataFrame(out_list)
        os.makedirs(outpath, exist_ok=True)
        to_save.to_csv(os.path.join(outpath, f'{self.task_name}_triggers.csv'), index=False)

# ----- NOSE Task: (Ring-Index) -----
class NOSE(Task):

    def __init__(self, tracking_data, task_name, cadence, handedness):
        self.__check_coordinates(tracking_data)
        super().__init__(tracking_data, task_name, handedness)
        self.cadence = cadence

    def __check_coordinates(self, tracking_data):
        try:
            self.nose = tracking_data["Hand:Ring"]
            self.index = tracking_data["Hand:Index"]
            self.thumb = tracking_data["Hand:Thumb"]
            self.wrist = (tracking_data["Hand:Wrist_in"] + tracking_data["Hand:Wrist_out"]) / 2
            self.palm = tracking_data["Hand:Palm"]
        except KeyError as e:
            raise ValueError(f"Expected marker {e} not found in header mapping.")

    def compute_ref_signal(self):
        self.ref_signal = moving_average_filter(np.array(self.thumb)[:, 2])

    def compute_triggers(self, cadence):
        # For NOSE, extract both minima (inward) and maxima (outward)
        minima, maxima = find_peaks(-self.ref_signal, distance=cadence, height=-np.mean(self.ref_signal))[0], find_peaks(self.ref_signal, distance=cadence, height=np.mean(self.ref_signal))[0]
        self.triggers = {'min': minima, 'max': maxima}

    def plot_reference(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.tracking_data['Time'], self.ref_signal, label="Z-variation of Thumb landmark")
        if self.triggers:
            for typ, idx_arr in self.triggers.items():
                marker = 'rv' if typ=='min' else 'r^'
                label = "Inward" if typ=='max' else "Outward"
                plt.plot(self.tracking_data['Time'].iloc[idx_arr], self.ref_signal[idx_arr],
                         marker, markersize=8, label=label)
        plt.xlabel("Time")
        plt.ylabel("Distance")
        plt.title(f"{self.task_name} Trace with Detected NOSE Events")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def compute_triggers_with_grd(self, cadence):
        # For PS, extract both minima (palm orientation) and maxima (forehand orientation)
        # Compute gradient using np.gradient (central differences)
        grad = np.gradient(self.ref_signal)
        N = len(grad)
        # Define threshold: mean absolute value of gradient.
        threshold = np.mean(np.abs(grad))

        # Set minimum distance in samples.
        min_distance = int(cadence)
        zero_tol = 0.001
        # Find candidate peaks in the gradient.
        # For maximum candidates: we want points where grad is a local maximum above threshold.
        candidate_max, _ = find_peaks(grad, height=threshold, distance=min_distance)
        # For minimum candidates: find peaks on -grad.
        candidate_min, _ = find_peaks(-grad, height=threshold, distance=min_distance)
        plot_debug = True
        if plot_debug:
            plt.figure(figsize=(12, 5))
            plt.plot(grad, label="Gradient")
            plt.plot(candidate_max, grad[candidate_max], "bo", label="Candidate Max")
            plt.plot(candidate_min, grad[candidate_min], "go", label="Candidate Min")
            plt.axhline(zero_tol, color='r', linestyle='--', label=f"Zero tol (+{zero_tol})")
            plt.axhline(-zero_tol, color='r', linestyle='--', label=f"Zero tol (-{zero_tol})")
            plt.xlabel("Frame")
            plt.ylabel("Gradient Value")
            plt.title("Candidate Extrema on Gradient (Pre-Moving)")
            plt.legend()
            plt.tight_layout()
            plt.show()

        # For each candidate, find the first index where the gradient crosses zero (with tolerance).

        def find_trigger(candidate_indices, is_max):
            triggers = []
            for idx in candidate_indices:
                # For a maximum candidate, we expect the gradient to eventually fall below +zero_tol.
                # For a minimum candidate, we expect the gradient to eventually rise above -zero_tol.
                # We check for a sign change between consecutive samples relative to the tolerance.
                for j in range(idx, N - 1):
                    if is_max:
                        # If the gradient at j is above zero_tol and at j+1 it is below or equal to zero_tol,
                        # we consider that a zero crossing.
                        if grad[j] > zero_tol and grad[j + 1] <= zero_tol:
                            triggers.append(j + 1)
                            break
                    else:
                        if grad[j] < -zero_tol and grad[j + 1] >= -zero_tol:
                            triggers.append(j + 1)
                            break
            return np.array(triggers, dtype=int)

        triggers_max = find_trigger(candidate_max, True)
        triggers_min = find_trigger(candidate_min, False)

        if plot_debug:
            plt.figure(figsize=(12, 5))
            plt.plot(grad, label="Gradient")
            plt.plot(triggers_max, grad[triggers_max], "bo", label="Final Max Trigger")
            plt.plot(triggers_min, grad[triggers_min], "go", label="Final Min Trigger")
            plt.axhline(zero_tol, color='r', linestyle='--', label=f"Zero tol (+{zero_tol})")
            plt.axhline(-zero_tol, color='r', linestyle='--', label=f"Zero tol (-{zero_tol})")
            plt.xlabel("Frame")
            plt.ylabel("Gradient Value")
            plt.title("Final Trigger Positions on Gradient (Post-Moving)")
            plt.legend()
            plt.tight_layout()
            plt.show()

        self.triggers = {'max': triggers_max, 'min': triggers_min}

    def save_triggers(self, outpath):
        out_list = []
        for typ, idx_arr in self.triggers.items():
            gesture = "inward" if typ=='min' else "outward"
            for idx in idx_arr:
                out_list.append({
                    'Frame': self.tracking_data['Frame'].iloc[idx],
                    'Time': self.tracking_data['Time'].iloc[idx],
                    'Gesture': gesture
                })
        to_save = pd.DataFrame(out_list)
        os.makedirs(outpath, exist_ok=True)
        to_save.to_csv(os.path.join(outpath, f'{self.task_name}_triggers.csv'), index=False)

# ----- PS Task: Palm Rotation Angle (Thumb x coordinate) -----
class PS(Task):

    def __init__(self, tracking_data, task_name, cadence, handedness):
        self.__check_coordinates(tracking_data)
        super().__init__(tracking_data, task_name, handedness)
        self.cadence = cadence

    def __check_coordinates(self, tracking_data):
        try:
            self.thumb = tracking_data["Hand:Thumb"]
            self.pinkie = tracking_data["Hand:Pinkie"]
            self.middle = tracking_data["Hand:Pinkie"]
            self.wrin = tracking_data["Hand:Wrist_in"]
            self.wrout= tracking_data["Hand:Wrist_out"]
            self.palm = tracking_data["Hand:Palm"]
        except KeyError as e:
            raise ValueError(f"Expected marker {e} not found in header mapping.")

    def compute_ref_signal(self):
        self.ref_signal = moving_average_filter(np.array(self.thumb)[:, 0])

    def compute_triggers(self, cadence):
        # For PS, extract both minima (palm orientation) and maxima (forehand orientation)
        minima, maxima = find_peaks(-self.ref_signal, distance=(cadence), height=-np.mean(self.ref_signal))[0], find_peaks(self.ref_signal, distance=(cadence), height=np.mean(self.ref_signal))[0]
        self.triggers = {'min': minima, 'max': maxima}

    def compute_triggers_with_grd(self, cadence):
        # For PS, extract both minima (palm orientation) and maxima (forehand orientation)
        # Compute gradient using np.gradient (central differences)
        grad = np.gradient(self.ref_signal)
        N = len(grad)

        # Define threshold: mean absolute value of gradient.
        threshold = np.mean(np.abs(grad))

        # Set minimum distance in samples.
        min_distance = int(cadence)
        zero_tol = 0.0005
        # Find candidate peaks in the gradient.
        # For maximum candidates: we want points where grad is a local maximum above threshold.
        candidate_max, _ = find_peaks(grad, height=threshold, distance=min_distance)
        # For minimum candidates: find peaks on -grad.
        candidate_min, _ = find_peaks(-grad, height=threshold, distance=min_distance)
        plot_debug=True
        if plot_debug:
            plt.figure(figsize=(12, 5))
            plt.plot(grad, label="Gradient")
            plt.plot(candidate_max, grad[candidate_max], "bo", label="Candidate Max")
            plt.plot(candidate_min, grad[candidate_min], "go", label="Candidate Min")
            plt.axhline(zero_tol, color='r', linestyle='--', label=f"Zero tol (+{zero_tol})")
            plt.axhline(-zero_tol, color='r', linestyle='--', label=f"Zero tol (-{zero_tol})")
            plt.xlabel("Frame")
            plt.ylabel("Gradient Value")
            plt.title("Candidate Extrema on Gradient (Pre-Moving)")
            plt.legend()
            plt.tight_layout()
            plt.show()


        # For each candidate, find the first index where the gradient crosses zero (with tolerance).

        def find_trigger(candidate_indices, is_max):
            triggers = []
            for idx in candidate_indices:
                # For a maximum candidate, we expect the gradient to eventually fall below +zero_tol.
                # For a minimum candidate, we expect the gradient to eventually rise above -zero_tol.
                # We check for a sign change between consecutive samples relative to the tolerance.
                for j in range(idx, N - 1):
                    if is_max:
                        # If the gradient at j is above zero_tol and at j+1 it is below or equal to zero_tol,
                        # we consider that a zero crossing.
                        if grad[j] > zero_tol and grad[j + 1] <= zero_tol:
                            triggers.append(j + 1)
                            break
                    else:
                        if grad[j] < -zero_tol and grad[j + 1] >= -zero_tol:
                            triggers.append(j + 1)
                            break
            return np.array(triggers, dtype=int)

        triggers_max = find_trigger(candidate_max, True)
        triggers_min = find_trigger(candidate_min, False)

        if plot_debug:
            plt.figure(figsize=(12, 5))
            plt.plot(grad, label="Gradient")
            plt.plot(triggers_max, grad[triggers_max], "bo", label="Final Max Trigger")
            plt.plot(triggers_min, grad[triggers_min], "go", label="Final Min Trigger")
            plt.axhline(zero_tol, color='r', linestyle='--', label=f"Zero tol (+{zero_tol})")
            plt.axhline(-zero_tol, color='r', linestyle='--', label=f"Zero tol (-{zero_tol})")
            plt.xlabel("Frame")
            plt.ylabel("Gradient Value")
            plt.title("Final Trigger Positions on Gradient (Post-Moving)")
            plt.legend()
            plt.tight_layout()
            plt.show()

        self.triggers = {'max': triggers_max, 'min': triggers_min}

    def plot_reference(self):

        plt.figure(figsize=(10, 6))
        plt.plot(self.tracking_data['Time'], self.ref_signal, label="Thumb-Pinkie Angle")
        if self.triggers:
            for typ, idx_arr in self.triggers.items():
                marker = 'rv' if typ=='min' else 'r^'
                label = "Palm" if typ=='min' else "Forehand"
                plt.plot(self.tracking_data['Time'].iloc[idx_arr], self.ref_signal[idx_arr],
                         marker, markersize=8, label=label)
        plt.xlabel("Time")
        plt.ylabel("Angle (deg)")
        plt.title(f"{self.task_name} Trace with Detected PS Events")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def save_triggers(self, outpath):
        out_list = []
        for type, idx_arr in self.triggers.items():
            gesture = "palm" if ((type=='min' and self.handedness=='LEFT') or (type=='max' and self.handedness=='RIGHT')) else "forehand"
            for idx in idx_arr:
                out_list.append({
                    'Frame': self.tracking_data['Frame'].iloc[idx],
                    'Time': self.tracking_data['Time'].iloc[idx],
                    'Gesture': gesture
                })
        to_save = pd.DataFrame(out_list)
        os.makedirs(outpath, exist_ok=True)
        to_save.to_csv(os.path.join(outpath, f'{self.task_name}_triggers.csv'), index=False)

# ----- TR Task: Sway Area -----
class TR(Task):

    def __init__(self, tracking_data, task_name, handedness):
        self.__check_coordinates(tracking_data)
        super().__init__(tracking_data, task_name, handedness)

    def __check_coordinates(self, tracking_data):
        try:
            self.thumb = tracking_data["Hand:Thumb"]
        except KeyError as e:
            raise ValueError(f"Expected marker {e} not found in header mapping.")

    def compute_ref_signal(self):
        # Compute sway area as convex hull volume for the entire thumb trajectory.
        points = np.column_stack((np.array(self.thumb)[:,0], np.array(self.thumb)[:,1], np.array(self.thumb)[:,2]))
        self.ref_signal = compute_convex_hull_volume(points)
        # For TR, ref_signal is a single value.

    def compute_triggers(self, cadence=None):
        # TR is a single computed value, so we do not have a time-series trigger.
        print("TR cannot be used for triggering, this method call is unaffective.") #TODO: check a nicer way to do this
        self.triggers = None

    def plot_reference(self):
        plt.figure(figsize=(10, 6))
        plt.title(f"{self.task_name} Sway Area: {self.ref_signal:.3f}")
        plt.text(0.5, 0.5, f"Sway Area: {self.ref_signal:.3f}", ha='center', va='center', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def save_triggers(self, outpath):
        # For TR, save the single computed value.
        print("TR cannot be used for triggering, this method call is unaffective.")
