import numpy as np
import mediapipe as mp
mp_hands = mp.solutions.hands

class FingerTapping:
    def compute_metrics(self, landmarks, image_shape):
        thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        thumb_pos = np.array([thumb_tip.x * image_shape[1], thumb_tip.y * image_shape[0]])
        index_pos = np.array([index_tip.x * image_shape[1], index_tip.y * image_shape[0]])

        distance = np.linalg.norm(thumb_pos - index_pos)
        return distance

    def aggregate_metrics(self, metrics):
        return [(a + b) / 2 for a, b in zip(metrics[0], metrics[1])]

class NoseTapping:
    def compute_metrics(self, landmarks, image_shape):
        index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        x, y = index_tip.x * image_shape[1], index_tip.y * image_shape[0]
        return x
    
    
    def aggregate_metrics(self, metrics):
        return [(b - a) / 2 for a, b in zip(metrics[0], metrics[1])]

class OpenClose:
    def compute_metrics(self, landmarks, image_shape):
        wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
        middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

        wrist_pos = np.array([wrist.x * image_shape[1], wrist.y * image_shape[0]])
        middle_tip_pos = np.array([middle_tip.x * image_shape[1], middle_tip.y * image_shape[0]])

        distance = np.linalg.norm(wrist_pos - middle_tip_pos)
        return distance
    
    def aggregate_metrics(self, metrics):
        return [(a + b) / 2 for a, b in zip(metrics[0], metrics[1])]
    
class PronoSupination:
    def compute_metrics(self, landmarks, image_shape):
        index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        distance_x = index_tip.x - pinky_tip.x
        return distance_x
    
    def aggregate_metrics(self, metrics):
        window_size = 11
        aggregated = [(-a -b) / 2 for a, b in zip(metrics[0], metrics[1])]
        smoothed = np.convolve(aggregated, np.ones(window_size)/window_size, mode='same')
        return smoothed