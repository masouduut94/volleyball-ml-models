from typing import List
import numpy as np
import cv2
import supervision as sv

from ..core.data_structures import Detection, PlayerKeyPoints


def det2supervision(detections: List[Detection | PlayerKeyPoints], tracking_instance: sv.ByteTrack = None):
    if detections:
        boxes = []
        confidences = []
        class_ids = []
        labels = []

        for detection in detections:
            if detection.bbox is not None:
                bbox = detection.bbox
                boxes.append([bbox.x1, bbox.y1, bbox.x2, bbox.y2])
                confidences.append(detection.confidence)
                class_ids.append(detection.class_id)
                labels.append(f"{detection.class_name}: {detection.confidence:.2f}")

        if boxes:
            sv_detections = sv.Detections(
                xyxy=np.array(boxes),
                confidence=np.array(confidences),
                class_id=np.array(class_ids)
            )
            if tracking_instance:
                sv_detections = tracking_instance.update_with_detections(sv_detections)
            return sv_detections

    return sv.Detections.empty()


def draw_trajectory(annotated_frame: np.ndarray, ball_trajectory: List, trail_num=8):
    # Keep only last 8 points for trailing effect
    recent_trajectory = ball_trajectory[-trail_num:]
    if len(recent_trajectory) > 1:
        # Draw trajectory line using OpenCV
        for i in range(1, len(recent_trajectory)):
            # Calculate alpha (transparency) for fading effect
            alpha = i / len(recent_trajectory)
            thickness = max(1, int(3 * alpha))  # TODO: Test what happens with bigger values than 3

            # Convert points to integers
            pt1 = (int(recent_trajectory[i - 1][0]), int(recent_trajectory[i - 1][1]))
            pt2 = (int(recent_trajectory[i][0]), int(recent_trajectory[i][1]))

            # Draw line segment with varying thickness for trailing effect
            cv2.line(annotated_frame, pt1, pt2, (0, 255, 255), thickness)  # Yellow trail

        # Draw current ball position as a circle
        if recent_trajectory:
            current_pos = (int(recent_trajectory[-1][0]), int(recent_trajectory[-1][1]))
            cv2.circle(annotated_frame, current_pos, 5, (0, 0, 255), -1)  # Red dot

