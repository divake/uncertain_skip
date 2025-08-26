"""
Utility functions for MOT format handling and evaluation
"""

import numpy as np
import pandas as pd
from pathlib import Path
import cv2


def load_mot_gt(gt_file):
    """
    Load MOT ground truth file
    Format: frame_id, person_id, bbox_left, bbox_top, bbox_width, bbox_height, conf, class, visibility
    """
    if not Path(gt_file).exists():
        print(f"Warning: Ground truth file {gt_file} not found")
        return []
    
    gt_data = []
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                # Only include person class (class 1 in MOT)
                if len(parts) >= 8 and int(parts[7]) != 1:
                    continue
                
                frame_id = int(parts[0])
                track_id = int(parts[1])
                bbox_left = float(parts[2])
                bbox_top = float(parts[3])
                bbox_width = float(parts[4])
                bbox_height = float(parts[5])
                
                # Only include visible objects (visibility > 0.25 if provided)
                if len(parts) >= 9:
                    visibility = float(parts[8])
                    if visibility < 0.25:
                        continue
                
                gt_data.append([frame_id, track_id, bbox_left, bbox_top, bbox_width, bbox_height])
    
    return gt_data


def save_mot_results(output_file, tracks):
    """
    Save tracking results in MOT format
    Format: frame_id, track_id, bbox_left, bbox_top, bbox_width, bbox_height, conf, -1, -1, -1
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for track in tracks:
            # Ensure we have the right format
            if len(track) >= 6:
                line = f"{int(track[0])},{int(track[1])},{track[2]:.2f},{track[3]:.2f},"
                line += f"{track[4]:.2f},{track[5]:.2f}"
                
                # Add confidence if available
                if len(track) > 6:
                    line += f",{track[6]:.3f}"
                else:
                    line += ",1.0"
                
                # Add MOT format placeholders
                line += ",-1,-1,-1\n"
                f.write(line)


def convert_bbox_format(bbox, from_format='xyxy', to_format='xywh'):
    """
    Convert bounding box format
    Formats:
        xyxy: [x1, y1, x2, y2] - top-left and bottom-right corners
        xywh: [x, y, w, h] - top-left corner and width/height
        xcyc: [xc, yc, w, h] - center point and width/height
    """
    bbox = np.array(bbox)
    
    if from_format == 'xyxy' and to_format == 'xywh':
        # Convert from corners to top-left + dimensions
        x1, y1, x2, y2 = bbox[:4]
        w = x2 - x1
        h = y2 - y1
        return [x1, y1, w, h]
    
    elif from_format == 'xywh' and to_format == 'xyxy':
        # Convert from top-left + dimensions to corners
        x, y, w, h = bbox[:4]
        x2 = x + w
        y2 = y + h
        return [x, y, x2, y2]
    
    elif from_format == 'xyxy' and to_format == 'xcyc':
        # Convert from corners to center + dimensions
        x1, y1, x2, y2 = bbox[:4]
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return [xc, yc, w, h]
    
    elif from_format == 'xcyc' and to_format == 'xyxy':
        # Convert from center + dimensions to corners
        xc, yc, w, h = bbox[:4]
        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2
        return [x1, y1, x2, y2]
    
    elif from_format == 'xywh' and to_format == 'xcyc':
        # Convert from top-left + dimensions to center + dimensions
        x, y, w, h = bbox[:4]
        xc = x + w / 2
        yc = y + h / 2
        return [xc, yc, w, h]
    
    elif from_format == 'xcyc' and to_format == 'xywh':
        # Convert from center + dimensions to top-left + dimensions
        xc, yc, w, h = bbox[:4]
        x = xc - w / 2
        y = yc - h / 2
        return [x, y, w, h]
    
    else:
        # No conversion needed or unsupported conversion
        return bbox.tolist() if isinstance(bbox, np.ndarray) else bbox


def calculate_iou(bbox1, bbox2, format='xywh'):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    """
    if format == 'xywh':
        # Convert to xyxy format for easier calculation
        bbox1 = convert_bbox_format(bbox1, 'xywh', 'xyxy')
        bbox2 = convert_bbox_format(bbox2, 'xywh', 'xyxy')
    
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    # Avoid division by zero
    if union == 0:
        return 0
    
    return intersection / union


def format_detection_for_tracking(detection, frame_num=None):
    """
    Format detection for tracking input
    Input: detection dict or array
    Output: numpy array [x1, y1, x2, y2, confidence]
    """
    if isinstance(detection, dict):
        # Handle dictionary format from YOLO
        bbox = detection.get('bbox', detection.get('box', []))
        conf = detection.get('confidence', detection.get('conf', 1.0))
        
        # Check bbox format
        if len(bbox) == 4:
            if 'format' in detection:
                if detection['format'] == 'xywh':
                    bbox = convert_bbox_format(bbox, 'xywh', 'xyxy')
                elif detection['format'] == 'xcyc':
                    bbox = convert_bbox_format(bbox, 'xcyc', 'xyxy')
            # Default assume xywh
            else:
                bbox = convert_bbox_format(bbox, 'xywh', 'xyxy')
        
        return np.array(bbox + [conf])
    
    elif isinstance(detection, (list, np.ndarray)):
        # Handle array format
        detection = np.array(detection)
        
        # If bbox is in xywh format (MOT standard), convert to xyxy
        if len(detection) >= 4:
            # Assume first 4 values are bbox in xywh format
            bbox = convert_bbox_format(detection[:4], 'xywh', 'xyxy')
            
            # Get confidence if available
            if len(detection) > 4:
                conf = detection[4]
            else:
                conf = 1.0
            
            return np.array(bbox + [conf])
    
    return np.array(detection)


def interpolate_tracks(tracks, max_gap=10):
    """
    Interpolate missing detections in tracks
    """
    if not tracks:
        return tracks
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(tracks, columns=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'x1', 'x2', 'x3'])
    
    interpolated = []
    
    for track_id in df['id'].unique():
        track_data = df[df['id'] == track_id].sort_values('frame')
        
        # Check for gaps
        frames = track_data['frame'].values
        
        if len(frames) > 1:
            for i in range(len(frames) - 1):
                gap = frames[i+1] - frames[i]
                
                if gap > 1 and gap <= max_gap:
                    # Interpolate missing frames
                    start_row = track_data.iloc[i]
                    end_row = track_data.iloc[i+1]
                    
                    for j in range(1, gap):
                        alpha = j / gap
                        interp_row = start_row.copy()
                        interp_row['frame'] = frames[i] + j
                        interp_row['x'] = start_row['x'] * (1-alpha) + end_row['x'] * alpha
                        interp_row['y'] = start_row['y'] * (1-alpha) + end_row['y'] * alpha
                        interp_row['w'] = start_row['w'] * (1-alpha) + end_row['w'] * alpha
                        interp_row['h'] = start_row['h'] * (1-alpha) + end_row['h'] * alpha
                        interp_row['conf'] = min(start_row['conf'], end_row['conf']) * 0.9
                        
                        interpolated.append(interp_row.values.tolist())
    
    # Combine original and interpolated tracks
    all_tracks = tracks + interpolated
    all_tracks.sort(key=lambda x: (x[0], x[1]))  # Sort by frame then ID
    
    return all_tracks


def filter_low_confidence_tracks(tracks, min_confidence=0.3, min_length=5):
    """
    Filter out low confidence and short tracks
    """
    if not tracks:
        return tracks
    
    # Group tracks by ID
    tracks_by_id = {}
    for track in tracks:
        track_id = int(track[1])
        if track_id not in tracks_by_id:
            tracks_by_id[track_id] = []
        tracks_by_id[track_id].append(track)
    
    # Filter tracks
    filtered_tracks = []
    for track_id, track_list in tracks_by_id.items():
        # Check track length
        if len(track_list) < min_length:
            continue
        
        # Check average confidence
        avg_conf = np.mean([t[6] if len(t) > 6 else 1.0 for t in track_list])
        if avg_conf < min_confidence:
            continue
        
        filtered_tracks.extend(track_list)
    
    return filtered_tracks


def visualize_tracks_on_image(image, tracks, frame_num, gt_tracks=None):
    """
    Visualize tracks on an image
    """
    img = image.copy()
    
    # Draw predicted tracks
    frame_tracks = [t for t in tracks if t[0] == frame_num]
    
    for track in frame_tracks:
        track_id = int(track[1])
        x, y, w, h = track[2:6]
        
        # Generate color based on track ID
        color = ((track_id * 50) % 255, (track_id * 100) % 255, (track_id * 150) % 255)
        
        # Draw bounding box
        cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
        
        # Draw track ID
        cv2.putText(img, f"ID:{track_id}", (int(x), int(y-10)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw ground truth if available
    if gt_tracks:
        frame_gt = [t for t in gt_tracks if t[0] == frame_num]
        
        for gt in frame_gt:
            x, y, w, h = gt[2:6]
            
            # Draw GT in green
            cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 1)
    
    return img


def calculate_track_metrics(tracks, ground_truth):
    """
    Calculate basic tracking metrics
    """
    if not tracks or not ground_truth:
        return {
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'num_tracks': 0,
            'num_gt': 0
        }
    
    # Group by frame
    tracks_by_frame = {}
    gt_by_frame = {}
    
    for track in tracks:
        frame = int(track[0])
        if frame not in tracks_by_frame:
            tracks_by_frame[frame] = []
        tracks_by_frame[frame].append(track)
    
    for gt in ground_truth:
        frame = int(gt[0])
        if frame not in gt_by_frame:
            gt_by_frame[frame] = []
        gt_by_frame[frame].append(gt)
    
    # Calculate matches
    tp = 0
    fp = 0
    fn = 0
    
    for frame in set(list(tracks_by_frame.keys()) + list(gt_by_frame.keys())):
        frame_tracks = tracks_by_frame.get(frame, [])
        frame_gt = gt_by_frame.get(frame, [])
        
        # Simple matching based on IoU
        matched_gt = set()
        
        for track in frame_tracks:
            best_iou = 0
            best_gt_idx = -1
            
            for idx, gt in enumerate(frame_gt):
                iou = calculate_iou(track[2:6], gt[2:6], format='xywh')
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            
            if best_iou > 0.5:  # IoU threshold for matching
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
        
        fn += len(frame_gt) - len(matched_gt)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'num_tracks': len(set([t[1] for t in tracks])),
        'num_gt': len(set([g[1] for g in ground_truth]))
    }