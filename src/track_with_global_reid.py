import os
import cv2
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torchreid

VIDEO_PATH = "../assets/15sec_input_720p.mp4"
MODEL_PATH = "../assets/best.pt"
OUTPUT_DIR = "../output/tracked_frames"
OUTPUT_VIDEO_PATH = "../output/tracked_video_reid_final.mp4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

global_id_counter = 0
active_tracks = {}
inactive_gallery = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reid_model = torchreid.models.build_model('osnet_x1_0', num_classes=1000, pretrained=True)
reid_model.to(device)
reid_model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def extract_features(image_crop):
    try:
        img = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = reid_model(tensor)
        return features.cpu().numpy().flatten()
    except:
        return None

def match_in_gallery(features, used_global_ids, threshold=0.7):
    if not inactive_gallery:
        return None
    filtered = [g for g in inactive_gallery if g['global_id'] not in used_global_ids]
    if not filtered:
        return None
    gallery_features = [g['features'] for g in filtered]
    gallery_ids = [g['global_id'] for g in filtered]
    sims = cosine_similarity([features], gallery_features)[0]
    best_idx = np.argmax(sims)
    if sims[best_idx] > threshold:
        return gallery_ids[best_idx]
    return None

def assign_global_id(track_id, bbox, frame, used_global_ids):
    global global_id_counter
    if track_id in active_tracks:
        global_id = active_tracks[track_id]['global_id']
        if global_id not in used_global_ids:
            return global_id
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    features = extract_features(crop)
    if features is None:
        return None
    matched_global_id = match_in_gallery(features, used_global_ids)
    if matched_global_id is not None:
        global_id = matched_global_id
    else:
        global_id_counter += 1
        global_id = global_id_counter
        inactive_gallery.append({'global_id': global_id, 'features': features})
    active_tracks[track_id] = {'global_id': global_id, 'features': features}
    used_global_ids.add(global_id)
    return global_id

def retire_lost_tracks(current_track_ids):
    lost_ids = set(active_tracks.keys()) - set(current_track_ids)
    for tid in lost_ids:
        if 'features' in active_tracks[tid]:
            inactive_gallery.append({
                'global_id': active_tracks[tid]['global_id'],
                'features': active_tracks[tid]['features']
            })
    for tid in lost_ids:
        del active_tracks[tid]

def draw_frame(frame, results, class_names):
    annotated = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color_map = {
        'player': (255, 255, 255),
        'referee': (0, 215, 255),
        'goalkeeper': (0, 0, 255)
    }
    used_global_ids = set()
    current_frame_track_ids = []
    if results.boxes.id is not None:
        boxes = results.boxes.data.cpu().numpy()
        for *xyxy, track_id, conf, cls_id in boxes:
            x1, y1, x2, y2 = map(int, xyxy)
            track_id = int(track_id)
            cls_id = int(cls_id)
            label = class_names.get(cls_id, f"class{cls_id}")
            if label == 'ball':
                continue
            current_frame_track_ids.append(track_id)
            global_id = assign_global_id(track_id, (x1, y1, x2, y2), frame, used_global_ids)
            color = color_map.get(label, (128, 128, 128))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            text = f"ID: {global_id}"
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, text, (x1 + 2, y1 - 4), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    retire_lost_tracks(current_frame_track_ids)
    return annotated

def run_tracking(video_path, model_path, output_dir, output_video_path):
    model = YOLO(model_path)
    class_names = model.names
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(source=frame, persist=True, conf=0.4, verbose=False)[0]
        output_frame = draw_frame(frame, results, class_names)
        cv2.imwrite(os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg"), output_frame)
        out_writer.write(output_frame)
        frame_idx += 1
    cap.release()
    out_writer.release()
    print(f"Processed {frame_idx} frames. Video saved to {output_video_path}")

if __name__ == "__main__":
    run_tracking(VIDEO_PATH, MODEL_PATH, OUTPUT_DIR, OUTPUT_VIDEO_PATH)
