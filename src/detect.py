import os
import cv2
from ultralytics import YOLO

VIDEO_PATH = "../assets/15sec_input_720p.mp4"
MODEL_PATH = "../assets/best.pt"
OUTPUT_DIR = "../output/detected_frames"
OUTPUT_VIDEO_PATH = "../output/detected_video_custom.mp4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def draw_annotations(frame, results, class_names):
    annotated_frame = frame.copy()
    colors = {
        'player': (255, 255, 255),
        'referee': (32, 165, 218),
        'ball': (203, 192, 255),
        'goalkeeper': (255, 99, 99)
    }
    default_color = (144, 128, 112)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 1
    box_thickness = 2
    text_color = (0, 0, 0)

    if results is not None and results.boxes is not None:
        for *xyxy, conf, cls in results.boxes.data:
            x1, y1, x2, y2 = map(int, xyxy)
            class_id = int(cls)
            label = class_names.get(class_id, f"Class {class_id}")
            draw_color = colors.get(label, default_color)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), draw_color, box_thickness)
            confidence_str = f"{conf:.2f}"
            text = f"{label} {confidence_str}"
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_bg_y1 = max(0, y1 - text_height - 5)
            cv2.rectangle(annotated_frame, (x1, text_bg_y1), (x1 + text_width + 5, y1), draw_color, -1)
            cv2.putText(annotated_frame, text, (x1 + 2, y1 - 5 if y1 - 5 >= text_height else y1 + text_height + 5),
                        font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    return annotated_frame

def run_detection(video_path, model_path, output_dir, output_video_path):
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
        results = model.predict(source=frame, conf=0.4, verbose=False)[0]
        annotated_frame = draw_annotations(frame, results, class_names)
        frame_filename = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(frame_filename, annotated_frame)
        out_writer.write(annotated_frame)
        frame_idx += 1
    cap.release()
    out_writer.release()
    print(f"Processed {frame_idx} frames. Output saved to: {output_video_path}")

if __name__ == "__main__":
    run_detection(VIDEO_PATH, MODEL_PATH, OUTPUT_DIR, OUTPUT_VIDEO_PATH)
