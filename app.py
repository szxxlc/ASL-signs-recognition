import argparse
import os
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox

ALLOWED_EXT = {".jpg", ".jpeg", ".png"}

def get_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if api_key is None:
        raise RuntimeError("ROBOFLOW_API_KEY was not found")
    return api_key

def create_client(api_key: str) -> InferenceHTTPClient:
    return InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    )
    
def choose_image_path() -> str | None:
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Choose an image",
        filetypes=(
            ("Images", "*.jpg *.jpeg *.png"),
            ("All files", "*.*"),
        ),
    )

    if not file_path:
        return None

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ALLOWED_EXT:
        messagebox.showerror("Eror", f"Unsupported format: {ext}")
        return None

    return file_path

def get_detections(client: InferenceHTTPClient, image_path: str):
    result = client.run_workflow(
        workspace_name="p40lab5yolo",
        workflow_id="custom-workflow",
        images={
            "image": image_path
        },
        use_cache=True # Speeds up repeated requests
    )

    if isinstance(result, list):
        result = result[0]
        
    return result["predictions"]["predictions"]

def draw_detections(detections, input_path: str, output_path: str,):
    img = Image.open(input_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    for i, det in enumerate(detections, start=1):
        det_class = det["class"]
        conf = det["confidence"]
        x = det["x"]
        y = det["y"]
        w = det["width"]
        h = det["height"]

        print(f"\nDetection no. {i}:")
        print(f"  class: {det_class}")
        print(f"  confidence: {conf:.2f}")
        print(f"  bbox (x,y,w,h): {x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}")
        
        left = int(x - w / 2)
        top = int(y - h / 2)
        right = int(x + w / 2)
        bottom = int(y + h / 2)
        
        draw.rectangle([left, top, right, bottom], outline="purple", width=3)
        label = f"{det_class} {conf:.2f}"
        
        bbox = draw.textbbox((left, top), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        draw.rectangle([left, top - text_h, left + text_w, top], fill="purple")
        draw.text((left, top - text_h), label, fill="white", font=font)

    img.save(output_path)
    print(f"\nLabeled image saved to: {output_path}")
    
def get_detections_from_frame(client: InferenceHTTPClient, frame, tmp_path="webcam_frame.jpg"):
    cv2.imwrite(tmp_path, frame)
    result = client.run_workflow(
        workspace_name="p40lab5yolo",
        workflow_id="custom-workflow",
        images={
            "image": tmp_path
        },
        use_cache=True
    )

    if isinstance(result, list):
        result = result[0]

    return result["predictions"]["predictions"]

def draw_detections_on_frame(frame, detections):
    for det in detections:
        det_class = det["class"]
        conf = det["confidence"]
        x = det["x"]
        y = det["y"]
        w = det["width"]
        h = det["height"]

        left = int(x - w / 2)
        top = int(y - h / 2)
        right = int(x + w / 2)
        bottom = int(y + h / 2)

        cv2.rectangle(frame, (left, top), (right, bottom), (128, 0, 128), 2)
        label = f"{det_class} {conf:.2f}"

        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (left, top - text_h - 4), (left + text_w, top), (128, 0, 128), -1)
        cv2.putText(frame, label, (left, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def run_webcam(client: InferenceHTTPClient):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    frame_count = 0
    detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # requests limited to every 5 frames
        if frame_count % 5 == 0:
            try:
                detections = get_detections_from_frame(client, frame)
            except Exception as e:
                print("Error during inference:", e)

        if detections:
            draw_detections_on_frame(frame, detections)

        cv2.imshow("Roboflow live detection", frame)

        # exit with esc
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    api_key = get_api_key()
    client = create_client(api_key)
    
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--mode", choices=["image", "video"], default="image")
    args = parser.parse_args()
    
    if args.mode == "video":
        # live video recognition
        print("Click ESC to exit the video mode")
        run_webcam(client)
    else:
        # local image recognition
        input_path = choose_image_path()
        if input_path:
            output_path = "output_with_boxes.jpg"
            detections = get_detections(client, input_path)
            draw_detections(detections, input_path, output_path)
    
if __name__ == "__main__":
    main()