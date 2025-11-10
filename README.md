# ASL Recognition (Roboflow Demo)

A simple Python script that:

- lets the user pick a local image through a file dialog and runs object detection on it,
- or runs live detection from the webcam,
- draws bounding boxes with class and confidence,
- uses a model deployed on Roboflow (via `inference-sdk`).

The script decides which mode to run based on a CLI argument (`--mode`). Default is `image`.

## Requirements

- Python 3.10+ (recommended)
- A Roboflow account and a working workflow (you need `workspace_name` and `workflow_id`)
- Installed packages from `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file in the project root:

```text
ROBOFLOW_API_KEY=RF_your_api_key_here
```

You can find the API key in your Roboflow dashboard (Project / Workspace → Settings → API Keys).

The script loads this variable with `python-dotenv`.

## Running the Script

The script accepts one optional argument: `--mode`, with two allowed values:

- `image` (default) – opens a file picker, runs detection on the chosen image, saves the labeled image
- `video` – opens the webcam and overlays the latest detections on the live stream

### 1. Image mode (default)

```bash
python asl_recognition.py
```

or explicitly:

```bash
python asl_recognition.py --mode image
```

What happens:

1. A file dialog opens – pick a `.jpg/.jpeg/.png`.
2. The image is sent to your Roboflow workflow.
3. Detections are printed in the console, e.g.:

   ```text
   Detection no. 1:
     class: A
     confidence: 0.91
     bbox (x,y,w,h): 945.4, 453.1, 149.0, 156.9
   ```

4. A new image with drawn boxes and labels is saved as:

```text
output_with_boxes.jpg
```

### 2. Video mode (webcam)

```bash
python asl_recognition.py --mode video
```

What happens:

- The webcam opens in a window.
- Every few frames, the current frame is sent to Roboflow (to avoid spamming the API and to keep it responsive).
- The last detections are drawn on every frame.
- Press `ESC` to exit.

## Notes

- You can change how often inference is run in the webcam loop (default here is every 5th frame).
- Amongs the recognised signs are the first 5 letters of the alphabet: `A`, `B`, `C`, `D` and `E`.
