# Basketball Shooting Form Analyzer

A computer vision tool that analyzes basketball shooting form from video. It uses pose estimation to track joint angles (elbow extension, knee bend) throughout a shot, and object detection to track the ball's trajectory, estimate shot arc, and detect make/miss outcomes.

## How it works

1. **Pose estimation** — YOLOv8-Pose detects 17 body keypoints per frame, from which elbow and knee angles are calculated using vector geometry.
2. **Ball detection** — A YOLOv8 object detector (COCO's "sports ball" class) tracks the basketball's position frame by frame.
3. **Hoop marking** — On the first frame, the user manually marks the hoop/rim location once via a simple click-and-drag interface.
4. **Shot arc estimation** — The ball's early trajectory points are fitted to estimate the shot's launch angle.
5. **Make/miss detection** — A heuristic checks whether the ball's path passes through the marked hoop region while moving downward (make) or bounces back upward (miss).
6. **Visual output** — An annotated video is generated with a color-coded skeleton overlay, joint angle readouts, ball trajectory trail, and shot outcome label.

## Tech stack

- Python
- Ultralytics YOLOv8 (pose estimation + object detection)
- OpenCV

## Setup

1. Clone this repo and create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the analyzer on a video clip:
   ```
   python shooting_form_analyzer.py your_video.mp4 right
   ```
   (use `left` instead of `right` if you shoot left-handed)

4. When prompted, drag a box around the hoop/rim in the pop-up window, then press ENTER.

5. Check the terminal for a shot summary, and open `output_annotated.mp4` for the annotated video.

## Known limitations

- **Ball detection accuracy** depends heavily on lighting, contrast, and camera distance — a basketball that blends into the background or is partially obscured will be detected in fewer frames, which reduces the reliability of the shot arc and make/miss detection.
- **Make/miss detection is a heuristic**, not a physics simulation — it works best on clear, unobstructed shots and can be fooled by unusual bounces or sparse ball detections.
- Pose detection assumes a single person clearly visible in frame.

## Future improvements

- Use a larger/more accurate detection model and interpolate ball position between sparse detections
- Automatic hoop detection instead of manual marking
- Support for tracking multiple shots in one longer video clip
- Compare shooting form against benchmark data or a reference video
