import cv2
from ultralytics import YOLO

import pygame
import threading

# Load YOLO model
model = YOLO("model/best.pt")

# --- Video Capture Setup ---
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("Detection_Test.mp4")  # Use video file instead of webcam

# Set camera resolution to 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera opened. Detecting signs. Press 'q' to quit.")

# Create window and set it to a specific size
window_name = "Video Demo"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

# Variables for text display
detected_sign_meaning = "NO TRAFFIC SIGN DETECTED"
last_detected_sign_id = None
display_fade_timer = 0
FADE_OUT_TIME = 30
last_alerted_sign_id = None  # Add this near your other state variables

def play_alert(sign_name):
    import os
    alert_filename = sign_name.replace(" ", "_") + ".mp3"
    alert_path = os.path.join("alert", alert_filename)
    if os.path.exists(alert_path):
        def play():
            pygame.mixer.init()
            pygame.mixer.music.load(alert_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.quit()
        threading.Thread(target=play, daemon=True).start()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLO detection on the frame
    results = model(frame)

    # Process YOLO detections
    detected_signs = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Get confidence and class
            confidence = float(box.conf)
            sign_id = int(box.cls)

            detected_signs.append(((x1, y1, x2, y2), sign_id, confidence))

    # --- Process Detections ---
    current_frame_detected_sign_id = None
    if detected_signs:
        most_confident_sign = max(detected_signs, key=lambda x: x[2])
        bbox, sign_id, confidence = most_confident_sign

        current_frame_detected_sign_id = sign_id
        # Get the class name from results
        detected_sign_meaning = results[0].names[sign_id]
        display_fade_timer = FADE_OUT_TIME

        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{detected_sign_meaning} ({confidence:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    else:
        if display_fade_timer > 0:
            display_fade_timer -= 1
        if display_fade_timer <= 0:
            detected_sign_meaning = "NO TRAFFIC SIGN DETECTED"
            last_detected_sign_id = None

    h, w, _ = frame.shape
    text_bg_height = 40
    cv2.rectangle(frame, (0, h - text_bg_height), (w, h), (0, 0, 0), -1)

    # --- Sound Alert ---
    current_frame_detected_sign_id = None
    if detected_signs:
        most_confident_sign = max(detected_signs, key=lambda x: x[2])
        bbox, sign_id, confidence = most_confident_sign

        current_frame_detected_sign_id = sign_id
        detected_sign_meaning = results[0].names[sign_id]
        display_fade_timer = FADE_OUT_TIME

        # --- SOUND ALERT LOGIC ---
        if last_alerted_sign_id != sign_id:
            play_alert(detected_sign_meaning)
            last_alerted_sign_id = sign_id

        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{detected_sign_meaning} ({confidence:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    else:
        if display_fade_timer > 0:
            display_fade_timer -= 1
        if display_fade_timer <= 0:
            detected_sign_meaning = "NO TRAFFIC SIGN DETECTED"
            last_detected_sign_id = None
            last_alerted_sign_id = None  # Reset alert when no sign is detected


    # --- Display sign image in bottom left ---
    if detected_sign_meaning != "NO TRAFFIC SIGN DETECTED":
        import os
        # Replace spaces with underscores for file name
        sign_img_filename = detected_sign_meaning.replace(" ", "_") + ".png"
        sign_img_path = os.path.join("sign", sign_img_filename)
        if os.path.exists(sign_img_path):
            sign_img = cv2.imread(sign_img_path, cv2.IMREAD_UNCHANGED)
            if sign_img is not None:
                # Resize to small thumbnail (e.g., 70x70)
                thumb_size = 70
                sign_img = cv2.resize(sign_img, (thumb_size, thumb_size))
                # Handle alpha channel if present
                y_offset = h - text_bg_height - thumb_size - 10
                x_offset = 10
                if sign_img.shape[2] == 4:  # PNG with alpha
                    alpha_s = sign_img[:, :, 3] / 255.0
                    alpha_l = 1.0 - alpha_s
                    for c in range(3):
                        frame[y_offset:y_offset+thumb_size, x_offset:x_offset+thumb_size, c] = (
                            alpha_s * sign_img[:, :, c] +
                            alpha_l * frame[y_offset:y_offset+thumb_size, x_offset:x_offset+thumb_size, c]
                        )
                else:
                    frame[y_offset:y_offset+thumb_size, x_offset:x_offset+thumb_size] = sign_img


    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    text_color = (255, 255, 255)

    text_size = cv2.getTextSize(
        detected_sign_meaning, font, font_scale, font_thickness
    )[0]
    text_x = (w - text_size[0]) // 2
    text_y = h - (text_bg_height // 2) + (text_size[1] // 2)

    cv2.putText(
        frame,
        detected_sign_meaning,
        (text_x, text_y),
        font,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )

    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
