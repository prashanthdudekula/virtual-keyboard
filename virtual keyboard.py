import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import os

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

# Keyboard layout
ROWS = [
    list("1234567890"),
    list("QWERTYUIOP"),
    list("ASDFGHJKL;"),
    ["Shift"] + list("ZXCVBNM,./") + ["Backspace"],
    ["Ctrl", "Alt", "Space", "Return", "✓"]
]

typed_text     = ""
pinched        = False
last_action_ms = 0
DEBOUNCE_MS    = 300
notepad_opened = False
shift_active   = False  # Shift toggle state

# Draw stylish keyboard
def draw_keyboard(frame, key_boxes, highlight_key=None):
    h, w = frame.shape[:2]
    KEY_H = 65
    M     = 8
    overlay = frame.copy()

    for r, row in enumerate(ROWS):
        key_w = w // (len(row) + 2)
        total_w = len(row) * key_w + (len(row)+1)*M
        x = (w - total_w)//2 + M
        y = h - (KEY_H * len(ROWS) + M * (len(ROWS)+1)) + r*(KEY_H+M) + M

        for key in row:
            box_w = key_w
            if key == "Space":
                box_w = key_w * 4
            x1, y1 = int(x), int(y)
            x2, y2 = x1 + box_w, y1 + KEY_H
            key_boxes[key] = (x1, y1, x2, y2)

            # Fill color
            if key == highlight_key:
                fill = (180, 0, 180)  # purple highlight
            elif key == "Shift" and shift_active:
                fill = (0, 100, 255)  # blue when active
            else:
                fill = (50, 50, 50)   # default gray
            cv2.rectangle(overlay, (x1, y1), (x2, y2), fill, -1)

            # Border
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Label
            label = {
                "Shift": "⇧",
                "Backspace": "⌫",
                "Return": "↵",
                "Space": "␣",
                "✓": "✓"
            }.get(key, key)

            font_scale = 1.2 if len(label) <= 2 else 0.8
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            tx = x1 + (box_w - tw)//2
            ty = y1 + (KEY_H + th)//2
            cv2.putText(frame, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
            x = x2 + M

    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

# Open camera
def open_cam():
    for idx in (0, 1, 2, -1):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
    raise RuntimeError("Cannot open camera")

cap = open_cam()
win = "Virtual Gesture Keyboard"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, 1280, 720)  # ✅ Resizable window instead of full-screen

# Wrap text into lines
def wrap_text(text, max_chars=40):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    vis = frame.copy()
    h, w = frame.shape[:2]

    # Typed text area
    text_box_height = 200
    cv2.rectangle(vis, (0, 0), (w, text_box_height), (180, 0, 180), -1)
    lines = wrap_text(typed_text, max_chars=40)
    for i, line in enumerate(lines[:4]):
        y = 50 + i * 45
        cv2.putText(vis, line, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    # Draw keyboard
    key_boxes = {}
    draw_keyboard(vis, key_boxes)

    # Hand detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    highlight = None
    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(vis, lm, mp_hands.HAND_CONNECTIONS)

        ix = int(lm.landmark[8].x * w)
        iy = int(lm.landmark[8].y * h)
        mx = int(lm.landmark[12].x * w)
        my = int(lm.landmark[12].y * h)
        cv2.circle(vis, (ix, iy), 10, (255, 0, 255), -1)

        dist = np.hypot(ix - mx, iy - my)
        tapping = dist < w * 0.04
        now_ms = int(time.time() * 1000)

        if tapping and not pinched and now_ms - last_action_ms > DEBOUNCE_MS:
            for key, (x1, y1, x2, y2) in key_boxes.items():
                if x1 < ix < x2 and y1 < iy < y2:
                    pinched = True
                    last_action_ms = now_ms
                    highlight = key

                    if key == "Backspace":
                        typed_text = typed_text[:-1]
                        pyautogui.press("backspace")
                    elif key == "Return":
                        typed_text += "\n"
                        pyautogui.press("enter")
                    elif key == "Space":
                        typed_text += " "
                        pyautogui.press("space")
                    elif key == "✓":
                        pyautogui.press("enter")
                    elif key == "Shift":
                        shift_active = not shift_active
                    else:
                        char = key.upper() if shift_active else key.lower()
                        typed_text += char
                        pyautogui.press(char)

                    # Save to file
                    with open("typed_output.txt", "w", encoding="utf-8") as f:
                        f.write(typed_text)

                    # Open Notepad once
                    if not notepad_opened:
                        os.system("start notepad typed_output.txt")
                        notepad_opened = True
                    break

        if not tapping:
            pinched = False

    if highlight:
        draw_keyboard(vis, key_boxes, highlight)

    cv2.imshow(win, vis)
    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
