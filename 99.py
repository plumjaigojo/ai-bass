import logging 
import cv2
import numpy as np
import time
import os
from collections import deque
from ultralytics import YOLO

# ตั้งค่า logging
logging.basicConfig(filename='travel_detection_log.txt', level=logging.INFO)

# โหลดโมเดลที่ฝึกเสร็จแล้ว
pose_model = YOLO('yolov8s-pose.pt')  # สำหรับการตรวจจับท่าทาง
ball_model = YOLO('basketballModel.pt')  # สำหรับการตรวจจับลูกบาส

# เปิดการใช้งานกล้องเว็บแคม
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# ตัวแปรสำหรับการนับลูกบาสและการเดินผิดกติกา
dribble_count = 0
bounce_count = 0  # ตัวแปรสำหรับการนับลูกบาส
step_count = 0
prev_x_center = None
prev_y_center = None
prev_left_ankle_y = None
prev_right_ankle_y = None
prev_delta_y = None
ball_not_detected_frames = 0
max_ball_not_detected_frames = 20
dribble_threshold = 18
step_threshold = 5
min_wait_frames = 7
wait_frames = 0
travel_detected = False
travel_timestamp = None
total_dribble_count = 0
total_step_count = 0

# Define the body part indices
body_index = {"left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16}

# Define the frame dimensions and fps
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec using VideoWriter_fourcc and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# Make directory to save travel footage
if not os.path.exists("travel_footage"):
    os.makedirs("travel_footage")

# Initialize frame buffer and frame saving settings
frame_buffer = deque(maxlen=30)
save_frames = 60
frame_save_counter = 0
saving = False
out = None

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # บันทึกเฟรมใน buffer
        frame_buffer.append(frame)

        # การปรับความคมชัดของภาพเพื่อลดสัญญาณรบกวน
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        frame = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel)

        # ตรวจจับลูกบาส
        ball_results = ball_model(frame, verbose=False, conf=0.45)

        ball_detected = False
        for results in ball_results:
            for bbox in results.boxes.xyxy:
                x1, y1, x2, y2 = bbox[:4]

                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2

                if prev_y_center is not None:
                    delta_y = y_center - prev_y_center

                    if (
                        prev_delta_y is not None
                        and prev_delta_y > dribble_threshold
                        and delta_y < -dribble_threshold
                    ):
                        dribble_count += 1
                        bounce_count += 1  # เพิ่มการนับลูกบาสกระเด้ง
                        total_dribble_count += 1
                        logging.info(f"Bounce Count: {bounce_count}")

                    prev_delta_y = delta_y

                prev_x_center = x_center
                prev_y_center = y_center

                ball_detected = True
                ball_not_detected_frames = 0

            annotated_frame = results.plot()

        if not ball_detected:
            ball_not_detected_frames += 1

        if ball_not_detected_frames >= max_ball_not_detected_frames:
            step_count = 0

        # ตรวจจับท่าทาง
        pose_results = pose_model(frame, verbose=False, conf=0.5)

        if len(pose_results) > 0 and hasattr(pose_results[0], 'keypoints'):
            try:
                rounded_results = np.round(pose_results[0].keypoints.numpy(), 1)

                left_ankle = rounded_results[0][body_index["left_ankle"]]
                right_ankle = rounded_results[0][body_index["right_ankle"]]

                if (
                    prev_left_ankle_y is not None
                    and prev_right_ankle_y is not None
                    and wait_frames == 0
                ):
                    left_diff = abs(left_ankle[1] - prev_left_ankle_y)
                    right_diff = abs(right_ankle[1] - prev_right_ankle_y)

                    if max(left_diff, right_diff) > step_threshold:
                        step_count += 1
                        total_step_count += 1
                        print(f"Step taken: {step_count}")
                        wait_frames = min_wait_frames

                prev_left_ankle_y = left_ankle[1]
                prev_right_ankle_y = right_ankle[1]

                if wait_frames > 0:
                    wait_frames -= 1

            except:
                print("No human detected.")

            pose_annotated_frame = pose_results[0].plot()

            # รวมเฟรม
            combined_frame = cv2.addWeighted(annotated_frame, 0.6, pose_annotated_frame, 0.4, 0)

            # แสดงผลจำนวนการกระเด้งและการก้าว
            cv2.putText(
                combined_frame,
                f"Dribble count: {total_dribble_count}",
                (50, 950),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                combined_frame,
                f"Bounce count: {bounce_count}",
                (50, 1000),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                4,
                cv2.LINE_AA,
            )

            # ตรวจจับ Travel
            if ball_detected and step_count >= 2 and dribble_count == 0:
                print("Travel detected!")
                step_count = 0
                travel_detected = True
                travel_timestamp = time.time()

                if not saving:
                    filename = os.path.join("travel_footage", f"travel_{time.strftime('%Y%m%d-%H%M%S')}.mp4")
                    out = cv2.VideoWriter(filename, fourcc, 9, (frame_width, frame_height))

                    for f in frame_buffer:
                        out.write(f)

                    saving = True

            if travel_detected and time.time() - travel_timestamp > 3:
                travel_detected = False
                total_dribble_count = 0
                total_step_count = 0

            if travel_detected:
                blue_tint = np.full_like(combined_frame, (255, 0, 0), dtype=np.uint8)
                combined_frame = cv2.addWeighted(combined_frame, 0.7, blue_tint, 0.3, 0)
                cv2.putText(combined_frame, "Travel Detected!", (250, 300), 
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)

            if saving:
                out.write(frame)
                frame_save_counter += 1

                if frame_save_counter >= save_frames:
                    saving = False
                    frame_save_counter = 0
                    out.release()

            cv2.imshow("Basketball Detection + Travel Detection", combined_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

if out is not None:
    out.release()

cap.release()
cv2.destroyAllWindows()
