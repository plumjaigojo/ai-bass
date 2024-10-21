import logging
from ultralytics import YOLO
import cv2
import numpy as np
import time

# ตั้งค่า logging
logging.basicConfig(filename='double_dribble_log.txt', level=logging.INFO)

# โหลดโมเดลที่ฝึกเสร็จแล้ว
pose_model = YOLO('yolov8s-pose.pt')  # สำหรับการตรวจจับท่าทาง
ball_model = YOLO('basketballModel.pt')  # สำหรับการตรวจจับลูกบาส

# เปิดการใช้งานกล้องเว็บแคม
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# ตัวแปร
bounce_count = 0
double_dribble = False
prev_y_center = None
curr_y_center = None
is_holding = False
last_hold_time = None
holding_threshold = 0.8

def check_double_dribble(ball_x, ball_y, left_wrist, right_wrist):
    global double_dribble, is_holding, last_hold_time
    
    left_distance = np.hypot(ball_x - left_wrist[0], ball_y - left_wrist[1])
    right_distance = np.hypot(ball_x - right_wrist[0], ball_y - right_wrist[1])
    
    if min(left_distance, right_distance) < 50:
        if not is_holding:
            is_holding = True
            last_hold_time = time.time()
        elif time.time() - last_hold_time > holding_threshold:
            double_dribble = True
    else:
        is_holding = False

while cap.isOpened():
    success, frame = cap.read()
    if success:
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        frame = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel)
        
        ball_results = ball_model.predict(source=frame, conf=0.4, verbose=False)
        pose_results = pose_model.predict(source=frame, conf=0.5, verbose=False)

        ball_annotated_frame = ball_results[0].plot()
        pose_annotated_frame = pose_results[0].plot()

        final_frame = cv2.addWeighted(ball_annotated_frame, 0.5, pose_annotated_frame, 0.5, 0)

        for ball in ball_results:
            for bbox in ball.boxes.xyxy:
                x1, y1, x2, y2 = bbox[:4]
                curr_y_center = (y1 + y2) / 2

                if prev_y_center is not None and curr_y_center is not None:
                    if prev_y_center - curr_y_center > 10:
                        bounce_count += 1
                        logging.info(f"Bounce Count: {bounce_count}")

                if pose_results and len(pose_results[0].keypoints) > 10:
                    left_wrist = pose_results[0].keypoints[9]
                    right_wrist = pose_results[0].keypoints[10]
                    check_double_dribble((x1 + x2) / 2, curr_y_center, left_wrist, right_wrist)
                prev_y_center = curr_y_center

        cv2.putText(final_frame, f"Bounce Count: {bounce_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(final_frame, f"Double Dribble: {'Yes' if double_dribble else 'No'}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('YOLO Basketball Detection + Pose Estimation + Bounce Count', final_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
