from ultralytics import YOLO
import cv2
import numpy as np
import time

# โหลดโมเดลที่ฝึกเสร็จแล้ว
pose_model = YOLO('yolov8s-pose.pt')  # สำหรับการตรวจจับท่าทาง
ball_model = YOLO('basketballModel.pt')  # สำหรับการตรวจจับลูกบาส

# เปิดการใช้งานกล้องเว็บแคม
cap = cv2.VideoCapture(0)

# ตั้งค่าความละเอียดของกล้อง
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# ตัวแปรสำหรับการนับลูกบาส
bounce_count = 0
double_dribble = False
holding_threshold = 0.8  # ระยะเวลาการถือบอล
last_hold_time = None
is_holding = False

# ตัวแปรสำหรับเก็บตำแหน่ง Y ก่อนหน้าและตำแหน่งปัจจุบัน
prev_y_center = None
curr_y_center = None
prev_x_center = None
curr_x_center = None

# ฟังก์ชันตรวจจับการถือบอล
def check_holding(ball_x, ball_y, left_wrist, right_wrist):
    global is_holding, last_hold_time, double_dribble
    # ตรวจสอบระยะห่างระหว่างลูกบาสและข้อมือ
    left_distance = np.hypot(ball_x - left_wrist[0], ball_y - left_wrist[1])
    right_distance = np.hypot(ball_x - right_wrist[0], ball_y - right_wrist[1])

    # ถ้าลูกบาสอยู่ใกล้มือและไม่มีการ dribble
    if min(left_distance, right_distance) < 50:
        if not is_holding:
            is_holding = True
            last_hold_time = time.time()
        elif time.time() - last_hold_time > holding_threshold:
            double_dribble = True  # ตั้งค่าเป็น Double Dribble
    else:
        is_holding = False

# ปรับปรุงการตรวจจับการกระเด้งของลูกบาส
def update_bounce_count(curr_x, curr_y, prev_x, prev_y):
    global bounce_count
    if prev_y is not None and prev_x is not None:
        delta_y = prev_y - curr_y
        delta_x = abs(prev_x - curr_x)
        if delta_y > 10 and delta_x < 50:  # เงื่อนไขการตรวจจับการกระเด้ง
            bounce_count += 1

# วนลูปเพื่อรับและประมวลผลภาพจากกล้อง
while cap.isOpened():
    success, frame = cap.read()  # อ่านเฟรมจากกล้อง
    if success:
        # การเพิ่ม preprocessing ให้กับภาพเพื่อลดสัญญาณรบกวน
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Sharpening filter
        frame = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel)

        # ใช้โมเดล YOLO ในการพยากรณ์ตำแหน่งลูกบาส (ปรับ conf เป็น 0.4)
        ball_results = ball_model.predict(source=frame, conf=0.4, verbose=False)

        # ใช้โมเดล YOLO ในการพยากรณ์ท่าทางของผู้เล่น
        pose_results = pose_model.predict(source=frame, conf=0.6, verbose=False)

        # วาด bounding boxes ของลูกบาสเกตบอลบนเฟรม
        ball_annotated_frame = ball_results[0].plot()

        # วาด keypoints ของการตรวจจับท่าทางลงบนเฟรมที่ถูกแยกออกจากกัน
        pose_annotated_frame = pose_results[0].plot()

        # รวมเฟรมทั้งสองเข้าด้วยกันโดยใช้ฟังก์ชัน addWeighted
        final_frame = cv2.addWeighted(ball_annotated_frame, 0.5, pose_annotated_frame, 0.5, 0)

        # ตรวจสอบ bounding boxes ว่ามีลูกบาสถูกตรวจจับหรือไม่
        for ball in ball_results:
            for bbox in ball.boxes.xyxy:
                # คำนวณจุดกึ่งกลางของ bounding box
                x1, y1, x2, y2 = bbox[:4]
                curr_y_center = (y1 + y2) / 2  # ตำแหน่ง Y ของจุดกึ่งกลาง
                curr_x_center = (x1 + x2) / 2  # ตำแหน่ง X ของจุดกึ่งกลาง

                # ถ้ามีตำแหน่ง Y ก่อนหน้าและปัจจุบัน
                if prev_y_center is not None and curr_y_center is not None:
                    update_bounce_count(curr_x_center, curr_y_center, prev_x_center, prev_y_center)

                # ตรวจจับ Double Dribble
                left_wrist, right_wrist = [pose_results[0].keypoints[9]], [pose_results[0].keypoints[10]]
                check_holding(curr_x_center, curr_y_center, left_wrist, right_wrist)

                # อัปเดตตำแหน่ง Y และ X ก่อนหน้า
                prev_y_center = curr_y_center
                prev_x_center = curr_x_center

        # แสดงจำนวนครั้งที่ลูกบาสกระเด้งพื้นบนเฟรม
        cv2.putText(final_frame, f"Bounce Count: {bounce_count}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # แสดงข้อความ Double Dribble และกรอบสีแดงรอบผู้เล่น
        if double_dribble:
            cv2.putText(final_frame, "DOUBLE DRIBBLE!", (200, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6, cv2.LINE_AA)

            # วาดกรอบรอบผู้เล่นเป็นสีแดง
            for pose in pose_results:
                for box in pose.boxes.xyxy:
                    x1, y1, x2, y2 = box[:4]
                    cv2.rectangle(final_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)

        # แสดงผลลัพธ์ (ภาพที่มี bounding boxes, keypoints และจำนวนครั้งที่ลูกบาสกระทบพื้น)
        cv2.imshow('YOLO Basketball Detection + Pose Estimation + Bounce Count', final_frame)

        # หากกดปุ่ม 'q' จะหยุดการทำงาน
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# ปล่อยทรัพยากรเมื่อเสร็จสิ้น
cap.release()
cv2.destroyAllWindows()
