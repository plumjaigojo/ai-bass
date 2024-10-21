from ultralytics import YOLO
import cv2
import numpy as np
import time
import threading

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
holding_threshold = 0.8  # ระยะเวลาในการถือบอลที่ถือว่าเป็นการถือบอล (หน่วยเป็นวินาที)
last_hold_time = None  # เวลาที่ผู้เล่นหยุด dribble
is_holding = False  # สถานะการถือบอล

# ตัวแปรสำหรับเก็บตำแหน่ง Y ก่อนหน้าและตำแหน่งปัจจุบัน
prev_y_center = None
curr_y_center = None

# ฟังก์ชันตรวจจับการถือบอล
def check_holding(ball_x, ball_y, left_wrist, right_wrist):
    global is_holding, last_hold_time, double_dribble
    # ตรวจสอบระยะห่างระหว่างลูกบาสและข้อมือ
    left_distance = np.hypot(ball_x - left_wrist[0], ball_y - left_wrist[1])
    right_distance = np.hypot(ball_x - right_wrist[0], ball_y - right_wrist[1])
    
    # ถ้าลูกบอลอยู่ใกล้มือและไม่มีการ dribble เกิดขึ้น
    if min(left_distance, right_distance) < 50:  # ปรับค่า threshold ให้เหมาะสม
        if not is_holding:
            is_holding = True
            last_hold_time = time.time()  # บันทึกเวลาในการถือบอล
        elif time.time() - last_hold_time > holding_threshold:
            # ถ้าผู้เล่นถือบอลนานกว่า threshold และเริ่ม dribble ใหม่
            double_dribble = True  # ตั้งสถานะว่าเป็น Double Dribble
    else:
        is_holding = False  # รีเซ็ตสถานะการถือบอล

# ฟังก์ชันปรับปรุงการตรวจจับการกระเด้งของลูกบาส
def update_bounce_count(curr_x, curr_y, prev_x, prev_y):
    global bounce_count
    if prev_y is not None and prev_x is not None:
        delta_y = prev_y - curr_y
        delta_x = prev_x - curr_x
        if delta_y > 10 and abs(delta_x) < 50:  # เพิ่มเงื่อนไขการตรวจสอบแนว X เพื่อลด false positives
            bounce_count += 1

# ฟังก์ชันสำหรับการประมวลผลภาพแต่ละเฟรม
def process_frame(frame):
    global prev_y_center, curr_y_center

    # การเพิ่ม preprocessing ให้กับภาพเพื่อลดสัญญาณรบกวน
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Sharpening filter
    frame = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel)
    
    # ใช้โมเดล YOLO ในการพยากรณ์ตำแหน่งลูกบาส (ปรับ conf เป็น 0.4)
    ball_results = ball_model.predict(source=frame, conf=0.4, verbose=False)
    
    # ใช้โมเดล YOLO ในการพยากรณ์ท่าทางของผู้เล่น
    pose_results = pose_model.predict(source=frame, conf=0.5, verbose=False)

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

            # ถ้ามีตำแหน่ง Y ก่อนหน้าและปัจจุบัน
            if prev_y_center is not None and curr_y_center is not None:
                # ตรวจสอบว่าลูกบาสมีการเคลื่อนที่ลง (แสดงถึงการกระทบพื้น)
                update_bounce_count(x1, curr_y_center, prev_y_center, prev_y_center)
                if double_dribble:
                    # แสดงข้อความ Double Dribble สีแดงขนาดใหญ่
                    cv2.putText(final_frame, "DOUBLE DRIBBLE!", (200, 300),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6, cv2.LINE_AA)

                    # วาดกรอบรอบผู้เล่นเป็นสีแดง
                    if len(pose_results[0].keypoints) > 10:  # ตรวจสอบว่ามีการตรวจจับ keypoints อย่างน้อย 11 จุด
                        for box in pose_results[0].boxes.xyxy:
                            x1, y1, x2, y2 = box[:4]
                            cv2.rectangle(final_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)

            # อัปเดตตำแหน่ง Y ก่อนหน้า
            prev_y_center = curr_y_center

    # แสดงจำนวนครั้งที่ลูกบาสกระทบพื้นบนเฟรม
    cv2.putText(final_frame, f"Bounce Count: {bounce_count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    return final_frame

# วนลูปเพื่อรับและประมวลผลภาพจากกล้อง
while cap.isOpened():
    success, frame = cap.read()  # อ่านเฟรมจากกล้อง
    if success:
        # ใช้ threading สำหรับการประมวลผลเฟรม
        thread = threading.Thread(target=process_frame, args=(frame,))
        thread.start()
        thread.join()

        # แสดงผลลัพธ์ (ภาพที่มี bounding boxes, keypoints และจำนวนครั้งที่ลูกบาสกระทบพื้น)
        final_frame = process_frame(frame)
        cv2.imshow('YOLO Basketball Detection + Pose Estimation + Bounce Count', final_frame)

        # หากกดปุ่ม 'q' จะหยุดการทำงาน
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# ปล่อยทรัพยากรเมื่อเสร็จสิ้น
cap.release()
cv2.destroyAllWindows()
