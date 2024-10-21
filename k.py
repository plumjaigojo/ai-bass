from ultralytics import YOLO
import cv2
import time

# โหลดโมเดลที่ฝึกเสร็จแล้ว
pose_model = YOLO('yolov8s-pose.pt')  # สำหรับการตรวจจับท่าทาง
ball_model = YOLO('basketballModel.pt')  # สำหรับการตรวจจับลูกบาส

# เปิดการใช้งานกล้องเว็บแคม
cap = cv2.VideoCapture(0)

# ตัวแปรสำหรับการนับลูกบาสและตรวจจับ Double Dribble
bounce_count = 0
prev_y_center = None
is_holding = False
hold_start_time = None
dribbling = False
double_dribble_detected = False
hold_duration = 0.75  # ระยะเวลาที่ถือบาสก่อนที่จะถือว่าผิด
bounce_threshold = 10  # ความต่างแกน Y เพื่อตรวจสอบการกระเด้ง

# วนลูปเพื่อรับและประมวลผลภาพจากกล้อง
while cap.isOpened():
    success, frame = cap.read()  # อ่านเฟรมจากกล้อง
    if success:
        # ใช้โมเดล YOLO ในการพยากรณ์ตำแหน่งลูกบาส
        ball_results = ball_model.predict(source=frame, conf=0.5, verbose=False)
        
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
                    delta_y = prev_y_center - curr_y_center

                    # ตรวจสอบว่าลูกบาสมีการเคลื่อนที่ลง (แสดงถึงการกระทบพื้น)
                    if delta_y > bounce_threshold:  # ลูกบาสกำลังเคลื่อนที่ลง
                        dribbling = True
                        is_holding = False
                        hold_start_time = None
                        double_dribble_detected = False
                        bounce_count += 1  # นับการกระเด้งเพิ่มขึ้น

                    elif abs(delta_y) < 1 and not dribbling:
                        # ลูกบาสหยุดเคลื่อนไหว (holding)
                        if not is_holding:
                            hold_start_time = time.time()
                            is_holding = True
                        elif is_holding and (time.time() - hold_start_time > hold_duration):
                            # ถ้าหยุดเกินระยะเวลาที่กำหนดและผู้เล่นเริ่ม dribbling ใหม่ -> Double Dribble
                            if not double_dribble_detected and dribbling:
                                # แสดงข้อความ Double Dribble สีแดงและกรอบของผู้เล่นเป็นสีแดง
                                cv2.putText(final_frame, "DOUBLE DRIBBLE!", (50, 100),
                                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6, cv2.LINE_AA)
                                print("Double Dribble Detected!")
                                double_dribble_detected = True
                                is_holding = False

                    # รีเซ็ตสถานะหากลูกบาสหยุดแล้ว
                    if delta_y < bounce_threshold and dribbling:
                        dribbling = False

                # อัปเดตตำแหน่ง Y ก่อนหน้า
                prev_y_center = curr_y_center

        # วาดกรอบสีแดงสำหรับผู้เล่นที่ตรวจจับการทำผิด Double Dribble
        if double_dribble_detected:
            for pose in pose_results:
                for bbox in pose.boxes.xyxy:
                    x1, y1, x2, y2 = bbox[:4]
                    cv2.rectangle(final_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)  # วาดกรอบสีแดง

        # แสดงจำนวนครั้งที่ลูกบาสกระทบพื้นบนเฟรม
        cv2.putText(final_frame, f"Bounce Count: {bounce_count}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

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
