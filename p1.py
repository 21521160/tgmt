# import cv2
# import mediapipe as mp
# import numpy as np
# import math
# import imageio
#
#
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(max_num_faces=3, min_detection_confidence=0.5)
#
# mp_hand_mesh = mp.solutions.hands
# hand_mesh = mp_hand_mesh.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
#
#
#
# def rotate_image(sticker_image, angle):
#     height, width = sticker_image.shape[:2]
#
#     # Calculate the dimensions of the rotated image
#     angle_rad = np.deg2rad(angle)
#     cos_theta = np.abs(np.cos(angle_rad))
#     sin_theta = np.abs(np.sin(angle_rad))
#     new_width = int(width * cos_theta + height * sin_theta)
#     new_height = int(width * sin_theta + height * cos_theta)
#
#     # Calculate the center point for rotation
#     center_x = width // 2
#     center_y = height // 2
#
#     # Compute the rotation matrix
#     rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
#
#     # Adjust the translation component of the rotation matrix
#     rotation_matrix[0, 2] += (new_width - width) // 2
#     rotation_matrix[1, 2] += (new_height - height) // 2
#
#     # Rotate the sticker
#     rotated_sticker = cv2.warpAffine(sticker_image, rotation_matrix, (new_width, new_height))
#
#     return rotated_sticker
#
# chroma_key_color = (0, 255, 0)  # Green color
#
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(max_num_faces=3, min_detection_confidence=0.5)
#
# mp_hand_mesh = mp.solutions.hands
# hand_mesh = mp_hand_mesh.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
#
# # Function to apply chroma keying and remove the background
# def remove_background(frame, color):
#     # Convert the frame to HSV color space
#     hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     # Create a binary mask for the chroma key color
#     lower_color = np.array([color[0] - 20, 100, 100])
#     upper_color = np.array([color[0] + 20, 255, 255])
#     mask = cv2.inRange(hsv_frame, lower_color, upper_color)
#
#     # Apply the mask to the frame to remove the background
#     result = cv2.bitwise_and(frame, frame, mask=mask)
#
#     return result
#
# sticker = []
# img = cv2.imread('crown2.png', cv2.IMREAD_UNCHANGED)
# sticker.append(img)
#
# img_gif = imageio.mimread('fire.gif', memtest=False)
#
# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)
#
# sticker_image = sticker[0]
# idx = 0
# idx_gif = 0
#
# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         break
#     if cv2.waitKey(10) & 0xFF == ord('f'):
#         idx = (idx + 1) % len(sticker)
#         sticker_image = sticker[idx]
#
#     image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     results = face_mesh.process(image_rgb)
#     results1 = hand_mesh.process(image_rgb)
#
#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             left_eye = face_landmarks.landmark[159]  # Điểm ở phía trái mắt (góc trên cùng bên trái)
#             right_eye = face_landmarks.landmark[386]  # Điểm ở phía phải mắt (góc trên cùng bên phải)
#             eye_distance = np.sqrt((right_eye.x - left_eye.x) ** 2 + (right_eye.y - left_eye.y) ** 2)
#             head_tilt_angle = math.degrees(math.atan((right_eye.y - left_eye.y) / (right_eye.x - left_eye.x)))
#             # print(head_tilt_angle)
#             for i, landmark in enumerate(face_landmarks.landmark):
#                 if i == 9:
#                     x = int(landmark.x * frame.shape[1])
#                     y = int(landmark.y * frame.shape[0])
#                     len_sticker = int(2 * eye_distance * frame.shape[1])
#                     sticker_image_copy = cv2.resize(sticker_image, (len_sticker, len_sticker))
#                     sticker_image_copy = rotate_image(sticker_image_copy, head_tilt_angle * -1)
#                     height, width = sticker_image_copy.shape[:2]
#
#                     x_min = x - int(height * np.tan(np.deg2rad(-head_tilt_angle)) / 2) - int(width / 2)
#                     y_min = y - int(height) - 20
#                     x_max = x_min + width
#                     y_max = y_min + height
#
#                     # Kiểm tra xem sticker có nằm trong khung hình hay không
#                     if x_max >= 0 and y_max >= 0 and x_min < frame.shape[1] and y_min < frame.shape[0]:
#                         # Chỉnh lại vị trí nếu sticker ra khỏi khung hình
#                         if x_min < 0:
#                             sticker_image_copy = sticker_image_copy[:, -x_min:]
#                             width = sticker_image_copy.shape[1]
#                             x_min = 0
#                         if y_min < 0:
#                             sticker_image_copy = sticker_image_copy[-y_min:, :]
#                             height = sticker_image_copy.shape[0]
#                             y_min = 0
#                         if x_max > frame.shape[1]:
#                             sticker_image_copy = sticker_image_copy[:, :frame.shape[1] - x_min]
#                             width = sticker_image_copy.shape[1]
#                         if y_max > frame.shape[0]:
#                             sticker_image_copy = sticker_image_copy[:frame.shape[0] - y_min, :]
#                             height = sticker_image_copy.shape[0]
#
#                         try:
#                             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
#                             alpha_sticker = sticker_image_copy[:, :, 3] / 255.0
#                             alpha_frame = 1 - alpha_sticker
#                             for c in range(3):
#                                 frame[y_min:y_min + height, x_min:x_min + width, c] = (alpha_sticker *
#                                                                                         sticker_image_copy[:, :, c] +
#                                                                                         alpha_frame *
#                                                                                         frame[y_min:y_min + height,
#                                                                                         x_min:x_min + width, c])
#                         except:
#                             pass
#
#     if results1.multi_hand_landmarks:
#         for hand_landmarks in results1.multi_hand_landmarks:
#             thumb_tip = hand_landmarks.landmark[4]  # Ngon cai landmark
#             index_finger_tip = hand_landmarks.landmark[8]  # Ngon tro landmark
#             middle_finger_tip = hand_landmarks.landmark[12]  # Ngon giua landmark
#             ring_finger_tip = hand_landmarks.landmark[16]  # Ngon ap ut landmark
#             pinky_tip = hand_landmarks.landmark[20]  # Ngon ut landmark
#
#             blend_point = hand_landmarks.landmark[9] # Chon diem de blen anh gif
#
#             # Calculate the Euclidean distance between thumb and other finger tips
#             thumb_index_distance = np.sqrt((thumb_tip.x - index_finger_tip.x) ** 2 + (thumb_tip.y - index_finger_tip.y) ** 2)
#             thumb_middle_distance = np.sqrt((thumb_tip.x - middle_finger_tip.x) ** 2 + (thumb_tip.y - middle_finger_tip.y) ** 2)
#             thumb_ring_distance = np.sqrt((thumb_tip.x - ring_finger_tip.x) ** 2 + (thumb_tip.y - ring_finger_tip.y) ** 2)
#             thumb_pinky_distance = np.sqrt((thumb_tip.x - pinky_tip.x) ** 2 + (thumb_tip.y - pinky_tip.y) ** 2)
#
#             threshold = 0.65
#
#             # Check if the hand is open or closed based on the distances
#             if (
#                 (thumb_index_distance + thumb_middle_distance + thumb_ring_distance + thumb_pinky_distance) > threshold
#             ):
#                 gif_size = np.sqrt((middle_finger_tip.x - pinky_tip.x) ** 2 + (middle_finger_tip.y - pinky_tip.y) ** 2)
#                 gif_size = int(gif_size * frame.shape[1])
#                 gif = img_gif[idx_gif]
#                 gif = cv2.resize(gif, (gif_size, gif_size))
#                 gif = cv2.cvtColor(gif, cv2.COLOR_BGRA2RGBA)
#                 idx_gif = (idx_gif+1)%len(img_gif)
#
#                 x = int(blend_point.x * frame.shape[1])
#                 y = int(blend_point.y * frame.shape[0])
#                 x = x - int(gif.shape[1]/2)
#                 y = y - int(gif.shape[0]/2)
#
#                 green_range_lower = np.array([0, 100, 0], dtype=np.uint8)
#                 green_range_upper = np.array([100, 255, 100], dtype=np.uint8)
#                 mask = cv2.inRange(gif[:, :, :3], green_range_lower, green_range_upper)
#                 gif[:, :, 3] = np.where(mask == 255, 0, gif[:, :, 3])
#
#                 try:
#                     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
#                     alpha_sticker = gif[:,:,3]/255.0
#                     alpha_frame = 1 - alpha_sticker
#                     for c in range(3):
#                         frame[y:y+gif_size, x:x+gif_size, c] = (alpha_sticker * gif[:, :, c] + alpha_frame * frame[y:y+gif_size, x:x+gif_size, c])
#                 except:
#                     pass
#
#
#     cv2.imshow('Webcam', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#







# import cv2
# import mediapipe as mp
# import imageio
# import numpy as np
#
# # Khởi tạo đối tượng Mediapipe
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose
#
# # Khởi tạo đối tượng Pose
# pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
#
# # Đọc video từ webcam
# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)
#
# # Đường dẫn đến file GIF filter
# filter_path = "bomb.gif"
# filter_duration = 1  # Thời gian chạy của filter (s)
#
# # Đọc file GIF filter
# filter_frames = imageio.mimread(filter_path)
# filter_frames = [frame[:, :, :3] for frame in filter_frames]  # Loại bỏ kênh alpha (nếu có)
#
# # Thiết lập các biến cho việc chạy filter
# frame_count = 0
# filter_start_frame = None
# idx_gif = 0
#
# # Cấu hình VideoWriter
# output_path = "outputgif.mp4"
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Định dạng video
# out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
#
# # Biến để kiểm tra có lỗi xảy ra trong try hay không
# error_occurred = False
#
# while True:
#     # Đọc từng khung hình
#     success, image = cap.read()
#     if not success:
#         break
#
#     try:
#         # Chuyển đổi hình ảnh sang không gian màu RGB
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#         # Phát hiện các điểm mốc trên cơ thể
#         results = pose.process(image_rgb)
#
#         # Vẽ các điểm mốc và các kết nối giữa chúng lên hình ảnh
#         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                   mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
#                                   mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
#
#         # Kiểm tra xem cánh tay có đang được duỗi thẳng
#         if results.pose_landmarks:
#             left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
#             left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
#             left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
#             right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
#             right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
#             right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
#
#             # Kiểm tra nếu cánh tay đang được duỗi thẳng
#             if (left_wrist.y < left_elbow.y and left_elbow.y < left_shoulder.y) or \
#                     (right_wrist.y < right_elbow.y and right_elbow.y < right_shoulder.y):
#                 # Lấy khung hình hiện tại để vẽ filter
#                 frame_for_filter = filter_frames[idx_gif % len(filter_frames)]
#                 frame_for_filter = cv2.resize(frame_for_filter, (image.shape[1], image.shape[0]))
#
#                 # Chạy filter
#                 if filter_start_frame is None:
#                     filter_start_frame = frame_count
#                 alpha = min(1.0, (frame_count - filter_start_frame) / (filter_duration * fps))
#                 filtered_image = cv2.addWeighted(image, 1 - alpha, frame_for_filter, alpha, 0)
#
#                 # Gán lại hình ảnh đã được lọc để hiển thị
#                 image = filtered_image
#
#                 # Tăng biến idx_gif lên 1 để chọn khung hình filter tiếp theo
#                 idx_gif += 1
#                 frame_count += 1
#
#         # Ghi khung hình vào video
#         out.write(image)
#
#         # Hiển thị hình ảnh kết quả
#         cv2.imshow('MediaPipe Pose', image)
#
#         # Nhấn phím 'q' để thoát
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     except Exception as e:
#         # Nếu có lỗi xảy ra, đặt biến error_occurred thành True
#         error_occurred = True
#         print("An error occurred:", str(e))
#         break
#
# # Nếu không có lỗi xảy ra, tiếp tục ghi các khung hình từ khối `except` vào video
# if not error_occurred:
#     while idx_gif < len(filter_frames):
#         frame_for_filter = filter_frames[idx_gif % len(filter_frames)]
#         frame_for_filter = cv2.resize(frame_for_filter, (image.shape[1], image.shape[0]))
#         out.write(frame_for_filter)
#         idx_gif += 1
#
# # Giải phóng tài nguyên
# cap.release()
# out.release()
# cv2.destroyAllWindows()











import cv2
import mediapipe as mp
import math
import time
import imageio
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)


def main():
    # Khởi tạo trình xử lý Mediapipe Hand
    # Đọc ảnh vương miện
    head_filter_image = cv2.imread("power1.jpg")
    if head_filter_image is None:
        print("Không thể đọc tệp ảnh vương miện")
        return
    show_filter = False
    # Kích thước của filter cho đỉnh đầu
    head_filter_height, head_filter_width, _ = head_filter_image.shape

    # Tạo đối tượng Mediapipe Hand
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:

        # Tạo đối tượng Mediapipe Face Mesh
        with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
            # Đọc video từ webcam
            cap = cv2.VideoCapture(0)
            cap.set(3, 1280)
            cap.set(4, 720)
            dist = 0.0

            while cap.isOpened():
                # Đọc từng khung hình
                success, frame = cap.read()
                if not success:
                    print("Không thể đọc khung hình")
                    break
                # Chuyển đổi khung hình từ BGR sang RGB để sử dụng với Mediapipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Dò tìm bàn tay trong khung hình
                results = hands.process(frame_rgb)
                hand_landmarks = results.multi_hand_landmarks

                # Dò tìm khuôn mặt trong khung hình
                face_results = face_mesh.process(frame_rgb)
                face_landmarks = face_results.multi_face_landmarks

                if hand_landmarks:
                    if len(hand_landmarks) == 2:
                        for hand_lm in hand_landmarks:
                            middle_finger = hand_lm.landmark[12]
                            x, y = int(middle_finger.x * frame.shape[1]), int(middle_finger.y * frame.shape[0])
                            cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

                        middle_finger_1 = hand_landmarks[0].landmark[12]
                        middle_finger_2 = hand_landmarks[1].landmark[12]
                        dist = math.sqrt(
                            (middle_finger_2.x - middle_finger_1.x) ** 2 + (middle_finger_2.y - middle_finger_1.y) ** 2)

                    # Hiển thị đơn vị khoảng cách trên khung hình
                    cv2.putText(frame, f"Distance: {dist:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Hiển thị filter nếu dist > 0.1 và show_filter = True
                    if dist < 0.05:
                        head_filter_height, head_filter_width, _ = head_filter_image.shape

                        # Tính toán tọa độ đỉnh đầu
                        head_top = (int(face_landmarks[0].landmark[10].x * frame.shape[1]),
                                    int(face_landmarks[0].landmark[10].y * frame.shape[0]))

                        # Tính toán vị trí để gắn filter lên đỉnh đầu
                        head_filter_x = head_top[0] - head_filter_width + 80
                        head_filter_y = head_top[1] - head_filter_height + 30

                        # Giới hạn vùng cần gắn filter cho đỉnh đầu trong khung hình
                        head_region_width = min(head_filter_width, frame.shape[1] - head_filter_x)
                        head_region_height = min(head_filter_height, frame.shape[0] - head_filter_y)

                        # Lấy vùng cần gắn filter cho đỉnh đầu trên khung hình
                        head_roi = frame[head_filter_y:head_filter_y + head_region_height,
                                   head_filter_x:head_filter_x + head_region_width]

                        # Điều chỉnh kích thước filter cho đỉnh đầu để khớp với kích thước vùng cần gắn
                        head_filter_resized = cv2.resize(head_filter_image, (head_region_width, head_region_height))

                        # Kiểm tra kích thước của filter đã điều chỉnh
                        if head_filter_resized.shape[0] > head_roi.shape[0] or head_filter_resized.shape[1] > \
                                head_roi.shape[1]:
                            head_filter_resized = head_filter_resized[:head_roi.shape[0], :head_roi.shape[1]]

                        # Gắn filter cho đỉnh đầu vào vùng cần gắn trong khung hình
                        blended_head_roi = cv2.addWeighted(head_roi, 1, head_filter_resized, 1, 0)

                        # Gắn filter cho đỉnh đầu vào vùng cần gắn trong khung hình
                        if blended_head_roi is not None and head_region_height > 0 and head_region_width > 0:
                            frame[head_filter_y:head_filter_y + head_region_height,
                            head_filter_x:head_filter_x + head_region_width] = blended_head_roi


                # Hiển thị khung hình kết quả
                cv2.imshow('Hand Tracking', frame)

                # Nhấn phím 'q' để thoát
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Giải phóng tài nguyên và đóng video
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

