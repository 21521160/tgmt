import cv2
import mediapipe as mp
import imageio
import numpy as np

# Khởi tạo đối tượng Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Khởi tạo đối tượng Pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Đọc video từ webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Đường dẫn đến file GIF filter
filter_path = "bomb.gif"
filter_duration = 1  # Thời gian chạy của filter (s)

# Đọc file GIF filter
filter_frames = imageio.mimread(filter_path)
filter_frames = [frame[:, :, :3] for frame in filter_frames]  # Loại bỏ kênh alpha (nếu có)

# Thiết lập các biến cho việc chạy filter
frame_count = 0
filter_start_frame = None
idx_gif = 0

# Cấu hình VideoWriter
output_path = "outputgif.mp4"
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Định dạng video
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    # Đọc từng khung hình
    success, image = cap.read()
    if not success:
        break

    # Chuyển đổi hình ảnh sang không gian màu RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Phát hiện các điểm mốc trên cơ thể
    results = pose.process(image_rgb)

    # # Vẽ các điểm mốc và các kết nối giữa chúng lên hình ảnh
    # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
    #                           mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
    #                           mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))

    # Kiểm tra xem cánh tay có đang được duỗi thẳng
    if results.pose_landmarks:
        left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]

        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Tính toán và xác định các cử chỉ của cánh tay
        left_wrist_elbow_length = left_wrist.x - left_elbow.x
        left_elbow_shoulder_length = left_elbow.x - left_shoulder.x

        right_wrist_elbow_length = right_wrist.x - right_elbow.x
        right_elbow_shoulder_length = right_elbow.x - right_shoulder.x

        co_threshold = 0.1
        duoi_threshold = 0.1

        if left_wrist_elbow_length < co_threshold and left_elbow_shoulder_length < duoi_threshold:
            left_gesture = "CO"
        else:
            left_gesture = "DUOI"

        if right_wrist_elbow_length < co_threshold and right_elbow_shoulder_length < duoi_threshold:
            right_gesture = "CO"
        else:
            right_gesture = "DUOI"

        # Hiển thị kết quả lên màn hình
        cv2.putText(image, left_gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, right_gesture, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Chèn filter vào vị trí cách cánh tay
        if left_gesture == "DUOI":
            # Tính toán vị trí để hiển thị filter cách cánh tay trái
            left_arm_x = int(left_elbow.x * image.shape[1])
            left_arm_y = int(left_elbow.y * image.shape[0])
            blend_distance = 200  # Khoảng cách từ cánh tay
            x = left_arm_x - int(filter_frames[0].shape[1] / 2) + blend_distance + 500
            y = left_arm_y - int(filter_frames[0].shape[0] / 2) + 50

            # Chèn filter vào khung hình
            gif_size = 200  # Kích thước mới của filter (kích thước phóng to)
            gif = filter_frames[idx_gif]
            gif = cv2.resize(gif, (gif_size, gif_size))
            gif = cv2.cvtColor(gif, cv2.COLOR_BGRA2RGBA)
            idx_gif = (idx_gif + 1) % len(filter_frames)

            red_range_lower = np.array([0, 0, 255], dtype=np.uint8)
            red_range_upper = np.array([200, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(gif[:, :, :3], red_range_lower, red_range_upper)
            gif[:, :, 3] = np.where(mask == 255, 0, gif[:, :, 3])

            try:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
                alpha_sticker = gif[:, :, 3] / 255.0
                for c in range(3):
                    image[y:y + gif_size, x:x + gif_size, c] = (
                            alpha_sticker * gif[:, :, c] + alpha_sticker * image[y:y + gif_size, x:x + gif_size, c])
            except:
                pass

        if right_gesture == "DUOI":
            # Tính toán vị trí để hiển thị filter cách cánh tay phải
            right_arm_x = int(right_elbow.x * image.shape[1])
            right_arm_y = int(right_elbow.y * image.shape[0])
            blend_distance = 200  # Khoảng cách từ cánh tay
            x = right_arm_x - int(filter_frames[0].shape[1] / 2) + blend_distance + 500
            y = right_arm_y - int(filter_frames[0].shape[0] / 2) + 50

            # Chèn filter vào khung hình
            gif_size = 200  # Kích thước mới của filter (kích thước phóng to)
            gif = filter_frames[idx_gif]
            gif = cv2.resize(gif, (gif_size, gif_size))
            gif = cv2.cvtColor(gif, cv2.COLOR_BGRA2RGBA)
            idx_gif = (idx_gif + 1) % len(filter_frames)

            red_range_lower = np.array([0, 0, 255], dtype=np.uint8)
            red_range_upper = np.array([200, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(gif[:, :, :3], red_range_lower, red_range_upper)
            gif[:, :, 3] = np.where(mask == 255, 0, gif[:, :, 3])

            try:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
                alpha_sticker = gif[:, :, 3] / 255.0
                for c in range(3):
                    image[y:y + gif_size, x:x + gif_size, c] = (
                            alpha_sticker * gif[:, :, c] + alpha_sticker * image[y:y + gif_size, x:x + gif_size, c])
            except:
                pass
    # Hiển thị hình ảnh kết quả
    cv2.imshow('MediaPipe Pose', image)

    # Viết video xuất ra file
    out.write(image)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()



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
# while True:
#     # Đọc từng khung hình
#     success, image = cap.read()
#     if not success:
#         break
#
#     # Chuyển đổi hình ảnh sang không gian màu RGB
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     # Phát hiện các điểm mốc trên cơ thể
#     results = pose.process(image_rgb)
#
#     # Vẽ các điểm mốc và các kết nối giữa chúng lên hình ảnh
#     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                               mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
#                               mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
#
#     # Kiểm tra xem cánh tay có đang được duỗi thẳng
#     if results.pose_landmarks:
#         left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
#         left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
#         left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
#
#         right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
#         right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
#         right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
#
#         # Tính toán độ dài từ cổ tay đến khuỷu tay và từ khuỷu tay đến vai (sử dụng tọa độ x)
#         left_wrist_elbow_length = left_wrist.x - left_elbow.x
#         left_elbow_shoulder_length = left_elbow.x - left_shoulder.x
#
#         right_wrist_elbow_length = right_wrist.x - right_elbow.x
#         right_elbow_shoulder_length = right_elbow.x - right_shoulder.x
#
#         # Sử dụng ngưỡng để xác định khi nào cánh tay được co và khi nào cánh tay được duỗi
#         co_threshold = 0.1
#         duoi_threshold = 0.1
#
#         if left_wrist_elbow_length < co_threshold and left_elbow_shoulder_length < duoi_threshold:
#             left_gesture = "CO"
#         else:
#             left_gesture = "DUOI"
#
#         if right_wrist_elbow_length < co_threshold and right_elbow_shoulder_length < duoi_threshold:
#             right_gesture = "CO"
#         else:
#             right_gesture = "DUOI"
#
#         # Hiển thị kết quả lên màn hình
#         cv2.putText(image, left_gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         cv2.putText(image, right_gesture, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#
#         # Chèn filter vào vị trí cách cánh tay
#         if left_gesture == "DUOI":
#             # Tính toán vị trí để hiển thị filter cách cánh tay trái
#             left_arm_x = int(left_elbow.x * image.shape[1])
#             left_arm_y = int(left_elbow.y * image.shape[0])
#             blend_distance = 200  # Khoảng cách từ cánh tay
#             x = left_arm_x - int(filter_frames[0].shape[1] / 2) + blend_distance + 500
#             y = left_arm_y - int(filter_frames[0].shape[0] / 2) + 50
#
#             # Chèn filter vào khung hình
#             gif_size = 200  # Kích thước mới của filter (kích thước phóng to)
#             gif = filter_frames[idx_gif]
#             gif = cv2.resize(gif, (gif_size, gif_size))
#             gif = cv2.cvtColor(gif, cv2.COLOR_BGRA2RGBA)
#             idx_gif = (idx_gif + 1) % len(filter_frames)
#
#             red_range_lower = np.array([0, 0, 255], dtype=np.uint8)
#             red_range_upper = np.array([200, 255, 255], dtype=np.uint8)
#             mask = cv2.inRange(gif[:, :, :3], red_range_lower, red_range_upper)
#             gif[:, :, 3] = np.where(mask == 255, 0, gif[:, :, 3])
#
#             try:
#                 image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
#                 alpha_sticker = gif[:, :, 3] / 255.0
#                 for c in range(3):
#                     image[y:y + gif_size, x:x + gif_size, c] = (
#                             alpha_sticker * gif[:, :, c] + alpha_sticker * image[y:y + gif_size, x:x + gif_size, c])
#             except:
#                 pass
#
#         if right_gesture == "DUOI":
#             # Tính toán vị trí để hiển thị filter cách cánh tay phải
#             right_arm_x = int(right_elbow.x * image.shape[1])
#             right_arm_y = int(right_elbow.y * image.shape[0])
#             blend_distance = 200  # Khoảng cách từ cánh tay
#             x = right_arm_x - int(filter_frames[0].shape[1] / 2) + blend_distance + 500
#             y = right_arm_y - int(filter_frames[0].shape[0] / 2) + 50
#
#             # Chèn filter vào khung hình
#             gif_size = 200  # Kích thước mới của filter (kích thước phóng to)
#             gif = filter_frames[idx_gif]
#             gif = cv2.resize(gif, (gif_size, gif_size))
#             gif = cv2.cvtColor(gif, cv2.COLOR_BGRA2RGBA)
#             idx_gif = (idx_gif + 1) % len(filter_frames)
#
#             red_range_lower = np.array([0, 0, 255], dtype=np.uint8)
#             red_range_upper = np.array([200, 255, 255], dtype=np.uint8)
#             mask = cv2.inRange(gif[:, :, :3], red_range_lower, red_range_upper)
#             gif[:, :, 3] = np.where(mask == 255, 0, gif[:, :, 3])
#
#             try:
#                 image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
#                 alpha_sticker = gif[:, :, 3] / 255.0
#                 for c in range(3):
#                     image[y:y + gif_size, x:x + gif_size, c] = (
#                             alpha_sticker * gif[:, :, c] + alpha_sticker * image[y:y + gif_size, x:x + gif_size, c])
#             except:
#                 pass
#     # Hiển thị hình ảnh kết quả
#     cv2.imshow('MediaPipe Pose', image)
#
#     # Nhấn phím 'q' để thoát
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Giải phóng tài nguyên
# cap.release()
# cv2.destroyAllWindows()
