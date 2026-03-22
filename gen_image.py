import cv2
import os

def extract_frames(video_path, output_folder="duc_face", interval=5):
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(output_folder, exist_ok=True)

    # Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Không thể mở video:", video_path)
        return

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Hết video

        # Mỗi 5 frame lưu 1 ảnh
        if frame_count % interval == 0:
            filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Đã lưu {saved_count} ảnh vào thư mục '{output_folder}'.")

extract_frames("/home/ntduc/doan/78637757288126213.mp4")