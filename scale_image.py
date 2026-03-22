import os
from PIL import Image

# Đường dẫn đến các thư mục
folders = ['/home/ntduc/doan/casia_webface/casia-webface/010572', '/home/ntduc/doan/casia_webface/casia-webface/010573', '/home/ntduc/doan/casia_webface/casia-webface/010574']

# Kích thước mới
new_size = (112, 112)

# Hàm thay đổi kích thước ảnh trong folder
def resize_images_in_folder(folder_path, new_size):
    for filename in os.listdir(folder_path):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):  # Kiểm tra định dạng ảnh
            image_path = os.path.join(folder_path, filename)
            try:
                # Mở ảnh
                img = Image.open(image_path)
                # Thay đổi kích thước ảnh
                img_resized = img.resize(new_size)
                # Lưu ảnh đã thay đổi kích thước
                img_resized.save(image_path)
                print(f'Resized {filename} in {folder_path}')
            except Exception as e:
                print(f'Error processing {filename}: {e}')

# Lặp qua tất cả các thư mục và thay đổi kích thước các ảnh
for folder in folders:
    resize_images_in_folder(folder, new_size)
