
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tkinter import Tk, filedialog

def select_image():
    root = Tk()
    root.withdraw()  
    file_path = filedialog.askopenfilename(title="Chọn một ảnh", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
    return file_path

# 1. Ảnh âm tính (áp dụng cho từng kênh màu)
def negative_transform(image):
    return 255 - image

# 2. Tăng độ tương phản (co giãn histogram cho từng kênh màu)
def increase_contrast(image):
    contrast_image = np.zeros_like(image)
    for i in range(3):  # 3 kênh màu: B, G, R
        min_val = np.min(image[:, :, i])
        max_val = np.max(image[:, :, i])
        contrast_image[:, :, i] = (image[:, :, i] - min_val) * (255 / (max_val - min_val))
    return contrast_image.astype(np.uint8)

# 3. Biến đổi log với chuẩn hóa
def log_transform(image):
    # Chuyển đổi kiểu dữ liệu sang float32 để tránh tràn số
    image_float = image.astype(np.float32)
    
    # Áp dụng biến đổi logarit
    c = 255 / np.log(1 + np.max(image_float))
    log_image = c * np.log(1 + image_float)
    
    # Chuẩn hóa giá trị pixel về khoảng 0-255
    log_image = cv2.normalize(log_image, None, 0, 255, cv2.NORM_MINMAX)
    
    # Chuyển đổi lại về kiểu uint8 để hiển thị ảnh
    return log_image.astype(np.uint8)

# 4. Cân bằng histogram (áp dụng cho từng kênh màu)
def equalize_hist_color(image):
    hist_equalized_image = np.zeros_like(image)
    for i in range(3):  # 3 kênh màu: B, G, R
        hist_equalized_image[:, :, i] = cv2.equalizeHist(image[:, :, i])
    return hist_equalized_image

# Gọi hàm chọn ảnh
image_path = select_image()

if image_path:
    # Đọc ảnh đã chọn (ảnh màu)
    image = cv2.imread(image_path)

    # Áp dụng các phép biến đổi
    negative_image = negative_transform(image)          # Ảnh âm tính
    contrast_image = increase_contrast(image)           # Tăng độ tương phản
    log_image = log_transform(image)                    # Biến đổi log với chuẩn hóa
    hist_equalized_image = equalize_hist_color(image)   # Cân bằng histogram

    # Hiển thị kết quả
    titles = ['Original Image', 'Negative Image', 'Increased Contrast', 'Log Transform', 'Histogram Equalization']
    images = [image, negative_image, contrast_image, log_image, hist_equalized_image]

    plt.figure(figsize=(10, 8))

    for i in range(5):
        plt.subplot(2, 3, i+1), plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()
else:
    print("Không có ảnh nào được chọn.")