import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def parse_arguments():
    """Hàm phân tích đối số dòng lệnh."""
    parser = argparse.ArgumentParser(description="Edge Detection using Sobel and LoG")
    parser.add_argument("input_image", type=str, help="Đường dẫn đến ảnh đầu vào")
    parser.add_argument("--blur", type=int, default=5, help="Kích thước kernel Gaussian Blur (mặc định: 5)")
    return parser.parse_args()

def sobel_edge_detection(image):
    """Hàm thực hiện dò biên bằng toán tử Sobel."""
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient theo trục X
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient theo trục Y
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)  # Tổng hợp magnitude
    return cv2.convertScaleAbs(sobel_magnitude)  # Chuyển về kiểu uint8

def log_edge_detection(image, blur_size):
    """Hàm thực hiện dò biên bằng Laplacian of Gaussian (LoG)."""
    blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 0)  # Làm mờ ảnh
    log_result = cv2.Laplacian(blurred, cv2.CV_64F)  # Áp dụng Laplacian
    return cv2.convertScaleAbs(log_result)  # Chuyển về kiểu uint8

def display_results(original, sobel_result, log_result):
    """Hàm hiển thị ảnh gốc và kết quả dò biên."""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title('Ảnh Gốc', fontsize=14)
    plt.imshow(original, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Toán tử Sobel', fontsize=14)
    plt.imshow(sobel_result, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Laplacian of Gaussian (LoG)', fontsize=14)
    plt.imshow(log_result, cmap='gray')
    plt.axis('off')

    plt.show()

def main():
    """Hàm chính."""
    args = parse_arguments()

    # Kiểm tra ảnh đầu vào có tồn tại không
    if not os.path.exists(args.input_image):
        print(f"Lỗi: Không tìm thấy ảnh tại '{args.input_image}'")
        return

    # Đọc ảnh đầu vào
    image = cv2.imread(args.input_image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Lỗi: Không thể đọc ảnh. Đảm bảo file là ảnh hợp lệ.")
        return

    # Thực hiện dò biên
    sobel_result = sobel_edge_detection(image)
    log_result = log_edge_detection(image, args.blur)

    # Hiển thị kết quả
    display_results(image, sobel_result, log_result)

if __name__ == "__main__":
    main()
