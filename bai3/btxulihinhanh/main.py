import tkinter as tk
from tkinter import filedialog, Label, Frame, messagebox
from PIL import ImageTk, Image
from src.data_preprocessing import load_dataset, split_data, load_and_preprocess_image
from src.model_training import train_svm, train_knn, train_decision_tree
from src.model_evaluation import evaluate_model
from src.utils import log_results

def predict_image(image_path, model):
    """
    Dự đoán lớp của một ảnh dựa trên mô hình đã huấn luyện.
    """
    image = load_and_preprocess_image(image_path)
    prediction = model.predict([image])  # Chuyển ảnh thành mảng để mô hình dự đoán
    return prediction[0]  # Trả về nhãn dự đoán (0 hoặc 1)

def classify_image():
    """
    Xử lý việc phân loại khi người dùng chọn ảnh từ máy tính
    """
    file_path = filedialog.askopenfilename()

    if file_path:
        # Hiển thị ảnh được chọn
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img

        # Dự đoán lớp của ảnh
        prediction = predict_image(file_path, svm_model)  # Sử dụng SVM để phân loại

        # Hiển thị kết quả
        if prediction == 0:
            result_label.config(text="Dự đoán: Hoa")
        else:
            result_label.config(text="Dự đoán: Động vật")

        # Hiển thị các chỉ số của mô hình
        model_info.config(text=f"SVM Model:\nAccuracy: {svm_accuracy:.4f}\nPrecision: {svm_precision:.4f}\nRecall: {svm_recall:.4f}\nTraining time: {svm_time:.4f} seconds")

def train_models():
    """
    Huấn luyện mô hình từ tập dữ liệu
    """
    global svm_model, svm_time, svm_accuracy, svm_precision, svm_recall
    
    # Load dữ liệu và chia tập huấn luyện và kiểm tra
    dataset_dir = 'data/dataset'
    X, y = load_dataset(dataset_dir)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Huấn luyện mô hình SVM
    svm_model, svm_time = train_svm(X_train, y_train)
    
    # Đánh giá mô hình SVM
    svm_accuracy, svm_precision, svm_recall = evaluate_model(svm_model, X_test, y_test)

    # In ra kết quả cho SVM
    log_results('SVM', svm_accuracy, svm_precision, svm_recall, svm_time)

    messagebox.showinfo("Thông báo", "Huấn luyện hoàn tất!")

# Tạo giao diện người dùng
root = tk.Tk()
root.title("Phân loại hoa và động vật")
root.geometry("800x600")  # Kích thước cửa sổ lớn hơn

# Khung chứa các phần của giao diện
frame = Frame(root, bg="lightblue", padx=10, pady=10)
frame.pack(fill="both", expand=True)

# Nút để người dùng tải ảnh lên
upload_button = tk.Button(frame, text="Tải ảnh lên", font=("Arial", 16), command=classify_image, bg="green", fg="white")
upload_button.pack(pady=20)

# Khung hiển thị ảnh
panel = Label(frame, bg="white", width=300, height=300)
panel.pack(pady=10)

# Nhãn hiển thị kết quả dự đoán
result_label = tk.Label(frame, text="Kết quả phân loại sẽ hiển thị ở đây", font=("Arial", 18), bg="lightblue")
result_label.pack(pady=10)

# Khung hiển thị các chỉ số của mô hình
model_info = tk.Label(frame, text="Chỉ số mô hình sẽ hiển thị ở đây", font=("Arial", 14), bg="lightblue")
model_info.pack(pady=10)

# Khởi động quá trình huấn luyện mô hình
train_models()

# Chạy vòng lặp chính của giao diện Tkinter
root.mainloop()
