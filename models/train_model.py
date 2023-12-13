import joblib
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import os
import cv2

def load_images_from_folder(folder):
    images = []
    labels = []

    for class_folder in os.listdir(folder):
        class_path = os.path.join(folder, class_folder)

        try:
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)

                try:
                    img = cv2.imread(img_path)

                    if img is not None and img.size != 0:
                        img = cv2.resize(img, (256, 256))
                        img_array = np.array(img)
                        images.append(img_array.flatten())
                        labels.append(class_folder)
                    else:
                        print(f"Error processing image {img_path}: Unable to read or empty image.")
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
        except Exception as e:
            print(f"Error accessing folder {class_path}: {e}")

    return np.array(images), np.array(labels)

# Path untuk dataset
folder_path_train = "CLASIFICATION SAMPAH/train"
X_train, y_train = load_images_from_folder(folder_path_train)

# Membuat direktori untuk menyimpan model jika belum ada
model_directory = 'D:\\Perkuliahan\\Semester 5\\TEKCERTAN\\ProyekTekcertan2\\app\\models'
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

# Simpan model dalam folder 'models'
model_path = os.path.join(model_directory, 'classifier_model.joblib')

# Inisialisasi model SVM
svm_model = svm.SVC(kernel='linear', C=1)

# Melatih model dengan data pelatihan
svm_model.fit(X_train, y_train)

# Simpan model ke dalam berkas menggunakan joblib
joblib.dump(svm_model, model_path)

# Jumlah Sampel Train
print("Jumlah Sampel Train:", len(X_train))

# Melakukan prediksi pada data uji
y_pred = svm_model.predict(X_train)

# Mengukur akurasi
accuracy = accuracy_score(y_train, y_pred)
print("Akurasi SVM pada data pelatihan:", accuracy)
