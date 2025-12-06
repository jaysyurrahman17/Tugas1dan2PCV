## 🎓 Tugas PCV: Image Filtering & Color Detection pada VTuber

Laporan ini mendokumentasikan integrasi tugas mata kuliah Pengolahan Citra Digital ke dalam sistem VTuber yang telah dibuat sebelumnya.

Tujuan dari tugas ini adalah menambahkan fungsionalitas:

1. **Tugas 1:** Implementasi Filter Konvolusi (Blurring & Sharpening) pada input kamera.

2. **Tugas 2:** Deteksi Objek Berwarna (Warna Biru) berbasis HSV untuk memicu interaksi pada avatar.

## 📋 Spesifikasi Tugas & Implementasi

### Tugas 1: Image Filtering (Konvolusi)

Fitur ini memungkinkan pengguna mengubah tampilan feed kamera asli menggunakan keyboard.

| **Mode**      | **Filter**        | **Implementasi Teknis**                                                                                                                                 |
| --------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tekan '1'** | **Average Blur**  | Menggunakan kernel box filter ukuran 5x5 (```cv2.blur```).                                                                                                |
| **Tekan '2'** | **Gaussian Blur** | **(Wajib Manual)** Membuat kernel Gaussian 2D sendiri menggunakan perkalian outer product dari vektor Gaussian 1D, lalu diterapkan dengan ```cv2.filter2D```. |
| **Tekan '3'** | **Sharpening**    | Menggunakan kernel matriks Laplacian-based untuk mempertajam tepi citra.                                                                            |
| **Tekan '0'** | **Normal**        | Reset ke tampilan asli.                                                                                                                             |

### Code Snippet (Pembuatan Kernel Gaussian Manual):
```
# Sesuai syarat tugas untuk memahami konsep konvolusi
k_1d = cv2.getGaussianKernel(ksize, sigma)
k_2d = np.outer(k_1d, k_1d) # Menghasilkan matriks 2D
output = cv2.filter2D(frame, -1, k_2d)
```

### Tugas 2: Color Detection & Morphological Operations

Fitur ini mendeteksi keberadaan objek berwarna BIRU (misal: tutup botol atau pulpen) untuk mengubah latar belakang Avatar secara otomatis.

**Metode:**

1. **Color Space Conversion:** Mengubah frame dari BGR ke **HSV (Hue, Saturation, Value)** agar deteksi warna tahan terhadap perubahan intensitas cahaya.

2. **Thresholding:** Membuat *Binary Mask* menggunakan ```cv2.inRange``` dengan batas:

    - Lower Blue: ```[100, 150, 50]```

    - Upper Blue: ```[140, 255, 255]```

3. **Morphological Operations (Pembersihan Noise):**

    - **Opening:** Menghapus noise (bintik putih kecil) di latar belakang.

    - **Closing:** Menutup lubang-lubang kecil pada objek yang terdeteksi.

4. **Action Trigger:** Jika kontur objek > 500 pixel terdeteksi, latar belakang VTuber berubah menjadi ```background_special.jpg.```

## 🎥 Demo Video

Klik gambar di bawah ini untuk melihat video demo:

[![Judul](https://drive.google.com/uc?export=view&id=1ec4OYlgAChrX0uRQ1pdKbCUWf-B17GpY)](https://drive.google.com/file/d/1uwOrMdrb7CxE1BvQ3hZS9UBdKVKxavBX/view?usp=drive_link)


## 🖥️ Tampilan Antarmuka

Aplikasi menjalankan dua jendela sekaligus:

1. **Jendela Kamera Input:** Menampilkan video asli pengguna yang telah diberi efek **Filter (Tugas 1)**.

2. **Jendela Avatar Output:** Menampilkan karakter VTuber yang bereaksi terhadap **Deteksi Warna (Tugas 2)**.

## 📂 Perubahan Struktur Kode

Kode dimodifikasi menjadi modular untuk memisahkan logika core VTuber dan logika tugas kuliah:

- ```config.py```: Ditambahkan konfigurasi Kernel Sharpening dan Range Warna HSV.

- ```utils.py```:

  - Added: ```create_gaussian_kernel()``` (Logika manual Gaussian).

  - Added: ```detect_blue_object()``` (Pipeline HSV + Morfologi).

  - Added: ```apply_filter_mode()``` (Switch case untuk filter).

- ```main.py```: Mengintegrasikan input keyboard dan loop deteksi warna ke dalam *main loop* VTuber.

## 🚀 Cara Pengujian

1. Jalankan ```main.py```.

2. **Uji Tugas 1**: Tekan angka ```1```, ```2```, atau ```3``` pada keyboard. Perhatikan perubahan pada jendela "Kamera Input".

3. **Uji Tugas 2**: Arahkan benda berwarna biru tua ke depan kamera. Perhatikan latar belakang pada jendela "VTuber Output" akan berubah.
