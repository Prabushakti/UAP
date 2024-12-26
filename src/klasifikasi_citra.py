import numpy as np
import tensorflow as tf
from pathlib import Path
import streamlit as st
import base64

def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: local;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

# Set background
set_background('src/static/background/Untitled Project3.jpg')

# Judul aplikasi
st.markdown(    """
    <div style="text-align: center; margin-bottom: 20px; font-size: 40px">
        Ship Classification
    </div>
    """,
    unsafe_allow_html=True
)

# Tambahkan CSS untuk latar belakang

st.markdown(
    """
    <div style="text-align: justify; margin-bottom: 20px;">
            Kapal adalah kendaraan yang dirancang untuk beroperasi di atas air, seperti laut, sungai, dan danau. Kapal digunakan untuk berbagai tujuan, termasuk transportasi penumpang dan barang, eksplorasi, kegiatan militer, penyelamatan, hingga rekreasi.
Secara teknis, kapal biasanya memiliki ukuran yang lebih besar dibandingkan perahu, dilengkapi dengan mesin, layar, atau alat penggerak lainnya untuk bergerak di air.

Kapal memainkan peran penting dalam perdagangan internasional, transportasi, dan eksplorasi global sejak zaman kuno hingga era modern.
             Fungsi kapal sangat beragam tergantung pada jenis dan tujuan penggunaannya seperti Transportasi Barang, tranportasi penumpang, eksplorasi dan penelitian, kegiatan militer dan keamanan, perikanan, kegiatan rekreasi, layanan penyelamatan, pengangkutan energi, dan konstruksi pada industri

Aplikasi ini dibuat untuk membantu masyarakat mengetahui berbagai macam jenis kapal dengan mudah.
        Silakan unggah gambar kapal, pilih model yang tersedia, dan tekan tombol Predict untuk mendapatkan hasil prediksi.
    </div>
    """,
    unsafe_allow_html=True
)

# Fungsi prediksi
def predict(uploaded_image, model_path):
    # Daftar kelas
    class_names = [
        "Aircraft Carrier",
        "Auxiliary Ship",
        "Barge",
        "Cargo",
        "Commander",
        "Container Ship",
        "Cruiser",
        "Destroyer",
        "Dock",
        "Ferry",
        "Fishing Vessel",
        "Frigate",
        "Hovercraft",
        "Landing",
        "Motorboat",
        "Oil Tanker",
        "Other Merchant",
        "Other Ship",
        "Other Warship",
        "Patrol",
        "RoRo",
        "Sailboat",
        "Submarine",
        "Tugboat",
        "Yacht",
    ]


    # Muat dan preprocess citra
    img = tf.keras.utils.load_img(uploaded_image, target_size=(224, 224))  # Pastikan ukuran sesuai dengan model
    img = tf.keras.utils.img_to_array(img) / 255.0  # Normalisasi
    img = np.expand_dims(img, axis=0)  # Tambahkan dimensi batch

    # Muat model
    model = tf.keras.models.load_model(model_path)

    # Prediksi
    output = model.predict(img)
    score = tf.nn.softmax(output[0])  # Hitung probabilitas
    return class_names[np.argmax(score)], 100 * np.max(score)  # Prediksi label dan confidence

# Fungsi prediksi dengan TFLite
def predict_tflite(uploaded_image, model_path):
    class_names = [
        "Aircraft Carrier", "Auxiliary Ship", "Barge", "Cargo", "Commander",
        "Container Ship", "Cruiser", "Destroyer", "Dock", "Ferry",
        "Fishing Vessel", "Frigate", "Hovercraft", "Landing", "Motorboat",
        "Oil Tanker", "Other Merchant", "Other Ship", "Other Warship", "Patrol",
        "RoRo", "Sailboat", "Submarine", "Tugboat", "Yacht"
    ]

    img = tf.keras.utils.load_img(uploaded_image, target_size=(224, 224))
    img = tf.keras.utils.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32) # Penting: tipe data float32

    # Load TFLite model dan alokasikan tensor
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Dapatkan detail input dan output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img)

    # Jalankan inferensi
    interpreter.invoke()

    # Dapatkan output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    score = tf.nn.softmax(output_data[0])
    return class_names[np.argmax(score)], 100 * np.max(score)

def predict_tflite_quantized(image_path, model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img = tf.keras.utils.img_to_array(img)

    print(f"Tipe data sebelum normalisasi: {img.dtype}") # Debugging

    img = img / 255.0  # Normalisasi *sebelum* konversi ke uint8
    print(f"Tipe data setelah normalisasi: {img.dtype}") # Debugging

    input_scale, input_zero_point = input_details[0]["quantization"]
    print(f"input_scale: {input_scale}, input_zero_point: {input_zero_point}")

    img = img / input_scale + input_zero_point
    print(f"Tipe data setelah scaling dan zero point: {img.dtype}") # Debugging

    img = img.astype(np.uint8) # Konversi ke uint8 *setelah* normalisasi dan scaling
    print(f"Tipe data setelah konversi ke uint8: {img.dtype}") # Debugging

    img = np.expand_dims(img, axis=0) # Tambahkan dimensi batch setelah konversi
    print(f"Shape gambar setelah expand_dims: {img.shape}") # Debugging

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    output_scale, output_zero_point = output_details[0]["quantization"]
    output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

    score = tf.nn.softmax(output_data[0])
    class_names = [
        "Aircraft Carrier", "Auxiliary Ship", "Barge", "Cargo", "Commander",
        "Container Ship", "Cruiser", "Destroyer", "Dock", "Ferry",
        "Fishing Vessel", "Frigate", "Hovercraft", "Landing", "Motorboat",
        "Oil Tanker", "Other Merchant", "Other Ship", "Other Warship", "Patrol",
        "RoRo", "Sailboat", "Submarine", "Tugboat", "Yacht",
    ]
    return class_names[np.argmax(score)], 100 * np.max(score)
    
# Pilihan model
model_option = st.selectbox("Pilih model untuk prediksi:", ("Resnet", "MobileNetV2", "CNN"))

# Tentukan path model berdasarkan pilihan
if model_option == "Resnet":
    #model_path = Path(__file__).parent / "Model/Image/Resnet/transfer_resnet50_model.h5"
    model_path = "src/transfer_resnet50_model.tflite"
    predict_func = predict_tflite_quantized
elif model_option == "MobileNetV2":
    model_path = "src/model_MNV2.h5"
    predict_func = predict
  #  model_path = Path(__file__).parent / "Model/Image/MobileNetV2/model_MNV2.h5"
else:
    #model_path = Path(__file__).parent / "Model/Image/CNN/CNN.h5"
    model_path = "src/CNN.tflite"
    predict_func = predict_tflite

# Komponen file uploader untuk banyak file
uploads = st.file_uploader("Unggah citra untuk mendapatkan hasil prediksi", type=["png", "jpg"], accept_multiple_files=True)

# Tombol prediksi
if st.button("Predict", type="primary"):
    if uploads:
        st.subheader("Hasil prediksi:")

        for upload in uploads:
            # Tampilkan setiap citra yang diunggah
            st.image(upload, caption=f"Citra yang diunggah: {upload.name}", use_container_width=True)

            with st.spinner(f"Memproses citra {upload.name} untuk prediksi..."):
                # Panggil fungsi prediksi
                try:
                    label, confidence = predict_func(upload, model_path)
                    st.write(f"Image: **{upload.name}**")
                    st.write(f"Label : **{label}**")
                    st.write(f"Confidence: **{confidence:.5f}%**")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses {upload.name}: {e}")
    else:
        st.error("Unggah setidaknya satu citra terlebih dahulu!")
