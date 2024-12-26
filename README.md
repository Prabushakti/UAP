**UAP Machine Learning 7B**

![download](https://github.com/user-attachments/assets/ee8601aa-a332-40d8-b315-f9874423611a)

**Ship Classification**

Nama : Prabu Shakti Parama Purusa Satum

NIM  : 202110370311491

***Deskripsi Project***

Kapal adalah kendaraan yang dirancang untuk beroperasi di atas air, seperti laut, sungai, dan danau. Kapal digunakan untuk berbagai tujuan, termasuk transportasi penumpang dan barang, eksplorasi, kegiatan militer, penyelamatan, hingga rekreasi.
Secara teknis, kapal biasanya memiliki ukuran yang lebih besar dibandingkan perahu, dilengkapi dengan mesin, layar, atau alat penggerak lainnya untuk bergerak di air.

Kapal memainkan peran penting dalam perdagangan internasional, transportasi, dan eksplorasi global sejak zaman kuno hingga era modern.
Fungsi kapal sangat beragam tergantung pada jenis dan tujuan penggunaannya seperti Transportasi Barang, tranportasi penumpang, eksplorasi dan penelitian, kegiatan militer dan keamanan, perikanan, kegiatan rekreasi, layanan penyelamatan, pengangkutan energi, dan konstruksi pada industri

Aplikasi ini dibuat untuk membantu masyarakat mengetahui berbagai macam jenis kapal dengan mudah.

Link dataset dapat diakses melalui link berikut : https://universe.roboflow.com/smu-gkbdw/ship-classification-ks6cf

***Overview Dataset***

Dataset ini untuk mengetahui jenis kapal yang bersumber dari Roboflow dan crawling dengan total 25 class dan total dataset berjumlah 3.144 ribu dataset. Train : 2499 (80%), Validation : 303 (10%), dan Test : 332 (10%)

0 : Aircraft Carrier  :  120

1 : Auxiliary Ship    :  187

2 : Barge             :  137

3 : Cargo             :  120

4 : Commander         :  23

5 : Container Ship    :  112

6 : Cruiser           :  110

7 : Destroyer         :  154

8 : Dock              :  131

9 : Ferry             :  115

10 : Fishing Vessel   :  120

11 : Frigate          :  118 

12 : Hovercraft       :  113

13 : Landing          :  114

14 : Motorboat        :  293

15 : Oil Tanker       :  120

16 : Other Merchant   :  29

17 : Other Ship       :  165

18 : Other Warship    :  139

19 : Patrol           :  93

20 : RoRo             :  27

21 : Sailboat         :  177

22 : Submarine        :  158

23 : Tugboat          :  110

24 : Yacht            :  160

***Langkah Instalasi Dependencies***

Berikut adalah Library yang digunakan pada project ini

![Screenshot 2024-12-25 213251](https://github.com/user-attachments/assets/1164659e-4d5e-413b-bae9-b55b1a73c138)

![Screenshot 2024-12-25 213120](https://github.com/user-attachments/assets/1ae9fe45-2894-432c-9c21-69b1fa389a8d)

***Preprocessing***

Preprocessing yang dilakukan meliputi beberapa langkah berikut: pertama, resizing gambar ke ukuran (224, 224), diikuti dengan rescale atau normalisasi nilai piksel ke rentang 1./255. Selanjutnya, dataset dibagi menjadi tiga subset, yaitu Training, Validation, dan Testing, sesuai dengan deskripsi pada bagian dataset. 

Pada model CNN, dilakukan augmentasi data pada folder training dengan parameter `rotation_range=45`, `zoom_range=0.2`, `horizontal_flip=True`, dan `vertical_flip=True`. Sementara itu, pada model MobileNetV2, augmentasi data dilakukan pada folder training dengan parameter `rotation_range=20`, `zoom_range=0.1`, `horizontal_flip=True`, dan `vertical_flip=True`.

***Deskripsi Model dan Hasil Analisis Project***

**1. Resnet50**

![Architecture Resnet 50](https://github.com/user-attachments/assets/b3ef513b-7baa-46db-b8ad-9771afa09b63)

***1.1 Modelling Resnet50***

![Screenshot 2024-12-25 214539](https://github.com/user-attachments/assets/47df4f5c-3bd5-46ab-b103-a0be0aeb1f9b)

***1.2 Evaluasi Model Resnet50***

![Grafik](https://github.com/user-attachments/assets/f56bb159-4fb2-40f1-b461-be7545b6b4cd)

***1.3 Confussion Matrix ResNet50***

![Confussion Matrix](https://github.com/user-attachments/assets/19ac7c73-f9b9-4e82-865f-ad1604bea355)

![Classification report](https://github.com/user-attachments/assets/f86fc9c2-2c48-4d0d-af82-b1765ab43836)

**2. MobileNetV2**

![MNV2 Aarchitecture](https://github.com/user-attachments/assets/d5901155-f4fb-4138-ad2f-b52bd2df015c)

***2.1 Modelling MNV2***

![Screenshot 2024-12-25 214322](https://github.com/user-attachments/assets/01527eb7-3893-4581-887b-6799dfd7cf33)

***2.2 Evaluasi Model MNV2***

![Grafik](https://github.com/user-attachments/assets/4938f4dd-a44b-44c9-ad6f-ecc5dac12813)

***2.3 Confussion Matrix***

![Confussion Matrix](https://github.com/user-attachments/assets/5cbca4f9-9f80-432b-b136-0ecfdc9e5119)


![Classification Report](https://github.com/user-attachments/assets/8eb22de3-19cc-4275-89b2-f75fe84a0c72)

**3. Convolutional Neural Network**

![Arsitektur CNN](https://github.com/user-attachments/assets/e4933dcf-099f-4c3e-bd86-7b9cfd4798d1)

***3.1 Modelling CNN***

![Screenshot 2024-12-25 214156](https://github.com/user-attachments/assets/2ed9f82d-d818-4290-b90b-ce1f4dccf8b3)


***3.2 Evaluasi Model CNN***


![Grafik](https://github.com/user-attachments/assets/e97b8b20-5eb9-464b-8b44-8c11e1eb9714)

***3.3 Confussion Matrix***

![confussion matrix](https://github.com/user-attachments/assets/886a0c68-f58d-46a1-a994-e4283acfd82b)

![Classification Report](https://github.com/user-attachments/assets/aa355f85-23b9-4afd-a9ed-81864002fcd0)


**Hasil**

![Screenshot 2024-12-25 220614](https://github.com/user-attachments/assets/237bb392-e187-4763-9645-603b4e11046b)

![Screenshot 2024-12-25 220714](https://github.com/user-attachments/assets/601c06df-e19c-4f4a-9e3d-75d41093cbdd)


***Link Aplikasi***
https://shipclassification.streamlit.app/
