# Joint Pixel-Level and Superpixel-Level Segmentation for Parsing Clothing in Fashion Photographs

<img width="1310" alt="image" src="https://github.com/migellamp/clothing-parsing/assets/80758137/cac85daf-e165-415e-83a3-d6b0d2fe1c1a">

Project ini akan terbagi menjadi beberapa tahap yaitu :
1. Pelatihan Model menggunakan metode segmentasi pixel-level
2. Deteksi dan visualisasi objek Fesyen Item (Clothing Segmentation)
3. Deteksi dan Visualisasi objek Tubuh Manusia (Body Segmentation)
4. Penguraian Objek
5. Penerapan Superpixel


## 1. Pelatihan Model menggunakan Metode Segmentasi Pixel-Level
### Mempersiapkan Dataset
Sebelum dapat melakukan proses pelatihan atau pembelajaran perlu mempersiapkan dataset terlebih dahulu. Dataset yang digunakan yaitu dataset [ModaNet](https://github.com/eBay/modanet)
Dataset yang telah berhasil didownload masukkan kedalam folder berikut.
```
/custom/dataset/train/...
/custom/dataset/test/....
/custom/dataset/val/...
```
Dikarenakan metode segmentasi pixel-level menggunakan referensi dibawah, langkah selanjutnya ialah dengan melakukan pembagian pada dataset terhadap file anotasinya. Karena proses pembacaan gambar saat proses pelatihan akan dibaca melalui nama file yang terletak pada file anotasi, bukan dari gambar yang tersimpan pada suatu folder.
```
@misc{Soumya_Maskrcnn_2020,
  title={Mask R-CNN for custom object detection and instance segmentation on Keras and TensorFlow},
  author={Soumya Yadav},
  year={2020},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/soumyaiitkgp/Mask_RCNN/}},
}
```
### Proses Pembelajaran
Setelah dataset berhasil didownload dan dilakukan pembagian data, selanjutnya proses pembelajaran akan dilakukan. Untuk menjalankan proses pembelajaran dapat menjalankan command sebagai berikut :
* Jika memiliki Weigth Model, dapat menggunakannya command berikut
```
python ./custom/Program_TrainingModel.py train --dataset=/path --weigths=/path
```
* Jika tidak memiliki Weigth Model, dapat memasukkan nama yang diinginkan pada parameter `--weigths`. Mask R-CNN akan otomatis mengunduh pre-trained model.
```
python ./custom/Program_TrainingModel.py train --dataset=/path --weigths=(nama_model_bebas)
```
Selain itu, masih terdapat beberapa hal yang perlu dilakukan agar proses pembelajaran dapat menghasilkan hasil yang akurat. Yaitu dengan memilih value pada Hyperparameternya.
|     Parameter                     |     Nilai                 |
|-----------------------------------|---------------------------|
|     Backbone                      |     resnet101             |
|     Backbone_strides              |     [4, 8, 16, 32, 64]    |
|     Batch_size                    |     4                     |
|     Detection_min_confidence      |     0.9                   |
|     Detection_nms_threshold       |     0.3                   |
|     FPN_classif_fc_layers_size    |     1024                  |
|     Image_per_gpu                 |     4                     |
|     Image_max_dim                 |     1024                  |
|     Image_min_dim                 |     800                   |
|     Learning_rate                 |     0.001                 |
|     Num_classes                   |     14                    |
|     RPN_nms_threshold             |     0.7                   |
|     Steps_per_epoch               |     100                   |
|     Validation_steps              |     50                    |
|     Weight_decay                  |     0.0001                |

## 2. Deteksi dan Visualisasi Objek Fesyen Item
Untuk menjalankan proses deteksi dan visualisasi objek berupa fesyen item dapat menjalankan file `.ipynb` pada :
* Menjalankan file pada cell 1-5
```
./custom/Program_Main.ipynb
```
Hasil yang didapatkan ketika objek fesyen item dideteksi menggunakan model yang telah dilakukan pembelajaran pada metode segmentasi pixel-level ditampilkan pada gambar dibawah:
<img width="1065" alt="image" src="https://github.com/migellamp/clothing-parsing/assets/80758137/0f9cf24e-1531-4414-9546-1056b6b9ec72">

## 3. Deteksi dan Visualisasi Objek Bentuk Tubuh Manusia
Pada proses Body Segmentation, proses seperti mempersiapkan data dan melatih model dapat dilewati. Hal itu karena, untuk mendapatkan model pembelajaran pada dataset COCO, pada penelitian ini tidak perlu dilakukan tahap pembelajaran. 	
Saat menjalankan tahap pembelajaran model pada dataset ModaNet, untuk menjalankan proses tersebut pre-trained model tidak menjadi syarat dalam melakukan proses pembelajaran. Sehingga ketika pre-trained model tidak dipersiapkan terlebih dahulu maka otomatis weight model yang digunakan dalam proses pembelajaran akan diunduh terlebih dahulu secara otomatis. 

Pre-trained model yang dihasilkan merupakan model yang sudah dilakukan pembelajaran pada dataset COCO. Sehingga dengan menggunakan satu buah proses pembelajaran model saja yang berfokus pada dataset ModaNet, dapat menghasilkan dua buah weight model yaitu pre-trained model yang didapatkan pada proses Clothing segmentation dan weight model yang didapatkan juga setelah proses pembelajaran model dilakukan. 

Untuk menjalankan proses deteksi dan visualisasi objek berupa bentuk tubuh manusia dapat menjalankan file `.ipynb` pada :
* Menjalankan file pada cell 9
```
./custom/Program_Main.ipynb
```
Hasil yang didapatkan ketika objek tubuh manusia dideteksi menggunakan model yang telah dilakukan pembelajaran pada metode segmentasi pixel-level ditampilkan pada gambar dibawah:
<img width="775" alt="image" src="https://github.com/migellamp/clothing-parsing/assets/80758137/a175f81d-4594-4522-9801-74a7dbd1ea6e">


## 4. Penguraian Objek
Setelah model yang telah dilakukan pembelajaran didapatkan, dan proses deteksi objek dijalankan terhadap kedua proses segmentasi, yaitu proses Clothing Segmentation dan Body Segmentation. Selanjutnya objek yang teridentifikasi pada saat proses deteksi objek, perlu diuraikan terlebih dahulu. Proses penguraian tersebut bertujuan untuk mengambil semua objek yang terdeteksi, serta informasi berupa koordinat setiap objek tersebut juga diperlukan. 

Untuk menjalankan proses penguraian objek yang didapatkan dapat menjalankan file `./custom/Program_Main.ipynb` pada cell 7 dan 10.
```
ClassIndex = []
for i in r1['class_ids']:
    print(class_names_orang[i])
    ClassIndex.append(class_names_orang[i])
indexPerson = ClassIndex.index("person")
print(indexPerson)
# print(r1['rois'].shape[0])

mask2 = r1['masks'][:,:,indexPerson]
make_segmentation_mask(image2, mask2)
# get_coordinates(number)
```
Hasil yang didapatkan ketika kedua objek diuraikan atau diambil dalam bentuk masks ditampilkan pada gambar dibawah:
<img width="1144" alt="image" src="https://github.com/migellamp/clothing-parsing/assets/80758137/38baab4b-e0bf-4f1d-9d79-8d82746fe5e7">



