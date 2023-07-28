<div align="center">
  <h1 align="center">Tugas Akhir</h1>

  <h3 align="center">
     Penggabungan Segmentasi Pixel-Level dan Superpixel-Level untuk Pelabelan Fesyen Item
  </h3>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#struktur-projek">Struktur Projek</a>
    </li>
        <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#pelatihan-model-menggunakan-metode-segmentasi-pixel-level">Pelatihan Model menggunakan Metode Segmentasi Pixel-Level</a>
    </li>
    <li>
      <a href="#deteksi-dan-visualisasi-objek-fesyen-item">Deteksi dan Visualisasi Objek Fesyen Item</a>
    </li>
        <li>
      <a href="#deteksi-dan-visualisasi-objek-bentuk-tubuh-manusia">Deteksi dan Visualisasi Objek Bentuk Tubuh Manusia</a>
    </li>
    <li>
      <a href="#penguraian-objek">Penguraian Objek</a>
    </li>
    <li>
      <a href="#penerapan-superpixel">Penerapan Superpixel</a>
    </li>
        <li>
      <a href="#perbandingan-metode">Perbandingan Metode</a>
    </li>
  </ol>
</details>
<img width="1310" alt="image" src="https://github.com/migellamp/clothing-parsing/blob/main/images/judulFix.png">

## Alur Penelitian Projek

<img width="1310" alt="image" src="https://github.com/migellamp/clothing-parsing/blob/main/images/alur.png">

1. Pelatihan Model menggunakan metode segmentasi pixel-level
2. Deteksi dan visualisasi objek Fesyen Item (Clothing Segmentation)
3. Deteksi dan Visualisasi objek Tubuh Manusia (Body Segmentation)
4. Penguraian Objek
5. Penerapan Superpixel


## Installation
1. Clone repositori ini
2. Untuk menjalankan metode segmentasi Pixel-level, dapat menginstall dependency pertama menggunakan command berikut :
   ```bash
   pip install -r requirements.txt
   ```
3. Untuk menjalankan metode segmentasi Superpixel-level, dapat menginstall dependency kedua yang terletak pada file :
   ```
   dependencies.txt
   ```
4.  Jalankan file `setup.py` pada root directory
    ```bash
    python setup.py install
    ```
5.  Jalankan file `Program_TestingModel.ipynb` untuk melakukan testing pada Metode Segmentasi Superpixel-Level
    ```bash
    Jalankan satu per satu cell yang terletak didalam file tersebut
    ```
6.  Jalankan file `Program_TrainingModel.py` untuk melakukan training model menggunakan Metode Segmentasi Pixel-Level
    ```bash
    python ./custom/Program_TrainingModel.py train --dataset=/path --weigths=/path
    ```
7.  Jalankan file `Program_AplikasiTKINTER.py` untuk menjalankan program yang menyediakan UI
    ```bash
    python ./path/Program_AplikasiTKINTER.py
    ```
   

## Pelatihan Model menggunakan Metode Segmentasi Pixel-Level
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

## Deteksi dan Visualisasi Objek Fesyen Item
Untuk menjalankan proses deteksi dan visualisasi objek berupa fesyen item dapat menjalankan file `.ipynb` pada :
* Menjalankan file pada cell 1-5
```
./custom/Program_Main.ipynb
```
Hasil yang didapatkan ketika objek fesyen item dideteksi menggunakan model yang telah dilakukan pembelajaran pada metode segmentasi pixel-level ditampilkan pada gambar dibawah:
<img width="1065" alt="image" src="https://github.com/migellamp/clothing-parsing/blob/main/images/pixel-level.png">

## Deteksi dan Visualisasi Objek Bentuk Tubuh Manusia
Pada proses Body Segmentation, proses seperti mempersiapkan data dan melatih model dapat dilewati. Hal itu karena, untuk mendapatkan model pembelajaran pada dataset COCO, pada penelitian ini tidak perlu dilakukan tahap pembelajaran. 	
Saat menjalankan tahap pembelajaran model pada dataset ModaNet, untuk menjalankan proses tersebut pre-trained model tidak menjadi syarat dalam melakukan proses pembelajaran. Sehingga ketika pre-trained model tidak dipersiapkan terlebih dahulu maka otomatis weight model yang digunakan dalam proses pembelajaran akan diunduh terlebih dahulu secara otomatis. 

Pre-trained model yang dihasilkan merupakan model yang sudah dilakukan pembelajaran pada dataset COCO. Sehingga dengan menggunakan satu buah proses pembelajaran model saja yang berfokus pada dataset ModaNet, dapat menghasilkan dua buah weight model yaitu pre-trained model yang didapatkan pada proses Clothing segmentation dan weight model yang didapatkan juga setelah proses pembelajaran model dilakukan. 

Untuk menjalankan proses deteksi dan visualisasi objek berupa bentuk tubuh manusia dapat menjalankan file `.ipynb` pada :
* Menjalankan file pada cell 9
```
./custom/Program_Main.ipynb
```
Hasil yang didapatkan ketika objek tubuh manusia dideteksi menggunakan model yang telah dilakukan pembelajaran pada metode segmentasi pixel-level ditampilkan pada gambar dibawah:
<img width="775" alt="image" src="https://github.com/migellamp/clothing-parsing/blob/main/images/body">


## Penguraian Objek
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

## Penerapan Superpixel
Untuk menjalankan proses penerapan superpixel dapat menjalankan semua cell yang terdapat pada file `./custom/Program_Main.ipynb`.
Proses yang terjadi selama metode segmentasi superpixel-level berlangsung ialah sebagai berikut:

### 1. Menerapkan Superpixel
Algoritma SLIC merupakan algoritma yang dipilih untuk diterapkan kedalam metode superpixel pada penelitian ini. Algoritma SLIC menjalankan algoritma K-Means yang mana merupakan salah satu algoritma yang digunakan untuk mengelompokkan data kedalam suatu cluster.
```
AlgorithmSLIC = slic(image, segments, compactness, convert2lab , sigma)
```
<img width="638" alt="image" src="https://github.com/migellamp/clothing-parsing/assets/80758137/1c150435-654d-424e-90fe-58324f53559c">

### 2. Akses Setiap Piksel
Setiap piksel yang terbentuk akan dilakukan pengecekan satu per satu. Proses pengecekan bertujuan untuk melakukan identifikasi terhadap setiap pikselnya. Ketika piksel merupakan bagian dari suatu objek maka piksel tersebut akan diidentifikasikan sebagai objek baru.
```
  if(maskClothing[int(cx)][int(cy)] == True
    or maskClothing[int(cx-1)][int(cy)] == True
      or maskClothing[int(cx+1)][int(cy)] == True
        or maskClothing[int(cx)][int(cy-1)] == True
          or maskClothing[int(cx)][int(cy+1)] == True):

```
Apabila dilihat pada Gambar dibawah, bahwa hasil yang didapatkan dengan menerapkan metode segmentasi superpixel-level masih sangat bergantung pada objek yang dihasilkan pada metode segmentasi pixel-level. 

Hal tersebut dikarenakan, metode segmentasi hanya mengambil objek yang sebelumnya sudah dideteksi dan diterapkan dengan metode superpixel-level. Oleh karena itu, agar metode segmentasi superpixel-level tidak bergantung penuh dengan hasil yang didapatkan pada metode segmentasi pixel-level, maka proses selanjutnya ialah melakukan deteksi terhadap objek yang tidak terdeteksi pada metode segmentasi pixel- level.

<img width="831" alt="image" src="https://github.com/migellamp/clothing-parsing/assets/80758137/ae35adff-e71e-4608-b82a-1581786968f9">

### 3. Deteksi Objek atau Masks yang Tidak Sempurna

### Fungsi Deteksi Warna dan Neihbor Status
Fungsi deteksi warna akan melakukan pengecekan warna pada setiap piksel, warna pada setiap piksel yang tidak terdeteksi akan dilakukan pengecekan terhadap warna semua piksel yang terdeteksi. Ketika suatu piksel memiliki warna yang sama, superpixel tidak akan langsung menandai piksel tersebut terlebih dahulu. Namun untuk dengan pemanfaatan fungsi neighbor status, piksel yang memilki warna yang sama tersebut akan dilakukan pengecekan terlebih dahulu, apakah piksel sebelum dan sesudahnya memilki nama objek atau kelas yang sama. Ketika Piksel tetangganya memilki nama objek yang sama, maka baru piksel tersebut akan diidentifikasikan sebagai objek baru. 

### Penyempunaan Pengelompokkan Piksel
<img width="737" alt="image" src="https://github.com/migellamp/clothing-parsing/blob/main/images/penyempurnaan">


### Deteksi Warna Kulit
Fungsi deteksi warna akan digunakan untuk mengecek warna setiap piksel, ketika warna setiap piksel berada pada range warna kulit manusia. Maka metode superpixel-level tidak akan mengidentifikasi piksel tersebut sebagai objek fesyen item. Ketika warna piksel merupakan warna kulit manusia, maka piksel tersebut akan dilewati dan tidak akan di lakukan pengecekan.
```
dataset = CustomDataset()
dataset.load_custom("datasetTesting/dataset", "test")
dataset.prepare()


min_YCrCb = np.array([0,133,77],np.uint8)
max_YCrCb = np.array([235,173,127],np.uint8)  

print(len(dataset.image_ids))
for image_id in dataset.image_ids:
    plt.figure(figsize=(10, 10))
    _, ax = plt.subplots(1, figsize=(8,8))
    imageName = dataset.image_reference(image_id)
    image = cv2.imread(imageName)
    imageYCrCb = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)
    skinRegionYCrCb = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)

    skinYCrCb = cv2.bitwise_and(image, image, mask = skinRegionYCrCb)
    screen = cv2.cvtColor(skinYCrCb, cv2.COLOR_RGB2BGR)
    plt.imshow(screen)
```

### Hasil Akhir Penerapan Metode Superpixel-Level
<img width="1134" alt="image" src="https://github.com/migellamp/clothing-parsing/blob/main/images/hasil.png">


## Perbandingan Metode 
Perbandingan Metode Segmentasi Pixel Level dan Metode Segmentasi Pixel Level + Superpixel Level
Metode Segmentasi Pixel Level         |  Metode Segmentasi Pixel Level + Superpixel Level
:-------------------------:|:-------------------------:
<img width="289" alt="image" src="https://github.com/migellamp/clothing-parsing/blob/main/images/bf1.png"> | <img width="289" alt="image" src="https://github.com/migellamp/clothing-parsing/blob/main/images/af1.png">
<img width="289" alt="image" src="https://github.com/migellamp/clothing-parsing/blob/main/images/bf2.png"> |  <img width="289" alt="image" src="https://github.com/migellamp/clothing-parsing/blob/main/images/af2.png">
<img width="289" alt="image" src="https://github.com/migellamp/clothing-parsing/blob/main/images/bf3.png"> | <img width="289" alt="image" src="https://github.com/migellamp/clothing-parsing/blob/main/images/af3.png">







