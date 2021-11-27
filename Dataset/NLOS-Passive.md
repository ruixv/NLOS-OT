# NLOS-Passive

You can download each group in NLOS-Passive through the link below. Please note that a compressed package (.zip or .z01+.zip) represents a group of measured data.


- Since the size of NLOS-Passive exceeds 1TB, we use Baidu Netdisk to store it. If you cannot use Baidu Netdisk, please contact us for data.
- If the link fails, please feel free to contact us.

link：https://pan.baidu.com/s/19Q48BWm1aJQhIt6BF9z-uQ 

code：j3p2

In the following, we will explain the meaning of each directory and file in the link, and their corresponding parts in the article.


## Step2_pretrained and Step1_pretrained
These two folders save the weight files of the networks in Step1 and Step2. You can use them to evaluate our network directly. Strictly speaking, they are not part of NLOS-Passive. We put them in the link just for convenience.

## Dataset
``Dataset`` is the NLOS-Passive directory. It contains four subfolders:
- Anime Faces
- MNIST
- Supermodel Faces
- STL-10

They respectively represent the collected datasets when using anime faces/ MNIST/ supermodel faces/ STL-10 as the hidden pictures. In each subfolder, there are several compressed files. They can be divided into two groups
- Groundtruth (hidden images) starting with ``GT``
- Projection images collected under different conditions.

### STL-10
| File Name | Description|
| ------ | ------ |
| GT_stl10_allimages.zip | Ground-truth: STL-10 and real-word images played on the screen. |
| stl10_dark_1_d180_wall_occluder.zip/.z01 | Using stl-10 (and real-world images), the collected projection images in the dark, angle $\angle \alpha 1$, $D = 180cm$, wall and partial occluder condition. |
