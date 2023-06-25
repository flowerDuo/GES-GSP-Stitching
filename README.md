# Geometric Structure Preserving Warp for Natural Image Stitching

This repository contains our dataset and C++ implementation of the CVPR 2022 paper, ***Geometric Structure Preserving Warp for Natural Image Stitching***. If you use any code or data from our work, please cite our paper.


<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://github.com/flowerDuo/GES-GSP-Stitching/blob/master/Images/CAVE-PLAYGROUND.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Figure 1. An example of stitching 10 images. (a) The AutoStitch's [8] result is severely distorted. (b) The person on the right side is distorted in the APAP's [7] result. (c) Several misalignments (red and green closeup) in the ELA’s [9] result. (d) The SPW's [10] result exhibits significant wrong scale at the right end. (e) There are distortions in the red box, e.g., the floor and carpet are curved in the result obtained by GSP [2]. (f) Our result preserves the salient geometric structures in scene.</div>
</center>

## Download

1. [Paper+Supplementary](https://openaccess.thecvf.com/content/CVPR2022/html/Du_Geometric_Structure_Preserving_Warp_for_Natural_Image_Stitching_CVPR_2022_paper.html)
2. Code
	* [GitHub](https://github.com/flowerDuo/GES-GSP-Stitching/tree/master/Code)
	* [Baidu NetDisk](https://pan.baidu.com/s/16OoMjeEiOLHIxv8shB1dFw?pwd=h5vi)
3. DataSet (GES-50)
	* [GitHub](https://github.com/flowerDuo/GES-GSP-Stitching/tree/master/Dataset)
	* [Google NetDisk](https://drive.google.com/file/d/1SlQ2P9nW9PW4hUGemDvv6uCy75byDPq8/view?usp=sharing)
	* [Baidu NetDisk](https://pan.baidu.com/s/1ok-yYw1Ww77ARZ6tiHxgjA?pwd=7zjv)
4. [Android Appication(Harmony)](http://mosica.nat300.top/aic/mosaic.apk)
    * [Server download](http://mosica.nat300.top/aic/mosaic.apk)
    * [Baidu NetDisk](https://pan.baidu.com/s/1f4HkxKF7I71Be4W6awnpKw?pwd=s8vi )

## Code

### 1. Usage

	(1). Download code and comile.
		You need Opencv 4.4.0, VLFEAT, Eigen.
	(2). Download dataset to "input-data" folder.
	(3). Run project.

Or

	(4). We provide scripts that make it easier to test data. The following are the steps:
	(5). Edit "RUN_EXE.bat". 
		Change "file=\RUN_FILE.txt" and "\GES_Stitching.exe" to corresponding path.
	(6). List dataset names you want to test in "RUN_FILE.txt".
	(7). Click "RUN_EXE.bat".

Notice:
* If you make changes to the code, you can copy .exe from the "x64" to the root directory and rename it to "GES_Stitching.exe" after running project.
* If the .exe output errors, try to run the project to get a new .exe.
	
You can find results in folder "input-data".

  


## Dataset

### 1. Introduction
There are 50 diversified and challenging dataset (26 from [1–7] and 24 collected by ourselves). The numbers of images range from 2 to 35.

### 2. Usage
	(1). Copy dataset to folder "input-data" in project.
	(2). Make sure the file "xxx-STITCH-GRAPH.txt" in each dataset correspond to the name of this dataset.
	(3). You can change the relation between the images by modifying the file "xxx-STITCH-GRAPH.txt".

## Android(Harmony) Application

### 1. Introduction
Based on the C++ implementation of the CVPR 2022 paper, 
Geometric Structure Preserving Warp for Natural Image Stitching, we have developed an Android(Harmony) application.

With our Android(Harmony) application,
you can easily perform image stitching and obtain large-scale images in various fields 
such as cultural tourism, smart agriculture, and security monitoring. 
You can effortlessly complete the stitching process with astonishing speed while ensuring high-quality results

We feel sorry, but currently this application only supports Chinese. However, you can follow our instructions to use it.

### 2. Guide
	(1). Download and install the package on an Android(Harmony) phone.
	(2). Apply for a trial account and log in.
	(3). Select to import from the gallery or capture images for stitching.
	(4). Select 'Speed Priority'(Left) or 'Quality Priority'(Right) and then click on the top-right corner to start the stitching process.
	(5). After obtaining the stitching result, you can choose to perform operations such as cropping, saving, sharing, and more.

### 3. Part of the software screenshot
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://github.com/flowerDuo/GES-GSP-Stitching/blob/master/Images/guide4.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Figure 2 Part of the software screenshot</div>
</center>

### 4. Account application
Welcome to download the Android application. If you need a trial account, please contact us via email(lin9@nwafu.edu.cn)


## Contact

Feel free to contact me if there is any question (peng-du@nwafu.edu.cn).

## Reference
> 1. *Che-Han Chang, Yoichi Sato, and Yung-Yu Chuang. Shapepreserving half-projective warps for image stitching. In Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition, pages 3254–3261, 2014.*
> 2. *Yu-Sheng Chen and Yung-Yu Chuang. Natural image stitching with the global similarity prior. In European conference
on computer vision, pages 186–201. Springer, 2016.*
> 3. *Junhong Gao, Seon Joo Kim, and Michael S Brown. Constructing image panoramas using dual-homography warping. In Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition, pages 49–56. IEEE, 2011.*
> 4. *Qi Jia, ZhengJun Li, Xin Fan, Haotian Zhao, Shiyu Teng,Xinchen Ye, and Longin Jan Latecki. Leveraging line-point consistence to preserve structures for wide parallax image
stitching. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 12186–12195,2021.*
> 5. *Chung-Ching Lin, Sharathchandra U Pankanti, Karthikeyan Natesan Ramamurthy, and Aleksandr Y Aravkin. Adaptive as-natural-as-possible image stitching. In Proceedings of the
IEEE Conference on Computer Vision and Pattern Recognition, pages 1155–1163, 2015.*
> 6. *Yoshikuni Nomura, Li Zhang, and Shree K Nayar. Scene collages and flexible camera arrays. In Proceedings of the 18th Eurographics conference on Rendering Techniques, pages
127–138, 2007.*
> 7. *Julio Zaragoza, Tat-Jun Chin, Michael S Brown, and David Suter. As-projective-as-possible image stitching with moving dlt. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2339–2346, 2013.*
> 8. *Matthew Brown and David G Lowe. Automatic panoramic image stitching using invariant features. International journal of computer vision, 74(1):59–73, 2007.* 
> 9. *Jing Li, Zhengming Wang, Shiming Lai, Yongping Zhai, and Maojun Zhang. Parallax-tolerant image stitching based on robust elastic warping. IEEE Transactions on multimedia,
20(7):1672–1687, 2017.*
> 10. *Tianli Liao and Nan Li. Single-perspective warps in natural image stitching. IEEE Transactions on Image Processing, 29:724–735, 2019.*



