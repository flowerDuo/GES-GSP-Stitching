# Geometric Structure Preserving Warp for Natural Image Stitching

This repository contains our dataset and C++ implementation of the CVPR 2022 paper, ***Geometric Structure Preserving Warp for Natural Image Stitching***. If you use any code or data from our work, please cite our paper.


<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://github.com/aalallalalal/GES-GSP-Stitching/blob/master/Images/CAVE-PLAYGROUND.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Figure 1. An example of 10 images stitching. (a) The AutoStitch's [8] result is severely distorted. (b) The person on the right side is distorted in the APAP's [7] result. (c) Several misalignments (red and green closeup) in the ELA's [9] result. (d) The SPW's [10] result exhibits significant wrong scale at the right end. (e) There are distortions in the red box, for example, the floor and carpet are curved in the result of GSP [2]. (f) Our result preserves the salient geometric structures in scene.</div>
</center>

## Download

1. [Paper](...)
2. [Supplementary](...)

## Code

### Usage

1. Download code and comile.
	* You need  **Opencv 4.4.0**,**VLFEAT**,**Eigen**
2. Download dataset to "input-data" folder.
3. Run project.
4. We provide scripts that make it easier to test data. The following are the steps:
5. Edit "RUN_EXE.bat". 
	* Change "file=\RUN_FILE.txt" and "\GES_Stitching.exe" to corresponding path.
5. List dataset names you want to test in "RUN_FILE.txt".
6. Click "RUN_EXE.bat".

You can find results in folder "input-data".

Notice: If you make changes to the code and run it, you can copy .exe from the "x64" to the root directory, and rename it to "GES_Stitching.exe".  


## Dataset

### Introduction
There are 50 diversified and challenging dataset (26 from [1–7] and 24 collected by ourselves). The numbers of images range from 2 to 35.

### Usage
1. Copy dataset to folder "input-data" in project.
2. Make sure the file "xxx-STITCH-GRAPH.txt" in each dataset correspond to the name of this dataset.
3. You can change the relation between the images by modifying the file "xxx-STITCH-GRAPH.txt".

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



