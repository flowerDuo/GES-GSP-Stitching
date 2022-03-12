# Geometric Structure Preserving Warp for Natural Image Stitching



This repository contains our dataset and C++ implementation of the CVPR 2022 paper, ***Geometric Structure Preserving Warp for Natural Image Stitching***. If you use any code or data from our work, please cite our paper.

## Code

### Download

1. [Paper](...)
2. [Supplementary](...)

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
> 7. *Julio Zaragoza, Tat-Jun Chin, Michael S Brown, and David Suter. As-projective-as-possible image stitching with moving dlt. In Proceedings of the IEEE Conference on Computer
Vision and Pattern Recognition, pages 2339–2346, 2013.*


