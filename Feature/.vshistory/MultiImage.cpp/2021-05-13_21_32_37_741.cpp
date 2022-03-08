//
//  MultiImages.cpp
//  UglyMan_Stitching
//
//  Created by uglyman.nothinglo on 2015/8/15.
//  Copyright (c) 2015 nothinglo. All rights reserved.
//

#include "MultiImages.h"

MultiImages::MultiImages(const string& _file_name,
	LINES_FILTER_FUNC* _width_filter,
	LINES_FILTER_FUNC* _length_filter) : parameter(_file_name) {

	/*将文件夹下面的每个图片都实例化为 ImageData ,并加入images_data 列表*/
	for (int i = 0; i < parameter.image_file_full_names.size(); ++i) {
#ifndef DP_NO_LOG
		images_data.emplace_back(parameter.file_dir,
			parameter.image_file_full_names[i],
			_width_filter,
			_length_filter,
			&parameter.debug_dir);
#else
		images_data.emplace_back(parameter.file_dir,
			parameter.image_file_full_names[i],
			_width_filter,
			_length_filter);
#endif
	}
}

/// <summary>
/// 特征点找,匹配,剔除,局部单应性矩阵;
//<image url="$(ProjectDir)CommentImage\MultiImage-doFeatureMatching.jpg" scale="0.1"/>
/// </summary>
void MultiImages::doFeatureMatching() const {
	//将 配对的两两图片的 对应两个index 放入 列表中,并返回.一对图片并不会出现1--2加2--1的情况,就一个 1--2
	const vector<pair<int, int> >& images_match_graph_pair_list = parameter.getImagesMatchGraphPairList();
	//images_features 对应保存每张图片的ImageFeatures 数据.只包括网格点;
	images_features.resize(images_data.size());
	//images_features_mask 为 bool 的二维数组.第一维对应 一张图像;第二维对应一张图像中的网格点.;
	images_features_mask.resize(images_data.size());

	//将网格点都作为特征点 加入每张图像对应的列表. 
	for (int i = 0; i < images_data.size(); ++i) {
		// 获取网格点 对应的 像素点数据.
		const vector<Point2>& vertices = images_data[i].mesh_2d->getVertices();
		// images_features_mask 每一维,记录一张图像中 网格点的状态.
		images_features_mask[i].resize(vertices.size(), false);
		//将网格点 加入列表.
		for (int j = 0; j < vertices.size(); ++j) {
			images_features[i].keypoints.emplace_back(vertices[j], 0);
		}
	}
	//一维数组,来表示两两图像 对应数据.包括H,匹配网格对.
	pairwise_matches.resize(images_data.size() * images_data.size());
	//3维数组
	apap_homographies.resize(images_data.size());
	//3维数组
	apap_overlap_mask.resize(images_data.size());
	//3维数组
	apap_matching_points.resize(images_data.size());
	for (int i = 0; i < images_data.size(); ++i) {
		apap_homographies[i].resize(images_data.size());
		apap_overlap_mask[i].resize(images_data.size());
		apap_matching_points[i].resize(images_data.size());
	}

	//得到三维数组,记录了两两图像之间对应的正确的特征点对.
	//feature_matches[i][j]:指向的是第i张图像对应第j张图像 i上的特征点;
	//feature_matches[j][i]:指向的是第i张图像对应第j张图像 j上的特征点;
	const vector<vector<vector<Point2>>>& feature_matches = getFeatureMatches();
	//特征点匹配阶段完成!
	//
	for (int i = 0; i < images_match_graph_pair_list.size(); ++i) {
		//得到配对的图像index 对.
		const pair<int, int>& match_pair = images_match_graph_pair_list[i];
		//m1:第一张图像  m2:第二张图像  m1 变形向 m2靠齐
		const int& m1 = match_pair.first, & m2 = match_pair.second;

		//<image url="$(ProjectDir)CommentImage\APAP_Stitching_apap_project.png" scale="0.5"/>
		//apap_matching_points[][]存放对应图为m2,变形图为m1,m1的新网格点数据 ;
		//传入参数:m1上正确的特征点坐标集, m2上正确的特征点坐标集, m1的网格点, m1变形对应m2后的新网格点,存放m1图像 所有的网格点对应H,个数为网格点数;
		APAP_Stitching::apap_project(feature_matches[m1][m2],
			feature_matches[m2][m1],
			images_data[m1].mesh_2d->getVertices(), apap_matching_points[m1][m2], apap_homographies[m1][m2]);
		//apap_matching_points[][]存放对应图为m1,变形图为m2,m2的新网格点数据 ;
		//传入参数:m2上正确的特征点坐标集, m1上正确的特征点坐标集, m2的网格点, m2变形对应m1后的新网格点,存放m2图像 所有的网格点对应H,个数为网格点数;
		APAP_Stitching::apap_project(feature_matches[m2][m1],
			feature_matches[m1][m2],
			images_data[m2].mesh_2d->getVertices(), apap_matching_points[m2][m1], apap_homographies[m2][m1]);

		const int PAIR_SIZE = 2;

		//存放都已变形完的图一图二的网格点位置 [vector<point2>,vector<point2>] = [图1中网格点1,2,3...的point2 , 图2中网格点1,2,3...的point2]
		const vector<Point2>* out_dst[PAIR_SIZE] = { &apap_matching_points[m1][m2], &apap_matching_points[m2][m1] };

		apap_overlap_mask[m1][m2].resize(apap_homographies[m1][m2].size(), false);
		apap_overlap_mask[m2][m1].resize(apap_homographies[m2][m1].size(), false);

		//相当于,为每张图片都留出 了imagesize的位置: 123..n 123...n........（n*n的矩阵）
		const int pm_index = m1 * (int)images_data.size() + m2;
		const int m_index[PAIR_SIZE] = { m2, m1 };
		//读取m1-m2图像 位置的vector<DMatch>,表示m1对应到m2 的匹配网格对. 
		//存放m1扭到m2以及m2扭到m1的所有自己网格点加扭来的网格点.
		vector<DMatch>& D_matches = pairwise_matches[pm_index].matches;

		//将匹配信息放入一维数组(表示二维数组)
		//下面这for就是:将扭曲图片的网格点 加入到 对应图片网格点数据上:

		for (int j = 0; j < PAIR_SIZE; ++j) {
			for (int k = 0; k < out_dst[j]->size(); ++k) {
				//遍历第1或2张图变换后的所有网格点
				//图像j的第k个网格点的x,y 均>=0 ;   图像j的第k个网格点(变形点) 要在另一个图像(参考图像) 的区域.即最后为重叠区域.;
				if ((*out_dst[j])[k].x >= 0 && (*out_dst[j])[k].y >= 0 &&
					(*out_dst[j])[k].x <= images_data[m_index[j]].img.cols &&
					(*out_dst[j])[k].y <= images_data[m_index[j]].img.rows) {

					if (j) {
						//m2扭曲对应m1

						apap_overlap_mask[m2][m1][k] = true;
						//将m1图片上次最后加入的点index+1 对应 扭曲图片--m2的k位;

						D_matches.emplace_back(images_features[m_index[j]].keypoints.size(), k, 0);
						images_features_mask[m2][k] = true;
					}
					else {
						//m1扭曲对应m2

						apap_overlap_mask[m1][m2][k] = true;
						//将m2图片上次最后加入的点index+1 对应 扭曲图片--m1的k位;
						D_matches.emplace_back(k, images_features[m_index[j]].keypoints.size(), 0);
						images_features_mask[m1][k] = true;
					}

					//将m1的变换后的网格点数据加入m2的网格点列表;,相反:将m2的变换后的网格点数据加入m1的网格点列表
					//images_features[m1] 放m1自己的网格点加上扭曲的m2网格点;images_features[m2] 放m2自己的网格点加上扭曲的m1网格点;
					images_features[m_index[j]].keypoints.emplace_back((*out_dst[j])[k], 0);
				}
			}
		}
		pairwise_matches[pm_index].confidence = 2.; /*** need > 1.f ***/
		pairwise_matches[pm_index].src_img_idx = m1;
		pairwise_matches[pm_index].dst_img_idx = m2;
		pairwise_matches[pm_index].inliers_mask.resize(D_matches.size(), 1);
		pairwise_matches[pm_index].num_inliers = (int)D_matches.size();
		pairwise_matches[pm_index].H = apap_homographies[m1][m2].front(); /*** for OpenCV findMaxSpanningTree funtion ***/
		//pairwise_matches[pm_index].ransacDifference = ransacDiff[m1][m2];


		//最后pairwise_matches[pm_index]存放:m1->m2: 
		// m1 扭曲 到 m2 对应的m1点的H    +     m1扭曲到m2:(m2的自己网格点index+ m1扭曲过来网格点的index)  +    m2扭曲到m1:(m1的自己网格点index+ m2扭曲过来网格点的index)


	}
}

/// <summary>
/// 获取每个图像的自己变换前的网格点数据和匹配图像网格变换后的点数据.
/// </summary>
/// <returns></returns>
const vector<detail::ImageFeatures>& MultiImages::getImagesFeaturesByMatchingPoints() const {
	if (images_features.empty()) {
		doFeatureMatching();
	}
	return images_features;
}

/// <summary>
/// 每对匹配图像 的匹配信息.注意:例如:pairwise_matches:长度为4,如果图1和图2有关系,那么只有1--->2有数据,2--->1是没数据的.
/// </summary>
/// <returns></returns>
const vector<detail::MatchesInfo>& MultiImages::getPairwiseMatchesByMatchingPoints() const {
	if (pairwise_matches.empty()) {
		doFeatureMatching();
	}
	return pairwise_matches;
}

/// <summary>
/// 获取相机参数
/// </summary>
/// <returns></returns>
const vector<detail::CameraParams>& MultiImages::getCameraParams() const {
	if (camera_params.empty()) {
		camera_params.resize(images_data.size());
		/*** Focal Length ***/
		const vector<vector<vector<bool> > >& apap_overlap_mask = getAPAPOverlapMask();
		const vector<vector<vector<Mat> > >& apap_homographies = getAPAPHomographies();

		vector<Mat> translation_matrix;
		translation_matrix.reserve(images_data.size());
		for (int i = 0; i < images_data.size(); ++i) {
			//	图像坐标系-->像素坐标系的矩阵
			//<image url="$(ProjectDir)CommentImage\MultiImage-getCameraParams.png" scale="1"/>
			Mat T(3, 3, CV_64FC1);
			T.at<double>(0, 0) = T.at<double>(1, 1) = T.at<double>(2, 2) = 1;
			T.at<double>(0, 2) = images_data[i].img.cols * 0.5;
			T.at<double>(1, 2) = images_data[i].img.rows * 0.5;
			T.at<double>(0, 1) = T.at<double>(1, 0) = T.at<double>(2, 0) = T.at<double>(2, 1) = 0;
			translation_matrix.emplace_back(T);
		}
		vector<vector<double> > image_focal_candidates;
		image_focal_candidates.resize(images_data.size());
		for (int i = 0; i < images_data.size(); ++i) {
			for (int j = 0; j < images_data.size(); ++j) {
				for (int k = 0; k < apap_overlap_mask[i][j].size(); ++k) {
					if (apap_overlap_mask[i][j][k]) {//(图i扭曲 对应 图j,k为i上的处于重叠区域的点index)
						double f0, f1;
						bool f0_ok, f1_ok;
						//注:这H写错了, 应该是R;
						//<image url="$(ProjectDir)CommentImage\MultiImage-getCameraParams_estimate.png" scale="1"/>
						//且,把f放到了R上:<image url="$(ProjectDir)CommentImage\MultiImage-getCameraParams_estimate3.png" scale="1"/>
						Mat H = translation_matrix[j].inv() * apap_homographies[i][j][k] * translation_matrix[i];

						detail::focalsFromHomography(H / H.at<double>(2, 2),
							f0, f1, f0_ok, f1_ok);
						if (f0_ok && f1_ok) {
							image_focal_candidates[i].emplace_back(f0);
							image_focal_candidates[j].emplace_back(f1);
						}
					}
				}
			}
		}
		for (int i = 0; i < camera_params.size(); ++i) {
			if (image_focal_candidates[i].empty()) {
				camera_params[i].focal = images_data[i].img.cols + images_data[i].img.rows;
			}
			else {
				Statistics::getMedianWithoutCopyData(image_focal_candidates[i], camera_params[i].focal);
			}
		}
		/********************/
		/*** 3D Rotations ***/
		vector<vector<Mat> > relative_3D_rotations;
		relative_3D_rotations.resize(images_data.size());
		for (int i = 0; i < relative_3D_rotations.size(); ++i) {
			relative_3D_rotations[i].resize(images_data.size());
		}
		const vector<detail::ImageFeatures>& images_features = getImagesFeaturesByMatchingPoints();
		const vector<detail::MatchesInfo>& pairwise_matches = getPairwiseMatchesByMatchingPoints();
		const vector<pair<int, int> >& images_match_graph_pair_list = parameter.getImagesMatchGraphPairList();
		for (int i = 0; i < images_match_graph_pair_list.size(); ++i) {
			const pair<int, int>& match_pair = images_match_graph_pair_list[i];
			const int& m1 = match_pair.first, & m2 = match_pair.second;
			const int m_index = m1 * (int)images_data.size() + m2;
			const detail::MatchesInfo& matches_info = pairwise_matches[m_index];
			const double& focal1 = camera_params[m1].focal;
			const double& focal2 = camera_params[m2].focal;

			MatrixXd A = MatrixXd::Zero(matches_info.num_inliers * DIMENSION_2D,
				HOMOGRAPHY_VARIABLES_COUNT);

			for (int j = 0; j < matches_info.num_inliers; ++j) {
				Point2d p1 = Point2d(images_features[m1].keypoints[matches_info.matches[j].queryIdx].pt) -
					Point2d(translation_matrix[m1].at<double>(0, 2), translation_matrix[m1].at<double>(1, 2));
				Point2d p2 = Point2d(images_features[m2].keypoints[matches_info.matches[j].trainIdx].pt) -
					Point2d(translation_matrix[m2].at<double>(0, 2), translation_matrix[m2].at<double>(1, 2));
				A(2 * j, 0) = p1.x;
				A(2 * j, 1) = p1.y;
				A(2 * j, 2) = focal1;
				A(2 * j, 6) = -p2.x * p1.x / focal2;
				A(2 * j, 7) = -p2.x * p1.y / focal2;
				A(2 * j, 8) = -p2.x * focal1 / focal2;

				A(2 * j + 1, 3) = p1.x;
				A(2 * j + 1, 4) = p1.y;
				A(2 * j + 1, 5) = focal1;
				A(2 * j + 1, 6) = -p2.y * p1.x / focal2;
				A(2 * j + 1, 7) = -p2.y * p1.y / focal2;
				A(2 * j + 1, 8) = -p2.y * focal1 / focal2;
			}
			JacobiSVD<MatrixXd, HouseholderQRPreconditioner> jacobi_svd(A, ComputeThinV);
			MatrixXd V = jacobi_svd.matrixV();
			Mat R(3, 3, CV_64FC1);
			for (int j = 0; j < V.rows(); ++j) {
				R.at<double>(j / 3, j % 3) = V(j, V.rows() - 1);
			}
			SVD svd(R, SVD::FULL_UV);
			relative_3D_rotations[m1][m2] = svd.u * svd.vt;
		}
		queue<int> que;
		vector<bool> labels(images_data.size(), false);
		const int& center_index = parameter.center_image_index;
		const vector<vector<bool> >& images_match_graph = parameter.getImagesMatchGraph();

		que.push(center_index);
		relative_3D_rotations[center_index][center_index] = Mat::eye(3, 3, CV_64FC1);

		while (que.empty() == false) {
			int now = que.front();
			que.pop();
			labels[now] = true;
			for (int i = 0; i < images_data.size(); ++i) {
				if (labels[i] == false) {
					if (images_match_graph[now][i]) {
						relative_3D_rotations[i][i] = relative_3D_rotations[now][i] * relative_3D_rotations[now][now];
						que.push(i);
					}
					if (images_match_graph[i][now]) {
						relative_3D_rotations[i][i] = relative_3D_rotations[i][now].inv() * relative_3D_rotations[now][now];
						que.push(i);
					}
				}
			}
		}
		/********************/
		for (int i = 0; i < camera_params.size(); ++i) {
			camera_params[i].aspect = 1;
			camera_params[i].ppx = translation_matrix[i].at<double>(0, 2);
			camera_params[i].ppy = translation_matrix[i].at<double>(1, 2);
			camera_params[i].t = Mat::zeros(3, 1, CV_64FC1);
			camera_params[i].R = relative_3D_rotations[i][i].inv();
			camera_params[i].R.convertTo(camera_params[i].R, CV_32FC1);
		}

		//Ptr<detail::BundleAdjusterBase> adjuster = makePtr<detail::BundleAdjusterReproj>();
		//adjuster->setTermCriteria(TermCriteria(TermCriteria::EPS, CRITERIA_MAX_COUNT, CRITERIA_EPSILON));

		//Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
		//refine_mask(0, 0) = 1; /* (0, 0)->focal, (0, 2)->ppx, (1, 2)->ppy, (1, 1)->aspect */
		//adjuster->setConfThresh(1.f);
		//adjuster->setRefinementMask(refine_mask);

		//if (!(*adjuster)(images_features, pairwise_matches, camera_params)) {
		//	printError("F(getCameraParams) camera parameters adjuster failed");
		//}

		Mat center_rotation_inv = camera_params[parameter.center_image_index].R.inv();
		for (int i = 0; i < camera_params.size(); ++i) {
			camera_params[i].R = center_rotation_inv * camera_params[i].R;
		}
		/* wave correction */
		if (WAVE_CORRECT != WAVE_X) {
			vector<Mat> rotations;
			rotations.reserve(camera_params.size());
			for (int i = 0; i < camera_params.size(); ++i) {
				rotations.emplace_back(camera_params[i].R);
			}
			waveCorrect(rotations, ((WAVE_CORRECT == WAVE_H) ? detail::WAVE_CORRECT_HORIZ : detail::WAVE_CORRECT_VERT));
			for (int i = 0; i < camera_params.size(); ++i) {
				camera_params[i].R = rotations[i];
			}
		}
		/*******************/
	}
	return camera_params;
}

//<image url="$(ProjectDir)CommentImage\MultiImage_getImagesFeaturesMaskByMatchingPoints1.png" scale="1"/>
const vector<vector<bool> >& MultiImages::getImagesFeaturesMaskByMatchingPoints() const {
	if (images_features_mask.empty()) {
		doFeatureMatching();
	}
	return images_features_mask;
}

/// <summary>
/// 获取
/// </summary>
/// <returns></returns>
const vector<vector<vector<bool> > >& MultiImages::getAPAPOverlapMask() const {
	if (apap_overlap_mask.empty()) {
		doFeatureMatching();
	}
	return apap_overlap_mask;
}
const vector<vector<vector<Mat> > >& MultiImages::getAPAPHomographies() const {
	if (apap_homographies.empty()) {
		doFeatureMatching();
	}
	return apap_homographies;
}

const vector<vector<vector<Point2> > >& MultiImages::getAPAPMatchingPoints() const {
	if (apap_matching_points.empty()) {
		doFeatureMatching();
	}
	return apap_matching_points;
}

/// <summary>
/// 算出 每张图像的 每个点 (自己的网格点和另一个图片投影过来的网格点) 处于第几个网格点 和 该网格点4个点的权重.
/// </summary>
/// <returns></returns>
const vector<vector<InterpolateVertex> >& MultiImages::getInterpolateVerticesOfMatchingPoints() const {
	if (mesh_interpolate_vertex_of_matching_pts.empty()) {

		mesh_interpolate_vertex_of_matching_pts.resize(images_data.size());
		//images_features[m1] 放m1自己的网格点加上扭曲的m2网格点;images_features[m2] 放m2自己的网格点加上扭曲的m1网格点;
		const vector<detail::ImageFeatures>& images_features = getImagesFeaturesByMatchingPoints();
		for (int i = 0; i < mesh_interpolate_vertex_of_matching_pts.size(); ++i) {
			//第i张图
			mesh_interpolate_vertex_of_matching_pts[i].reserve(images_features[i].keypoints.size());
			for (int j = 0; j < images_features[i].keypoints.size(); ++j) {
				//算出 第i张图像的每个点处于第几个网格点 和 该网格点4个点的权重.
				mesh_interpolate_vertex_of_matching_pts[i].emplace_back(images_data[i].mesh_2d->getInterpolateVertex(images_features[i].keypoints[j].pt));
			}
		}
	}
	return mesh_interpolate_vertex_of_matching_pts;
}

/// <summary>
/// 返回为每个图像的第一个网格起始点index.顺序是图像文件顺序,和图像关系无关. 且注意返回数据index是*2了.
/// </summary>
/// <returns></returns>
const vector<int>& MultiImages::getImagesVerticesStartIndex() const {
	if (images_vertices_start_index.empty()) {
		images_vertices_start_index.reserve(images_data.size());
		int index = 0;
		for (int i = 0; i < images_data.size(); ++i) {
			images_vertices_start_index.emplace_back(index);
			index += images_data[i].mesh_2d->getVertices().size() * DIMENSION_2D;
		}
	}
	return images_vertices_start_index;
}

/// <summary>
/// 先用LSD求出图像对旋转角度,.为了更好的得到旋转角度解,加限制(认为如果用LSD得到的旋转角度得到的合理旋转角度范围包括0度,则此图像就是0度.).最后最小二乘法得到,最佳的图像旋转角度. 以及scale
/// 详见论文.
/// </summary>
/// <param name=""></param>
/// <returns></returns>
const vector<SimilarityElements>& MultiImages::getImagesSimilarityElements(const enum GLOBAL_ROTATION_METHODS& _global_rotation_method) const {
	const vector<vector<SimilarityElements>*>& images_similarity_elements = {
		&images_similarity_elements_2D, &images_similarity_elements_3D
	};
	vector<SimilarityElements>& result = *images_similarity_elements[_global_rotation_method];
	if (result.empty()) {
		result.reserve(images_data.size());
		//获取相机参数
		const vector<detail::CameraParams>& camera_params = getCameraParams();

		for (int i = 0; i < images_data.size(); ++i) {
			//第一项:第i张图像与中心图像的焦距相比,确定scale.第二项:根据相机外参的R,确定图像的旋转角度.
			result.emplace_back(fabs(camera_params[parameter.center_image_index].focal / camera_params[i].focal),
				-getEulerZXYRadians<float>(camera_params[i].R)[2]);
		}

		//中心图像的旋转角度.默认为0π;
		double rotate_theta = parameter.center_image_rotation_angle;
		for (int i = 0; i < images_data.size(); ++i) {
			double a = (result[i].theta - rotate_theta) * 180 / M_PI;
			//控制旋转角度不超过180度.
			result[i].theta = normalizeAngle(a) * M_PI / 180;
			//记录每张图像的旋转角度,单位是xπ.而不是度数.
		}

		const vector<pair<int, int> >& images_match_graph_pair_list = parameter.getImagesMatchGraphPairList();
		//图像对中每个图像的旋转角度范围.
		const vector<vector<pair<double, double> > >& images_relative_rotation_range = getImagesRelativeRotationRange();

		switch (_global_rotation_method) {
		case GLOBAL_ROTATION_2D_METHOD:
		{
			class RotationNode {
			public:
				int index, parent;
				RotationNode(const int _index, const int _parent) {
					index = _index, parent = _parent;
				}
			private:

			};

			/// <summary>
			/// 以下为BFS沿着邻接图法传播旋转范围.
			/// </summary>
			/// <param name=""></param>
			/// <returns></returns>
			const double TOLERANT_THETA = TOLERANT_ANGLE * M_PI / 180;
			vector<pair<int, double> > theta_constraints;
			//记录哪些图像已经被确定了旋转角度
			vector<bool> decided(images_data.size(), false);
			vector<RotationNode> priority_que;
			theta_constraints.emplace_back(parameter.center_image_index, result[parameter.center_image_index].theta);
			decided[parameter.center_image_index] = true;
			priority_que.emplace_back(parameter.center_image_index, -1);
			const vector<vector<bool> >& images_match_graph = parameter.getImagesMatchGraph();
			while (priority_que.empty() == false) {
				RotationNode node = priority_que.front();
				priority_que.erase(priority_que.begin());
				if (!decided[node.index]) {
					decided[node.index] = true;
					//根据上一个图像的旋转角度,确定此图像的旋转角.使用LSD得到旋转角度。
					result[node.index].theta = result[node.parent].theta + getImagesMinimumLineDistortionRotation(node.parent, node.index);
				}
				for (int i = 0; i < decided.size(); ++i) {
					if (!decided[i]) {
						const int e[EDGE_VERTEX_SIZE] = { node.index, i };
						for (int j = 0; j < EDGE_VERTEX_SIZE; ++j) {
							if (images_match_graph[e[j]][e[!j]]) {
								//两个图像有关联
								RotationNode new_node(i, node.index);
								if (isRotationInTheRange<double>(0, result[node.index].theta + images_relative_rotation_range[node.index][i].first - TOLERANT_THETA,
									result[node.index].theta + images_relative_rotation_range[node.index][i].second + TOLERANT_THETA)) {
									//如果下张图像旋转范围包括0度,则下张旋转角度设置为0;
									priority_que.insert(priority_que.begin(), new_node);
									result[i].theta = 0;
									decided[i] = true;
									theta_constraints.emplace_back(i, 0);
								}
								else {
									priority_que.emplace_back(new_node);
								}
								break;
							}
						}
					}
				}
			}


			const int equations_count = (int)(images_match_graph_pair_list.size() + theta_constraints.size()) * DIMENSION_2D;
			SparseMatrix<double> A(equations_count, images_data.size() * DIMENSION_2D);
			VectorXd b = VectorXd::Zero(equations_count);
			vector<Triplet<double> > triplets;
			triplets.reserve(theta_constraints.size() * 2 + images_match_graph_pair_list.size() * 6);

			int equation = 0;
			for (int i = 0; i < theta_constraints.size(); ++i) {
				triplets.emplace_back(equation, DIMENSION_2D * theta_constraints[i].first, STRONG_CONSTRAINT);
				triplets.emplace_back(equation + 1, DIMENSION_2D * theta_constraints[i].first + 1, STRONG_CONSTRAINT);
				b[equation] = STRONG_CONSTRAINT * cos(theta_constraints[i].second);
				b[equation + 1] = STRONG_CONSTRAINT * sin(theta_constraints[i].second);
				equation += DIMENSION_2D;
			}
			for (int i = 0; i < images_match_graph_pair_list.size(); ++i) {
				const pair<int, int>& match_pair = images_match_graph_pair_list[i];
				const int& m1 = match_pair.first, & m2 = match_pair.second;
				const FLOAT_TYPE& MLDR_theta = getImagesMinimumLineDistortionRotation(m1, m2);
				triplets.emplace_back(equation, DIMENSION_2D * m1, cos(MLDR_theta));
				triplets.emplace_back(equation, DIMENSION_2D * m1 + 1, -sin(MLDR_theta));
				triplets.emplace_back(equation, DIMENSION_2D * m2, -1);
				triplets.emplace_back(equation + 1, DIMENSION_2D * m1, sin(MLDR_theta));
				triplets.emplace_back(equation + 1, DIMENSION_2D * m1 + 1, cos(MLDR_theta));
				triplets.emplace_back(equation + 1, DIMENSION_2D * m2 + 1, -1);
				equation += DIMENSION_2D;
			}
			assert(equation == equations_count);
			A.setFromTriplets(triplets.begin(), triplets.end());
			LeastSquaresConjugateGradient<SparseMatrix<double> > lscg(A);

			VectorXd x = lscg.solve(b);

			for (int i = 0; i < images_data.size(); ++i) {
				result[i].theta = atan2(x[DIMENSION_2D * i + 1], x[DIMENSION_2D * i]);
			}
		}
		break;
		case GLOBAL_ROTATION_3D_METHOD:
		{
			const int equations_count = (int)images_match_graph_pair_list.size() * DIMENSION_2D + DIMENSION_2D;
			SparseMatrix<double> A(equations_count, images_data.size() * DIMENSION_2D);
			VectorXd b = VectorXd::Zero(equations_count);
			vector<Triplet<double> > triplets;
			triplets.reserve(images_match_graph_pair_list.size() * 6 + DIMENSION_2D);

			b[0] = STRONG_CONSTRAINT * cos(result[parameter.center_image_index].theta);
			b[1] = STRONG_CONSTRAINT * sin(result[parameter.center_image_index].theta);
			triplets.emplace_back(0, DIMENSION_2D * parameter.center_image_index, STRONG_CONSTRAINT);
			triplets.emplace_back(1, DIMENSION_2D * parameter.center_image_index + 1, STRONG_CONSTRAINT);
			int equation = DIMENSION_2D;
			for (int i = 0; i < images_match_graph_pair_list.size(); ++i) {
				const pair<int, int>& match_pair = images_match_graph_pair_list[i];
				const int& m1 = match_pair.first, & m2 = match_pair.second;
				const double guess_theta = result[m2].theta - result[m1].theta;
				FLOAT_TYPE decision_theta, weight;
				if (isRotationInTheRange(guess_theta,
					images_relative_rotation_range[m1][m2].first,
					images_relative_rotation_range[m1][m2].second)) {
					decision_theta = guess_theta;
					weight = LAMBDA_GAMMA;
				}
				else {
					decision_theta = getImagesMinimumLineDistortionRotation(m1, m2);
					weight = 1;
				}
				triplets.emplace_back(equation, DIMENSION_2D * m1, weight * cos(decision_theta));
				triplets.emplace_back(equation, DIMENSION_2D * m1 + 1, weight * -sin(decision_theta));
				triplets.emplace_back(equation, DIMENSION_2D * m2, -weight);
				triplets.emplace_back(equation + 1, DIMENSION_2D * m1, weight * sin(decision_theta));
				triplets.emplace_back(equation + 1, DIMENSION_2D * m1 + 1, weight * cos(decision_theta));
				triplets.emplace_back(equation + 1, DIMENSION_2D * m2 + 1, -weight);

				equation += DIMENSION_2D;
			}
			assert(equation == equations_count);
			A.setFromTriplets(triplets.begin(), triplets.end());
			LeastSquaresConjugateGradient<SparseMatrix<double> > lscg(A);
			VectorXd x = lscg.solve(b);

			for (int i = 0; i < images_data.size(); ++i) {
				result[i].theta = atan2(x[DIMENSION_2D * i + 1], x[DIMENSION_2D * i]);
			}
		}
		break;
		default:
			printError("F(getImagesSimilarityElements) NISwGSP_ROTATION_METHOD");
			break;
		}
	}
	return result;
}

/// <summary>
/// 算出图像对的旋转角度范围.范围就是根据对应的网格边的角度,找出最小旋转角度,和最大旋转角度 为上下限
/// </summary>
/// <returns></returns>
const vector<vector<pair<double, double> > >& MultiImages::getImagesRelativeRotationRange() const {
	if (images_relative_rotation_range.empty()) {
		images_relative_rotation_range.resize(images_data.size());
		for (int i = 0; i < images_relative_rotation_range.size(); ++i) {
			images_relative_rotation_range[i].resize(images_relative_rotation_range.size(), make_pair(0, 0));
		}
		//图片匹配对
		const vector<pair<int, int> >& images_match_graph_pair_list = parameter.getImagesMatchGraphPairList();
		//重叠区域mask
		const vector<vector<vector<bool> > >& apap_overlap_mask = getAPAPOverlapMask();
		//存放变形后的数据:apap_matching_points[变形图像][对应图像][变形后的网格点] == 变形后的每个网格点的坐标.
		const vector<vector<vector<Point2> > >& apap_matching_points = getAPAPMatchingPoints();

		for (int i = 0; i < images_match_graph_pair_list.size(); ++i) {
			const pair<int, int>& match_pair = images_match_graph_pair_list[i];
			const int& m1 = match_pair.first, & m2 = match_pair.second;
			//变形图像的边
			const vector<Edge>& m1_edges = images_data[m1].mesh_2d->getEdges();
			//对应图像的边
			const vector<Edge>& m2_edges = images_data[m2].mesh_2d->getEdges();

			const vector<const vector<Edge>*>& edges = { &m1_edges, &m2_edges };

			const vector<pair<int, int> > pair_index = { make_pair(m1, m2), make_pair(m2, m1) };
			//将:m1图像的原网格点 与 m1对应m2扭曲后的网格点 凑成一对;  m2图像的原网格点 与 m2对应m1扭曲后的网格点 凑成一对.  最后再凑成一个两个元素的列表.
			const vector<pair<const vector<Point2>*, const vector<Point2>*> >& vertices_pair = {
				make_pair(&images_data[m1].mesh_2d->getVertices(), &apap_matching_points[m1][m2]),
				make_pair(&images_data[m2].mesh_2d->getVertices(), &apap_matching_points[m2][m1])
			};
			vector<double> positive, negative;
			const vector<bool> sign_mapping = { false, true, true, false };
			for (int j = 0; j < edges.size(); ++j) {
				for (int k = 0; k < edges[j]->size(); ++k) {
					const Edge& e = (*edges[j])[k];
					if (apap_overlap_mask[pair_index[j].first][pair_index[j].second][e.indices[0]] &&
						apap_overlap_mask[pair_index[j].first][pair_index[j].second][e.indices[1]]) { //整条边都在重叠区域
						//扭曲图像的自己的(未变形)一条网格边x,y的distance
						const Point2d a = (*vertices_pair[j].first)[e.indices[0]] - (*vertices_pair[j].first)[e.indices[1]];
						//扭曲图像的已变形 的一条网格边x,y的distance
						const Point2d b = (*vertices_pair[j].second)[e.indices[0]] - (*vertices_pair[j].second)[e.indices[1]];
						//求未变现和变形后的 边的旋转角度.
						const double theta = acos(a.dot(b) / (norm(a) * norm(b)));
						//旋转方法
						const double direction = a.x * b.y - a.y * b.x;
						//(0,1)*2 + (0,1)
						int map = ((direction > 0) << 1) + j;
						if (sign_mapping[map]) {
							positive.emplace_back(theta);
						}
						else {
							negative.emplace_back(-theta);
						}
					}
				}
			}
			//排序旋转角度.
			sort(positive.begin(), positive.end());
			sort(negative.begin(), negative.end());

			//根据得到的每个边的旋转角度,算出图像对最佳的旋转角度范围
			if (positive.empty() == false && negative.empty() == false) {//正旋转和负旋转都有
				if (positive.back() - negative.front() < M_PI) {
					images_relative_rotation_range[m1][m2].first = negative.front() + 2 * M_PI;
					images_relative_rotation_range[m1][m2].second = positive.back() + 2 * M_PI;
					images_relative_rotation_range[m2][m1].first = 2 * M_PI - positive.back();
					images_relative_rotation_range[m2][m1].second = 2 * M_PI - negative.front();
				}
				else {
					images_relative_rotation_range[m1][m2].first = positive.front();
					images_relative_rotation_range[m1][m2].second = negative.back() + 2 * M_PI;
					images_relative_rotation_range[m2][m1].first = -negative.back();
					images_relative_rotation_range[m2][m1].second = 2 * M_PI - positive.front();

				}
			}
			else if (positive.empty() == false) {
				images_relative_rotation_range[m1][m2].first = positive.front();
				images_relative_rotation_range[m1][m2].second = positive.back();
				images_relative_rotation_range[m2][m1].first = 2 * M_PI - positive.back();
				images_relative_rotation_range[m2][m1].second = 2 * M_PI - positive.front();
			}
			else {
				images_relative_rotation_range[m1][m2].first = negative.front() + 2 * M_PI;
				images_relative_rotation_range[m1][m2].second = negative.back() + 2 * M_PI;
				images_relative_rotation_range[m2][m1].first = -negative.back();
				images_relative_rotation_range[m2][m1].second = -negative.front();
			}
		}
	}
	//images_relative_rotation_range[m1][m2]:图1对图2,图1的旋转角度范围;images_relative_rotation_range[m2][m1]:图2对图1,图2的旋转角度范围.
	//<image url="$(ProjectDir)CommentImage\MultiImage-getImageRelativeRotationRange.png" scale="1"/>
	return images_relative_rotation_range;
}


/// <summary>
/// 用LSD,通过apap,得到图像对最小直线扭曲旋转角度(MLDR),和scale
/// </summary>
/// <param name="_from"></param>
/// <param name="_to"></param>
/// <returns></returns>
FLOAT_TYPE MultiImages::getImagesMinimumLineDistortionRotation(const int _from, const int _to) const {
	if (images_minimum_line_distortion_rotation.empty()) {
		images_minimum_line_distortion_rotation.resize(images_data.size());
		for (int i = 0; i < images_minimum_line_distortion_rotation.size(); ++i) {
			images_minimum_line_distortion_rotation[i].resize(images_data.size(), FLT_MAX);
		}
	}
	if (images_minimum_line_distortion_rotation[_from][_to] == FLT_MAX) {
		//LSD获取from 图像的直线.
		const vector<LineData>& from_lines = images_data[_from].getLines();
		//LSD获取to 图像的直线.
		const vector<LineData>& to_lines = images_data[_to].getLines();

		//_from变形为_to,_from中直线变形后的线数据
		const vector<Point2>& from_project = getImagesLinesProject(_from, _to);
		//_to变形为_from,_to中直线变形后的线数据
		const vector<Point2>& to_project = getImagesLinesProject(_to, _from);

		//两张图像变形前的直线数据
		const vector<const vector<LineData>*>& lines = { &from_lines,   &to_lines };
		//两张图像对应变形后的直线数据
		const vector<const vector<Point2  >*>& projects = { &from_project, &to_project };
		const vector<int>& img_indices = { _to, _from };
		const vector<int> sign_mapping = { -1, 1, 1, -1 };

		//根据两图像直线长,宽,密度得到每对直线(变前变后) 得到图像旋转角度 对整体图像旋转角度的影响权重;
		vector<pair<double, double> > theta_weight_pairs;
		for (int i = 0; i < lines.size(); ++i) {
			const int& rows = images_data[img_indices[i]].img.rows;
			const int& cols = images_data[img_indices[i]].img.cols;
			const vector<pair<Point2, Point2> >& boundary_edgs = {
				make_pair(Point2(0,       0), Point2(cols,    0)),
				make_pair(Point2(cols,    0), Point2(cols, rows)),
				make_pair(Point2(cols, rows), Point2(0, rows)),
				make_pair(Point2(0, rows), Point2(0,    0))
			};
			for (int j = 0; j < lines[i]->size(); ++j) {
				//得到变形后直线中的一个点坐标
				const Point2& p1 = (*projects[i])[EDGE_VERTEX_SIZE * j];
				//得到变形后直线中的下一个点坐标
				const Point2& p2 = (*projects[i])[EDGE_VERTEX_SIZE * j + 1];
				const bool p1_in_img = (p1.x >= 0 && p1.x <= cols && p1.y >= 0 && p1.y <= rows);
				const bool p2_in_img = (p2.x >= 0 && p2.x <= cols && p2.y >= 0 && p2.y <= rows);

				const bool p_in_img[EDGE_VERTEX_SIZE] = { p1_in_img, p2_in_img };

				Point2 p[EDGE_VERTEX_SIZE] = { p1, p2 };

				if (!p1_in_img || !p2_in_img) {
					vector<double> scales;
					for (int k = 0; k < boundary_edgs.size(); ++k) {
						double s1;
						if (isEdgeIntersection(p1, p2, boundary_edgs[k].first, boundary_edgs[k].second, &s1)) {
							scales.emplace_back(s1);
						}
					}
					assert(scales.size() <= EDGE_VERTEX_SIZE);
					if (scales.size() == EDGE_VERTEX_SIZE) {
						assert(!p1_in_img && !p2_in_img);
						if (scales.front() > scales.back()) {
							iter_swap(scales.begin(), scales.begin() + 1);
						}
						for (int k = 0; k < scales.size(); ++k) {
							p[k] = p1 + scales[k] * (p2 - p1);
						}
					}
					else if (!scales.empty()) {
						for (int k = 0; k < EDGE_VERTEX_SIZE; ++k) {
							if (!p_in_img[k]) {
								p[k] = p1 + scales.front() * (p2 - p1);
							}
						}
					}
					else {
						continue;
					}
				}
				//变形前第j条直线的斜率
				const Point2d a = (*lines[i])[j].data[1] - (*lines[i])[j].data[0];
				//变形后第j条直线的斜率
				const Point2d b = p2 - p1;
				//计算旋转角度
				const double theta = acos(a.dot(b) / (norm(a) * norm(b)));
				const double direction = a.x * b.y - a.y * b.x;
				const int map = ((direction > 0) << 1) + i;
				const double b_length_2 = sqrt(b.x * b.x + b.y * b.y);
				//根据线长,宽,点密度 计算权值.
				theta_weight_pairs.emplace_back(theta * sign_mapping[map],
					(*lines[i])[j].length * (*lines[i])[j].width * b_length_2);
			}
		}
		Point2 dir(0, 0);
		for (int i = 0; i < theta_weight_pairs.size(); ++i) {
			const double& theta = theta_weight_pairs[i].first;
			dir += (theta_weight_pairs[i].second * Point2(cos(theta), sin(theta)));
		}
		//根据每对直线旋转角度,得到图像对最小旋转角度.
		images_minimum_line_distortion_rotation[_from][_to] = acos(dir.x / (norm(dir))) * (dir.y > 0 ? 1 : -1);
		images_minimum_line_distortion_rotation[_to][_from] = -images_minimum_line_distortion_rotation[_from][_to];
	}
	return images_minimum_line_distortion_rotation[_from][_to];
}



/// <summary>
/// 将_from图像中的直线,经apap投影为 对应_to 的扭曲线.
/// </summary>
/// <param name="_from"></param>
/// <param name="_to"></param>
/// <returns></returns>
const vector<Point2>& MultiImages::getImagesLinesProject(const int _from, const int _to) const {
	if (images_lines_projects.empty()) {
		images_lines_projects.resize(images_data.size());
		for (int i = 0; i < images_lines_projects.size(); ++i) {
			images_lines_projects[i].resize(images_data.size());
		}
	}
	if (images_lines_projects[_from][_to].empty()) {
		const vector<vector<vector<Point2> > >& feature_matches = getFeatureMatches();
		const vector<LineData>& lines = images_data[_from].getLines();
		vector<Point2> points, project_points;
		points.reserve(lines.size() * EDGE_VERTEX_SIZE);
		for (int i = 0; i < lines.size(); ++i) {
			for (int j = 0; j < EDGE_VERTEX_SIZE; ++j) {
				points.emplace_back(lines[i].data[j]);
			}
		}
		vector<Mat> not_be_used;
		//将from的直线位置,通过apap得到 扭曲对齐后的 直线位置.
		APAP_Stitching::apap_project(feature_matches[_from][_to], feature_matches[_to][_from], points, images_lines_projects[_from][_to], not_be_used);
	}
	return images_lines_projects[_from][_to];
}

/// <summary>
/// 得到所有的图像内容.
/// </summary>
/// <returns></returns>
const vector<Mat>& MultiImages::getImages() const {
	if (images.empty()) {
		images.reserve(images_data.size());
		for (int i = 0; i < images_data.size(); ++i) {
			images.emplace_back(images_data[i].img);
		}
	}
	return images;
}

class dijkstraNode {
public:
	int from, pos;
	double dis;
	dijkstraNode(const int& _from,
		const int& _pos,
		const double& _dis) : from(_from), pos(_pos), dis(_dis) {
	}
	bool operator < (const dijkstraNode& rhs) const {
		return dis > rhs.dis;
	}
};

/// <summary>
/// 通过以上dijkstra算法,得到:重叠区域对应的网格的权重为0,非重叠区域的网格的权重为距离重叠区域最近距离.
//<image url="$(ProjectDir)CommentImage\MultiImage-getImagesGridSpaceMatchingPointsWeight.jpg" scale="0.1"/>
/// </summary>
/// <param name="_global_weight_gamma"></param>
/// <returns></returns>
const vector<vector<double> >& MultiImages::getImagesGridSpaceMatchingPointsWeight(const double _global_weight_gamma) const {
	if (_global_weight_gamma && images_polygon_space_matching_pts_weight.empty()) {

		images_polygon_space_matching_pts_weight.resize(images_data.size());

		//得到 每个图像自己的网格点 哪些是 要扭曲(index).(就是图像自己哪些网格点 是 处于重叠部分.重叠部分的点是true)
		const vector<vector<bool > >& images_features_mask = getImagesFeaturesMaskByMatchingPoints();

		//得到 每张图像的 每个点(自己的 和 别图扭过来的)处于第几个网格点 和 该网格点4个点的权重.
		const vector<vector<InterpolateVertex> >& mesh_interpolate_vertex_of_matching_pts = getInterpolateVerticesOfMatchingPoints();

		//
		for (int i = 0; i < images_polygon_space_matching_pts_weight.size(); ++i) {//图像个数,遍历每个图像
			//一个图像的网格数量
			const int polygons_count = (int)images_data[i].mesh_2d->getPolygonsIndices().size();

			//给每个网格设置mask,默认都false
			vector<bool> polygons_has_matching_pts(polygons_count, false);

			//将属于重叠区域的网格mask设置为true
			for (int j = 0; j < images_features_mask[i].size(); ++j) {
				//图像i的 重叠区域的点
				if (images_features_mask[i][j]) {
					//将图像i 中属于重叠区域的网格mask设置为true
					polygons_has_matching_pts[mesh_interpolate_vertex_of_matching_pts[i][j].polygon] = true;
				}
			}
			//每个网格的权重
			images_polygon_space_matching_pts_weight[i].reserve(polygons_count);

			priority_queue<dijkstraNode> que;//优先级最高的node优先出队.top():访问队头元素. pop()弹出队头元素. 

			//遍历每个网格的mask,将遍历过的网格mask都设置为false.
			//如果网格mask为true,则将其权重设为0,如果网格mask为false,则将其权重设为MAX.
			//如果网格mask为true,new dijkstraNode(j,j,0) 点加入que;
			for (int j = 0; j < polygons_has_matching_pts.size(); ++j) {
				if (polygons_has_matching_pts[j]) {
					polygons_has_matching_pts[j] = false;
					images_polygon_space_matching_pts_weight[i].emplace_back(0);
					que.push(dijkstraNode(j, j, 0));
				}
				else {
					images_polygon_space_matching_pts_weight[i].emplace_back(FLT_MAX);
				}
			}
			//经上面的for,所有的mask都变成的false; 但重叠区域的权值为0,且加入队列;非重叠区域的权值为MAX,不加入队列.;

			//以下的while为迪杰斯特拉求解 图像所有网格 距离 重叠区域最近的距离.
			//得到每个网格点的左上角点的 4邻居点的index 列表.
			const vector<Indices>& polygons_neighbors = images_data[i].mesh_2d->getPolygonsNeighbors();
			//得到每个网格点的中心点的坐标.
			const vector<Point2>& polygons_center = images_data[i].mesh_2d->getPolygonsCenter();
			while (que.empty() == false) {//队列不为空,就一直循环.
				//队头的元素.
				const dijkstraNode now = que.top();
				//pos网格 属于 第几个网格
				const int index = now.pos;
				que.pop();
				//第一遍,将所有的重叠区域的网格进行....
				if (polygons_has_matching_pts[index] == false) {
					//将处理过得网格mask设置为true.
					polygons_has_matching_pts[index] = true;
					//将此网格的....
					for (int j = 0; j < polygons_neighbors[index].indices.size(); ++j) {//循环得到此网格左上角顶点的邻居点index
						//得到此网格左上角顶点的一个邻居点的index
						const int n = polygons_neighbors[index].indices[j];
						if (polygons_has_matching_pts[n] == false) { //判断此网格的 一个邻居网格mask是否为false(即这个网格没有被处理过)
							const double dis = norm(polygons_center[n] - polygons_center[now.from]);
							if (images_polygon_space_matching_pts_weight[i][n] > dis) {
								images_polygon_space_matching_pts_weight[i][n] = dis;
								que.push(dijkstraNode(now.from, n, dis));
							}
						}
					}
				}
			}

			//将得到的权重数据进行归一化. 除数为图像的斜对角距离.
			const double normalize_inv = 1. / norm(Point2i(images_data[i].img.cols, images_data[i].img.rows));
			for (int j = 0; j < images_polygon_space_matching_pts_weight[i].size(); ++j) {
				images_polygon_space_matching_pts_weight[i][j] = images_polygon_space_matching_pts_weight[i][j] * normalize_inv;
			}
		}
	}
	return images_polygon_space_matching_pts_weight;
}

/// <summary>
/// 初始化3维数据
/// </summary>
void MultiImages::initialFeaturePairsSpace() const {
	feature_pairs.resize(images_data.size());
	for (int i = 0; i < images_data.size(); ++i) {
		feature_pairs[i].resize(images_data.size());
	}
}

//#初始化ransacdiff 数据
void MultiImages::initialRansacDiffPairs() const {
	ransacDiff.resize(images_data.size());
	for (int i = 0; i < images_data.size(); ++i) {
		ransacDiff[i].resize(images_data.size());
	}
	ransacAvgDiff.resize(images_data.size());
	for (int i = 0; i < images_data.size(); ++i) {
		ransacAvgDiff[i].resize(images_data.size());
	}
	ransacDiffWeight.resize(images_data.size());
	for (int i = 0; i < images_data.size(); ++i) {
		ransacDiffWeight[i].resize(images_data.size());
	}
}


/// <summary>
/// 循环将 匹配的图像 进行ransac特征点匹配,得到正确的特征点匹配对index 列表,最后加入3维数组.
/// </summary>
/// <returns>返回三维数组:即(图像一 X 图像二 X 正确特征点匹配对index列表)</returns>;
const vector<vector<vector<pair<int, int> > > >& MultiImages::getFeaturePairs() const {
	if (feature_pairs.empty()) {
		//3维数组,先初始化前二维的大小.
		initialFeaturePairsSpace();
		//获取到两两配对的图像的 index对.
		const vector<pair<int, int> >& images_match_graph_pair_list = parameter.getImagesMatchGraphPairList();
		for (int i = 0; i < images_match_graph_pair_list.size(); ++i) {
			//拿到其中的一对图像
			const pair<int, int>& match_pair = images_match_graph_pair_list[i];
			//得到的差距值最小或最小+次小的特征点描述匹配对 index
			const vector<pair<int, int>>& initial_indices = getInitialFeaturePairs(match_pair);
			//存放第一个图像的特征点
			const vector<Point2>& m1_fpts = images_data[match_pair.first].getFeaturePoints();
			//存放另一个图像的特征点
			const vector<Point2>& m2_fpts = images_data[match_pair.second].getFeaturePoints();
			//X存放 特征点index, Y存放对应X的 特征点index.
			vector<Point2> X, Y;
			X.reserve(initial_indices.size());
			Y.reserve(initial_indices.size());
			for (int j = 0; j < initial_indices.size(); ++j) {
				const pair<int, int> it = initial_indices[j];
				X.emplace_back(m1_fpts[it.first]);
				Y.emplace_back(m2_fpts[it.second]);
			}
			//result 为 对应两图像对 对应的vector<pair<int,int>>;用来存放:真正匹配的点对的index.(引用的方式也可以赋值.)
			vector<pair<int, int> >& result = feature_pairs[match_pair.first][match_pair.second];
			//获取到真正匹配的点对index
			result = getFeaturePairsBySequentialRANSAC(match_pair, X, Y, initial_indices);

			assert(result.empty() == false);
#ifndef DP_NO_LOG
			//将图像和匹配对画圈线,保存在本地.
			//writeImageOfFeaturePairs("sRANSAC", match_pair, result);
#endif
		}
		//#将得到的图像对的ransac误差均值进行数据处理，得到权值数据。
		generateRansacDiffWeight(images_match_graph_pair_list);
	}
	return feature_pairs;
}

/// <summary>
/// 得到三维数组,记录了两两图像之间对应的正确的特征点对.
/// </summary>
/// <returns>
///        0      1      2      3     
/// 0          p1(...)
/// 1	p2(..)	      p1(...)
/// 2			p2(..)		 p1(...)
/// 3				   p2(..)
/// 
/// p1(...)表示第一张图像的特征点们; p2(...)表示第二张图像的特征点们; 
/// 
/// 即:feature_matches[i][j]:指向的是第i张图像对应第j张图像 i上的特征点;
/// feature_matches[j][i]:指向的是第i张图像对应第j张图像 j上的特征点;
/// </returns>
const vector<vector<vector<Point2> > >& MultiImages::getFeatureMatches() const {
	if (feature_matches.empty()) {
		//得到 三维数组:即(图像一 对 图像二 的 正确特征点匹配对index列表) 注:1对2则[1][2]有数据,[2][1],[1][1],[2][2]等没数据.
		const vector<vector<vector<pair<int, int> > > >& feature_pairs = getFeaturePairs();
		//获取图像对应对 index
		const vector<pair<int, int> >& images_match_graph_pair_list = parameter.getImagesMatchGraphPairList();
		feature_matches.resize(images_data.size());
		for (int i = 0; i < images_data.size(); ++i) {
			feature_matches[i].resize(images_data.size());
		}
		//接下来这么一堆,就是一个将上面的v<v<v<pair<int,int>>>> 转化为 v<v<v<Point2>>>;
		for (int i = 0; i < images_match_graph_pair_list.size(); ++i) {
			const pair<int, int>& match_pair = images_match_graph_pair_list[i];
			const int& m1 = match_pair.first, & m2 = match_pair.second;
			feature_matches[m1][m2].reserve(feature_pairs[m1][m2].size());
			feature_matches[m2][m1].reserve(feature_pairs[m1][m2].size());
			const vector<Point2>& m1_fpts = images_data[m1].getFeaturePoints();
			const vector<Point2>& m2_fpts = images_data[m2].getFeaturePoints();
			for (int j = 0; j < feature_pairs[m1][m2].size(); ++j) {
				feature_matches[m1][m2].emplace_back(m1_fpts[feature_pairs[m1][m2][j].first]);
				feature_matches[m2][m1].emplace_back(m2_fpts[feature_pairs[m1][m2][j].second]);
			}
		}
	}
	return feature_matches;
}

/// <summary>
/// 利用Ransac,多次循环匹配内点对,最后得到内点匹配点对的index(真正匹配的点对的index);
/// </summary>
/// <param name="_match_pair"></param>
/// <param name="_X">匹配好的 图像一 的特征点坐标</param>
/// <param name="_Y">匹配好的 图像二 的特征点坐标.与X中一一对应</param>
/// <param name="_initial_indices">初始的匹配点对.有可能匹配对不是真的匹配点(外点)</param>
/// <returns></returns>
vector<pair<int, int> > MultiImages::getFeaturePairsBySequentialRANSAC(const pair<int, int>& _match_pair,
	const vector<Point2>& _X,
	const vector<Point2>& _Y,
	const vector<pair<int, int> >& _initial_indices) const {
	//#初始化ransacdiff数据的pair对3维表
	initialRansacDiffPairs();

	const int HOMOGRAPHY_MODEL_MIN_POINTS = 4;
	//尝试次数 : log(a)/log(b) 是 log以b为底,a的对数.   log(a)是以e为底a的对数. 2064次
	const int GLOBAL_MAX_ITERATION = log(1 - OPENCV_DEFAULT_CONFIDENCE) / log(1 - pow(GLOBAL_TRUE_PROBABILITY, HOMOGRAPHY_MODEL_MIN_POINTS));
	//final_mask的index是与initial_indices是对应的;
	vector<char> final_mask(_initial_indices.size(), 0);

	//#得到第一次ransac的单应性矩阵
	Mat H(3, 3, CV_64F);
	//global_homography_max_inliers_dist:如果文件预设没有,默认为5
	H = findHomography(_X, _Y, RANSAC, parameter.global_homography_max_inliers_dist, final_mask, GLOBAL_MAX_ITERATION);
	updataRansacDiff(_match_pair, _X, _Y, final_mask, H);

#ifndef DP_NO_LOG
	//drawRansac(_match_pair.first, _match_pair.second, _initial_indices, final_mask);
#endif

	vector<Point2> tmp_X = _X, tmp_Y = _Y;

	//mask_indices就是用来控制下次循环的点的index;下次循环前,列表前依次存放的是上次得到的外点index;
	vector<int> mask_indices(_initial_indices.size(), 0);
	//mask_indices为[0,1,2,3,4,5,......]
	for (int i = 0; i < mask_indices.size(); ++i) {
		mask_indices[i] = i;
	}


	//对tmp_X 和tmp_Y中的 points 进行配对.
	//主要流程:第一遍Ransac得到一批内点.然后将外点再进行一次Ransac,循环往复,最后直到匹配的内点数量不足,则停止循环.
	//最后会得到final_mask:存放内点外点的标志mask;
	while (tmp_X.size() >= HOMOGRAPHY_MODEL_MIN_POINTS &&
		parameter.local_homogrpahy_max_inliers_dist < parameter.global_homography_max_inliers_dist) {

		const int LOCAL_MAX_ITERATION = log(1 - OPENCV_DEFAULT_CONFIDENCE) / log(1 - pow(LOCAL_TRUE_PROBABILITY, HOMOGRAPHY_MODEL_MIN_POINTS));
		vector<Point2> next_X, next_Y;
		//mask 存放 对应index inlier还是outlier
		vector<char> mask(tmp_X.size(), 0);

		//ptsetreg.cpp line:162
		//fundam.cpp line:352
		H = findHomography(tmp_X, tmp_Y, RANSAC, parameter.local_homogrpahy_max_inliers_dist, mask, LOCAL_MAX_ITERATION);
		updataRansacDiff(_match_pair, tmp_X, tmp_Y, mask, H);

		//执行findHomography()后.掩图来指定inlier和outlier。
		int inliers_count = 0;
		for (int i = 0; i < mask.size(); ++i) {
			//true为inlier
			if (mask[i]) { ++inliers_count; }
		}
		//如果匹配内点数量小于最小阈值(文件读入,如果没有默认40),那么就不再进行下一次外点配对循环.
		if (inliers_count < parameter.local_homography_min_features_count) {
			break;
		}
		//匹配内点数量符合要求了.
		//
		for (int i = 0, shift = -1; i < mask.size(); ++i) {
			//mask对应的i处,为内点还是外点?
			//i处mask为内点:
			if (mask[i]) {
				//对应的final_mask[n]处设置为1;n为多少呢:对应点在initial_indices(最初点列表)的index.
				final_mask[mask_indices[i]] = 1;
			}
			//i处mask为外点:
			else {
				//把外点的point分别加入next_X,next_Y两个列表中
				next_X.emplace_back(tmp_X[i]);
				next_Y.emplace_back(tmp_Y[i]);
				//将外点的index依次记录在mask_indices列表前面;
				mask_indices[++shift] = mask_indices[i];

			}
		}

		//最后mask_indices假如为: [2,5,2,3,4,5,6,.....]其中2,5就是外点.shift是不会超越i的.
#ifndef DP_NO_LOG
		//drawRansac(_match_pair.first, _match_pair.second, _initial_indices, final_mask);
		cout << "Local true Probabiltiy = " << next_X.size() / (float)tmp_X.size() << endl;
#endif
		//将外点Point列表数据 赋值给tmp_X,tmp_Y,while循环对所有的外点再进行一次匹配.
		tmp_X = next_X;
		tmp_Y = next_Y;
	}

	//最后将所有的内点对index 加入result列表
	vector<pair<int, int> > result;
	for (int i = 0; i < final_mask.size(); ++i) {
		if (final_mask[i]) {
			result.emplace_back(_initial_indices[i]);
		}
	}
	double ransacDst = generateRansacAvgDiff(_match_pair);
#ifndef DP_NO_LOG
	//drawRansac(_match_pair.first, _match_pair.second, _initial_indices, final_mask);
	cout << "Global true Probabiltiy = " << result.size() / (float)_initial_indices.size() << endl;
	//cout << "DUP!!!!ransac diff " << _match_pair.first << " + " << _match_pair.second << " : " << ransacDst;

#endif
	return result;
}

//#计算出某一对图像之间的每一对内点特征点对之间的差距
void MultiImages::updataRansacDiff(const pair<int, int>& _index_pair, const vector<Point2> srcPoints, const vector<Point2> dstPoints, const vector<char> final_mask, const Mat H) const {
	vector<double>* diffList = &ransacDiff[_index_pair.first][_index_pair.second];
	ofstream mycout(txtName + to_string(_index_pair.first) + "_" + to_string(_index_pair.second) + ".txt", ios::app);
	mycout << "****************" << endl;
	for (int i = 0; i < final_mask.size(); i++) {
		char isIn = final_mask[i];
		Point2 pointSrc, pointTransDst, pointDst;
		if (isIn == 1) {
			pointSrc = srcPoints[i];
			pointDst = dstPoints[i];
			pointTransDst = applyTransform3x3(pointSrc.x, pointSrc.y, H);
			Point2 d = pointTransDst - pointDst;
			double dst = sqrt(d.x * d.x + d.y * d.y);
			diffList->emplace_back(dst);
			mycout << dst << "," << 1 << endl;
		}
		else {
			pointSrc = srcPoints[i];
			pointDst = dstPoints[i];
			pointTransDst = applyTransform3x3(pointSrc.x, pointSrc.y, H);
			Point2 d = pointTransDst - pointDst;
			double dst = sqrt(d.x * d.x + d.y * d.y);
			mycout << dst << "," << 0 << endl;
		}

	}
	mycout.close();
}

//#获取某一对图像之间的ransac差距
const double MultiImages::generateRansacAvgDiff(const pair<int, int>& _index_pair) const
{
	vector<double> diffList = ransacDiff[_index_pair.first][_index_pair.second];
	double sum = 0, avg;
	for (int i = 0; i < diffList.size(); i++) {
		sum += diffList[i];
	}
	avg = sum / diffList.size();
	ransacAvgDiff[_index_pair.first][_index_pair.second] = avg;

#ifndef DP_NO_LOG
	ofstream mycout(txtName + to_string(_index_pair.first) + "_" + to_string(_index_pair.second) + ".txt", ios::app);
	mycout << "****************" << endl;
	mycout << avg << endl;
	mycout.close();
#endif
	return avg;
}

//#将得到的图像对的ransac误差均值进行数据处理，得到权值数据。
//步骤为先归一化使用exp(-diff*diff),控制数据在0-1，并且实现diff越大权值越小。然后求数据平均值，将新数据= 数据+1-平均值，保证权值数据均值为1.
const void MultiImages::generateRansacDiffWeight(vector<pair<int, int> > pairList) const
{
	vector<double> avgDiffs;
	double sumWeight = 0;
	for (int i = 0; i < pairList.size(); ++i) {
		//拿到其中的一对图像
		const pair<int, int>& match_pair = pairList[i];
		avgDiffs.emplace_back(ransacAvgDiff[match_pair.first][match_pair.second]);

#ifndef DP_NO_LOG
		ofstream mycoutAvg(txtName + "ransacAvg.txt", ios::app);
		mycoutAvg << match_pair.first << "--" << match_pair.second << "," << ransacAvgDiff[match_pair.first][match_pair.second] << endl;
		mycoutAvg.close();
#endif

		double weightTemp = exp(-ransacAvgDiff[match_pair.first][match_pair.second] * ransacAvgDiff[match_pair.first][match_pair.second]);
		sumWeight += weightTemp;
		ransacDiffWeight[match_pair.first][match_pair.second] = weightTemp;

#ifndef DP_NO_LOG
		ofstream mycoutWeightTemp(txtName + "weightTemp.txt", ios::app);
		mycoutWeightTemp << match_pair.first << "--" << match_pair.second << "," << weightTemp << endl;
		mycoutWeightTemp.close();
#endif
	}
	double avgWeight = sumWeight / pairList.size();
	for (int i = 0; i < pairList.size(); ++i) {
		//拿到其中的一对图像
		const pair<int, int>& match_pair = pairList[i];
		ransacDiffWeight[match_pair.first][match_pair.second] = ransacDiffWeight[match_pair.first][match_pair.second] + 1 - avgWeight;
		/*if (ransacDiffWeight[match_pair.first][match_pair.second] >= 1) {
			ransacDiffWeight[match_pair.first][match_pair.second] = (ransacDiffWeight[match_pair.first][match_pair.second] - 1) * (0.9 / 0.5) + 1;
		}
		else {
			ransacDiffWeight[match_pair.first][match_pair.second] = 1 - (1 - ransacDiffWeight[match_pair.first][match_pair.second]) * (0.9 / 0.5);
		}*/

		//ransacDiffWeight[match_pair.first][match_pair.second] = ransacDiffWeight[match_pair.first][match_pair.second] / avgWeight;
#ifndef DP_NO_LOG

		ofstream mycout(txtName + "weightFinal.txt", ios::app);
		mycout << match_pair.first << "--" << match_pair.second << "," << ransacDiffWeight[match_pair.first][match_pair.second] << endl;
		mycout.close();
#endif
	}

	return void();
}

//#返回权重数据。数据保证误差越大，权重越小，且数据集中在0.5-1.5。
const double MultiImages::getRansacDiffWeight(const pair<int, int>& _index_pair) const
{
	return ransacDiffWeight[_index_pair.first][_index_pair.second];
}

//获取重叠度先验性
const float MultiImages::getOverlap(pair<int, int> _mask_pair) const {
	//float L_dif = abs(images_data[_mask_pair.first].LatALon.first - images_data[_mask_pair.second].LatALon.first) * 3600.0 * 30.9;
	//float W_dif = cos(images_data[_mask_pair.second].LatALon.first * M_PI / 180.0) * 6370114 * M_PI * abs(images_data[_mask_pair.second].LatALon.second - images_data[_mask_pair.first].LatALon.second) / 180;
	//float GSD = Hight * Pixe / Focal_;

	//double fov_L = images_data[_mask_pair.first].img.cols * GSD;
	//double fov_w = images_data[_mask_pair.first].img.rows * GSD;

	//float lv = abs((fov_L - L_dif)) * abs((fov_L - W_dif)) / (fov_L * fov_w);

	//return lv;
	return 0;
}

bool compareFeaturePair(const FeatureDistance& fd_1, const FeatureDistance& fd_2) {
	return
		(fd_1.feature_index[0] == fd_2.feature_index[0]) ?
		(fd_1.feature_index[1] < fd_2.feature_index[1]) :
		(fd_1.feature_index[0] < fd_2.feature_index[0]);
}

/// <summary>
/// 得到的差距值最小或者 最小+次小的特征点描述匹配对 index
/// </summary>
/// <param name="_match_pair">图像匹配对 index</param>
/// <returns>差距值最小和次小的特征点描述匹配对 index</returns>
vector<pair<int, int> > MultiImages::getInitialFeaturePairs(const pair<int, int>& _match_pair) const {
	const int nearest_size = 2, pair_count = 1;
	const bool ratio_test = true, intersect = true;

	assert(nearest_size > 0);

	//获取第一张图片的特征点(也会求出特征描述)
	const int feature_size_1 = (int)images_data[_match_pair.first].getFeaturePoints().size();
	//获取第二张图片的特征点(也会求出特征描述)
	const int feature_size_2 = (int)images_data[_match_pair.second].getFeaturePoints().size();
	const int PAIR_COUNT = 2;
	//存放两张图片的特征点个数.
	const int feature_size[PAIR_COUNT] = { feature_size_1, feature_size_2 };
	//存放两张图片的index
	const int pair_match[PAIR_COUNT] = { _match_pair.first , _match_pair.second };

	// vector<FeatureDistance> 数组
	vector<FeatureDistance> feature_pairs[PAIR_COUNT];

	for (int p = 0; p < pair_count; ++p) {
		//另一个图像特征点的数量.
		const int another_feature_size = feature_size[1 - p];

		//控制得到最大/最小距离.
		const int nearest_k = min(nearest_size, another_feature_size);
		//获取此图像的每个特征点的描述子
		const vector<FeatureDescriptor>& feature_descriptors_1 = images_data[pair_match[p]].getFeatureDescriptors();
		//获取另一个图像的每个特征点的描述子
		const vector<FeatureDescriptor>& feature_descriptors_2 = images_data[pair_match[!p]].getFeatureDescriptors();

		//此图像的特征点描述子先一个一个 来 与另一个图像的特征点描述子 计算差距.
		for (int f1 = 0; f1 < feature_size[p]; ++f1) {
			//记录此图像 此特征点 与 另一个图像其他特征点的差距.set会自动排序(平衡搜索二叉树).
			//但注意:FeatureDistance重载了大于小于号.
			set<FeatureDistance> feature_distance_set;
			//先存一个差距巨大的临时值.
			feature_distance_set.insert(FeatureDistance(FLT_MAX, p, -1, -1));

			//此图像的特征点描述子再一个一个 和另一个图像的 特征描述子进行计算 差距.
			for (int f2 = 0; f2 < feature_size[!p]; ++f2) {
				//获取这两个特征点描述子的 差距.
				const double dist = FeatureDescriptor::getDistance(feature_descriptors_1[f1], feature_descriptors_2[f2], feature_distance_set.begin()->distance);
				//如果这两个差距 比 之前比较过的 差距还小,那就更新set.
				if (dist < feature_distance_set.begin()->distance) {
					//如果set大小已经到达最大容量,那就把之前的最大的去除
					if (feature_distance_set.size() == nearest_k) {
						feature_distance_set.erase(feature_distance_set.begin());
					}
					//将最小的更新进去.
					feature_distance_set.insert(FeatureDistance(dist, p, f1, f2));
				}
			}
			//最后,feature_distance_set会保留一个最小差距和次小差距.

			//次小的
			set<FeatureDistance>::const_iterator it = feature_distance_set.begin();
			if (ratio_test && feature_distance_set.size() >= 2) {
				//最小的
				const set<FeatureDistance>::const_iterator it2 = std::next(it, 1);
				if (nearest_k == nearest_size &&
					it2->distance * FEATURE_RATIO_TEST_THRESHOLD > it->distance) {
					//最小的差距 x Feature_ration_test_threshold(1.5) 大于次小的差距(意思就是最小和次小差距不大).则不交换it还是次小;
					//否则(意思就是最小和次小差距大)it为最小.
					continue;
				}
				it = it2;
			}
			//根据上面的计算:
			//如果最小次小的差距不大,都加入feature_paires[0]对应的vector<FeatureDistance>中.
			//如果差距大,只加入最小的距离.
			feature_pairs[p].insert(feature_pairs[p].end(), it, feature_distance_set.end());
		}
		//此图像所有的特征点描述子已完全匹配完了,每个特征点的描述子最小或者 最小+次小距离,以及特征点匹配对的index.已经保存在feature_pairs中.
	}
	vector<FeatureDistance> feature_pairs_result;
	if (pair_count == PAIR_COUNT) {
		sort(feature_pairs[0].begin(), feature_pairs[0].end(), compareFeaturePair);
		sort(feature_pairs[1].begin(), feature_pairs[1].end(), compareFeaturePair);
		if (intersect) {
			set_intersection(feature_pairs[0].begin(), feature_pairs[0].end(),
				feature_pairs[1].begin(), feature_pairs[1].end(),
				std::inserter(feature_pairs_result, feature_pairs_result.begin()),
				compareFeaturePair);
		}
		else {
			set_union(feature_pairs[0].begin(), feature_pairs[0].end(),
				feature_pairs[1].begin(), feature_pairs[1].end(),
				std::inserter(feature_pairs_result, feature_pairs_result.begin()),
				compareFeaturePair);
		}
	}
	else {
		//将数据移动到feature_pairs_result
		feature_pairs_result = std::move(feature_pairs[0]);
	}

	vector<double> distances;
	distances.reserve(feature_pairs_result.size());
	for (int i = 0; i < feature_pairs_result.size(); ++i) {
		distances.emplace_back(feature_pairs_result[i].distance);
	}
	double mean, std;
	//计算距离的平均值和方差(x^2 - mean^2)
	Statistics::getMeanAndSTD(distances, mean, std);

	//异常域
	const double OUTLIER_THRESHOLD = (INLIER_TOLERANT_STD_DISTANCE * std) + mean;
	vector<pair<int, int> > initial_indices;
	initial_indices.reserve(feature_pairs_result.size());
	for (int i = 0; i < feature_pairs_result.size(); ++i) {
		if (feature_pairs_result[i].distance < OUTLIER_THRESHOLD) {
			initial_indices.emplace_back(feature_pairs_result[i].feature_index[0],
				feature_pairs_result[i].feature_index[1]);
		}
	}
#ifndef DP_NO_LOG
	//writeImageOfFeaturePairs("init", _match_pair, initial_indices);
#endif
	//最后将得到的差距值最小和次小的特征点描述对 加入initial_indices 列表中.
	return initial_indices;
}

Mat MultiImages::textureMapping(const vector<vector<Point2> >& _vertices,
	const Size2& _target_size,
	const BLENDING_METHODS& _blend_method) const {
	vector<Mat> warp_images;
	return textureMapping(_vertices, _target_size, _blend_method, warp_images);
}

/// <summary>
/// 
/// </summary>
/// <param name="_vertices">每张图像的每个网格点的坐标.以大图左上为原点</param>
/// <param name="_target_size"></param>
/// <param name="_blend_method"></param>
/// <param name="_warp_images"></param>
/// <returns></returns>
Mat MultiImages::textureMapping(const vector<vector<Point2> >& _vertices,
	const Size2& _target_size,
	const BLENDING_METHODS& _blend_method,
	vector<Mat>& _warp_images) const {

	//weight_mask为原图的权值图,中心权值最大, 四周权值减小.
	vector<Mat> weight_mask, new_weight_mask;
	vector<Point2> origins;
	//得到每个图像的矩形的4点.
	vector<Rect_<FLOAT_TYPE> > rects = getVerticesRects<FLOAT_TYPE>(_vertices);

	switch (_blend_method) {
	case BLEND_AVERAGE:
		break;
	case BLEND_LINEAR:
		weight_mask = getMatsLinearBlendWeight(getImages());
		break;
	default:
		printError("F(textureMapping) BLENDING METHOD");;
	}
#ifndef DP_NO_LOG
	for (int i = 0; i < rects.size(); ++i) {
		cout << images_data[i].file_name << " rect = " << rects[i] << endl;
	}
#endif
	_warp_images.reserve(_vertices.size());
	origins.reserve(_vertices.size());
	new_weight_mask.reserve(_vertices.size());

	const int NO_GRID = -1, TRIANGLE_COUNT = 3, PRECISION = 0;
	const int SCALE = pow(2, PRECISION);

	for (int i = 0; i < images_data.size(); ++i) {//遍历图像
		//得到网格点
		const vector<Point2>& src_vertices = images_data[i].mesh_2d->getVertices();
		//得到每个网格的4个点
		const vector<Indices>& polygons_indices = images_data[i].mesh_2d->getPolygonsIndices();
		//得到第i张图像的左上角顶点的坐标
		const Point2 origin(rects[i].x, rects[i].y);

		const Point2 shift(0.5, 0.5);

		vector<Mat> affine_transforms;
		affine_transforms.reserve(polygons_indices.size() * (images_data[i].mesh_2d->getTriangulationIndices().size()));
		Mat polygon_index_mask(rects[i].height + shift.y, rects[i].width + shift.x, CV_32SC1, Scalar::all(NO_GRID));
		int label = 0;
		for (int j = 0; j < polygons_indices.size(); ++j) {//遍历网格
			//[ [0,1,2],[0,2,3] ]
			for (int k = 0; k < images_data[i].mesh_2d->getTriangulationIndices().size(); ++k) {
				//[0,1,2]
				const Indices& index = images_data[i].mesh_2d->getTriangulationIndices()[k];
				//第i个图像的第j个网格 的3个顶点,即一个网格的一半,是三角形.
				const Point2i contour[] = {
					//第i个图像的第j个网格 的↖/↗/↘ 三个顶点的坐标(以此图像↖为原点)  *  SCALE(=1)
					(_vertices[i][polygons_indices[j].indices[index.indices[0]]] - origin) * SCALE,
					(_vertices[i][polygons_indices[j].indices[index.indices[1]]] - origin) * SCALE,
					(_vertices[i][polygons_indices[j].indices[index.indices[2]]] - origin) * SCALE,
				};
				//将label填充至形状为contour的区域,返回polygon_index_mask 矩阵
				fillConvexPoly(polygon_index_mask, contour, TRIANGLE_COUNT, label, LINE_AA, PRECISION);//params:返回的图像,填充多边形,多边形顶点数,颜色,线条类型,顶点坐标的小数点位数.

				//网格的三角形的相对 这张图像 的坐标
				Point2f src[] = {
					_vertices[i][polygons_indices[j].indices[index.indices[0]]] - origin,
					_vertices[i][polygons_indices[j].indices[index.indices[1]]] - origin,
					_vertices[i][polygons_indices[j].indices[index.indices[2]]] - origin
				};
				//第i个图像 变形前的网格的三角形 坐标
				Point2f dst[] = {
					src_vertices[polygons_indices[j].indices[index.indices[0]]],
					src_vertices[polygons_indices[j].indices[index.indices[1]]],
					src_vertices[polygons_indices[j].indices[index.indices[2]]]
				};
				//得到 相对小图(该小图网格变换后的) 到 该小图原始网格点的 坐标变化的 仿射变换矩阵
				affine_transforms.emplace_back(getAffineTransform(src, dst));
				++label;
			}
		}
		//这个for循环结束后,可以得到第i个图像的 一个填满三角形的polygon_index_mask; 一堆从相对第i图像的三角形 仿射变换至 相对大图的三角形 的仿射变换矩阵.

		//存放变形后且画完像素点的图像.
		Mat image = Mat::zeros(rects[i].height + shift.y, rects[i].width + shift.x, CV_8UC4);

		Mat w_mask = (_blend_method != BLEND_AVERAGE) ? Mat::zeros(image.size(), CV_32FC1) : Mat();

		for (int y = 0; y < image.rows; ++y) {
			for (int x = 0; x < image.cols; ++x) {//遍历待填充的图像像素点.
				//(x,y)这个像素点在 polygon_index_mask 对应的数据(网格index或者No_Grid)
				int polygon_index = polygon_index_mask.at<int>(y, x);
				if (polygon_index != NO_GRID) {//该点应该是图像像素.
					//得到变化前对应的位置.即原图的对应像素点
					Point2 p_f = applyTransform2x3<FLOAT_TYPE>(x, y,
						affine_transforms[polygon_index]);
					if (p_f.x >= 0 && p_f.y >= 0 &&
						p_f.x <= images_data[i].img.cols &&
						p_f.y <= images_data[i].img.rows) {
						//返回图像 p_f点的alpha通道数据.返回为长度为1的列向量.
						Vec<uchar, 1> alpha = getSubpix<uchar, 1>(images_data[i].alpha_mask, p_f);
						//返回图像 p_f点的rgb通道数据.返回为长度为3的列向量.
						Vec3b c = getSubpix<uchar, 3>(images_data[i].img, p_f);

						//将得到的像素点画到最终图像对应像素点上
						image.at<Vec4b>(y, x) = Vec4b(c[0], c[1], c[2], alpha[0]);
						if (_blend_method != BLEND_AVERAGE) {
							w_mask.at<float>(y, x) = getSubpix<float>(weight_mask[i], p_f);
						}
					}
				}
			}
		}
		//到此,一张图像填充完毕.

		_warp_images.emplace_back(image);
		origins.emplace_back(rects[i].x, rects[i].y);
		if (_blend_method != BLEND_AVERAGE) {
			new_weight_mask.emplace_back(w_mask);
		}
	}

	return Blending(_warp_images, origins, _target_size, new_weight_mask, _blend_method == BLEND_AVERAGE);
}

void MultiImages::writeResultWithMesh(const Mat& _result,
	const vector<vector<Point2> >& _vertices,
	const string& _postfix,
	const bool _only_border) const {
	const int line_thickness = 2;
	const Mat result(_result.size() + Size(line_thickness * 6, line_thickness * 6), CV_8UC4);
	const Point2 shift(line_thickness * 3, line_thickness * 3);
	const Rect rect(shift, _result.size());
	_result.copyTo(result(rect));
	for (int i = 0; i < images_data.size(); ++i) {
		Mat item_img(_result.size() + Size(line_thickness * 6, line_thickness * 6), CV_8UC4);
		string mesh_img_name = "-Mesh";
		const Scalar& color = getBlueToRedScalar((2. * i / (images_data.size() - 1)) - 1) * 255;
		const vector<Edge>& edges = images_data[i].mesh_2d->getEdges();
		vector<int> edge_indices;
		if (_only_border) {
			edge_indices = images_data[i].mesh_2d->getBoundaryEdgeIndices();
			mesh_img_name = "-Border";
		}
		else {
			mesh_img_name = "-Mesh";
			edge_indices.reserve(edges.size());
			for (int j = 0; j < edges.size(); ++j) {
				edge_indices.emplace_back(j);
			}
		}
		for (int j = 0; j < edge_indices.size(); ++j) {
			line(result,
				_vertices[i][edges[edge_indices[j]].indices[0]] + shift,
				_vertices[i][edges[edge_indices[j]].indices[1]] + shift, color, line_thickness, LINE_8);

			line(item_img, _vertices[i][edges[edge_indices[j]].indices[0]] + shift,
				_vertices[i][edges[edge_indices[j]].indices[1]] + shift, color, line_thickness, LINE_8);

		}
		imwrite(parameter.debug_dir + parameter.file_name + images_data[i].file_name + _postfix + mesh_img_name + ".png", item_img);
	}

	imwrite(parameter.debug_dir + parameter.file_name + _postfix + ".png", result(rect));
}

const vector<vector<vector<Point2> > >& MultiImages::getTwoImgFeatureMatches(pair<int, int> _mask_pair_) const {

	vector<pair<int, int>> feature_pair_ = getTwoImgFeaturePairs(_mask_pair_);
	const int& m1 = _mask_pair_.first, & m2 = _mask_pair_.second;

	feature_matches[m1][m2].reserve(feature_pair_.size());
	feature_matches[m2][m1].reserve(feature_pair_.size());
	const vector<Point2>& m1_fpts = images_data[m1].getFeaturePoints();
	const vector<Point2>& m2_fpts = images_data[m2].getFeaturePoints();
	for (int j = 0; j < feature_pair_.size(); ++j) {
		feature_matches[m1][m2].emplace_back(m1_fpts[feature_pair_[j].first]);
		feature_matches[m2][m1].emplace_back(m2_fpts[feature_pair_[j].second]);
	}
	return feature_matches;
}


vector<pair<int, int>> MultiImages::getTwoImgFeaturePairs(pair<int, int> _mask_pair_) const {

	const vector<pair<int, int> >& initial_indices = getInitialFeaturePairs(_mask_pair_);
	const vector<Point2>& m1_fpts = images_data[_mask_pair_.first].getFeaturePoints();
	const vector<Point2>& m2_fpts = images_data[_mask_pair_.second].getFeaturePoints();
	vector<Point2> X, Y; // 分图像保存上一步初始匹配好的特征点
	X.reserve(initial_indices.size());
	Y.reserve(initial_indices.size());
	int flag = 0;
	for (int j = 0; j < initial_indices.size(); ++j) {
		const pair<int, int> it = initial_indices[j];
		if (it.first < 0 || it.second == -1) {
			flag = 1;
			continue;
		}
		X.emplace_back(m1_fpts[it.first]);
		Y.emplace_back(m2_fpts[it.second]);
	}
	vector<pair<int, int> > result;
	result = getFeaturePairsBySequentialRANSAC(_mask_pair_, X, Y, initial_indices);

	return  result;
}


/// <summary>
/// 得到RMSE指标
/// </summary>
/// <param name="_vertices"></param>
/// <returns></returns>
double MultiImages::getRMSE(vector<vector<Point2>> _vertices) const
{
	const vector<pair<int, int> >& images_match_graph_pair_list = parameter.getImagesMatchGraphPairList();

	for (int i = 0; i < images_match_graph_pair_list.size(); ++i) {
		const pair<int, int>& match_pair = images_match_graph_pair_list[i];
		const int& m1 = match_pair.first, & m2 = match_pair.second;
		if (feature_matches[m1][m2].size() == 0) {
			getTwoImgFeatureMatches(pair<int, int>(m1, m2));
		}
	}

	vector<Rect_<FLOAT_TYPE> > rects = getVerticesRects<FLOAT_TYPE>(_vertices);
	int feature_num = 0;
	double rmse_temp = 0.0;

	for (int i = 0; i < images_match_graph_pair_list.size(); ++i) {
		const pair<int, int>& match_pair = images_match_graph_pair_list[i];
		const int& m1 = match_pair.first, & m2 = match_pair.second;

		feature_num += feature_matches[m1][m2].size();
		double temp__ = rmse_temp;
		const vector<Indices>& m1_polygons_indice = images_data[m1].mesh_2d->getPolygonsIndices();//得到网格四个顶点索引
		const vector<Point2>& m1_vertices = images_data[m1].mesh_2d->getVertices();
		for (int j = 0; j < feature_matches[m1][m2].size(); ++j) {
			// get mesh indices and vertex coordinates 
			int m1_j_indice = images_data[m1].mesh_2d->getGridIndexOfPoint(feature_matches[m1][m2][j]);
			int m2_j_indice = images_data[m2].mesh_2d->getGridIndexOfPoint(feature_matches[m2][m1][j]);
			if ((m1_j_indice > images_data[m1].mesh_2d->nh * images_data[m1].mesh_2d->nw) || (m1_j_indice < 0) ||
				(m2_j_indice > images_data[m1].mesh_2d->nh * images_data[m1].mesh_2d->nw) || (m2_j_indice < 0)) {
				feature_num -= feature_matches[m1][m2].size();
				rmse_temp = temp__;
				break;
			}
			//确定特征点所在三角形顶点
			int m1Tom2_p1_temp = VerifyVertices(feature_matches[m1][m2][j], m1_j_indice, m1_polygons_indice, m1_vertices);
			int m1Tom2_p2_temp = VerifyVertices(feature_matches[m2][m1][j], m2_j_indice, m1_polygons_indice, m1_vertices);

			Mat m1Tom2_p1_affineTransform = _getAffineTransform(_vertices, m1, m1_polygons_indice, m1_j_indice, m1Tom2_p1_temp, m1_vertices);
			Mat m1Tom2_p2_affineTransform = _getAffineTransform(_vertices, m2, m1_polygons_indice, m2_j_indice, m1Tom2_p2_temp, m1_vertices);

			Point2 m1Tom2_p1_f = applyTransform2x3<FLOAT_TYPE>(feature_matches[m1][m2][j].x, feature_matches[m1][m2][j].y, m1Tom2_p1_affineTransform);
			Point2 m1Tom2_p2_f = applyTransform2x3<FLOAT_TYPE>(feature_matches[m2][m1][j].x, feature_matches[m2][m1][j].y, m1Tom2_p2_affineTransform);

			rmse_temp += sqrt(Point2dDis(m1Tom2_p1_f, m1Tom2_p2_f));
		}
	}

	return sqrt(rmse_temp / feature_num);
}

int lastEdgeInRow(int y, int nw) {
	return (1 + y) * (nw * 2) + y;
}

double MultiImages::getWarpingResidual(vector<vector<Point2>> _vertices) const
{
	for (int i = 0; i < images_data.size(); ++i) {

		const vector<Edge>& edges = images_data[i].mesh_2d->getEdges();

		int nw = images_data[i].mesh_2d->nw;
		int nh = images_data[i].mesh_2d->nh;
		vector<vector<Point2f> > rows, cols;
		rows.reserve(nw + 1);
		cols.reserve(nh + 1);

		int row_index = 0;
		int x = 0;
		for (int j = 0; j < edges.size(); j = j + 2, x++) {

			if (x >= nw) {//一行遍历完毕,转到下一行
				x = 0;
				j++;
				row_index++;
			}
			if (row_index == nh) {
				break;
			}
			else {
				vector<Point2f> row_item = rows[row_index];
				Point2f e_start_p = _vertices[i][edges[j].indices[0]];
				Point2f e_end_p = _vertices[i][edges[j].indices[1]];
				row_item.emplace_back(e_start_p);
				if (x == nw - 1) {
					row_item.emplace_back(e_end_p);
				}
			}

		}

	}

	return 0.0;
}

/// <summary>
/// 将图像和特征点画圈/线,保存到本地.
/// </summary>
/// <param name="_name">保存的图像名</param>
/// <param name="_match_pair">图像匹配对index</param>
/// <param name="_pairs">正确的特征点匹配对index</param>
void MultiImages::writeImageOfFeaturePairs(const string& _name,
	const pair<int, int>& _match_pair,
	const vector<pair<int, int> >& _pairs) const {
	cout << images_data[_match_pair.first].file_name << "-" <<
		images_data[_match_pair.second].file_name << " " << _name << " feature pairs = " << _pairs.size() << endl;

	//获取第一张图片的特征点
	const vector<Point2>& m1_fpts = images_data[_match_pair.first].getFeaturePoints();
	//获取第二张图片的特征点
	const vector<Point2>& m2_fpts = images_data[_match_pair.second].getFeaturePoints();
	vector<Point2> f1, f2;
	f1.reserve(_pairs.size());
	f2.reserve(_pairs.size());
	//将正确的特征点加入f1,f2;
	for (int i = 0; i < _pairs.size(); ++i) {
		//根据index,将第一张图像对应的特征点加入 f1;
		f1.emplace_back(m1_fpts[_pairs[i].first]);
		//根据index,将第二张图像对应的特征点加入 f2;
		f2.emplace_back(m2_fpts[_pairs[i].second]);
	}
	//得到画好的图像
	Mat image_of_feauture_pairs = getImageOfFeaturePairs(images_data[_match_pair.first].img,
		images_data[_match_pair.second].img,
		f1, f2);
	//将图像写到本地.
	imwrite(parameter.debug_dir +
		"feature_pairs-" + _name + "-" +
		images_data[_match_pair.first].file_name + "-" +
		images_data[_match_pair.second].file_name + "-" +
		to_string(_pairs.size()) +
		images_data[_match_pair.first].file_extension, image_of_feauture_pairs);
}

int timeransac = 0;
void MultiImages::drawRansac(const int img_index, const int img_index_second,
	const vector<pair<int, int> >& _initial_indices, const vector<char>& _mask) const {
	Mat img1 = images_data[img_index].img;
	Mat img2 = images_data[img_index_second].img;
	const int CIRCLE_RADIUS = 5;
	const int CIRCLE_THICKNESS = 1;
	const int LINE_THICKNESS = 1;
	const int RGB_8U_RANGE = 256;

	Mat result = Mat::zeros(max(img1.rows, img2.rows), img1.cols + img2.cols, CV_8UC3);
	Mat left(result, Rect(0, 0, img1.cols, img1.rows));
	Mat right(result, Rect(img1.cols, 0, img2.cols, img2.rows));

	Mat img1_8UC3, img2_8UC3;

	if (img1.type() == CV_8UC3) {
		img1_8UC3 = img1;
		img2_8UC3 = img2;
	}
	else {
		img1.convertTo(img1_8UC3, CV_8UC3);
		img2.convertTo(img2_8UC3, CV_8UC3);
	}
	img1_8UC3.copyTo(left);
	img2_8UC3.copyTo(right);

	//获取第一张图片的特征点
	const vector<Point2>& m1_fpts = images_data[img_index].getFeaturePoints();
	//获取第二张图片的特征点
	const vector<Point2>& m2_fpts = images_data[img_index_second].getFeaturePoints();
	vector<Point2> f1, f2;
	f1.reserve(_initial_indices.size());
	f2.reserve(_initial_indices.size());
	//将正确的特征点加入f1,f2;
	for (int i = 0; i < _initial_indices.size(); ++i) {
		//根据index,将第一张图像对应的特征点加入 f1;
		f1.emplace_back(m1_fpts[_initial_indices[i].first]);
		//根据index,将第二张图像对应的特征点加入 f2;
		f2.emplace_back(m2_fpts[_initial_indices[i].second]);
	}

	for (int i = 0; i < f1.size(); ++i) {

		Scalar colorLine(255, 255, 255);
		Scalar colorOut(0, 0, 0);
		Scalar colorIn(255, 0, 0);

		if (_mask[i] == 1) {
			circle(result, f1[i], CIRCLE_RADIUS, colorIn, CIRCLE_THICKNESS, LINE_AA);
			line(result, f1[i], f2[i] + Point2(img1.cols, 0), colorLine, LINE_THICKNESS, LINE_AA);
			circle(result, f2[i] + Point2(img1.cols, 0), CIRCLE_RADIUS, colorIn, CIRCLE_THICKNESS, LINE_AA);
		}
		else {
			circle(result, f1[i], CIRCLE_RADIUS, colorOut, CIRCLE_THICKNESS, LINE_AA);
			circle(result, f2[i] + Point2(img1.cols, 0), CIRCLE_RADIUS, colorOut, CIRCLE_THICKNESS, LINE_AA);
		}



	}

	imwrite(parameter.debug_dir +
		"ransac-" + to_string(timeransac) +
		images_data[img_index].file_extension, result);
	timeransac++;
}

/// <summary>
/// 得到所有图像的轮廓点采样数据(网格4点差值数据形式)
/// </summary>
/// <returns></returns>
const vector<vector<vector<InterpolateVertex>>>& MultiImages::getSamplesInterpolation() const
{
	if (content_mesh_interpolation.empty()) {
		content_mesh_interpolation.resize(images_data.size());
		//得到所有图像的轮廓点采样数据
		const vector<vector<vector<Point>>>& content_sample_points = getContentSamplePoints();

		//将直线采样点坐标,转化为网格差值的数据形式
		for (int i = 0; i < content_sample_points.size(); ++i) {
			content_mesh_interpolation[i].resize(content_sample_points[i].size());
			for (int j = 0; j < content_sample_points[i].size(); ++j) {
				content_mesh_interpolation[i][j].reserve(content_sample_points[i][j].size());
				for (int k = 0; k < content_sample_points[i][j].size(); ++k)
					content_mesh_interpolation[i][j].emplace_back(images_data[i].mesh_2d->getInterpolateVertex(content_sample_points[i][j][k]));
			}
		}
	}
	return content_mesh_interpolation;
}


/// <summary>
/// 获取所有图像的采样点的uv数据
/// </summary>
/// <returns></returns>
const vector < vector<vector<pair<double, double> >> >& MultiImages::getTermUV() const
{
	if (content_term_uv.empty()) {
		content_term_uv.reserve(images_data.size());
		const vector<vector<vector<Point> > >& content_sample_points = getContentSamplePoints();
		for (int i = 0; i < content_sample_points.size(); ++i) {
			content_term_uv.push_back(calcTriangleUV(content_sample_points[i]));
		}
	}
	return content_term_uv;
}

/// <summary>
/// 为每个图像获取轮廓采样点数据
/// </summary>
/// <returns></returns>
const vector<vector<vector<Point> > >& MultiImages::getContentSamplePoints() const {
	if (content_sample_points.empty()) {
		//		content_sample_points.resize(images_data.size());
		content_sample_points.resize(images_data.size());
		for (int i = 0; i < images_data.size(); i++) {
			//	if(images_data[i].getContentSamplesPoint().empty())
				//	content_sample_points.push_back(vector<vector<Point>>(1,vector<Point>(1,Point(65535,65535))));
			//	else
			vector<double> weight;
			content_sample_points[i] = images_data[i].getContentSamplesPoint(weight);
			content_line_weights.emplace_back(weight);
		}
	}
	return content_sample_points;
}

/// <summary>
/// 根据采样点,计算每个采样点对应的uv
/// </summary>
/// <param name="samples"></param>
/// <returns></returns>
const vector<vector<pair<double, double>>> MultiImages::calcTriangleUV(const vector<vector<Point>> samples) const
{
	vector<vector<pair<double, double>>> uvs;
	uvs.reserve(samples.size());
	if (samples.size() <= 0) {
		return uvs;
	}

	for (int i = 0; i < samples.size(); i++) {
		vector<pair<double, double>> item_uvs;

		vector<Point> item = samples[i];
		Point2 start = item[0], end = item[1];
		item_uvs.reserve(item.size() - 2);

		for (int j = 2; j < item.size(); j++) {
			//1.三角形3个点. 起始点|采样点|终点
			Point2 sample(item[j].x, item[j].y);
			//2.转化 start-->sample, start-->end 两个向量 为3维向量.
			Vector3d ab = trans2Vector(end - start), ac = trans2Vector(sample - start);
			double abNormal = ab.norm(), acNormal = ac.norm();

			//3.得到三角形面积后,得到过采样点的三角形高:h,然后再得到垂线点到始点的距离:stroke
			double s = ac.cross(ab).norm() / 2;
			double h = s * 2 / abNormal;
			double stroke = sqrt(ac.norm() * ac.norm() - h * h);

			//cout << "三角形面积:" << s << endl;
			//cout << "过采样点高的三角形高度h" << h << endl;
			//cout << "过采样点高的三角形垂点stroke:" << stroke << endl;

			//4.计算v,u
			double v = h / ab.norm();
			double u = stroke / ab.norm();

			//5.调整u正负.根据start-->sample, start-->end两向量cos值计算,如果夹角大于90度,为负值,小于90度,为正值.
			if (0 <= ab.dot(ac) / (ab.norm() * ac.norm())) {
				u = u;
			}
			else {
				u = -u;
			}

			//6.调整v正负.根据start-->sample, start-->end两向量叉乘值正负计算,如果叉乘z为负,则正时针,与start-->end 旋转角度同向,v为正.
			if (ab.cross(ac)(2) <= 0) {
				v = v;
			}
			else {
				v = -v;
			}

			item_uvs.emplace_back(u, v);
		}
		uvs.push_back(item_uvs);
	}

	return uvs;
}

/// <summary>
/// 计算出每个样本点的权重
/// </summary>
/// <returns></returns>
const vector<vector<vector<double> >>& MultiImages::getSamplesWeight() const {
	if (samplesWeight.empty()) {
		double throsholdMin = 0.01;
		const vector<vector<vector<InterpolateVertex> > >& content_interpolation = getSamplesInterpolation();
		samplesWeight.resize(content_interpolation.size());

		vector<vector<double> > images_polygon_distance_to_nonOverlap;
		images_polygon_distance_to_nonOverlap.resize(images_data.size());

		//得到 每个图像自己的网格点 哪些是 要扭曲(index).(就是图像自己哪些网格点 是 处于重叠部分.重叠部分的点是true)
		const vector<vector<bool > >& images_features_mask = getImagesFeaturesMaskByMatchingPoints();

		//得到 每张图像的 每个点(自己的 和 别图扭过来的)处于第几个网格点 和 该网格点4个点的权重.
		const vector<vector<InterpolateVertex> >& mesh_interpolate_vertex_of_matching_pts = getInterpolateVerticesOfMatchingPoints();


		for (int i = 0; i < images_polygon_distance_to_nonOverlap.size(); ++i) {//图像个数,遍历每个图像
			vector<vector<InterpolateVertex> > oneImageSamples = content_interpolation[i];
			samplesWeight[i].resize(oneImageSamples.size());

			//一个图像的网格数量
			const int polygons_count = (int)images_data[i].mesh_2d->getPolygonsIndices().size();

			//给每个网格设置mask,默认都false
			vector<bool> polygons_has_matching_pts(polygons_count, false);

			//将属于重叠区域的网格mask设置为true
			for (int j = 0; j < images_features_mask[i].size(); ++j) {
				//图像i的 非重叠区域的点
				if (!images_features_mask[i][j]) {
					//将图像i 中属于非重叠区域的网格mask设置为true
					polygons_has_matching_pts[mesh_interpolate_vertex_of_matching_pts[i][j].polygon] = true;
				}
			}
			//每个网格的权重
			images_polygon_distance_to_nonOverlap[i].reserve(polygons_count);

			priority_queue<dijkstraNode> que;//优先级最高的node优先出队.top():访问队头元素. pop()弹出队头元素. 

			//遍历每个网格的mask,将遍历过的网格mask都设置为false.
			//如果网格mask为true,则将其权重设为0,如果网格mask为false,则将其权重设为MAX.
			//如果网格mask为true,new dijkstraNode(j,j,0) 点加入que;
			for (int j = 0; j < polygons_has_matching_pts.size(); ++j) {
				if (polygons_has_matching_pts[j]) {
					polygons_has_matching_pts[j] = false;
					images_polygon_distance_to_nonOverlap[i].emplace_back(0);
					que.push(dijkstraNode(j, j, 0));
				}
				else {
					images_polygon_distance_to_nonOverlap[i].emplace_back(FLT_MAX);
				}
			}
			//经上面的for,所有的mask都变成的false; 但非重叠区域的权值为0,且加入队列;重叠区域的权值为MAX,不加入队列.;

			//以下的while为迪杰斯特拉求解 图像所有网格 距离 重叠区域最近的距离.
			//得到每个网格点的左上角点的 4邻居点的index 列表.
			const vector<Indices>& polygons_neighbors = images_data[i].mesh_2d->getPolygonsNeighbors();
			//得到每个网格点的中心点的坐标.
			const vector<Point2>& polygons_center = images_data[i].mesh_2d->getPolygonsCenter();
			while (que.empty() == false) {//队列不为空,就一直循环.
				//队头的元素.
				const dijkstraNode now = que.top();
				//pos网格 属于 第几个网格
				const int index = now.pos;
				que.pop();
				//第一遍,将所有的重叠区域的网格进行....
				if (polygons_has_matching_pts[index] == false) {
					//将处理过得网格mask设置为true.
					polygons_has_matching_pts[index] = true;
					//将此网格的....
					for (int j = 0; j < polygons_neighbors[index].indices.size(); ++j) {//循环得到此网格左上角顶点的邻居点index
						//得到此网格左上角顶点的一个邻居点的index
						const int n = polygons_neighbors[index].indices[j];
						if (polygons_has_matching_pts[n] == false) { //判断此网格的 一个邻居网格mask是否为false(即这个网格没有被处理过)
							const double dis = norm(polygons_center[n] - polygons_center[now.from]);
							if (images_polygon_distance_to_nonOverlap[i][n] > dis) {
								images_polygon_distance_to_nonOverlap[i][n] = dis;
								que.push(dijkstraNode(now.from, n, dis));
							}
						}
					}
				}
			}
			//计算每个样本点的权重
			for (int j = 0; j < oneImageSamples.size(); j++) {
				samplesWeight[i][j].resize(oneImageSamples[j].size(), 1.0);
				double maxWeight = -1;
				//遍历一条线的样本,算权重
				for (int k = 0; k < oneImageSamples[j].size(); k++) {
					InterpolateVertex item = oneImageSamples[j][k];
					//该样本点距离非重叠区域的距离
					double distanceToNonOverlop = images_polygon_distance_to_nonOverlap[i][item.polygon];
					//0:就在非重叠区域.
					if (distanceToNonOverlop == 0) {
						samplesWeight[i][j][k] = 1; //非重叠区域权重为1;
						maxWeight = 1;
						continue;
					}
					//该样本点在重叠区域.计算该样本距离图片边界,然后和非重叠区域的距离 进行计算得到权重.
					Point2 p = polygons_center[item.polygon];
					double distanceToImageBorder = min(p.x, p.y);
					//ratio : 0-1;1:距离图片边界近,0:距离非重叠区域近
					double ratio = distanceToNonOverlop / (distanceToImageBorder + distanceToNonOverlop);
					double sampleWeight = exp(-pow(ratio, 2) / (1 / log(1 / throsholdMin)));
					if (sampleWeight > maxWeight) {
						maxWeight = sampleWeight;
					}
					samplesWeight[i][j][k] = sampleWeight;
				}
				//解决全部处于重叠区域的曲线权重太小的问题.
				for (int k = 0; k < oneImageSamples[j].size(); k++) {
					samplesWeight[i][j][k] = samplesWeight[i][j][k] / maxWeight;
				}
			}
		}
	}

	return samplesWeight;
}
double Point2dDis(Point2d p1, Point2d p2) {
	return  (pow((p1.x - p2.x), 2) + pow((p1.y - p2.y), 2));
}

int VerifyVertices(Point2d p1, int m1Tom2_p1_index, const vector<Indices>& m1_polygons_indice, const vector<Point2>& m1_vertices) {
	int m1_temp;
	if (Point2dDis(p1, m1_vertices[m1_polygons_indice[m1Tom2_p1_index].indices[1]]) >
		Point2dDis(p1, m1_vertices[m1_polygons_indice[m1Tom2_p1_index].indices[3]]))
		m1_temp = m1_polygons_indice[m1Tom2_p1_index].indices[3];
	else
		m1_temp = m1_polygons_indice[m1Tom2_p1_index].indices[1];
	return  m1_temp;
}

Mat _getAffineTransform(vector<vector<Point2>>& _vertices, int m1, const vector<Indices>& m1_polygons_indice, int m1Tom2_p1_index, int m1Tom2_p1_temp, const vector<Point2>& m1_vertices) {
	Point2f src[] = {
		m1_vertices[m1_polygons_indice[m1Tom2_p1_index].indices[0]],
		m1_vertices[m1_polygons_indice[m1Tom2_p1_index].indices[2]],
		m1_vertices[m1Tom2_p1_temp]
	};

	Point2f dst[] = {
		_vertices[m1][m1_polygons_indice[m1Tom2_p1_index].indices[0]],
		_vertices[m1][m1_polygons_indice[m1Tom2_p1_index].indices[2]],
		_vertices[m1][m1Tom2_p1_temp]
	};
	Mat affineTransform = getAffineTransform(src, dst);
	return affineTransform;
}
