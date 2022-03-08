//
//  ImageData.cpp
//  UglyMan_Stitching
//
//  Created by uglyman.nothinglo on 2015/8/15.
//  Copyright (c) 2015 nothinglo. All rights reserved.
//

#include "ImageData.h"

LineData::LineData(const Point2& _a,
	const Point2& _b,
	const double _width,
	const double _length) {
	data[0] = _a;
	data[1] = _b;
	width = _width;
	length = _length;
}

const bool LINES_FILTER_NONE(const double _data,
	const Statistics& _statistics) {
	return true;
};

const bool LINES_FILTER_WIDTH(const double _data,
	const Statistics& _statistics) {
	return _data >= MAX(2.f, (_statistics.min + _statistics.mean) / 2.f);
	return true;
};

const bool LINES_FILTER_LENGTH(const double _data,
	const Statistics& _statistics) {
	return _data >= MAX(10.f, _statistics.mean);
	return true;
};


ImageData::ImageData(const string& _file_dir,
	const string& _file_full_name,
	LINES_FILTER_FUNC* _width_filter,
	LINES_FILTER_FUNC* _length_filter,
	const string* _debug_dir) {

	file_dir = &_file_dir;
	std::size_t found = _file_full_name.find_last_of(".");
	assert(found != std::string::npos);
	file_name = _file_full_name.substr(0, found);
	file_extension = _file_full_name.substr(found);
	debug_dir = _debug_dir;

	grey_img = Mat();

	width_filter = _width_filter;
	length_filter = _length_filter;

	/*读入图片*/
	/*if (read_LatALon == "_img_metadata") {
		if (file_extension == ".jpg" || file_extension == ".JPG")
			LatALon = ReadJPGLonALon(*file_dir + file_name + file_extension);
		else if (file_extension == ".tif" || file_extension == ".TIF")
			LatALon = ReadPhantom4TIFLonALon(*file_dir + file_name + file_extension);
		else
			LatALon = pair<double, double>(0, 0);
	}*/

	img = imread(*file_dir + file_name + file_extension); // 默认3通道,GBR读入
	rgba_img = imread(*file_dir + file_name + file_extension, IMREAD_UNCHANGED); //按解码得到的方式读入图像

	/*如果原图大于最低要求,就将原图降为DOWN_SAMPLE_IMAGE_SIZE(800*600)大小*/
	float original_img_size = img.rows * img.cols;
	if (original_img_size > DOWN_SAMPLE_IMAGE_SIZE) {
		float scale = sqrt(DOWN_SAMPLE_IMAGE_SIZE / original_img_size);
		resize(img, img, Size(), scale, scale);
		resize(rgba_img, rgba_img, Size(), scale, scale);
	}

	assert(rgba_img.channels() >= 3);
	/*如果按解码方法 得到的图像是3通道,则变为BGRA4通道.*/
	if (rgba_img.channels() == 3) {
		cvtColor(rgba_img, rgba_img, CV_BGR2BGRA);
	}
	vector<Mat> channels;
	/*将BGRA4通道图像,分割成4个Mat.每个Mat分别代表不同通道.*/
	split(rgba_img, channels);
	alpha_mask = channels[3];
	/*使用make方法来代替new,为一张图片生成MeshGrid实例*/
	mesh_2d = make_unique<MeshGrid>(img.cols, img.rows);
}

/*将BGR图像转为灰度图*/
const Mat& ImageData::getGreyImage() const {
	if (grey_img.empty()) {
		cvtColor(img, grey_img, CV_BGR2GRAY);
	}
	return grey_img;
}


const vector<LineData>& ImageData::getLines() const {
	if (img_lines.empty()) {
		const Mat& grey_image = getGreyImage();
		Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_STD);

		vector<Vec4f>  lines;
		vector<double> lines_width, lines_prec, lines_nfa;
		ls->detect(grey_image, lines, lines_width, lines_prec, lines_nfa);

		vector<double> lines_length;
		vector<Point2> lines_points[2];

		const int line_count = (int)lines.size();

		lines_length.reserve(line_count);
		lines_points[0].reserve(line_count);
		lines_points[1].reserve(line_count);

		for (int i = 0; i < line_count; ++i) {
			lines_points[0].emplace_back(lines[i][0], lines[i][1]);
			lines_points[1].emplace_back(lines[i][2], lines[i][3]);
			lines_length.emplace_back(norm(lines_points[1][i] - lines_points[0][i]));
		}

		const Statistics width_statistics(lines_width), length_statistics(lines_length);
		for (int i = 0; i < line_count; ++i) {
			if (width_filter(lines_width[i], width_statistics) &&
				length_filter(lines_length[i], length_statistics)) {
				img_lines.emplace_back(lines_points[0][i],
					lines_points[1][i],
					lines_width[i],
					lines_length[i]);
			}
		}
#ifndef DP_LOG
		vector<Vec4f> draw_lines;
		draw_lines.reserve(img_lines.size());
		for (int i = 0; i < img_lines.size(); ++i) {
			draw_lines.emplace_back(img_lines[i].data[0].x, img_lines[i].data[0].y,
				img_lines[i].data[1].x, img_lines[i].data[1].y);
		}
		Mat canvas = Mat::zeros(grey_image.rows, grey_image.cols, grey_image.type());
		ls->drawSegments(canvas, draw_lines);
		imwrite(*debug_dir + "line-result-" + file_name + file_extension, canvas);
#endif
	}
	return img_lines;
}

/*获取一个图像的所有特征点*/
const vector<Point2>& ImageData::getFeaturePoints() const {
	if (feature_points.empty()) {
		FeatureController::detect(getGreyImage(), feature_points, feature_descriptors);
	}
	return feature_points;
}
/*获取一个图像的说有特征点描述*/
const vector<FeatureDescriptor>& ImageData::getFeatureDescriptors() const {
	if (feature_descriptors.empty()) {
		FeatureController::detect(getGreyImage(), feature_points, feature_descriptors);
	}
	return feature_descriptors;
}


/// <summary>
/// 得到此图像的所有曲线的起终点,采样点数据.0:起点,1:终点,之后是采样点.
/// </summary>
/// <returns></returns>
const vector<vector<Point>> ImageData::getContentSamplesPoint(vector<double>& weights) const
{
	Mat imgRes = img.clone();
	Mat gray;
	cvtColor(imgRes, gray, COLOR_BGR2GRAY);
	Mat image;
	//1.调整到HED规定图片大小.
	resize(imgRes, image, cv::Size(500, 500 * (double)imgRes.rows / imgRes.cols));

	//2.HED图片边缘检测
	edgeDetection(image, image, 0.5);
	//3.回归原本大小
	resize(image, image, Size(imgRes.cols, imgRes.rows));
	//4.细化边缘图像
	thin(image, image, (double)imgRes.cols / 500);

	//4.1 角点检测
	std::vector<cv::Point2f> corners;

	int max_corners = 600;
	double quality_level = 0.1;
	double min_distance = 12.0;
	int block_size = 3;
	bool use_harris = false;
	double k = 0.04;
	cv::goodFeaturesToTrack(image,
		corners,
		max_corners,
		quality_level,
		min_distance,
		cv::Mat(),
		block_size,
		use_harris,
		k);
	//4.2删除角点处像素
	Point2f itemPoint;
	Rect roi;
	roi.width = 8;
	roi.height = 8;
	for (int i = 0; i < corners.size(); i++) {
		itemPoint = corners[i];
		roi.width = 8;
		roi.height = 8;
		roi.x = itemPoint.x - 4;
		roi.y = itemPoint.y - 4;
		roi &= Rect(0, 0, image.cols, image.rows);//防止越界

		Mat cover = Mat::zeros(roi.size(), CV_8UC1);
		cover.setTo(Scalar(0));
		cover.copyTo(image(roi));
	}



	//5.对细化后的边缘图像进行轮廓提取
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point());
	Mat imageContours = Mat::zeros(image.size(), CV_8UC3);//输出图
	drawContours(imageContours, contours, -1, Scalar(255), 1, 8, hierarchy);
	//将检测到的角点绘制到原图上
	for (int i = 0; i < corners.size(); i++)
	{
		cv::circle(imageContours, corners[i], 1, cv::Scalar(0, 0, 255), 2, 8, 0);
	}
	imwrite(*debug_dir + "After-findcontours" + file_name + file_extension, imageContours);


	//5.1将同方向相近线段连接
	vector<vector<Point>> contoursLineConnected;
	connectSmallLine(contours, hierarchy, contoursLineConnected);

	Mat imageCorn = Mat::zeros(image.size(), CV_8UC3);//输出图
	Mat Contours = Mat::zeros(image.size(), CV_8UC1);  //绘制  

	//6.轮廓数据剔除优化
	//大小阈值
	double min_size = min(image.cols, image.rows) * 0.1;
	//存储排除后的曲线集.
	vector<vector<Point>> res;
	Rect tempRect;
	for (vector<vector<Point>>::iterator iterator = contoursLineConnected.begin(); iterator != contoursLineConnected.end(); ++iterator) {
		tempRect = boundingRect(*iterator);
		float maxLength = max(tempRect.width, tempRect.height);
		if (maxLength <= min_size) {

		}
		else if (false) {

		}
		else {
			res.push_back(*iterator);
			//绘制轮廓  
			drawContours(imageCorn, contoursLineConnected, iterator - contoursLineConnected.begin(), Scalar(255), 1, 8);
		}
	}

	//6.1添加直线数据
	vector<Vec4f> lines = findLine(gray);

	//7.取采样点
	Mat imageSamples = Mat::zeros(image.size(), CV_8UC3);//输出图
	//存放所有曲线的起终点,采样点数据.0:起点,1:终点,之后是采样点.
	vector<vector<Point>> samplesData;
	samplesData.reserve(contoursLineConnected.size() + lines.size());
	weights.reserve(contoursLineConnected.size() + lines.size());

	//7.1直线采样
	//直线长度

	double lineLength, lineSampleDistX, lineSampleDistY;
	//分别为一条曲线点的数量,采样点个数,采样点间隔数
	int lineSampleNum, index = 0;
	for (vector<Vec4f>::iterator iterator = lines.begin(); iterator != lines.end(); ++iterator) {
		Vec4f item = *iterator;
		Point start = Point(item[0], item[1]);
		Point end = Point(item[2], item[3]);
		Point mid = start + (end - start) / 2;
		//1.计算曲线长度
		lineLength = PointDist(start, end);
		//2.根据总长度计算合适的采样点个数
		lineSampleNum = lineLength / (GRID_SIZE);
		//3.如果总长度还没有规定采样点距离长,这条线就不要了
		if (lineSampleNum == 0) {
			continue;
		}
		//存放采样点,起终点.
		vector<Point> itemLine;
		//4.放入起点,终点
		itemLine.reserve(lineSampleNum + 3);
		itemLine.push_back(start);
		itemLine.push_back(end);

		//6.计算采样点间隔数
		lineSampleDistX = (end.x - start.x) / (double)lineSampleNum;
		lineSampleDistY = (end.y - start.y) / (double)lineSampleNum;

		//7.将采样点加入采样点数据列表
		for (int i = 1; i <= lineSampleNum; i++) {
			Point sample;
			sample.x = start.x + i * lineSampleDistX;
			sample.y = start.y + i * lineSampleDistY;

			if (i == lineSampleNum) {//最后一个采样点
				if (itemLine.size() == 2) { // 还没有加入采样点
					//如果采样点到了终点,或靠近终点.
					if (sample == end || PointDist(sample, end) <= lineLength / 2) {
						//如果没有采样点,则曲线中间取为采样点
						itemLine.push_back(mid);
					}
					else {
						itemLine.push_back(sample);
					}

				}
				else {//已加入采样点
					itemLine.push_back(sample);
				}
			}
			//如果采样点不是最后一个点,就正常加入
			else {
				itemLine.push_back(sample);
			}
		}
		samplesData.push_back(itemLine);
		weights.emplace_back(1.4); //直线最大权重
		RNG rng(cvGetTickCount());
		Scalar s = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		//输出图像
		for (int i = 0; i < samplesData[index].size(); i++) {
			if (i == 0 || i == 1) {
				circle(imageSamples, samplesData[index][i], 5, s, FILLED);
			}
			else {
				circle(imageSamples, samplesData[index][i], 8, s);
			}
		}

		line(imageSamples, start, end, Scalar(255, 255, 255));

		index++;
	}

	//7.2曲线采样
	//曲线长度
	double contourLength;
	//分别为一条曲线点的数量,采样点个数,采样点间隔数
	int contourSize, sampleNum, sampleDist;

	for (vector<vector<Point>>::iterator iterator = res.begin(); iterator != res.end(); ++iterator) {
		//1.计算曲线长度
		contourLength = arcLength(*iterator, true);
		//2.根据总长度计算合适的采样点个数
		sampleNum = contourLength / (2.23 * GRID_SIZE);
		//3.如果总长度还没有规定采样点距离长,这条曲线就不要了
		if (sampleNum == 0) {
			//iterator = res.erase(iterator);
			continue;
		}
		//4.曲线点数据 排序去重.
		sort((*iterator).begin(), (*iterator).end(), sortForPoint);
		(*iterator).erase(unique((*iterator).begin(), (*iterator).end(), equalForPoint), (*iterator).end());

		//存放采样点,起终点.
		vector<Point> itemLine;
		//5.放入起点,终点
		itemLine.reserve(sampleNum + 3);
		itemLine.push_back((*iterator)[0]);
		itemLine.push_back((*iterator)[(*iterator).size() - 1]);

		//6.计算采样点间隔数
		contourSize = (*iterator).size();
		sampleDist = contourSize / sampleNum;

		//7.将采样点加入采样点数据列表
		for (int i = 1; i <= sampleNum; i++) {
			int sampleIndex = sampleDist * i;

			//如果采样点到了终点,或越界.
			if (sampleIndex >= (*iterator).size() - 1) {
				//如果没有采样点,则曲线中间取为采样点
				if (itemLine.size() == 2) {
					itemLine.push_back((*iterator)[(*iterator).size() / 2]);
				}
			}
			//如果采样点不是终点,就正常加入
			else {
				itemLine.push_back((*iterator)[sampleIndex]);
			}
		}
		samplesData.push_back(itemLine);
		weights.emplace_back(getLineWeight(*iterator));

		RNG rng(cvGetTickCount());
		Scalar s = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		//输出图像
		for (int i = 0; i < samplesData[index].size(); i++) {
			if (i == 0 || i == 1) {
				circle(imageSamples, samplesData[index][i], 6, s, FILLED);
			}
			else {
				circle(imageSamples, samplesData[index][i], 6, s);
			}
		}
		String weightStr = to_string(weights[index]);
		putText(imageSamples, weightStr, samplesData[index][0], FONT_HERSHEY_COMPLEX, 0.3, Scalar(0, 255, 255));
		for (int j = 0; j < (*iterator).size(); j++) {
			circle(imageSamples, (*iterator)[j], 1, s, FILLED);
		}
		index++;
	}

	imwrite(*debug_dir + "Samples" + file_name + file_extension, imageSamples);

	return samplesData;
}

void ImageData::clear() {
	img.release();
	grey_img.release();
	img_lines.clear();
	feature_points.clear();
	feature_descriptors.clear();
}
bool sortForPoint(Point a, Point b) {
	return (a.x < b.x || (a.x == b.x && a.y < b.y));
}

bool equalForPoint(Point a, Point b) {
	if (a.x == b.x && a.y == b.y) {
		return true;
	}
	return false;
}

/// <summary>
/// 连接线段.
/// </summary>
/// <param name="contours"></param>
/// <param name="hierarchy"></param>
/// <param name="contoursConnected"></param>
void connectSmallLine(vector<vector<Point>> contours, vector<Vec4i> hierarchy, vector<vector<Point>>& contoursConnected)
{
	for (vector<vector<Point>>::iterator iterator = contours.begin(); iterator != contours.end(); ++iterator) {
		sort((*iterator).begin(), (*iterator).end(), sortForPoint);
		(*iterator).erase(unique((*iterator).begin(), (*iterator).end(), equalForPoint), (*iterator).end());
	}

	//斜率阈值.20度
	double angleThrhold = (30.0 / 180) * acos(-1);

	//保存每条线的斜率
	vector<double> angles;
	//保存每条线的两端点
	vector< pair<Point, Point>> ses;

	//遍历所有线
	for (vector<vector<Point>>::iterator iterator = contours.begin(); iterator != contours.end();) {
		if ((*iterator).size() <= 2) {
			iterator = contours.erase(iterator);
			continue;
		}

		Vec4f line_para;
		//得到拟合直线
		fitLine(*iterator, line_para, DIST_L2, 0, 1e-2, 1e-2);
		//得到线的斜率,转化为角度
		double k = line_para[1] / line_para[0];
		angles.push_back(atan(k));

		//找到一条线的端点
		//pair<Point, Point> SEPoint = findStartEndPoint(*iterator);
		pair<Point, Point> SEPoint = findLineMinAndMax(*iterator);
		ses.push_back(SEPoint);
		++iterator;
	}

	//记录邻接线的关系,存放index 
	vector<vector<int>> connectIds;
	connectIds.resize(contours.size());

	for (int j = 0; j < contours.size(); j++) {
		for (int k = j; k < contours.size(); k++) {
			if (j == k) {
				continue;
			}
			//角度符合?
			if (abs(angles[j] - angles[k]) <= angleThrhold) {
				//距离符合?
				if (isClose(ses[j], ses[k])) {
					//这一对线可相连.
					connectIds[j].push_back(k);
				}
			}
		}
	}

	//最后得到二维index, 表示 前和后可连的id.
	for (int j = connectIds.size() - 1; j >= 0; j--) {
		if (connectIds[j].size() <= 0) {

			continue;
		}

		for (int k = 0; k < connectIds[j].size(); k++) {
			contours[j].insert(contours[j].end(), contours[connectIds[j][k]].begin(), contours[connectIds[j][k]].end());
			contours[connectIds[j][k]].clear();
		}
	}

	contoursConnected.reserve(contours.size());
	for (int j = 0; j < contours.size(); j++) {
		if (contours[j].size() >= 4) {
			contoursConnected.emplace_back(contours[j]);
		}
	}


}

/// <summary>
/// 找到曲线的最小x,y 最大x,y;
/// </summary>
/// <param name="contour"></param>
/// <returns>point1(最小x,最大x),point2(最小y,最大y)</returns>
pair<Point, Point> findLineMinAndMax(vector<Point > contour) {
	pair<Point, Point> pair;
	float minX, maxX, minY, maxY;
	for (int i = 0; i < contour.size(); i++) {
		Point item = contour[i];
		if (i == 0) {
			minX = item.x;
			maxX = item.x;
			minY = item.y;
			maxY = item.y;
		}

		if (item.x < minX) {
			minX = item.x;
		}
		if (item.x > maxX) {
			maxX = item.x;
		}
		if (item.y < minY) {
			minY = item.y;
		}
		if (item.y > maxY) {
			maxY = item.y;
		}
	}
	pair.first = Point(minX, maxX);
	pair.second = Point(minY, maxY);
	return pair;
}


/// <summary>
/// 找到线的两端点.
/// </summary>
/// <param name="contour"></param>
/// <returns></returns>
pair<Point, Point> findStartEndPoint(vector<Point > contour) {
	if (contour.size() <= 0) {
		return pair<Point, Point>(Point(-99, -99), Point(-99, -99));
	}

	pair<Point, Point> pair;
	//记录一点有几个相邻的点.
	int count = 0;
	//记录找到了几个端点了
	int resultCount = 0;
	for (int i = 0; i < contour.size(); i++) {
		Point ip;
		count = 0;
		ip = contour[i];
		for (int j = 0; j < contour.size(); j++) {
			//两点距离
			double dist = PointDist(ip, contour[j]);

			if (dist <= sqrt(2) && dist > 0) {
				//算一个相邻的点
				count++;
			}
			//如果相邻点有两个以上,则i点不是端点,继续下一个点.
			if (count >= 2) {
				break;
			}
		}
		//和所有点比较完毕,如果只有一个相邻点,则认为是候选端点.
		if (count == 1) {

			if (resultCount == 0) {
				pair.first = ip;
				resultCount++;
			}
			else if (resultCount == 1) {
				pair.second = ip;
				resultCount++;
				return pair;
			}
			else {
				return pair;
			}
		}
	}
	//异常处理.如果端点数不是两个的情况.
	if (resultCount == 0) {//没有端点,则设置数据
		pair.first = contour[0];
		pair.second = contour[contour.size() - 1];
	}
	else if (resultCount == 1) {
		//少一个端点
		pair.second = contour[0];
	}

	return pair;
}

double PointDist(Point p1, Point p2) {
	return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

/// <summary>
/// 判断两线是否是延伸关系.
/// </summary>
/// <param name="pair1"></param>
/// <param name="pair2"></param>
/// <returns></returns>
bool isExtend(pair<Point, Point> pair1, pair<Point, Point> pair2) {
	//距离阈值
	double distThrhold = 20; // 小于sqrt(8*8+8*8)

	Point jFirst = pair1.first, jSecond = pair1.second;
	Point kFirst = pair2.first, kSecond = pair2.second;
	//有一对端点靠的近.
	vector<double> dists;
	dists.emplace_back(PointDist(jFirst, kFirst));
	dists.emplace_back(PointDist(jFirst, kSecond));
	dists.emplace_back(PointDist(jSecond, kFirst));
	dists.emplace_back(PointDist(jSecond, kSecond));
	sort(dists.begin(), dists.end());
	if (dists[0] <= distThrhold) {//近点小于阈值,且原点大于两个长度和.保证了两线是延伸关系. && (*dists.end()) >= (PointDist(jFirst, jSecond) + PointDist(kFirst, kSecond))
		return true;
	}
	return false;
}


/// <summary>
/// 判断两线是否相近.
/// </summary>
/// <param name="pair1"></param>
/// <param name="pair2"></param>
/// <returns></returns>
bool isClose(pair<Point, Point> pair1, pair<Point, Point> pair2) {
	//距离阈值
	double distThrhold = 20; // 小于sqrt(8*8+8*8)

	Point jFirst = pair1.first, jSecond = pair1.second;
	Point kFirst = pair2.first, kSecond = pair2.second;
	//有一对端点靠的近.
	vector<double> dists;
	dists.emplace_back(PointDist(jFirst, kFirst));
	dists.emplace_back(PointDist(jFirst, kSecond));
	dists.emplace_back(PointDist(jSecond, kFirst));
	dists.emplace_back(PointDist(jSecond, kSecond));
	sort(dists.begin(), dists.end());
	if (dists[0] <= distThrhold) {//近点小于阈值,且原点大于两个长度和.保证了两线是延伸关系. && (*dists.end()) >= (PointDist(jFirst, jSecond) + PointDist(kFirst, kSecond))
		return true;
	}
	return false;
}


double getLineWeight(vector<Point> line) {
	double minWeight = 0.2;
	RotatedRect rrect = minAreaRect(line);
	Rect rect = rrect.boundingRect();
	double ratio = min((double)rect.width, (double)rect.height) / max((double)rect.width, (double)rect.height);
	double weight = exp(log(minWeight) * ratio) + (1 - minWeight) / 2; // 控制weight在1.4--0.6
	return weight;
}

vector<Vec4f> findLine(Mat& gray) {

	GaussianBlur(gray, gray, Size(15, 15), 1, 1);
	//大小阈值
	double min_size_img = min(gray.cols, gray.rows) * 0.12;
	double min_size_grid = 1.41 * GRID_SIZE;
	double min = max(min_size_grid, min_size_img);
	Ptr<FastLineDetector> fld = createFastLineDetector(min, 1.414213538F, 50.0, 50.0, 3, true);
	vector<Vec4f> lines_std;
	fld->detect(gray, lines_std);

	//delectParallaxAndNear(lines_std);

	Mat imageContours = Mat::zeros(gray.size(), CV_8UC3);//输出图
	fld->drawSegments(imageContours, lines_std);

	return lines_std;
}
