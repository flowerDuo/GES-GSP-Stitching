#include "Test.h"
#include <Eigen/Geometry> 

using namespace cv;
using namespace std;

/// <summary>
/// 测试三角形数学公式
/// </summary>
/// <returns></returns>
int TestSP::testTriangle(double startX = 0, double startY = 0, double endX = 6, double endY = 100, double sampleX = 33, double sampleY = -50)
{
	double clockStart = clock();

	double x1 = startX, y1 = startY; //start
	double x2 = endX, y2 = endY; //end
	double x3 = sampleX, y3 = sampleY; //sample
	cout << "采样点X" << sampleX << endl;
	cout << "采样点Y" << sampleY << endl;

	//1.三角形3个点. 起始点|终点|采样点
	Point2f a(x1, y1), b(x2, y2), c(x3, y3);

	//2.转化 start-->sample, start-->end 两个向量 为3维向量.
	Vector3d ab = trans2Vector(b - a), ac = trans2Vector(c - a);
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
	cout << "v:" << v << endl;
	cout << "u:" << u << endl;
	//7.u,v计算完成,扭曲前数据准备完成.

	//8.根据start,end,计算出样本点.
	double predictX3 = (1 - u) * x1 - v * y1 + u * x2 + v * y2;
	double predictY3 = (v * x1 + (1 - u) * y1 - v * x2 + u * y2);
	cout << "预测x:" << predictX3 << endl;
	cout << "预测y:" << predictY3 << endl;

	cout << "时间:" << clock() - clockStart << endl;
	return 0;
}


void TestSP::testContours()
{
	string path = R"(F:\Projects\C++\NISwGSP_Stitching\input-42-data\CAVE-times_square\02.jpg)";
	Mat imgRes = imread(path, 1);
	cout << imgRes.rows << "  " << imgRes.cols << endl;

	Mat image;
	//1.调整到HED规定图片大小.
	resize(imgRes, image, cv::Size(500, 500 * (double)imgRes.rows / imgRes.cols));
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	//2.HED图片边缘检测
	edgeDetection(image, image, 0.5);
	imshow("after EdgeDetection Image", image);
	//3.回归原本大小
	//resize(image, image, Size(imgRes.cols, imgRes.rows)); // TEST!!!临时注释
	//4.细化边缘图像
	thinTest(image, image, (double)imgRes.cols / 500);
	imshow("after Thin Image", image);

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
	imshow("after findcontours", imageContours); //轮廓  


	//5.1将同方向相近线段连接
	vector<vector<Point>> contoursLineConnected;
	connectSmallLine1(contours, hierarchy, contoursLineConnected);

	//5.2将拐度较大处断

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
	vector<Vec4f> lines = findLine1(gray);


	//7.取采样点
	Mat imageSamples = Mat::zeros(image.size(), CV_8UC3);//输出图
	//存放所有曲线的起终点,采样点数据.0:起点,1:终点,之后是采样点.
	vector<vector<Point>> samplesData;
	samplesData.reserve(contoursLineConnected.size() + lines.size());
	vector<double> weights;
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
		lineLength = PointDist1(start, end);
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
					if (sample == end || PointDist1(sample, end) <= lineLength / 2) {
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
				//circle(imageSamples, samplesData[index][i], 8, s);
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
		sort((*iterator).begin(), (*iterator).end(), sortForPoint1);
		(*iterator).erase(unique((*iterator).begin(), (*iterator).end(), equalForPoint1), (*iterator).end());

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
		weights.emplace_back(getLineWeight1(*iterator));
		RNG rng(cvGetTickCount());
		Scalar s = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		//输出图像
		for (int i = 0; i < samplesData[index].size(); i++) {
			if (i == 0 || i == 1) {
				circle(imageSamples, samplesData[index][i], 6, s, FILLED);
			}
			else {
				//circle(imageSamples, samplesData[index][i], 6, s);
			}
		}
		String weightStr = to_string(weights[index]);
		//putText(imageSamples, weightStr, samplesData[index][0], FONT_HERSHEY_COMPLEX, 0.35, Scalar(0, 255, 255));

		for (int j = 0; j < (*iterator).size(); j++) {
			circle(imageSamples, (*iterator)[j], 1, s, FILLED);
		}

		index++;
	}

	imshow("Samples", imageSamples);

	cv::waitKey(0);
	return;
}

void TestSP::testLineProcess() {

	string path = R"(F:\Projects\C++\NISwGSP_Stitching\input-42-data\CAVE-times_square\02.jpg)";
	Mat image = imread(path, IMREAD_GRAYSCALE);
	//4.1 角点检测
	std::vector<cv::Point2f> corners;

	int max_corners = 300;
	double quality_level = 0.1;
	double min_distance = 12.0;
	int block_size = 3;
	bool use_harris = true;
	cv::goodFeaturesToTrack(image,
		corners,
		max_corners,
		quality_level,
		min_distance,
		cv::Mat(),
		block_size,
		use_harris);
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

	//5.2将同方向相近线段连接
	vector<vector<Point>> contoursLineConnected;
	connectSmallLine(contours, hierarchy, contoursLineConnected);
	//6.1 连接共线直线.
	vector <double > lineslength; //每个线段的长度
	vector<vector<Point>> static_sample; //每个线段一定要加的样本点位置
	//存储排除后的曲线集.
	vector<vector<Point>> res;
	res = connectCollineationLine(res, lineslength, static_sample, image.cols, image.rows);
	Mat imageSamples = Mat::zeros(image.size(), CV_8UC3);//输出图
	for (vector<vector<Point>>::iterator iterator = res.begin(); iterator != res.end() && i < lineslength.size(); ++iterator) {
		RNG rng(cvGetTickCount());
		Scalar s = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		for (int j = 0; j < (*iterator).size(); j++) {
			circle(imageSamples, (*iterator)[j], 1, s, FILLED);
		}
	}
}

bool sortForPoint1(Point a, Point b) {
	return (a.x < b.x || (a.x == b.x && a.y < b.y));
}

bool equalForPoint1(Point a, Point b) {
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
void connectSmallLine1(vector<vector<Point>> contours, vector<Vec4i> hierarchy, vector<vector<Point>>& contoursConnected)
{
	for (vector<vector<Point>>::iterator iterator = contours.begin(); iterator != contours.end(); ++iterator) {
		sort((*iterator).begin(), (*iterator).end(), sortForPoint1);
		(*iterator).erase(unique((*iterator).begin(), (*iterator).end(), equalForPoint1), (*iterator).end());
	}



	//斜率阈值.20度
	double angleThrhold = (30.0 / 180) * acos(-1);

	//保存每条线的斜率
	vector<double> angles;
	//保存每条线的两端点
	vector< pair<Point, Point>> ses;

	//遍历所有线

	for (vector<vector<Point>>::iterator iterator = contours.begin(); iterator != contours.end();) {
		/*if ((*iterator).size() < 2) {
			iterator = contours.erase(iterator);
			continue;
		}*/

		Vec4f line_para;
		//得到拟合直线
		fitLine(*iterator, line_para, DIST_L2, 0, 1e-2, 1e-2);
		//得到线的斜率,转化为角度
		double k = line_para[1] / line_para[0];
		angles.push_back(atan(k));

		//找到一条线的端点
		pair<Point, Point> SEPoint = findStartEndPoint1(*iterator);
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
				if (isClose1(ses[j], ses[k])) {
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
/// 找到线的两端点.
/// </summary>
/// <param name="contour"></param>
/// <returns></returns>
pair<Point, Point> findStartEndPoint1(vector<Point > contour) {
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
			double dist = PointDist1(ip, contour[j]);

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

double PointDist1(Point p1, Point p2) {
	return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

/// <summary>
/// 判断两线是否相近.
/// </summary>
/// <param name="pair1"></param>
/// <param name="pair2"></param>
/// <returns></returns>
bool isClose1(pair<Point, Point> pair1, pair<Point, Point> pair2) {
	//距离阈值
	double distThrhold = 20; // 小于sqrt(8*8+8*8)

	Point jFirst = pair1.first, jSecond = pair1.second;
	Point kFirst = pair2.first, kSecond = pair2.second;
	//有一对端点靠的近.
	vector<double> dists;
	dists.emplace_back(PointDist1(jFirst, kFirst));
	dists.emplace_back(PointDist1(jFirst, kSecond));
	dists.emplace_back(PointDist1(jSecond, kFirst));
	dists.emplace_back(PointDist1(jSecond, kSecond));
	sort(dists.begin(), dists.end());
	if (dists[0] <= distThrhold) {//近点小于阈值,且原点大于两个长度和.保证了两线是延伸关系. && (*dists.end()) >= (PointDist1(jFirst, jSecond) + PointDist1(kFirst, kSecond))
		return true;
	}
	return false;
}

double getLineWeight1(vector<Point> line) {
	double minWeight = 0.2;
	RotatedRect rrect = minAreaRect(line);
	Rect rect = rrect.boundingRect();
	double ratio = min((double)rect.width, (double)rect.height) / max((double)rect.width, (double)rect.height);
	double weight = exp(log(minWeight) * ratio) + (1 - minWeight) / 2; // 控制weight在1.4--0.6
	return weight;
}


vector<double> getCurvature(std::vector<cv::Point> const& vecContourPoints, int step)
{
	std::vector< double > vecCurvature(vecContourPoints.size());

	if (vecContourPoints.size() < step)
		return vecCurvature;

	auto frontToBack = vecContourPoints.front() - vecContourPoints.back();

	bool isClosed = ((int)std::max(std::abs(frontToBack.x), std::abs(frontToBack.y))) <= 1;

	cv::Point2f pplus, pminus;
	cv::Point2f f1stDerivative, f2ndDerivative;
	for (int i = 0; i < vecContourPoints.size(); i++)
	{
		const cv::Point2f& pos = vecContourPoints[i];

		int maxStep = step;
		if (!isClosed)
		{
			maxStep = std::min(std::min(step, i), (int)vecContourPoints.size() - 1 - i);
			if (maxStep == 0)
			{
				vecCurvature[i] = std::numeric_limits<double>::infinity();
				continue;
			}
		}


		int iminus = i - maxStep;
		int iplus = i + maxStep;
		pminus = vecContourPoints[iminus < 0 ? iminus + vecContourPoints.size() : iminus];
		pplus = vecContourPoints[iplus > vecContourPoints.size() ? iplus - vecContourPoints.size() : iplus];


		f1stDerivative.x = (pplus.x - pminus.x) / (iplus - iminus);
		f1stDerivative.y = (pplus.y - pminus.y) / (iplus - iminus);
		f2ndDerivative.x = (pplus.x - 2 * pos.x + pminus.x) / ((iplus - iminus) / 2 * (iplus - iminus) / 2);
		f2ndDerivative.y = (pplus.y - 2 * pos.y + pminus.y) / ((iplus - iminus) / 2 * (iplus - iminus) / 2);

		double curvature2D;
		double divisor = f1stDerivative.x * f1stDerivative.x + f1stDerivative.y * f1stDerivative.y;
		if (std::abs(divisor) > 10e-8)
		{
			curvature2D = std::abs(f2ndDerivative.y * f1stDerivative.x - f2ndDerivative.x * f1stDerivative.y) /
				pow(divisor, 3.0 / 2.0);
		}
		else
		{
			curvature2D = std::numeric_limits<double>::infinity();
		}

		vecCurvature[i] = curvature2D;


	}
	return vecCurvature;
}

double dist(Point p, Vec4f l)
{
	//先算出三条边的长度a b c
	double a, b, c;
	double s;//面积
	double hl;//周长的一半
	double h;//距离
	a = sqrt(abs(p.x - l[0]) * abs(p.x - l[0]) + abs(p.y - l[1]) * abs(p.y - l[1]));
	b = sqrt(abs(p.x - l[2]) * abs(p.x - l[2]) + abs(p.y - l[3]) * abs(p.y - l[3]));
	c = sqrt(abs(l[0] - l[2]) * abs(l[0] - l[2]) + abs(l[1] - l[3]) * abs(l[1] - l[3]));
	hl = (a + b + c) / 2;
	s = sqrt(hl * (hl - a) * (hl - b) * (hl - c));
	h = (2 * s) / c;
	return h;
}

vector<Vec4f> delectParallaxAndNear(vector<Vec4f> lines) {
	vector<Vec4f> linesRes;
	linesRes.reserve(lines.size());
	double thresholdSlop = 5;
	double thresholdDist = 10;
	vector<double> slop;
	slop.reserve(lines.size());
	for (int i = 0; i < lines.size(); i++) {
		slop.emplace_back((lines[i][3] - lines[i][1]) / (lines[i][2] - lines[i][0]));
	}
	sort(slop.begin(), slop.end());
	for (int i = 0; i < slop.size(); i++) {
		if (i == 0) {
			linesRes.emplace_back(lines[i]);
		}
		else {
			if (abs(slop[i] - slop[i - 1]) <= thresholdSlop && dist(Point(lines[i - 1][0], lines[i - 1][1]), lines[i]) <= thresholdDist) {

			}
			else {
				linesRes.emplace_back(lines[i]);
			}
		}
	}
	return linesRes;
}

vector<Vec4f> findLine1(Mat& gray) {

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
	imshow("line detector", imageContours);

	return lines_std;
}

vector<Point>& getVector() {
	vector<Point> samples;
	samples.emplace_back(1, 1);
	samples.emplace_back(2, 2);
	return samples;
}

void TestSP::testVector()
{
	vector<vector<Point>> sampleSS;
	sampleSS.resize(2);
	sampleSS[1] = getVector();

	cout << "d " << endl;
}
