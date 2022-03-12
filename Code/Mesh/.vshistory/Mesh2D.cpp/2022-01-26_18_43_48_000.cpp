//
//  Mesh2D.cpp
//  UglyMan_Stitching
//
//  Created by uglyman.nothinglo on 2015/8/15.
//  Copyright (c) 2015 nothinglo. All rights reserved.
//

#include "./Mesh2D.h"

/*传入图像的像素宽高,每格像素大小GRID_SIZE(40).不足一格的算一格.*/
Mesh2D::Mesh2D(const int _cols, const int _rows) {
	//nw:网格宽数; nh:网格高数; lw:
	nw = _cols / GRID_SIZE + (_cols % GRID_SIZE != 0);
	nh = _rows / GRID_SIZE + (_rows % GRID_SIZE != 0);
	lw = _cols / (double)nw;
	lh = _rows / (double)nh;
}
Mesh2D::~Mesh2D() {

}

template int Mesh2D::getGridIndexOfPoint<float>(const Point_<float>& _p) const;
template int Mesh2D::getGridIndexOfPoint<double>(const Point_<double>& _p) const;


/// <summary>
/// 得到每个网格的中心点坐标.
/// </summary>
/// <returns></returns>
const vector<Point2>& Mesh2D::getPolygonsCenter() const {
	if (polygons_center.empty()) {
		//得到自己网格点的坐标
		const vector<Point2>& vertices = getVertices();
		//得到每个网格的4个顶点的index
		const vector<Indices>& polygons_indices = getPolygonsIndices();
		polygons_center.reserve(polygons_indices.size());

		//得到每个网格的中心点的坐标.
		for (int i = 0; i < polygons_indices.size(); ++i) {//遍历每个网格.
			Point2 center(0, 0);
			for (int j = 0; j < polygons_indices[i].indices.size(); ++j) {
				center += vertices[polygons_indices[i].indices[j]];
			}
			polygons_center.emplace_back(center / (FLOAT_TYPE)polygons_indices[i].indices.size());
		}
	}
	return polygons_center;
}

/// <summary>
/// 返回_p点是第几个网格(左上角点)
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="_p"></param>
/// <returns></returns>
template <typename T>
int Mesh2D::getGridIndexOfPoint(const Point_<T>& _p) const {
	Point2i grid_p(_p.x / lw, _p.y / lh);
	grid_p.x = grid_p.x - (grid_p.x == nw);
	grid_p.y = grid_p.y - (grid_p.y == nh);
	return grid_p.x + grid_p.y * nw;
}
