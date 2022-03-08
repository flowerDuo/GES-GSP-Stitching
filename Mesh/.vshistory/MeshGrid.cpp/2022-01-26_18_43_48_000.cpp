//
//  MeshGrid.cpp
//  UglyMan_Stitching
//
//  Created by uglyman.nothinglo on 2015/8/15.
//  Copyright (c) 2015 nothinglo. All rights reserved.
//

#include "./MeshGrid.h"
#include "./Mesh2D.cpp"

const int GRID_VERTEX_SIZE = 4;

MeshGrid::MeshGrid(const int _cols, const int _rows) : Mesh2D(_cols, _rows) {

}

/*根据网格分割数据,对应为图像的像素点.将网格点对应的 像素点 存入 列表中.*/
const vector<Point2>& MeshGrid::getVertices() const {
	if (vertices.empty()) {
		const int memory = (nh + 1) * (nw + 1);
		vertices.reserve(memory);
		for (int h = 0; h <= nh; ++h) {
			for (int w = 0; w <= nw; ++w) {
				vertices.emplace_back(w * lw, h * lh);
			}
		}
		assert(memory == vertices.size());
	}
	return vertices;
}

/*行优先顺序编号:将网格中所有相邻的点的边加入 列表.
顺序是:点是行优先顺序遍历,每个点先右后下边加入 列表.
*/
const vector<Edge>& MeshGrid::getEdges() const {
	if (edges.empty()) {
		const vector<Point2i> nexts = { Point2i(1, 0), Point2i(0, 1) };
		const int memory = DIMENSION_2D * nh * nw + nh + nw;
		edges.reserve(memory);
		for (int h = 0; h <= nh; ++h) {
			for (int w = 0; w <= nw; ++w) {
				const Point2i p1(w, h); //网格的每个顶点.
				for (int n = 0; n < nexts.size(); ++n) {
					const Point2i p2 = p1 + nexts[n]; //网格点p1的右/下 点.
					if (p2.x >= 0 && p2.y >= 0 && p2.x <= nw && p2.y <= nh) {
						//行优先顺序编号:p1的index 到 p2的index;
						edges.emplace_back(p1.x + p1.y * (nw + 1),
							p2.x + p2.y * (nw + 1));
					}
				}
			}
		}
		assert(memory == edges.size());
	}
	return edges;
}

/// <summary>
/// 返回每个网格4个点的index
/// </summary>
/// <returns></returns>
//放入vector的点的index数据的顺序:
//<image url="$(ProjectDir)CommentImage\MeshGrid-getPolygonsIndices.jpg" scale="0.05"/>
const vector<Indices>& MeshGrid::getPolygonsIndices() const {
	if (polygons_indices.empty()) {
		const Point2i nexts[GRID_VERTEX_SIZE] = {
			Point2i(0, 0), Point2i(1, 0), Point2i(1, 1), Point2i(0, 1)
		};
		const int memory = nh * nw;
		polygons_indices.resize(memory);
		int index = 0;
		for (int h = 0; h < nh; ++h) {
			for (int w = 0; w < nw; ++w) {
				const Point2i p1(w, h);
				polygons_indices[index].indices.reserve(GRID_VERTEX_SIZE);
				for (int n = 0; n < GRID_VERTEX_SIZE; ++n) {
					const Point2i p2 = p1 + nexts[n];
					polygons_indices[index].indices.emplace_back(p2.x + p2.y * (nw + 1));
				}
				++index;
			}
		}
		assert(memory == polygons_indices.size());
	}
	return polygons_indices;
}

// <image url="$(ProjectDir)CommentImage\MeshGrid-getPolygonsNeighbors.png" scale="2"/>
/// <summary>
/// 返回每个网格点的左上角点的4邻居点的index 列表.
/// </summary>
/// <returns></returns>
const vector<Indices>& MeshGrid::getPolygonsNeighbors() const {
	if (polygons_neighbors.empty()) {
		const vector<Point2i> nexts = {
			Point2i(1, 0), Point2i(0, 1), Point2i(-1, 0), Point2i(0, -1)
		};
		const int memory = nh * nw;
		polygons_neighbors.resize(memory);
		int index = 0;
		for (int h = 0; h < nh; ++h) {
			for (int w = 0; w < nw; ++w) {
				const Point2i p1(w, h);
				for (int n = 0; n < nexts.size(); ++n) {
					const Point2i p2 = p1 + nexts[n];
					if (p2.x >= 0 && p2.y >= 0 && p2.x < nw && p2.y < nh) {
						polygons_neighbors[index].indices.emplace_back(p2.x + p2.y * nw);
					}
				}
				++index;
			}
		}
		assert(memory == polygons_neighbors.size());
	}
	return polygons_neighbors;
}
const vector<Indices>& MeshGrid::getPolygonsEdges() const {
	if (polygons_edges.empty()) {
		const vector<int> nexts = {
			0, 1, 3, nw * 2 + 1
		};
		const int memory = nh * nw;
		polygons_edges.resize(memory);
		int index = 0, e_index = 0;;
		for (int h = 0; h < nh; ++h) {
			for (int w = 0; w < nw; ++w) {
				for (int n = 0; n < nexts.size(); ++n) {
					polygons_edges[index].indices.emplace_back(e_index + nexts[n]);
				}
				polygons_edges[index].indices.back() = polygons_edges[index].indices.back() - (h == nh - 1) * w;
				++index;
				e_index += 2;
			}
			polygons_edges[index - 1].indices[2] = polygons_edges[index - 1].indices[2] - 1;
			e_index += 1;
		}
		assert(memory == polygons_edges.size());
	}
	return polygons_edges;
}

/*行优先顺序存放每个点的4周点信息:最终保存数据结构为:
 vertex_structures[i] == indices: indices 是个int列表,存放 第i个点 右下左上4个点的 index ;
//<image url="$(ProjectDir)CommentImage\MeshGrid-getVertexStructures.png" scale="1" />
*/
const vector<Indices>& MeshGrid::getVertexStructures() const {
	if (vertex_structures.empty()) {
		const vector<Point2i> nexts = {
			Point2i(1, 0), Point2i(0, 1), Point2i(-1, 0), Point2i(0, -1)
		};
		const int memory = (nh + 1) * (nw + 1);
		vertex_structures.resize(memory);
		int index = 0;
		for (int h = 0; h <= nh; ++h) {
			for (int w = 0; w <= nw; ++w) {
				Point2i p1(w, h);
				for (int n = 0; n < nexts.size(); ++n) {
					Point2i p2 = p1 + nexts[n];
					if (p2.x >= 0 && p2.y >= 0 && p2.x <= nw && p2.y <= nh) {
						vertex_structures[index].indices.emplace_back(p2.x + p2.y * (nw + 1));
					}
				}
				++index;
			}
		}
		assert(memory == vertex_structures.size());
	}
	return vertex_structures;
}
const vector<Indices>& MeshGrid::getEdgeStructures() const {
	if (edge_structures.empty()) {
		const vector<Point2i>         nexts = { Point2i(1,  0), Point2i(0, 1) };
		const vector<Point2i> grid_neighbor = { Point2i(0, -1), Point2i(-1, 0) };
		const int memory = DIMENSION_2D * nh * nw + nh + nw;
		edge_structures.resize(memory);
		int index = 0;
		for (int h = 0; h <= nh; ++h) {
			for (int w = 0; w <= nw; ++w) {
				Point2i p1(w, h);
				for (int n = 0; n < nexts.size(); ++n) {
					//p2依次是此网格的右网格和下网格.
					Point2i p2 = p1 + nexts[n];
					if (p2.x >= 0 && p2.y >= 0 && p2.x <= nw && p2.y <= nh) {//此网格点不是最右下的网格
						for (int j = 0; j < grid_neighbor.size(); ++j) {
							Point2i p3 = p1 + grid_neighbor[n] * j;
							if (p3.x >= 0 && p3.y >= 0 && p3.x < nw && p3.y < nh) {
								edge_structures[index].indices.emplace_back(p3.x + p3.y * nw);
							}
						}
						++index;
					}
				}
			}
		}
		assert(memory == edge_structures.size());
	}
	return edge_structures;
}

/// <summary>
/// [ [0,1,2],[0,2,3] ] 
/// </summary>
/// <returns></returns>
const vector<Indices>& MeshGrid::getTriangulationIndices() const {
	if (triangulation_indices.empty()) {
		triangulation_indices.emplace_back(0, 1, 2);
		triangulation_indices.emplace_back(0, 2, 3);
	}
	return triangulation_indices;
}

const int& MeshGrid::getPolygonVerticesCount() const {
	return GRID_VERTEX_SIZE;
}

const vector<int>& MeshGrid::getBoundaryVertexIndices() const {
	if (boundary_vertex_indices.empty()) {
		const int memory = DIMENSION_2D * (nw + nh) + 1;
		boundary_vertex_indices.reserve(memory);
		for (int tw = 0; tw < nw; ++tw) {
			boundary_vertex_indices.emplace_back(tw);
		}
		const int right_bottom = nw * nh + nw + nh;
		for (int rh = nw; rh < right_bottom; rh += (nw + 1)) {
			boundary_vertex_indices.emplace_back(rh);
		}
		const int left_bottom = nh * (nw + 1);
		for (int bw = right_bottom; bw > left_bottom; --bw) {
			boundary_vertex_indices.emplace_back(bw);
		}
		for (int lh = left_bottom; lh >= 0; lh -= (nw + 1)) {
			boundary_vertex_indices.emplace_back(lh);
		}
		assert(memory == boundary_vertex_indices.size());
	}
	return boundary_vertex_indices;
}

const vector<int>& MeshGrid::getBoundaryEdgeIndices() const {
	if (boundary_edge_indices.empty()) {
		const int memory = DIMENSION_2D * (nh + nw);
		boundary_edge_indices.reserve(memory);
		const int bottom_shift = DIMENSION_2D * nh * nw + nh;
		for (int w = 0; w < nw; ++w) {
			boundary_edge_indices.emplace_back(2 * w);
			boundary_edge_indices.emplace_back(bottom_shift + w);
		}
		const int dh = 2 * nw + 1;
		for (int h = 0; h < nh; ++h) {
			int tmp = h * dh;
			boundary_edge_indices.emplace_back(tmp + 1);
			boundary_edge_indices.emplace_back(tmp + dh - 1);
		}
		assert(memory == boundary_edge_indices.size());
	}
	return boundary_edge_indices;
}

template InterpolateVertex MeshGrid::getInterpolateVertexTemplate< float>(const Point_< float>& _p) const;
template InterpolateVertex MeshGrid::getInterpolateVertexTemplate<double>(const Point_<double>& _p) const;

/// <summary>
/// 通过传入的点信息,得到 _p点是属于第几个网格 和 _p点所属网格点的4个点的权值
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="_p"></param>
/// <returns>_p点是属于第几个网格 和 _p点所属网格点的4个点的权值</returns>
template <typename T>
InterpolateVertex MeshGrid::getInterpolateVertexTemplate(const Point_<T>& _p) const {
	const vector<Point2>& vertices = getVertices();
	//每个网格4个点的index
	const vector<Indices>& grids = getPolygonsIndices();


	//返回_p是第几个网格点
	const int grid_index = getGridIndexOfPoint(_p);
	//const int grid_index = 1;
	//返回_p点为左上角点的这个网格的4个点的index.
	const Indices& g = grids[grid_index];

	const vector<int> diagonal_indices = { 2, 3, 0, 1 }; /* 0 1    2 3
															  ->
														  3 2    1 0 */
	assert(g.indices.size() == GRID_VERTEX_SIZE);

	//根据
	vector<double> weights(GRID_VERTEX_SIZE);
	double sum_inv = 0;
	for (int i = 0; i < diagonal_indices.size(); ++i) {
		//distX, distY
		Point2 tmp(_p.x - vertices[g.indices[diagonal_indices[i]]].x,
			_p.y - vertices[g.indices[diagonal_indices[i]]].y);
		//weight = |distX*distY|
		weights[i] = fabs(tmp.x * tmp.y);
		sum_inv += weights[i];
	}
	// 1/权重和
	sum_inv = 1. / sum_inv;
	//权重正则化 ,控制在每个点对于_p点的权重和 为 1 
	for (int i = 0; i < GRID_VERTEX_SIZE; ++i) {
		weights[i] = weights[i] * sum_inv;
	}
	return InterpolateVertex(grid_index, weights);
}

/// <summary>
/// 通过传入的点信息,得到 _p点是属于第几个网格 和 _p点所属网格点的4个点的权值
/// </summary>
InterpolateVertex MeshGrid::getInterpolateVertex(const Point_<float>& _p) const {
	return getInterpolateVertexTemplate(_p);
}
/// <summary>
/// 通过传入的点信息,得到 _p点是属于第几个网格 和 _p点所属网格点的4个点的权值
/// </summary>
InterpolateVertex MeshGrid::getInterpolateVertex(const Point_<double>& _p) const {
	return getInterpolateVertexTemplate(_p);
}

InterpolateVertex MeshGrid::getInterpolateVertex(const Point_<int>& _p) const {
	return getInterpolateVertexTemplate(_p);
}
