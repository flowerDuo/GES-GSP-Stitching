﻿//
//  NISwGSP_Stitching.h
//  UglyMan_Stitching
//
//  Created by uglyman.nothinglo on 2015/8/15.
//  Copyright (c) 2015 nothinglo. All rights reserved.
//

#ifndef __UglyMan_Stitiching__NISwGSP_Stitching__
#define __UglyMan_Stitiching__NISwGSP_Stitching__

#include "../Mesh/MeshOptimization.h"

class NISwGSP_Stitching : public MeshOptimization {
public:
	NISwGSP_Stitching(const MultiImages& _multi_images);
	/*交给MeshOptimization来做*/
	void setWeightToAlignmentTerm(const double _weight);
	/*交给MeshOptimization来做*/
	void setWeightToLocalSimilarityTerm(const double _weight);
	/*交给MeshOptimization来做*/
	void setWeightToGlobalSimilarityTerm(const double _weight_beta,
		const double _weight_gamma,
		const enum GLOBAL_ROTATION_METHODS _global_rotation_method);
	void setWeightToContentPreservingTerm(const double _weight);
	/*正式开始*/
	Mat solve(const BLENDING_METHODS& _blend_method);
	Mat solve_content(const BLENDING_METHODS& _blend_method);
	/*写出图片*/
	void writeImage(const Mat& _image, const string _blend_method_name) const;
private:
};

#endif /* defined(__UglyMan_Stitiching__NISwGSP_Stitching__) */
