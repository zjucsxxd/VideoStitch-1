
#include "stdafx.h"

#include <iostream>
#include <cmath>

#include "opencv2/features2d/features2d.hpp" 
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include "apap.h"


using namespace std;

APAPWarper::APAPWarper()
{
	cell_height_ = 15;
	cell_width_ = 20;
	gamma_ = 0.05;
	sigma_ = 8.5;

	H_s_ = NULL;
}


/*
 *	
 */
int APAPWarper::buildMaps(Mat src_img, ImageFeatures src_features, ImageFeatures dst_features, 
	MatchesInfo matches_info, Mat &xmap, Mat &ymap, Point &corner )
{
	int matches_size = matches_info.matches.size();

	/*
	 * 初始化矩阵A
	 */
	InitA(src_features, dst_features, matches_info);
	
	/*
	 * 每一个cell做一次DLT
	 */
	int inliner_nums = matches_info.num_inliers;
	// 初始化W_
	W_ = Mat_<apap_float>::zeros(2 * inliner_nums, 2 * inliner_nums);

	// 存放每一个cell对应的H
	if(H_s_ != NULL)
	{
		for(int i = 0; i < cell_rows_; i++)
			delete []H_s_[i];
		delete []H_s_;
		H_s_ = NULL;
	}
	int src_rows = src_img.rows;
	int src_cols = src_img.cols;
	cell_rows_ = cvCeil((double)(src_rows) / cell_height_);
	cell_cols_ = cvCeil((double)(src_cols) / cell_width_);

	H_s_ = new Mat_<apap_float> *[cell_rows_];
	for(int i = 0; i < cell_rows_; i++)
		H_s_[i] = new Mat_<apap_float> [cell_cols_];
	Mat_<apap_float> H(3, 3);

	for(int x = 0; x < src_cols; x += cell_width_)
		for(int y = 0; y < src_rows; y += cell_height_)
			CellDLT(x, y, src_features, dst_features, matches_info, H_s_[y / cell_height_][x / cell_width_]);

	cout << "apap: MDLT completed" << endl;

	/*
	 * Remap
	 * 反向映射（Reverse mapping，结果图片中的每个点对应src的坐标）
	 */
	//	1、计算映射结果图片的尺寸，只需要把四条边映射一下就可以了
	Point2d corner_2d(H_s_[0][0](0, 2) / H_s_[0][0](2, 2), H_s_[0][0](1, 2) / H_s_[0][0](2, 2));
	Point2d br(corner_2d);
	for(int x = 0; x < src_cols; x ++)
	{
		//上面的边
		updateMappedSize(H_s_[0][x / cell_width_], x, 0, corner_2d, br);
		//下面的边
		updateMappedSize(H_s_[(src_rows-1) / cell_height_][x / cell_width_], 
			x, src_rows - 1, corner_2d, br);
	}

	for(int y = 0; y < src_rows; y ++)
	{
		// 左边
		updateMappedSize(H_s_[y / cell_height_][0], 0, y, corner_2d, br);
		// 右边
		updateMappedSize(H_s_[y / cell_height_][(src_cols-1) / cell_width_],
			src_cols - 1, y, corner_2d, br);
	}

	cout << corner_2d << endl;

	//	2、反向映射
	corner.x = cvFloor(corner_2d.x);
	corner.y = cvFloor(corner_2d.y);
	int result_width = cvCeil(br.x) + 1 - corner.x;
	int result_height = cvCeil(br.y) + 1 - corner.y;

	xmap.create(result_height, result_width, CV_32F);
	ymap.create(result_height, result_width, CV_32F);
	float *xmap_rev_ptr = xmap.ptr<float>(0);
	float *ymap_rev_ptr = ymap.ptr<float>(0);
	for(int i = 0; i < result_width * result_height; i++)
		xmap_rev_ptr[i] = ymap_rev_ptr[i] = -1;

	for(int x = 0; x < src_cols; x += cell_width_)
	{
		for(int y = 0; y < src_rows; y += cell_height_)
		{
			int xx_max = std::min(x + cell_width_, src_cols);
			int yy_max = std::min(y + cell_height_, src_rows);
			BuildCellMap(x, y, xx_max, yy_max, H_s_[y / cell_height_][x / cell_width_], corner, 
				xmap, ymap);
		}
	}

	cout << "apap: build map completed" << endl;

	return 0;
}

int APAPWarper::buildMaps( vector<Mat> imgs, vector<ImageFeatures> features, 
	vector<MatchesInfo> pairwise_matches, vector<Mat> &xmaps, vector<Mat> &ymaps, vector<Point> &corners )
{
	int n = imgs.size();

	int mid_idx = n / 2;

	// 基准图片的map
	int mid_img_width = imgs[mid_idx].cols;
	int mid_img_height = imgs[mid_idx].rows;
	xmaps[mid_idx].create(imgs[mid_idx].size(), CV_32F);
	ymaps[mid_idx].create(imgs[mid_idx].size(), CV_32F);
	float *xmap_rev_ptr = xmaps[mid_idx].ptr<float>(0);
	float *ymap_rev_ptr = ymaps[mid_idx].ptr<float>(0);
	for(int y = 0; y < mid_img_height; y++)
	{
		for(int x = 0; x < mid_img_width; x++)
		{
			int idx = y * mid_img_width + x;
			xmap_rev_ptr[idx] = (float)x;
			ymap_rev_ptr[idx] = (float)y;
		}
	}
	corners[mid_idx].x = 0;
	corners[mid_idx].x = 0;

	for(int i = mid_idx+1; i >= (mid_idx-1); i--)
	{
		if(i == mid_idx)
			continue;
		int pair_idx = i * n + mid_idx;
		buildMaps(imgs[i], features[i], features[mid_idx], pairwise_matches[pair_idx], 
			xmaps[i], ymaps[i], corners[i]);
	}

	if(n < 4)
		return 0;

	ImageFeatures features_warped;
	int feat_num = features[1].keypoints.size();
	features_warped.keypoints.resize(feat_num);
	features_warped.img_idx = features[1].img_idx;
	features_warped.img_size = features[1].img_size;
	Mat_<apap_float> tmp_X(3, 1), tmp_res_X;
	for(int i = 0; i < feat_num; i++)
	{
		tmp_X(0, 0) = features[1].keypoints[i].pt.x;
		tmp_X(1, 0) = features[1].keypoints[i].pt.y;
		tmp_X(2, 0) = 1;
		int x = cvRound(tmp_X(0, 0));
		int y = cvRound(tmp_X(1, 0));
		tmp_res_X = H_s_[y / cell_height_][x / cell_width_] * tmp_X;
		features_warped.keypoints[i].pt.x = (double)(tmp_res_X(0, 0) / tmp_res_X(2, 0) - corners[1].x);
		features_warped.keypoints[i].pt.y = (double)(tmp_res_X(1, 0) / tmp_res_X(2, 0) - corners[1].y);
	}
	buildMaps(imgs[0], features[0], features_warped, pairwise_matches[1],
		xmaps[0], ymaps[0], corners[0]);
	corners[0].x += corners[1].x;
	corners[0].y += corners[1].y;

	return 0;
}


/*
 * src_img向dst_img配准
 *
 */
int APAPWarper::warp( Mat src_img, ImageFeatures src_features, ImageFeatures dst_features, 
	MatchesInfo matches_info, Mat &result_img, Point &corner )
{
	Mat xmap_rev, ymap_rev;

	//	1、建立映射
	buildMaps(src_img, src_features, dst_features, matches_info,
		xmap_rev, ymap_rev, corner);

	//	2、remap（TODO）
	remap(src_img, result_img, xmap_rev, ymap_rev, INTER_LINEAR);

	return 0;
}

int APAPWarper::CellDLT( int offset_x, int offset_y, ImageFeatures src_features, 
	ImageFeatures dst_features, MatchesInfo matches_info, Mat_<apap_float> &H )
{
	int matches_size = matches_info.matches.size();
	int inliner_nums = matches_info.num_inliers;
	
	double center_x = offset_x + cell_width_ / 2;
	double center_y = offset_y + cell_height_ / 2;

	for(int j = 0, inliner_idx = 0; j < matches_size; j ++)
	{
		if (!matches_info.inliers_mask[j])
			continue;

		const DMatch& m = matches_info.matches[j];
		Point2f p1 = src_features.keypoints[m.queryIdx].pt;
		Point2f p2 = dst_features.keypoints[m.trainIdx].pt;

		double weight = exp((-1 / (sigma_ * sigma_)) * 
			((center_x-p1.x) * (center_x-p1.x) + (center_y-p1.y) * (center_y-p1.y)) );
		int i = inliner_idx * 2;
		W_(i, i) = W_(i+1, i+1) = std::max(weight, gamma_);
		inliner_idx ++;
	}

	Mat WA = W_ * A_;
	Mat_<apap_float> h(9, 1);
	SVD::solveZ(WA, h);
	
	/*
	printf("(");
	for(int i = 0; i < 9; i++)
		printf("%lf, ", H(i, 0) / H(8, 0));
	printf(")\n");
	*/
	if(H.empty())
		H.create(3, 3);
	apap_float *H_ptr = H.ptr<apap_float>(0);
	apap_float *h_ptr = h.ptr<apap_float>(0);
	for(int i = 0; i < 9; i++)
		H_ptr[i] = h_ptr[i];
	
	return 0;
}

void APAPWarper::InitA( ImageFeatures src_features, ImageFeatures dst_features, MatchesInfo matches_info )
{
	printf("init A\n");
	int matches_size = matches_info.matches.size();
	int inliner_nums = matches_info.num_inliers;
	A_ = Mat_<apap_float>::zeros(2 * inliner_nums, 9);

	Mat src_points(1, inliner_nums, CV_32FC2);
	Mat dst_points(1, inliner_nums, CV_32FC2);

	for(int j = 0, inliner_idx = 0; j < matches_size; j++)
	{
		if (!matches_info.inliers_mask[j])
			continue;

		const DMatch& m = matches_info.matches[j];
		Point2f p1 = src_features.keypoints[m.queryIdx].pt;
		Point2f p2 = dst_features.keypoints[m.trainIdx].pt;
		src_points.at<Point2f>(0, inliner_idx) = p1;
		dst_points.at<Point2f>(0, inliner_idx) = p2;
		
		int i = inliner_idx * 2;

		A_(i, 0) = A_(i, 1) = A_(i, 2) = 0;
		A_(i, 3) = -p1.x;
		A_(i, 4) = -p1.y;
		A_(i, 5) = -1;
		A_(i, 6) = p2.y * p1.x;
		A_(i, 7) = p2.y * p1.y;
		A_(i, 8) = p2.y;
		
		A_(i + 1, 0) = p1.x;
		A_(i + 1, 1) = p1.y;
		A_(i + 1, 2) = 1;
		A_(i + 1, 3) = A_(i + 1, 4) = A_(i + 1, 5) = 0;
		A_(i + 1, 6) = -p2.x * p1.x;
		A_(i + 1, 7) = -p2.x * p1.y;
		A_(i + 1, 8) = -p2.x;

		inliner_idx ++;
	}
}

void APAPWarper::testh( ImageFeatures src_features, ImageFeatures dst_features, 
	MatchesInfo matches_info, Mat h )
{
	int matches_size = matches_info.matches.size();
	double *h_ptr = h.ptr<double>(0);
	for(int i = 0; i < matches_size; i ++)
	{
		if (!matches_info.inliers_mask[i])
			continue;
		const DMatch& m = matches_info.matches[i];
		Point2f p1 = src_features.keypoints[m.queryIdx].pt;
		Point2f p2 = dst_features.keypoints[m.trainIdx].pt;

		double _z = h_ptr[6] * p1.x + h_ptr[7] * p1.y + h_ptr[8];
		double _y = h_ptr[3] * p1.x + h_ptr[4] * p1.y + h_ptr[5];
		double _x = h_ptr[0] * p1.x + h_ptr[1] * p1.y + h_ptr[2];

		_x = _x / _z;
		_y = _y / _z;

		if(i < 20)
			printf("%.7lf, %.7lf => %.7lf, %.7lf\n", p2.x, p2.y, _x, _y);
	}
	printf("\n");
}

void APAPWarper::updateMappedSize( Mat_<apap_float> H, double x, double y, 
	Point2d &corner, Point2d &br )
{
	double _z = H(2, 0) * x + H(2, 1) * y + H(2, 2);
	double _y = (H(1, 0) * x + H(1, 1) * y + H(1, 2)) / _z;
	double _x = (H(0, 0) * x + H(0, 1) * y + H(0, 2)) / _z;

	corner.x = std::min(corner.x, _x);
	corner.y = std::min(corner.y, _y);
	br.x = std::max(br.x, _x);
	br.y = std::max(br.y, _y);
}

void APAPWarper::BuildCellMap( int x_1, int y_1, int x_2, int y_2, Mat H, Point corner, Mat &xmap, Mat &ymap )
{
	float *xmap_rev_ptr = xmap.ptr<float>(0);
	float *ymap_rev_ptr = ymap.ptr<float>(0);
	int res_width = xmap.size().width;
	int res_height = xmap.size().height;

	Mat_<apap_float> tmp_corner_X[4], tmp_corner_res_X[4];	// 四个角的坐标
	for(int i = 0; i < 4; i++)
		tmp_corner_X[i] = Mat_<apap_float>::ones(3, 1);
	tmp_corner_X[0](0, 0) = tmp_corner_X[3](0, 0) = x_1;
	tmp_corner_X[0](1, 0) = tmp_corner_X[1](1, 0) = y_1;
	tmp_corner_X[1](0, 0) = tmp_corner_X[2](0, 0) = x_2;
	tmp_corner_X[2](1, 0) = tmp_corner_X[3](1, 0) = y_2;

	for(int i = 0; i < 4; i++)
	{
		tmp_corner_res_X[i] = H * tmp_corner_X[i];
		tmp_corner_res_X[i] = tmp_corner_res_X[i] / tmp_corner_res_X[i](2, 0);
	}
	//cout << "1" << endl;
	// 先找到x和y的范围
	double y_min_d, y_max_d;
	double x_min_d, x_max_d;
	y_min_d = y_max_d = tmp_corner_res_X[0](1, 0);
	x_min_d = x_max_d = tmp_corner_res_X[0](0, 0);
	for(int i = 1; i < 4; i++)
	{
		double y = tmp_corner_res_X[i](1, 0);
		double x = tmp_corner_res_X[i](0, 0);
		y_min_d = std::min(y, y_min_d);
		y_max_d = std::max(y, y_max_d);
		x_min_d = std::min(x, x_min_d);
		x_max_d = std::max(x, x_max_d);
	}

	//cout << "2" << endl;
	// 扫描线
	int y_min = cvFloor(y_min_d), y_max = cvCeil(y_max_d);
	double x1, x2, y1, y2, x_start_d, x_end_d;
	int x_start, x_end;
	Mat_<apap_float> H_inv = H.inv(), tmp_X = Mat_<apap_float>::ones(3, 1), tmp_mapped_X(3, 1);
	double delta = 1.2;
	for(int y = y_min; y <= y_max; y++)
	{
		// 与4条线的交点
		x_start_d = x_max_d;
		x_end_d = x_min_d;
		for(int i = 0; i < 4; i++)
		{
			x2 = tmp_corner_res_X[i](0, 0);
			y2 = tmp_corner_res_X[i](1, 0);
			x1 = tmp_corner_res_X[(i+1) % 4](0, 0);
			y1 = tmp_corner_res_X[(i+1) % 4](1, 0);
			if(std::abs(y1 - y2) < 0.2)
				continue;

			if((y > (y1+delta) && y > (y2+delta)) || (y < (y1-delta) && y < (y2-delta)))
				continue;
			double p_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1);
			x_start_d = std::min(x_start_d, p_x);
			x_end_d = std::max(x_end_d, p_x);
		}
		x_start = cvFloor(x_start_d);
		x_end = cvCeil(x_end_d);
		tmp_X(1, 0) = y;
		//cout << "3" << endl;
		for(int x = x_start; x <= x_end; x++)
		{
			int x_idx = x - corner.x;
			int y_idx = y - corner.y;
			if(x_idx >= res_width || y_idx >= res_height || x_idx < 0 || y_idx < 0)
				continue;
			tmp_X(0, 0) = x;
			tmp_mapped_X = H_inv * tmp_X;
			int idx = y_idx * res_width + x_idx;
			xmap_rev_ptr[idx] = (float)(tmp_mapped_X(0, 0) / tmp_mapped_X(2, 0));
			ymap_rev_ptr[idx] = (float)(tmp_mapped_X(1, 0) / tmp_mapped_X(2, 0));
		}
	}

}

APAPWarper::~APAPWarper()
{
	if(H_s_ != NULL)
	{
		for(int i = 0; i < cell_rows_; i++)
			delete []H_s_[i];
		delete []H_s_;
		H_s_ = NULL;
	}
}