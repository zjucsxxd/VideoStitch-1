#ifndef __MY_VIDEO_STITCHER_H__
#define __MY_VIDEO_STITCHER_H__


#include <opencv2/opencv_modules.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching/detail/autocalib.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/camera.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/detail/util.hpp>
#include <opencv2/stitching/detail/warpers.hpp>
#include <opencv2/stitching/warpers.hpp>

#include "blender.h"
#include "exposure_compensator.h"

using namespace std;
using namespace cv;
using namespace cv::detail;

#define BUFFER_SIZE 1


/************************************************************************/
/*                          stitch status                               */
/************************************************************************/
#define STITCH_SUCCESS		0
#define STITCH_CONFIG_ERROR	-1
#define STITCH_NOISE		-2


typedef float* float_ptr;

class XYMap
{
public:
	vector<Mat> xmaps;
	vector<Mat> ymaps;
	int index;
};


class MyStitchStatus
{
public:
protected:
private:
};


class MyVideoStitcher
{
public:
	MyVideoStitcher();
	~MyVideoStitcher();

	void setPreview(bool is_preview) { is_preview_ = is_preview; };
	void setSave(bool is_save) { is_save_video_ = is_save; };
	void setRange(int start, int end = -1) { start_frame_index_ = std::max(1, start) - 1; end_frame_index_ = end; };
	void setTryGPU(bool try_gpu) { is_try_gpu_ = try_gpu; };
	void setTrim(bool is_trim) {
		if(is_trim)
			trim_type_ = MyVideoStitcher::TRIM_AUTO;
		else
			trim_type_ = MyVideoStitcher::TRIM_NO;
	};
	void setTrim(Rect trim_rect) { trim_rect_ = trim_rect; trim_type_ = MyVideoStitcher::TRIM_RECTANGLE; };
	void setWarpType(string warp_type) { warp_type_ = warp_type; };

	int stitch(vector<VideoCapture> &captures, string &writer_file_name);
	int stitchImage(vector<Mat> &src, Mat &pano);

	void setDebugDirPath(string dir_path);

	void saveCameraParam(string filename);
	int loadCameraParam(string filename);

protected:


private:
	int Prepare(vector<Mat> &src);
	int PrepareAPAP(vector<Mat> &src);
	int PrepareClassical(vector<Mat> &src);
	int StitchFrame(vector<Mat> &src, Mat &dst);
	int StitchFrameCPU(vector<Mat> &src, Mat &dst);
	int StitchFrameGPU(vector<Mat> &src, Mat &dst);

	void InitMembers(int num_images);

	/*
	 * 计算一些放缩的尺度，在特征检测和计算接缝的时候，为了提高程序效率，可以对源图像进行一些放缩
	 */
	void SetScales(vector<Mat> &src);

	int FindFeatures(vector<Mat> &src, vector<ImageFeatures> &features);
	
	/*
	 * 特征匹配，然后去除噪声图片。本代码实现时，一旦出现噪声图片，就终止算法
	 * 返回值：
	 *		0	——	正常
	 *		-2	——	存在噪声图片
	 */
	int MatchImages(vector<ImageFeatures> &features, vector<MatchesInfo> &pairwise_matches);

	/*
	 * 摄像机标定
	 */
	int CalibrateCameras(vector<ImageFeatures> &features,
		vector<MatchesInfo> &pairwise_matches, vector<CameraParams> &cameras);

	/*
	 *	计算水平视角
	 */
	double GetViewAngle(vector<Mat> &src, vector<CameraParams> &cameras);


	/*
	 * 为接缝的计算做Warp
	 */
	int WarpForSeam(vector<Mat> &src, vector<CameraParams> &cameras,
		vector<Mat> &masks_warped, vector<Mat> &images_warped);

	/*
	 * 计算接缝
	 */
	int FindSeam(vector<Mat> &images_warped, vector<Mat> &masks_warped);

	/*
	 *	把摄像机参数和masks还原到正常大小
	 */
	int Rescale(vector<Mat> &src, vector<CameraParams> &cameras, vector<Mat> &seam_masks);

	int RegistEvaluation(vector<ImageFeatures> &features,
		vector<MatchesInfo> &pairwise_matches, vector<CameraParams> &cameras);

	/*
	 *	解决360°拼接问题。对于横跨360°接缝的图片，找到最宽的inpaint区域[x1, x2]
	 */
	int FindWidestInpaintRange(Mat mask, int &x1, int &x2);

	/*
	 * 裁剪掉inpaint区域
	 */
	int TrimRect(Rect rect);
	int TrimInpaint(vector<Mat> &src);
	bool IsRowCrossInpaint(uchar *row, int width);

	/* 裁剪类型 */
	enum { TRIM_NO, TRIM_AUTO, TRIM_RECTANGLE };

	/* 参数 */
	bool is_preview_;
	bool is_save_video_;
	int start_frame_index_, end_frame_index_;
	bool is_try_gpu_;
	bool is_debug_;
	int trim_type_;
	Rect trim_rect_;

	double work_megapix_;
	double seam_megapix_;
	float conf_thresh_;
	string features_type_;
	string ba_cost_func_;
	string ba_refine_mask_;
	bool is_do_wave_correct_;
	WaveCorrectKind wave_correct_;
	bool is_save_graph_;
	string save_graph_to_;
	string warp_type_;
	int expos_comp_type_;
	float match_conf_;
	string seam_find_type_;
	int blend_type_;
	float blend_strength_;

	Ptr<WarperCreator> warper_creator_;
	double work_scale_, seam_scale_;
	double median_focal_len_;

	/* 第一帧计算出的参数，不用重复计算 */
	vector<CameraParams> cameras_;
	vector<int> src_indices_;
	vector<Point> corners_;
	vector<Size> sizes_;
	Rect dst_roi_;
	vector<Mat> final_warped_masks_;	//warp的mask
	vector<Mat> xmaps_;
	vector<Mat> ymaps_;
	vector<Mat_<float>> ec_weight_maps_;		//曝光补偿
	vector<Mat> blend_weight_maps_;
	vector<Mat_<float>> total_weight_maps_;
	vector<Mat> final_blend_masks_;	//blend_mask = seam_mask & warp_mask
	double view_angle_;

	MyExposureCompensator compensator_;
	MyFeatherBlender blender_;

	/* 缓存 */
	vector<Mat> final_warped_images_;

	int cur_frame_idx_;
	int parallel_num_;
	bool is_prepared_;

	/* Debug */
	string debug_dir_path_;

};


typedef struct frameInfo_
{
	vector<Mat> src;
	Mat dst;
	int frame_idx;
	int stitch_status;
}FrameInfo;

#endif