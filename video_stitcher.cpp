
#include "stdafx.h"

#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include <Windows.h>

#include "composer_gpu.h"
#include "video_stitcher.h"
#include "apap.h"
#include "gutil.h"


MyVideoStitcher::MyVideoStitcher()
{
	is_preview_ = true;
	is_save_video_ = false;
	start_frame_index_ = 0;
	end_frame_index_ = -1;
	is_try_gpu_ = false;
	is_debug_ = false;
	trim_type_ = MyVideoStitcher::TRIM_NO;

	work_megapix_ = 1.0;//-1;//
	seam_megapix_ = 0.2;//-1;//
	is_prepared_ = false;
	conf_thresh_ = 1.f;
	features_type_ = "orb";//"surf";//
	ba_cost_func_ = "ray";
	ba_refine_mask_ = "xxxxx";
	is_do_wave_correct_ = true;
	wave_correct_ = detail::WAVE_CORRECT_HORIZ;
	is_save_graph_ = false;
	warp_type_ = "cylindrical";//"plane";//"apap";//"paniniA2B1";//"transverseMercator";//"spherical";//
	expos_comp_type_ = ExposureCompensator::GAIN_BLOCKS;//ExposureCompensator::GAIN;//
	match_conf_ = 0.3f;
	seam_find_type_ = "gc_color";//"voronoi";//
	blend_type_ = Blender::FEATHER;//Blender::MULTI_BAND;//Blender::NO;//
	blend_strength_ = 5;

	//	获取当前系统的核数
	SYSTEM_INFO sys_info;
	GetSystemInfo(&sys_info);
	parallel_num_ = sys_info.dwNumberOfProcessors;
}

int MyVideoStitcher::stitch( vector<VideoCapture> &captures, string &writer_file_name )
{
	int video_num = captures.size();
	vector<Mat> src(video_num);
	Mat frame, dst, show_dst;

	//	Debug用信息
	bool is_save_input_frames = false;
	bool is_save_output_frames = true;

	double fps = captures[0].get(CV_CAP_PROP_FPS);

	// skip some frames
	for(int j = 0; j < video_num; j++)
		for(int i = 0; i < start_frame_index_; i++)
			captures[j].read(frame);

	// 第一帧，做一些初始化，并且确定结果视频的分辨率
	for(int j = 0; j < video_num; j++)
	{
		if( !captures[j].read(frame))
			return -1;
		frame.copyTo(src[j]);
		if(is_debug_)
		{
			char img_save_name[100];
			sprintf(img_save_name, "/%d.jpg", j+1);
			imwrite(debug_dir_path_ + img_save_name, src[j]);
		}
	}

	long prepare_start_clock = clock();
	int prepare_status = Prepare(src);
	//	先用ORB特征测试，错误的话再使用SURF，仍然错误则报错，输入视频不符合条件
	if( prepare_status == STITCH_CONFIG_ERROR )
	{
		cout << "video stitch config error!" << endl;
		return -1;
	}
	if( prepare_status != STITCH_SUCCESS)
	{
		features_type_ = "surf";
		cout << "video stitch first try failed, second try ... " << endl;
		if( Prepare(src) != STITCH_SUCCESS)
		{
			cout << "videos input are invalid. Initialization failed." << endl;
			return -1;
		}
	}
	long prepare_end_clock = clock();
	cout << "\tprepare time: " << prepare_end_clock - prepare_start_clock << "ms" << endl;
	StitchFrame(src, dst);
	if(is_debug_)	//保存第一帧拼接结果和mask
	{
		imwrite(debug_dir_path_ + "/res.jpg", dst);
		vector<Mat> img_masks(video_num);
		for(int i = 0; i < video_num; i++)
		{
			img_masks[i].create(src[i].rows, src[i].cols, CV_8UC3);
			img_masks[i].setTo(Scalar::all(255));
		}
		Mat dst_mask;
		StitchFrame(img_masks, dst_mask);
		imwrite(debug_dir_path_ + "/mask.jpg", dst_mask);
	}

	// 创建结果视频
	VideoWriter writer;
	if(is_save_video_)
	{
		writer.open(writer_file_name, CV_FOURCC('D', 'I','V', '3'), 20, Size(dst.cols, dst.rows));
		writer.write(dst);
	}


	// 开始拼接
	double stitch_time = 0;

	FrameInfo frame_info;
	frame_info.src.resize(video_num);

	int frameidx = 1;

	cout << "Stitching..." << endl;

	string window_name = "视频拼接";
	if(is_preview_)
		namedWindow(window_name);
	double show_scale = 1.0, scale_interval = 0.03;
	int frame_show_interval = cvFloor(1000 / fps);
	
	int failed_frame_count = 0;

	char log_string[1000];
	char log_file_name[200];
	SYSTEMTIME sys_time = {0};
	GetLocalTime(&sys_time);
	sprintf(log_file_name, "%d%02d%02d-%02d%02d%02d.log", 
		sys_time.wYear, sys_time.wMonth, sys_time.wDay, sys_time.wHour, sys_time.wMinute, sys_time.wSecond);
	ofstream log_file;
	if(is_debug_)
		log_file.open(debug_dir_path_ + log_file_name);
	long long startTime = clock();
	while(true)
	{
		long frame_time = 0;
		//	采集
		long cap_start_clock = clock();
		int j;
		for(j = 0; j < video_num; j++)
		{
			if( !captures[j].read(frame))
				break;
			frame.copyTo(frame_info.src[j]);
		}
		frame_info.frame_idx = frameidx;
		frameidx++;
		if(j != video_num || (end_frame_index_ >= 0 && frameidx >= end_frame_index_))	//有一个视频源结束，则停止拼接
			break;

		//	拼接
		long stitch_start_clock = clock();
		frame_info.stitch_status = StitchFrame(frame_info.src, frame_info.dst);
		long stitch_clock = clock();
		sprintf(log_string, "\tframe %d: stitch(%dms), capture(%dms)", 
			frame_info.frame_idx, stitch_clock - stitch_start_clock, stitch_start_clock - cap_start_clock);
		printf("%s", log_string);
		if(is_debug_)
			log_file << log_string << endl;
		stitch_time += stitch_clock - stitch_start_clock;
		frame_time += stitch_clock - cap_start_clock;

		//	拼接失败
		if(frame_info.stitch_status != 0)
		{
			cout << "failed\n";
			if(is_debug_)
				log_file << "failed" << endl;
			failed_frame_count++;
			break;
		}

		//	保存视频
		if(is_save_video_)
		{
			cout << ", write(";
			if(is_save_output_frames)
			{
				char img_save_name[100];
				sprintf(img_save_name, "/images/%d.jpg", frame_info.frame_idx);
				imwrite(debug_dir_path_ + img_save_name, frame_info.dst);
			}
			long write_start_clock = clock();
			writer.write(frame_info.dst);
			long write_clock = clock();
			cout << write_clock - write_start_clock << "ms)";
			frame_time += write_clock - write_start_clock;
		}
		cout << endl;

		//	显示---
		if(is_preview_)
		{
			int key = waitKey(std::max(1, (int)(frame_show_interval - frame_time)));
			if(key == 27)	//	ESC
				break;
			else if(key == 61 || key == 43)	//	+
				show_scale += scale_interval;
			else if(key == 45)				//	-
				if(show_scale >= scale_interval)
					show_scale -= scale_interval;
			resize(frame_info.dst, show_dst, Size(show_scale * dst.cols, show_scale * dst.rows));
			imshow(window_name, show_dst);
		}
	}
	long long endTime = clock();
	cout << "test " << endTime - startTime << endl;
	cout << "\nStitch over" << endl;
	cout << failed_frame_count << " frames failed." << endl;
	cout << "\tfull view angle is " << cvRound(view_angle_) << "°" << endl;
	if(is_debug_)
		log_file << "\tfull view angle is " << cvRound(view_angle_) << "°" << endl;
	writer.release();

	cout << "\ton average: stitch time = " << stitch_time / (frameidx-1) << "ms" << endl;
	cout << "\tcenter: (" << -dst_roi_.x << ", " << -dst_roi_.y << ")" << endl;
	if(is_debug_)
	{
		log_file << "\ton average: stitch time = " << stitch_time / (frameidx-1) << "ms" << endl;
		log_file << "\tcenter: (" << -dst_roi_.x << ", " << -dst_roi_.y << ")" << endl;
		log_file.close();
	}

	return 0;
}

void MyVideoStitcher::InitMembers(int num_images)
{
}

/*
 *	初始图像可能分辨率很高，先做一步降采样，可以提高时间效率
 */
void MyVideoStitcher::SetScales(vector<Mat> &src)
{
	if(work_megapix_ < 0)
		work_scale_ = 1.0;
	else
		work_scale_ = min(1.0, sqrt(work_megapix_ * 1e6 / src[0].size().area()));

	if(seam_megapix_ < 0)
		seam_scale_ = 1.0;
	else
		seam_scale_ = min(1.0, sqrt(seam_megapix_ * 1e6 / src[0].size().area()));
}

/*
 *	特征提取，支持SURF和ORB
 */
int MyVideoStitcher::FindFeatures(vector<Mat> &src, vector<ImageFeatures> &features)
{
	Ptr<FeaturesFinder> finder;
	if (features_type_ == "surf")
	{
#ifdef HAVE_OPENCV_GPU
		if (is_try_gpu_ && gpu::getCudaEnabledDeviceCount() > 0)
			finder = new SurfFeaturesFinderGpu();
		else
#endif
			finder = new SurfFeaturesFinder();
	}
	else if (features_type_ == "orb")
	{
		finder = new OrbFeaturesFinder();//Size(3,1), 1500, 1.3, 5);
	}
	else
	{
		cout << "Unknown 2D features type: '" << features_type_ << "'.\n";
		return STITCH_CONFIG_ERROR;
	}

	int num_images = static_cast<int>(src.size());
	Mat full_img, img;

	for (int i = 0; i < num_images; ++i)
	{
		full_img = src[i].clone();//

		if (work_megapix_ < 0)
			img = full_img;
		else
			resize(full_img, img, Size(), work_scale_, work_scale_);

		(*finder)(img, features[i]);
		//LOGLN("Features in image #" << i+1 << "("<<img.size()<< "): " << features[i].keypoints.size());
		features[i].img_idx = i;
	}

	finder->collectGarbage();
	full_img.release();
	img.release();

	return STITCH_SUCCESS;
}

/*
 * 特征匹配，然后去除噪声图片。本代码实现时，一旦出现噪声图片，就终止算法
 * 返回值：
 *		0	——	正常
 *		-2	——	存在噪声图片
 */
int MyVideoStitcher::MatchImages(vector<ImageFeatures> &features, vector<MatchesInfo> &pairwise_matches)
{
	int total_num_images = static_cast<int>(features.size());

	BestOf2NearestMatcher matcher(is_try_gpu_, match_conf_);
	matcher(features, pairwise_matches);
	matcher.collectGarbage();
	
	// 去除噪声图像
	vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh_);
	
	// 一旦出现噪声图片，就终止算法
	int num_images = static_cast<int>(indices.size());
	if (num_images != total_num_images)
	{
		LOGLN(total_num_images - num_images << " videos are invaild");
		return STITCH_NOISE;
	}
	
	return STITCH_SUCCESS;
}

/*
 * 摄像机标定
 */
int MyVideoStitcher::CalibrateCameras(vector<ImageFeatures> &features, vector<MatchesInfo> &pairwise_matches, vector<CameraParams> &cameras)
{
	HomographyBasedEstimator estimator;
	Ptr<detail::BundleAdjusterBase> adjuster;
	Mat_<uchar> refine_mask;
	vector<double> focals;

	estimator(features, pairwise_matches, cameras);

	for (size_t i = 0; i < cameras.size(); ++i)
	{
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
		//LOGLN("Initial intrinsics #" << i << ":\n" << cameras[i].K());
	}

	if (ba_cost_func_ == "reproj") adjuster = new detail::BundleAdjusterReproj();
	else if (ba_cost_func_ == "ray") adjuster = new detail::BundleAdjusterRay();
	else
	{
		cout << "Unknown bundle adjustment cost function: '" << ba_cost_func_ << "'.\n";
		return STITCH_CONFIG_ERROR;
	}
	adjuster->setConfThresh(conf_thresh_);
	refine_mask = Mat::zeros(3, 3, CV_8U);
	if (ba_refine_mask_[0] == 'x') refine_mask(0,0) = 1;
	if (ba_refine_mask_[1] == 'x') refine_mask(0,1) = 1;
	if (ba_refine_mask_[2] == 'x') refine_mask(0,2) = 1;
	if (ba_refine_mask_[3] == 'x') refine_mask(1,1) = 1;
	if (ba_refine_mask_[4] == 'x') refine_mask(1,2) = 1;
	adjuster->setRefinementMask(refine_mask);
	(*adjuster)(features, pairwise_matches, cameras);

	// Find median focal length
	for (size_t i = 0; i < cameras.size(); ++i)
	{
		focals.push_back(cameras[i].focal);
		//LOGLN("Camera #" << i+1 << ":\n" << cameras[i].t << cameras[i].R);
	}

	sort(focals.begin(), focals.end());
	if (focals.size() % 2 == 1)
		median_focal_len_ = static_cast<float>(focals[focals.size() / 2]);
	else
		median_focal_len_ = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

	if (is_do_wave_correct_)
	{
		vector<Mat> rmats;
		for (size_t i = 0; i < cameras.size(); ++i)
			rmats.push_back(cameras[i].R);
		waveCorrect(rmats, wave_correct_);
		for (size_t i = 0; i < cameras.size(); ++i)
			cameras[i].R = rmats[i];
	}

	if(is_debug_)
		this->saveCameraParam(debug_dir_path_ + "/camera_param.dat");

	return STITCH_SUCCESS;
}

/*
 *	计算水平视角，用于判断是否适用于平面投影
 */
double MyVideoStitcher::GetViewAngle(vector<Mat> &src, vector<CameraParams> &cameras)
{
	Ptr<WarperCreator> warper_creator = new cv::CylindricalWarper();
	Ptr<RotationWarper> warper = warper_creator->create(median_focal_len_);

	int num_images = static_cast<int>(src.size());
	vector<Point> corners;
	vector<Size> sizes;
	for (int i = 0; i < num_images; ++i)
	{
		Mat_<float> K;
		cameras[i].K().convertTo(K, CV_32F);
		Rect roi = warper->warpRoi(Size(src[i].cols * work_scale_, src[i].rows * work_scale_), K, cameras[i].R);
		corners.push_back(roi.tl());
		sizes.push_back(roi.size());
	}
	Rect result_roi = resultRoi(corners, sizes);
	double view_angle = result_roi.width * 180.0 / (median_focal_len_  * CV_PI);
	return view_angle;
}

/*
 *	计算接缝之前，需要先把原始图像和mask按照相机参数投影
 */
int MyVideoStitcher::WarpForSeam(vector<Mat> &src, vector<CameraParams> &cameras, vector<Mat> &masks_warped, vector<Mat> &images_warped)
{
	// Warp images and their masks
#ifdef HAVE_OPENCV_GPU
	if (is_try_gpu_ && gpu::getCudaEnabledDeviceCount() > 0)
	{
		if (warp_type_ == "plane") warper_creator_ = new cv::PlaneWarperGpu();
		else if (warp_type_ == "cylindrical") warper_creator_ = new cv::CylindricalWarperGpu();
		else if (warp_type_ == "spherical") warper_creator_ = new cv::SphericalWarperGpu();
	}
	else
#endif
	{
		if (warp_type_ == "plane") warper_creator_ = new cv::PlaneWarper();
		else if (warp_type_ == "cylindrical") warper_creator_ = new cv::CylindricalWarper();
		else if (warp_type_ == "spherical") warper_creator_ = new cv::SphericalWarper();
		else if (warp_type_ == "fisheye") warper_creator_ = new cv::FisheyeWarper();
		else if (warp_type_ == "stereographic") warper_creator_ = new cv::StereographicWarper();
		else if (warp_type_ == "compressedPlaneA2B1") warper_creator_ = new cv::CompressedRectilinearWarper(2, 1);
		else if (warp_type_ == "compressedPlaneA1.5B1") warper_creator_ = new cv::CompressedRectilinearWarper(1.5, 1);
		else if (warp_type_ == "compressedPlanePortraitA2B1") warper_creator_ = new cv::CompressedRectilinearPortraitWarper(2, 1);
		else if (warp_type_ == "compressedPlanePortraitA1.5B1") warper_creator_ = new cv::CompressedRectilinearPortraitWarper(1.5, 1);
		else if (warp_type_ == "paniniA2B1") warper_creator_ = new cv::PaniniWarper(2, 1);
		else if (warp_type_ == "paniniA1.5B1") warper_creator_ = new cv::PaniniWarper(1.5, 1);
		else if (warp_type_ == "paniniPortraitA2B1") warper_creator_ = new cv::PaniniPortraitWarper(2, 1);
		else if (warp_type_ == "paniniPortraitA1.5B1") warper_creator_ = new cv::PaniniPortraitWarper(1.5, 1);
		else if (warp_type_ == "mercator") warper_creator_ = new cv::MercatorWarper();
		else if (warp_type_ == "transverseMercator") warper_creator_ = new cv::TransverseMercatorWarper();
	}

	if (warper_creator_.empty())
	{
		cout << "Can't create the following warper '" << warp_type_ << "'\n";
		return STITCH_CONFIG_ERROR;
	}

	float warp_scale = static_cast<float>(median_focal_len_ * seam_scale_ / work_scale_);
	Ptr<RotationWarper> warper = warper_creator_->create(warp_scale);
	int full_pano_width = cvFloor(warp_scale * 2 * CV_PI);

	int num_images = static_cast<int>(src.size());
	Mat img, mask;
	for (int i = 0; i < num_images; ++i)
	{
		Mat_<float> K;
		cameras[i].K().convertTo(K, CV_32F);
		float swa = (float)seam_scale_ / work_scale_;
		K(0,0) *= swa; K(0,2) *= swa;
		K(1,1) *= swa; K(1,2) *= swa;

		if (seam_megapix_ < 0)
			img = src[i].clone();
		else
			resize(src[i], img, Size(), seam_scale_, seam_scale_);

		mask.create(img.size(), CV_8U);
		mask.setTo(Scalar::all(255));
		Mat tmp_mask_warped, tmp_img_warped;
		Point tmp_corner;
		Size tmp_size;
		warper->warp(mask, K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, tmp_mask_warped);

		//	考虑360度拼接的特殊情况
		tmp_corner = warper->warp(img, K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, tmp_img_warped);
		//cout << "warped width = " << tmp_mask_warped.cols << ", pano width = " << full_pano_width << endl;
		if(abs(tmp_mask_warped.cols - full_pano_width) <= 10)
		{
			int x1, x2;
			FindWidestInpaintRange(tmp_mask_warped, x1, x2);
			Mat mask1, mask2, img1, img2;
			Rect rect1(0, 0, x1, tmp_mask_warped.rows), rect2(x2+1, 0, tmp_mask_warped.cols-1-x2, tmp_mask_warped.rows);
			tmp_mask_warped(rect1).copyTo(mask1);
			tmp_mask_warped(rect2).copyTo(mask2);
			masks_warped.push_back(mask1);
			masks_warped.push_back(mask2);

			tmp_img_warped(rect1).copyTo(img1);
			tmp_img_warped(rect2).copyTo(img2);
			images_warped.push_back(img1);
			images_warped.push_back(img2);

			corners_.push_back(tmp_corner);
			corners_.push_back(tmp_corner + rect2.tl());

			sizes_.push_back(rect1.size());
			sizes_.push_back(rect2.size());
		}
		else
		{
			masks_warped.push_back(tmp_mask_warped);
			corners_.push_back(tmp_corner);
			images_warped.push_back(tmp_img_warped);
			sizes_.push_back(tmp_img_warped.size());
		}

	}
	return STITCH_SUCCESS;
}

/*
 *	解决360°拼接问题。对于横跨360°接缝的图片，找到最宽的inpaint区域[x1, x2]
 */
int MyVideoStitcher::FindWidestInpaintRange(Mat mask, int &x1, int &x2)
{
	vector<int> sum_row(mask.cols);
	uchar *mask_ptr = mask.ptr<uchar>(0);
	for(int x = 0; x < mask.cols; x++)
		sum_row[x] = 0;
	for(int x = 0; x < mask.cols; x++)
		for(int y = 0; y < mask.rows; y++)
			if(mask_ptr[y * mask.cols + x] != 0)
				sum_row[x] = 1;

	int cur_x1, cur_x2, max_range = 0;
	for(int x = 1; x < mask.cols; x++)	//	最左边肯定是1
	{
		if(sum_row[x - 1] == 1 && sum_row[x] == 0)
			cur_x1 = x;
		else if(sum_row[x - 1] == 0 && sum_row[x] == 1)
		{
			cur_x2 = x - 1;
			if(cur_x2 - cur_x1 > max_range)
			{
				x1 = cur_x1;
				x2 = cur_x2;
			}
		}
	}
	return 0;
}

/*
 *	计算接缝
 */
int MyVideoStitcher::FindSeam(vector<Mat> &images_warped, vector<Mat> &masks_warped)
{
	int num_images = static_cast<int>(images_warped.size());
	vector<Mat> images_warped_f(num_images);
	for (int i = 0; i < num_images; ++i)
		images_warped[i].convertTo(images_warped_f[i], CV_32F);

	Ptr<SeamFinder> seam_finder;

	if (seam_find_type_ == "no")
		seam_finder = new detail::NoSeamFinder();
	else if (seam_find_type_ == "voronoi")
		seam_finder = new detail::VoronoiSeamFinder();
	else if (seam_find_type_ == "gc_color")
	{
#ifdef HAVE_OPENCV_GPU
		if (is_try_gpu_ && gpu::getCudaEnabledDeviceCount() > 0)
			seam_finder = new detail::GraphCutSeamFinderGpu(GraphCutSeamFinderBase::COST_COLOR);
		else
#endif
			seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR);
	}
	else if (seam_find_type_ == "gc_colorgrad")
	{
#ifdef HAVE_OPENCV_GPU
		if (is_try_gpu_ && gpu::getCudaEnabledDeviceCount() > 0)
			seam_finder = new detail::GraphCutSeamFinderGpu(GraphCutSeamFinderBase::COST_COLOR_GRAD);
		else
#endif
			seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR_GRAD);
	}
	else if (seam_find_type_ == "dp_color")
		seam_finder = new detail::DpSeamFinder(DpSeamFinder::COLOR);
	else if (seam_find_type_ == "dp_colorgrad")
		seam_finder = new detail::DpSeamFinder(DpSeamFinder::COLOR_GRAD);
	if (seam_finder.empty())
	{
		cout << "Can't create the following seam finder '" << seam_find_type_ << "'\n";
		return STITCH_CONFIG_ERROR;
	}
	seam_finder->find(images_warped_f, corners_, masks_warped);

	images_warped_f.clear();
	return STITCH_SUCCESS;
}

/*
 *	恢复原始图像大小
 */
int MyVideoStitcher::Rescale(vector<Mat> &src, vector<CameraParams> &cameras, vector<Mat> &seam_masks)
{
	median_focal_len_ = median_focal_len_ / work_scale_;
	Ptr<RotationWarper> warper = warper_creator_->create(median_focal_len_);
	int full_pano_width = cvFloor(median_focal_len_ * 2 * CV_PI);

	//cout << "median focal length: " << median_focal_len_ << endl;

	// Update corners and sizes
	int num_images = static_cast<int>(src.size());
	Mat tmp_mask, tmp_dilated_mask, tmp_seam_mask;
	corners_.clear();
	sizes_.clear();
	for (int src_idx = 0, seam_idx = 0; src_idx < num_images; ++src_idx)
	{
		// Update intrinsics
		cameras[src_idx].focal	/= work_scale_;
		cameras[src_idx].ppx	/= work_scale_;
		cameras[src_idx].ppy	/= work_scale_;

		Mat K;
		cameras[src_idx].K().convertTo(K, CV_32F);

		// 计算最终image warp的坐标映射矩阵
		Mat tmp_xmap, tmp_ymap;
		warper->buildMaps(src[src_idx].size(), K, cameras[src_idx].R, tmp_xmap, tmp_ymap);

		// Warp the current image mask
		Mat tmp_mask_warped, tmp_final_blend_mask;
		tmp_mask.create(src[src_idx].size(), CV_8U);
		tmp_mask.setTo(Scalar::all(255));
		Point tmp_corner = warper->warp(tmp_mask, K, cameras[src_idx].R, INTER_NEAREST, BORDER_CONSTANT, tmp_mask_warped);
		
		//	考虑360度拼接的特殊情况
		if(abs(tmp_mask_warped.cols - full_pano_width) <= 10)
		{
			int x1, x2;
			FindWidestInpaintRange(tmp_mask_warped, x1, x2);
			Mat warped_mask[2], blend_mask[2], xmap[2], ymap[2];
			Rect rect[2];
			rect[0] = Rect(0, 0, x1, tmp_mask_warped.rows);
			rect[1] = Rect(x2+1, 0, tmp_mask_warped.cols-1-x2, tmp_mask_warped.rows);
			for(int j = 0; j < 2; j++)
			{
				tmp_mask_warped(rect[j]).copyTo(warped_mask[j]);
				final_warped_masks_.push_back(warped_mask[j]);

				tmp_xmap(rect[j]).copyTo(xmap[j]);
				xmaps_.push_back(xmap[j]);

				tmp_ymap(rect[j]).copyTo(ymap[j]);
				ymaps_.push_back(ymap[j]);

				// 计算总的mask = warp_mask & seam_mask
				dilate(seam_masks[seam_idx], tmp_dilated_mask, Mat());	//膨胀
				resize(tmp_dilated_mask, tmp_seam_mask, rect[j].size());
				final_blend_masks_.push_back(warped_mask[j] & tmp_seam_mask);

				corners_.push_back(tmp_corner + rect[j].tl());
				sizes_.push_back(rect[j].size());

				src_indices_.push_back(src_idx);

				seam_idx++;
			}
		}
		else
		{
			xmaps_.push_back(tmp_xmap);
			ymaps_.push_back(tmp_ymap);
			final_warped_masks_.push_back(tmp_mask_warped);
			corners_.push_back(tmp_corner);

			Size sz = tmp_mask_warped.size();
			sizes_.push_back(sz);

			//	计算总的mask = warp_mask & seam_mask
			dilate(seam_masks[seam_idx], tmp_dilated_mask, Mat());	//膨胀
			resize(tmp_dilated_mask, tmp_seam_mask, sz);
			final_blend_masks_.push_back(tmp_mask_warped & tmp_seam_mask);

			src_indices_.push_back(src_idx);

			seam_idx ++;
		}
	}
	
	dst_roi_ = resultRoi(corners_, sizes_);
	int parts_num = sizes_.size();
	final_warped_images_.resize(parts_num);
	for(int j = 0; j < parts_num; j++)
		final_warped_images_[j].create(sizes_[j], src[src_indices_[j]].type());

	tmp_dilated_mask.release();
	tmp_seam_mask.release();
	tmp_mask.release();

	return STITCH_SUCCESS;
}

/*
 *	拼接结果可能是不规则形状，裁剪成方形
 */
int MyVideoStitcher::TrimRect(Rect rect)
{
	// 计算每幅图像的rect，并修改xmap和ymap
	int top = rect.y;
	int left = rect.x;
	int bottom = rect.y + rect.height - 1;
	int right = rect.x + rect.width - 1;
	int num_images = xmaps_.size();
	for(int i = 0; i < num_images; i++)
	{
		int top_i, bottom_i, left_i, right_i;
		top_i = max(dst_roi_.y + top, corners_[i].y);
		left_i = max(dst_roi_.x + left, corners_[i].x);
		bottom_i = min(corners_[i].y + sizes_[i].height - 1, dst_roi_.y + bottom);
		right_i = min(corners_[i].x + sizes_[i].width - 1, dst_roi_.x + right);

		sizes_[i].height = bottom_i - top_i + 1;
		sizes_[i].width = right_i - left_i + 1;

		Rect map_rect(left_i - corners_[i].x, top_i - corners_[i].y,
			sizes_[i].width, sizes_[i].height);

		Mat tmp_map = xmaps_[i].clone();
		tmp_map(map_rect).copyTo(xmaps_[i]);
		tmp_map = ymaps_[i].clone();
		tmp_map(map_rect).copyTo(ymaps_[i]);

		Mat tmp_img = final_blend_masks_[i].clone();
		tmp_img(map_rect).copyTo(final_blend_masks_[i]);

		corners_[i].x = left_i;
		corners_[i].y = top_i;
	}

	dst_roi_.x		+= left;
	dst_roi_.y		+= top;
	dst_roi_.width	= right - left + 1;
	dst_roi_.height	= bottom - top + 1;
	return STITCH_SUCCESS;
}

/*
 *	如果是平面投影的话，可以自动去除未填充区域
 */
int MyVideoStitcher::TrimInpaint(vector<Mat> &src)
{
	int num_images = static_cast<int>(src.size());
	
	// 先计算最终图像的mask
	dst_roi_ = resultRoi(corners_, sizes_);
	Mat dst = Mat::zeros(dst_roi_.height, dst_roi_.width, CV_8UC1);
	for(int i = 0; i < num_images; i++)
	{
		int dx = corners_[i].x - dst_roi_.x;
		int dy = corners_[i].y - dst_roi_.y;
		int img_rows = sizes_[i].height;
		int img_cols = sizes_[i].width;
		for(int y = 0; y < img_rows; y++)
		{
			uchar *mask_row_ptr = final_warped_masks_[i].ptr<uchar>(y);
			uchar *dst_row_ptr = dst.ptr<uchar>(dy + y);
			for(int x = 0; x < img_cols; x++)
				dst_row_ptr[dx + x] += mask_row_ptr[x];
		}
	}

	int x, y;
	// top
	for(y = 0; y < dst_roi_.height; y++)
	{
		uchar *dst_row_ptr = dst.ptr<uchar>(y);
		if(!(this->IsRowCrossInpaint(dst_row_ptr, dst_roi_.width)))
			break;
	}
	int top = y;
	
	// bottom
	for(y = dst_roi_.height - 1; y >= 0 ; y--)
	{
		uchar *dst_row_ptr = dst.ptr<uchar>(y);
		if(!(this->IsRowCrossInpaint(dst_row_ptr, dst_roi_.width)))
			break;
	}
	int bottom = y;
	
	// left
	uchar *dst_ptr_00 = dst.ptr<uchar>(0);
	for(x = 0; x < dst_roi_.width; x++)
	{
		for(y = top; y < bottom; y++)
			if(dst_ptr_00[y * (dst_roi_.width) + x] == 0)
				break;
		if(y == bottom)
			break;
	}
	int left = x;
	
	// right
	for(x = dst_roi_.width-1; x >= 0 ; x--)
	{
		for(y = top; y < bottom; y++)
			if(dst_ptr_00[y * (dst_roi_.width) + x] == 0)
				break;
		if(y == bottom)
			break;
	}
	int right = x;
	
	// 计算每幅图像的rect，并修改xmap和ymap
	for(int i = 0; i < num_images; i++)
	{
		int top_i, bottom_i, left_i, right_i;
		top_i = max(dst_roi_.y + top, corners_[i].y);
		left_i = max(dst_roi_.x + left, corners_[i].x);
		bottom_i = min(corners_[i].y + sizes_[i].height - 1, dst_roi_.y + bottom);
		right_i = min(corners_[i].x + sizes_[i].width - 1, dst_roi_.x + right);

		sizes_[i].height = bottom_i - top_i + 1;
		sizes_[i].width = right_i - left_i + 1;

		Rect rect(left_i - corners_[i].x, top_i - corners_[i].y,
			sizes_[i].width, sizes_[i].height);
		
		Mat tmp_map = xmaps_[i].clone();
		tmp_map(rect).copyTo(xmaps_[i]);
		tmp_map = ymaps_[i].clone();
		tmp_map(rect).copyTo(ymaps_[i]);

		Mat tmp_img = final_blend_masks_[i].clone();
		tmp_img(rect).copyTo(final_blend_masks_[i]);
		
		corners_[i].x = left_i;
		corners_[i].y = top_i;
	}

	dst_roi_.x		+= left;
	dst_roi_.y		+= top;
	dst_roi_.width	= right - left + 1;
	dst_roi_.height	= bottom - top + 1;

	return 0;
}

/*
 *	判断一行中是否有未填充像素
 */
bool MyVideoStitcher::IsRowCrossInpaint(uchar *row, int width)
{
	bool is_have_entered_inpaint = false;
	int count0 = 0;
	for(int x = 1; x < width; x++)
	{
		if(row[x] == 0)
			count0++;
		if(row[x-1] != 0 && row[x] == 0)
			is_have_entered_inpaint = true;
		if((row[x-1] == 0 && row[x] != 0) && is_have_entered_inpaint)
			return true;
	}
	if(count0 >= (width/2))
		return true;
	return false;
}

int MyVideoStitcher::Prepare(vector<Mat> &src)
{
	cv::setBreakOnError(true);
	int num_images = static_cast<int>(src.size());
	if (num_images < 2)
	{
		LOGLN("Need more images");
		return STITCH_CONFIG_ERROR;
	}

	int cudaStatus = testGPU();
	if( is_try_gpu_ && cudaStatus != 0 )
	{
		LOGLN("GPU acceleration failed! Error code: " <<  cudaStatus << " Please ensure that you have a CUDA-capable GPU installed!");
		LOGLN("Stitching with CPU next ...");
		return STITCH_CONFIG_ERROR;
	}
	
	int flag;
	if(warp_type_ == "apap")
		flag = PrepareAPAP(src);
	else
		flag = PrepareClassical(src);
	
	if(flag == 0 && is_try_gpu_)
	{
		flag = initGPU(num_images);
		if(flag != 0)
			return flag;
		C2GInitData *c2g_data = new C2GInitData[num_images];
		for(int i = 0; i < num_images; i++)
		{
			c2g_data[i].xmap			= xmaps_[i].ptr<float>(0);
			c2g_data[i].ymap			= ymaps_[i].ptr<float>(0);
			c2g_data[i].ec_weight		= ec_weight_maps_[i].ptr<float>(0);
			c2g_data[i].blend_weight	= blend_weight_maps_[i].ptr<float>(0);
			c2g_data[i].total_weight	= total_weight_maps_[i].ptr<float>(0);
			c2g_data[i].height			= src[i].rows;
			c2g_data[i].width			= src[i].cols;
			c2g_data[i].warped_height	= xmaps_[i].rows;
			c2g_data[i].warped_width	= xmaps_[i].cols;
			c2g_data[i].corner_x		= corners_[i].x - dst_roi_.x;
			c2g_data[i].corner_y		= corners_[i].y - dst_roi_.y;
		}
		initdataCopy2GPU(c2g_data, dst_roi_.height, dst_roi_.width);
	}

	if(flag == STITCH_SUCCESS)
	{
		LOGLN("\t~Prepare complete");
		is_prepared_ = true;
	}

	return flag;
}

/*
 *	APAP算法的初始化
 */
int MyVideoStitcher::PrepareAPAP(vector<Mat> &src)
{
	int num_images = static_cast<int>(src.size());

	this->InitMembers(num_images);

	// 计算一些放缩的尺度，在特征检测和计算接缝的时候，为了提高程序效率，可以对源图像进行一些放缩
	work_megapix_ = -1;	//	先不考虑放缩
	seam_megapix_ = -1;	//	先不考虑放缩
	this->SetScales(src);

	// 特征检测
	vector<ImageFeatures> features(num_images);
	this->FindFeatures(src, features);

	// 特征匹配，并去掉噪声图片
	vector<MatchesInfo> pairwise_matches;
	this->MatchImages(features, pairwise_matches);

	// APAP算法
	APAPWarper apap_warper;
	apap_warper.buildMaps(src, features, pairwise_matches, xmaps_, ymaps_, corners_);
	for(int i = 0; i < num_images; i++)
		sizes_[i] = xmaps_[i].size();
	dst_roi_ = resultRoi(corners_, sizes_);

	// 计算接缝
	vector<Mat> seamed_masks(num_images);
	vector<Mat> images_warped(num_images);
	vector<Mat> init_masks(num_images);
	for(int i = 0; i < num_images; i++)
	{
		init_masks[i].create(src[i].size(), CV_8U);
		init_masks[i].setTo(Scalar::all(255));
		remap(src[i], images_warped[i], xmaps_[i], ymaps_[i], INTER_LINEAR);
		remap(init_masks[i], final_warped_masks_[i], xmaps_[i], ymaps_[i], INTER_NEAREST, BORDER_CONSTANT);
		seamed_masks[i] = final_warped_masks_[i].clone();
	}
	this->FindSeam(images_warped, seamed_masks);
	LOGLN("find seam");

	// 曝光补偿
	compensator_.createWeightMaps(corners_, images_warped, final_warped_masks_, ec_weight_maps_);
	// 曝光补偿时，各像素权值也要resize一下
	compensator_.gainMapResize(sizes_, ec_weight_maps_);
	LOGLN("compensate");

	images_warped.clear();

	// 计算融合时，各像素的权值
	Size dst_sz = dst_roi_.size();
	//cout << "dst size: " << dst_sz << endl;
	float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength_ / 100.f;
	blender_.setSharpness(1.f / blend_width);
	for(int i = 0; i < num_images; i++)
		final_blend_masks_[i] = final_warped_masks_[i] & seamed_masks[i];
	blender_.createWeightMaps(dst_roi_, corners_, seamed_masks, blend_weight_maps_);

	return STITCH_SUCCESS;
}

int MyVideoStitcher::PrepareClassical(vector<Mat> &src)
{
	int num_images = static_cast<int>(src.size());
	LOGLN("Preparing...");

	this->InitMembers(num_images);

	// 计算一些放缩的尺度，在特征检测和计算接缝的时候，为了提高程序效率，可以对源图像进行一些放缩
	this->SetScales(src);

	if((cameras_.size() == 0) || (cameras_.size() != num_images))
	{
		if((cameras_.size() != 0) && (cameras_.size() != num_images))
		{
			cameras_.clear();
			LOGLN("\t~load camera parameters error! Trying to calculate again ...");
		}

		// 特征检测
		LOGLN("\t~finding features...");
		vector<ImageFeatures> features(num_images);
		this->FindFeatures(src, features);

		// 特征匹配，并去掉噪声图片
		LOGLN("\t~matching images...");
		vector<MatchesInfo> pairwise_matches;
		int retrun_flag = this->MatchImages(features, pairwise_matches);
		if(retrun_flag != 0)
			return retrun_flag;

		// 摄像机标定
		LOGLN("\t~calibrating cameras...");
		cameras_.resize(num_images);
		this->CalibrateCameras(features, pairwise_matches, cameras_);
	}


	//	计算水平视角，判定平面投影的合法性
	LOGLN("\t~calculating view angle...");
	view_angle_ = this->GetViewAngle(src, cameras_);
	if(view_angle_ > 140 && warp_type_ == "plane")
		warp_type_ = "cylindrical";

	// 为接缝的计算做Warp
	LOGLN("\t~warping for seaming...");
	vector<Mat> masks_warped;
	vector<Mat> images_warped;
	this->WarpForSeam(src, cameras_, masks_warped, images_warped);

	// 曝光补偿
	LOGLN("\t~compensating...");
	compensator_.createWeightMaps(corners_, images_warped, masks_warped, ec_weight_maps_);

	// 计算接缝
	//LOGLN("\t~finding seam...");
	this->FindSeam(images_warped, masks_warped);
	images_warped.clear();

	// 把摄像机参数和masks还原到正常大小
	LOGLN("\t~rescaling...");
	this->Rescale(src, cameras_, masks_warped);

	// 裁剪掉inpaint区域
	if(trim_type_ == MyVideoStitcher::TRIM_AUTO)
		if(warp_type_ == "plane")
			this->TrimInpaint(src);
	if(trim_type_ == MyVideoStitcher::TRIM_RECTANGLE)
		this->TrimRect(trim_rect_);

	// 拼接评价
	//this->RegistEvaluation(features, pairwise_matches, cameras);

	// 曝光补偿时，各像素权值也要resize一下
	LOGLN("\t~resizing compensators' weight map...");
	compensator_.gainMapResize(sizes_, ec_weight_maps_);

	// 计算融合时，各像素的权值
	LOGLN("\t~blending...");
	Size dst_sz = dst_roi_.size();
	float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength_ / 100.f;
	blender_.setSharpness(1.f / blend_width);
	blender_.createWeightMaps(dst_roi_, corners_, final_blend_masks_, blend_weight_maps_);

	//	计算总权重
	num_images = sizes_.size();
	total_weight_maps_.resize(num_images);
	for(int i = 0; i < num_images; i++)
	{
		int n_pixel = sizes_[i].height * sizes_[i].width;
		float *blend_weight_ptr = blend_weight_maps_[i].ptr<float>(0);
		float *ec_weight_ptr	= ec_weight_maps_[i].ptr<float>(0);
		total_weight_maps_[i].create(sizes_[i]);
		float *total_weight_ptr	= total_weight_maps_[i].ptr<float>(0);
		for(int j = 0; j < n_pixel; j++)
			total_weight_ptr[j] = blend_weight_ptr[j] * ec_weight_ptr[j];
	}
	//	处理xmap和ymap，方便GPU核函数使用
	for(int i = 0; i < num_images; i++)
	{
		float *xmap = xmaps_[i].ptr<float>(0);
		float *ymap = ymaps_[i].ptr<float>(0);
		int n_pixel = sizes_[i].height * sizes_[i].width;
		int src_height = src[src_indices_[i]].rows;
		int src_width = src[src_indices_[i]].cols;
		for(int j = 0; j < n_pixel; j++)
		{
			float map_x = xmap[j];
			float map_y = ymap[j];
			int map_x1 = cvFloor(map_x);
			int map_y1 = cvFloor(map_y);
			int map_x2 = map_x1 + 1;
			int map_y2 = map_y1 + 1;
			if((map_x1 < 0) || (map_y1 < 0) || (map_x2 >= src_width) || (map_y2 >= src_height))
				xmap[j] = ymap[j] = -1;
		}
	}
	
	is_prepared_ = true;
	return STITCH_SUCCESS;
}

int MyVideoStitcher::StitchFrame( vector<Mat> &src, Mat &dst)
{
	if( !is_prepared_ )
	{
		int flag = Prepare(src);
		if(flag != 0)
			return flag;
	}

	if(is_try_gpu_)
		return StitchFrameGPU(src, dst);
	else
		return StitchFrameCPU(src, dst);
}

int MyVideoStitcher::StitchFrameGPU( vector<Mat> &src, Mat &dst )
{
	if(dst.empty())
		dst.create(dst_roi_.size(), CV_8UC3);

	int image_num = src.size();
	GPUImageData *images = new GPUImageData[image_num];
	for(int i = 0; i < image_num; i++)
		images[i].data = src[i].ptr<uchar>(0);
	int flag = composeGPU(images, dst.ptr<uchar>(0));
	free(images);
	return flag;
}

int MyVideoStitcher::StitchFrameCPU( vector<Mat> &src, Mat &dst)
{
	bool time_debug = false;//true;//
	long start_clock, end_clock;

	if(time_debug)
		start_clock = clock();

	int64 t;
	int num_images = src_indices_.size();

	int dst_width = dst_roi_.width;
	int dst_height = dst_roi_.height;
	if(dst.empty())
		dst.create(dst_roi_.size(), CV_8UC3);
	uchar *dst_ptr_00 = dst.ptr<uchar>(0);
	memset(dst_ptr_00, 0, dst_width * dst_height * 3);

	double warp_time[100], feed_time[100];

	for (int img_idx = 0; img_idx < num_images; ++img_idx)
	{
		if(time_debug)
			t = getTickCount();

		// Warp the current image
		remap(src[src_indices_[img_idx]], final_warped_images_[img_idx], xmaps_[img_idx], ymaps_[img_idx], 
			INTER_LINEAR);//, BORDER_REFLECT);
		if(time_debug)
			warp_time[img_idx] = 1000 * (getTickCount() - t) / getTickFrequency();

		if(time_debug)
			t = getTickCount();
		int dx = corners_[img_idx].x - dst_roi_.x;
		int dy = corners_[img_idx].y - dst_roi_.y;
		int img_rows = sizes_[img_idx].height;
		int img_cols = sizes_[img_idx].width;
		int src_rows = src[img_idx].rows;
		int src_cols = src[img_idx].cols;
		
		int rows_per_parallel = img_rows / parallel_num_;
#pragma omp parallel for
		for(int parallel_idx = 0; parallel_idx < parallel_num_; parallel_idx++)
		{
			int row_start = parallel_idx * rows_per_parallel;
			int row_end = row_start + rows_per_parallel;
			if(parallel_idx == parallel_num_ - 1)
				row_end = img_rows;
			
			uchar *dst_ptr;
			uchar *warped_img_ptr	= final_warped_images_[img_idx].ptr<uchar>(row_start);
			float *total_weight_ptr	= total_weight_maps_[img_idx].ptr<float>(row_start);
			for(int y = row_start; y < row_end; y++)
			{
				dst_ptr = dst_ptr_00 + ((dy + y) * dst_width + dx) * 3;
				for(int x = 0; x < img_cols; x++)
				{
					/* 曝光补偿和融合加权平均 */
					(*dst_ptr) += (uchar)(cvRound((*warped_img_ptr) * (*total_weight_ptr)));
					warped_img_ptr++;
					dst_ptr++;
					
					(*dst_ptr) += (uchar)(cvRound((*warped_img_ptr) * (*total_weight_ptr)));
					warped_img_ptr++;
					dst_ptr++;

					(*dst_ptr) += (uchar)(cvRound((*warped_img_ptr) * (*total_weight_ptr)));
					warped_img_ptr++;
					dst_ptr++;
				
					total_weight_ptr++;
				}
			}
		}


		if(time_debug)
			feed_time[img_idx] = 1000 * (getTickCount() - t) / getTickFrequency();
	}

	if(time_debug)
		for(int i = 0; i < num_images; i++)
			cout << "\twarp " << warp_time[i] << "ms, feed " << feed_time[i] << "ms" << endl;

	if(time_debug)
		cout << "(=" << clock()-start_clock << "ms)"<<endl;

	return 0;
}

void MyVideoStitcher::setDebugDirPath( string dir_path )
{
	is_debug_ = true;
	debug_dir_path_ = dir_path;
}

int MyVideoStitcher::RegistEvaluation( vector<ImageFeatures> &features, vector<MatchesInfo> &pairwise_matches, vector<CameraParams> &cameras )
{
	int num_images = features.size();
	Ptr<RotationWarper> warper = warper_creator_->create(median_focal_len_);

	MatchesInfo matches_info;
	vector<vector<Point2f>> warped_fpts;
	warped_fpts.resize(num_images);
	for(int i = 0; i < num_images; i++)
	{
		int fpts_num = features[i].keypoints.size();
		warped_fpts[i].resize(fpts_num);
		Mat K;
		cameras[i].K().convertTo(K, CV_32F);
		for(int j = 0; j < fpts_num; j++)
			warped_fpts[i][j] = warper->warpPoint(features[i].keypoints[j].pt, K, cameras[i].R);
	}

	double final_total_error, final_total_inliners;
	final_total_inliners = final_total_error = 0;

	for(int i = 0; i < num_images; i++)
	{
		for(int j = i+1; j < num_images; j++)
		{
			// 特征点对
			int idx = i * num_images + j;
			matches_info = pairwise_matches[idx];
			
			int inliner_nums = matches_info.num_inliers;
			if(inliner_nums < 50)// || j != i+1)
				continue;

			int matches_size = matches_info.matches.size();
			double total_error = 0;
			for(int k = 0; k < matches_size; k ++)
			{
				if (matches_info.inliers_mask[k])
				{
					const DMatch& m = matches_info.matches[k];
					Point2f p1 = warped_fpts[i][m.queryIdx];
					Point2f p2 = warped_fpts[j][m.trainIdx];
					total_error += ((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
				}
			}
			final_total_error += total_error;
			final_total_inliners += inliner_nums;
			LOGLN("\t\t~Image" << i << "-" << j << ": total error(" << total_error <<
				"), total inliners(" << inliner_nums << "), average error(" <<
				sqrt(total_error / inliner_nums) <<")");
		}
	}
	LOGLN("\t\t~all pairs' total error(" << final_total_error <<
		"), total inliners(" << final_total_inliners << "), average error(" <<
		sqrt(final_total_error / final_total_inliners) <<")");

	return 0;
}

MyVideoStitcher::~MyVideoStitcher()
{
	freeGPU();
}

int MyVideoStitcher::stitchImage( vector<Mat> &src, Mat &pano )
{
	Prepare(src);
	if(false)
	{
		char img_name[100];
		int img_num = corners_.size();
		cout << dst_roi_ << endl;
		for(int i = 0; i < img_num; i++)
		{
			cout << src_indices_[i] << ", " << corners_[i] << ", " << sizes_[i] << endl;
			sprintf(img_name, "/masks/%d.jpg", i);
			imwrite(debug_dir_path_ + img_name, this->final_blend_masks_[i]);

			sprintf(img_name, "/weight/%d.jpg", i);
			Mat weight_img_float = total_weight_maps_[i] * 255;
			Mat weight_img;
			weight_img_float.convertTo(weight_img, CV_8U);
			imwrite(debug_dir_path_ + img_name, weight_img);
		}
	}
	StitchFrame(src, pano);
	return 0;
}

//	保存摄像机参数，文件格式如下：
//	第一行是中间焦距median_focal_len_
//	之后每一行是一个相机--
//		数据依次是focal、aspect、ppx、ppy、R、t
void MyVideoStitcher::saveCameraParam( string filename )
{
	ofstream cp_file(filename.c_str());
	cp_file << median_focal_len_ << endl;
	for(int i = 0; i < cameras_.size(); i++)
	{
		CameraParams cp = cameras_[i];
		cp_file << cp.focal << " " << cp.aspect << " " << cp.ppx << " " << cp.ppy;
		for(int r = 0; r < 3; r++)
			for(int c = 0; c < 3; c++)
				cp_file << " " << cp.R.at<float>(r, c);
		for(int r = 0; r < 3; r++)
			cp_file << " " << cp.t.at<double>(r, 0);
		cp_file << endl;
	}
	cp_file.close();
}

int MyVideoStitcher::loadCameraParam( string filename )
{
	ifstream cp_file(filename.c_str());
	string line;

	//	median_focal_len_
	if(!getline(cp_file, line))
		return -1;
	stringstream mfl_string_stream;
	mfl_string_stream << line;
	mfl_string_stream >> median_focal_len_;

	//	每行一个摄像机
	cameras_.clear();
	while(getline(cp_file, line))
	{
		stringstream cp_string_stream;
		cp_string_stream << line;
		CameraParams cp;
		cp.R.create(3, 3, CV_32F);
		cp.t.create(3, 1, CV_64F);
		cp_string_stream >> cp.focal >> cp.aspect >> cp.ppx >> cp.ppy;
		for(int r = 0; r < 3; r++)
			for(int c = 0; c < 3; c++)
				cp_string_stream >> cp.R.at<float>(r, c);
		for(int r = 0; r < 3; r++)
			cp_string_stream >> cp.t.at<double>(r, 0);
		cameras_.push_back(cp);
	}
	return 0;
}
