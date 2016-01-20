// VideoStitchOpenCV.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#include <iostream>

#include <time.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching/stitcher.hpp"

#include "video_stitcher.h"

using namespace cv;
using namespace std;


static void printUsage()
{
	cout <<
		"视频拼接.\n\n"
		"VideoStitch [flags]\n"
		"flags:\n"
		"    --camera n width height	摄像机模式，n个，分辨率width*height\n"
		"    video1 video2 ...		视频模式，输入视频路径（视频模式和摄像机模式只能开启一种）\n"
		"    --save save_path		输出视频路径（不填则强行开启拼接预览）\n"
		"    -v				开启拼接预览\n"
		"    --range start end		拼接范围，从start到end帧，end=-1表示拼接到结尾\n"
		"    -gpu			尝试使用GPU加速\n"
		"    -plane			尝试平面投影，仅在拼接视角小于140°时可用\n"
		"    -trim			尝试裁剪未填充区域，仅在平面投影时可用\n"
		"    --trim  x1 y1 x2 y2		按照x1 y1 x2 y2构成的矩形裁剪最终结果\n"
		"    --debug debug_path	debug	模式，设定debug保存目录debug_path，debug数据中包含相机参数，要确保该路径存在！\n"
		"    --cp camera_param_path	使用camera_param_path的摄像机参数\n";
}

static bool is_camera = false, is_view = false, is_save = false, is_debug = false;
static int cam_width, cam_height, cam_num;
static vector<string> video_names;
static string save_path, debug_path, cam_param_path = "";
static bool is_try_gpu = false, is_trim = false, is_trim_rect = false;
static int range_start = 0, range_end = -1;
static string warp_type = "cylindrical";
static Rect trim_rect;
static int parseCmdArgs(int argc, char* argv[])
{
	if(argc == 1)
	{
		printUsage();
		return -1;
	}

	video_names.clear();
	for(int i = 1; i < argc; i++)
	{
		if (string(argv[i]) == "--help" || string(argv[i]) == "/?")
		{
			printUsage();
			return -1;
		}
		else if(string(argv[i]) == "--camera")
		{
			cam_num		= atoi(argv[i + 1]);
			cam_width	= atoi(argv[i + 2]);
			cam_height	= atoi(argv[i + 3]);
			is_camera	= true;
			i += 3;
		}
		else if(string(argv[i]) == "--save")
		{
			save_path = argv[i+1];
			is_save = true;
			i++;
		}
		else if(string(argv[i]) == "-v")
			is_view = true;
		else if(string(argv[i]) == "--range")
		{
			range_start	= atoi(argv[i + 1]);
			range_end	= atoi(argv[i + 2]);
			i += 2;
		}
		else if(string(argv[i]) == "-gpu")
			is_try_gpu = true;
		else if(string(argv[i]) == "-trim")
			is_trim = true;
		else if(string(argv[i]) == "--trim")
		{
			is_trim = is_trim_rect = true;
			int x1 = atoi(argv[i + 1]);
			int y1 = atoi(argv[i + 2]);
			int x2 = atoi(argv[i + 3]);
			int y2 = atoi(argv[i + 4]);
			trim_rect = Rect(x1, y1, x2 - x1, y2 - y1);
			i += 4;
		}
		else if(string(argv[i]) == "--debug")
		{
			is_debug = true;
			debug_path = argv[i+1];
			i++;
		}
		else if(string(argv[i]) == "--cp")
		{
			cam_param_path = argv[i+1];
			i++;
		}
		else if(string(argv[i]) == "-plane")
			warp_type = "plane";
		else
			video_names.push_back(argv[i]);
	}
	if(!is_save)
		is_view = true;
	if(!is_camera && video_names.size() == 0)
	{
		printUsage();
		return -1;
	}
	return 0;
}

//两个样例：
//	VideoStitch data/6-2/my_cam_0.avi data/6-2/my_cam_1.avi data/6-2/my_cam_2.avi data/6-2/my_cam_3.avi data/6-2/my_cam_4.avi data/6-2/my_cam_5.avi -v -gpu
//	VideoStitch --camera 5 1280 720 -v -gpu --debug data/tmp/ --cp data/tmp/camera_param_5.dat
static int VideoStitch(int argc, char* argv[])
{
	for(int i = 0; i < argc; i++)
		printf("%s\n", argv[i]);
	int retval = parseCmdArgs(argc, argv);
	if(retval)
		return retval;

	for(int i = 0; i < video_names.size(); i++)
		cout << video_names[i] << endl;

	//	输入视频流
	vector<VideoCapture> captures;
	if(is_camera)
	{
		for(int cam_idx = 0; cam_idx < cam_num; cam_idx++)
		{
			VideoCapture cam_cap;
			if(cam_cap.open(cam_idx))
			{
				cam_cap.set(CV_CAP_PROP_FRAME_WIDTH, cam_width);
				cam_cap.set(CV_CAP_PROP_FRAME_HEIGHT, cam_height);
				cam_cap.set(CV_CAP_PROP_FPS, 15);
				captures.push_back(cam_cap);
				cout << "camera " << cam_idx << " opened successfully." << endl;
			}
			else
				break;
		}
		if(captures.size() == 0)
		{
			cout << "No camera captured. Please check!" << endl;
			return -1;
		}
	}
	else
	{
		int video_num = video_names.size();
		captures.resize(video_num);
		for(int i = 0; i < video_num; i++)
		{
			captures[i].open(video_names[i]);
			if(!captures[i].isOpened())
			{
				cout << "Fail to open " << video_names[i] << endl;
				for(int j = 0; j < i; j++) captures[j].release();
				return -1;
			}
		}
	}
	cout << "Video capture success" << endl;

	MyVideoStitcher video_stitcher;

	//	显示/保存
	video_stitcher.setPreview(is_view);
	video_stitcher.setSave(is_save);
	video_stitcher.setRange(range_start, range_end);

	//	拼接参数
	video_stitcher.setTryGPU(is_try_gpu);
	video_stitcher.setTrim(is_trim);
	if(cam_param_path != "")
		video_stitcher.loadCameraParam(cam_param_path);
	if(is_debug)
		video_stitcher.setDebugDirPath(debug_path);
	if(is_trim_rect)
		video_stitcher.setTrim(trim_rect);
	video_stitcher.setWarpType(warp_type);

	//	拼接
	video_stitcher.stitch(captures, save_path);

	//	释放资源
	for(int i = 0; i < captures.size(); i++)
		captures[i].release();

	cout << "Released all" << endl;
	
	return 0;
}

int main(int argc, char* argv[])
{
	VideoStitch(argc, argv);
	system("pause");
	return 0;
}
