/*  Copyright 2017 onlyliu(997737609@qq.com).                                */
/*                                                                        */
/*  part of source code come from https://github.com/qibao77/cornerDetect */
/*  Automatic Camera and Range Sensor Calibration using a single Shot     */
/*  this project realize the papar: Automatic Camera and Range Sensor     */
/*  Calibration using a single Shot                                       */


#include "CornerDetAC.h"
#include "corealgmatlab.h"


using namespace cv;
using namespace std;

//#define show_ 

CornerDetAC::CornerDetAC()
{
}


CornerDetAC::~CornerDetAC()
{
}
CornerDetAC::CornerDetAC(cv::Mat img)
{
	//3 scales
	radius.push_back(4);
	radius.push_back(8);
	radius.push_back(12);

	templateProps.push_back(Point2f((dtype)0, (dtype)CV_PI / 2));
	templateProps.push_back(Point2f((dtype)CV_PI / 4, (dtype)-CV_PI / 4));
	templateProps.push_back(Point2f((dtype)0, (dtype)CV_PI / 2));
	templateProps.push_back(Point2f((dtype)CV_PI / 4, (dtype)-CV_PI / 4));
	templateProps.push_back(Point2f((dtype)0, (dtype)CV_PI / 2));
	templateProps.push_back(Point2f((dtype)CV_PI / 4, (dtype)-CV_PI / 4));
}

//Normal probability density function (pdf).
dtype CornerDetAC::normpdf(dtype dist, dtype mu, dtype sigma)
{
	dtype s = exp(-0.5*(dist - mu)*(dist - mu) / (sigma*sigma));
	s = s / (std::sqrt(2 * CV_PI)*sigma);
	return s;
}


//**************************生成扇形核模板*****************************//
// 功能：创建4个扇形核(kernelA/B/C/D)，用于检测棋盘格角点
// 原理：棋盘格角点由4个象限组成，相邻象限有明暗对比
//      通过两条边界线(angle1, angle2)将圆形区域划分为4个扇形
// 参数：
//   angle1, angle2: 两条分界线的角度（通常为45度和90度配置）
//   kernelSize: 核的半径，最终核大小为 (kernelSize*2+1) x (kernelSize*2+1)
//   kernelA~D: 输出的4个扇形核，对应角点的4个象限
//*************************************************************************//
void CornerDetAC::createkernel(float angle1, float angle2, int kernelSize, Mat &kernelA, Mat &kernelB, Mat &kernelC, Mat &kernelD)
{

	// 核的尺寸：以kernelSize为半径的正方形
	int width = (int)kernelSize * 2 + 1;
	int height = (int)kernelSize * 2 + 1;
	kernelA = cv::Mat::zeros(height, width, mtype);
	kernelB = cv::Mat::zeros(height, width, mtype);
	kernelC = cv::Mat::zeros(height, width, mtype);
	kernelD = cv::Mat::zeros(height, width, mtype);

	// 遍历核的每个像素，根据位置分配到不同的扇形核
	for (int u = 0; u < width; ++u){
		for (int v = 0; v < height; ++v){
			// 将坐标原点移动到核中心
			dtype vec[] = { u - kernelSize, v - kernelSize };
			
			// 计算当前点到核中心的距离
			dtype dis = std::sqrt(vec[0] * vec[0] + vec[1] * vec[1]);
			
			// 通过旋转变换计算点相对于两条分界线的位置
			// side1、side2的符号决定了点位于哪个象限
			dtype side1 = vec[0] * (-sin(angle1)) + vec[1] * cos(angle1);
			dtype side2 = vec[0] * (-sin(angle2)) + vec[1] * cos(angle2);
			
			// 根据side1和side2的符号，将点分配到4个扇形核中
			// 每个扇形核使用高斯权重，距离中心越近权重越大
			if (side1 <= -0.1&&side2 <= -0.1){
				// 第一象限（左下）
				kernelA.ptr<dtype>(v)[u] = normpdf(dis, 0, kernelSize / 2);
			}
			if (side1 >= 0.1&&side2 >= 0.1){
				// 第三象限（右上），与A对角
				kernelB.ptr<dtype>(v)[u] = normpdf(dis, 0, kernelSize / 2);
			}
			if (side1 <= -0.1&&side2 >= 0.1){
				// 第二象限（左上）
				kernelC.ptr<dtype>(v)[u] = normpdf(dis, 0, kernelSize / 2);
			}
			if (side1 >= 0.1&&side2 <= -0.1){
				// 第四象限（右下），与C对角
				kernelD.ptr<dtype>(v)[u] = normpdf(dis, 0, kernelSize / 2);
			}
		}
	}
	
	// 归一化：使每个核的和为1，确保响应值在合理范围内
	kernelA = kernelA / cv::sum(kernelA)[0];
	kernelB = kernelB / cv::sum(kernelB)[0];
	kernelC = kernelC / cv::sum(kernelC)[0];
	kernelD = kernelD / cv::sum(kernelD)[0];

}

//**************************计算两个矩阵的逐元素最小值*****************************//
// 功能：对两个矩阵的每个对应元素取最小值，存入目标矩阵
// 用途：在角点响应计算中，用于组合不同条件下的响应值
//*************************************************************************//
void CornerDetAC::getMin(Mat src1, Mat src2, Mat &dst){
	int rowsLeft = src1.rows;
	int colsLeft = src1.cols;
	int rowsRight = src2.rows;
	int colsRight = src2.cols;
	if (rowsLeft != rowsRight || colsLeft != colsRight)return;

	int channels = src1.channels();

	int nr = rowsLeft;
	int nc = colsLeft;
	if (src1.isContinuous()){
		nc = nc*nr;
		nr = 1;
		//std::cout<<"continue"<<std::endl;
	}
	for (int i = 0; i < nr; i++){
		const dtype* dataLeft = src1.ptr<dtype>(i);
		const dtype* dataRight = src2.ptr<dtype>(i);
		dtype* dataResult = dst.ptr<dtype>(i);
		for (int j = 0; j < nc*channels; ++j){
			dataResult[j] = (dataLeft[j] < dataRight[j]) ? dataLeft[j] : dataRight[j];
		}
	}
}
//**************************计算两个矩阵的逐元素最大值*****************************//
// 功能：对两个矩阵的每个对应元素取最大值，存入目标矩阵
// 用途：在角点响应计算中，用于合并多个尺度的响应结果
//*************************************************************************//
void CornerDetAC::getMax(Mat src1, Mat src2, Mat &dst)
{
	int rowsLeft = src1.rows;
	int colsLeft = src1.cols;
	int rowsRight = src2.rows;
	int colsRight = src2.cols;
	if (rowsLeft != rowsRight || colsLeft != colsRight)return;

	int channels = src1.channels();

	int nr = rowsLeft;
	int nc = colsLeft;
	if (src1.isContinuous()){
		nc = nc*nr;
		nr = 1;
		//std::cout<<"continue"<<std::endl;
	}
	for (int i = 0; i < nr; i++){
		const dtype* dataLeft = src1.ptr<dtype>(i);
		const dtype* dataRight = src2.ptr<dtype>(i);
		dtype* dataResult = dst.ptr<dtype>(i);
		for (int j = 0; j < nc*channels; ++j){
			dataResult[j] = (dataLeft[j] >= dataRight[j]) ? dataLeft[j] : dataRight[j];
		}
	}
}
//**************************计算图像梯度角度和权重*****************************//
// 功能：计算图像的梯度方向和梯度强度
// 原理：使用Sobel算子计算x和y方向的梯度，然后转换为极坐标
// 输出：
//   imgDu: x方向梯度（水平方向）
//   imgDv: y方向梯度（垂直方向）
//   imgAngle: 梯度方向角度（0~π）
//   imgWeight: 梯度强度（幅值）
//*************************************************************************//
void CornerDetAC::getImageAngleAndWeight(Mat img, Mat &imgDu, Mat &imgDv, Mat &imgAngle, Mat &imgWeight)
{
	
	Mat sobelKernel(3, 3, mtype);
	Mat sobelKernelTrs(3, 3, mtype);
	
	// 构造Sobel滤波核：[-1, 0, 1] 用于检测水平边缘
	sobelKernel.col(0).setTo(cv::Scalar(-1.0));
	sobelKernel.col(1).setTo(cv::Scalar(0.0));
	sobelKernel.col(2).setTo(cv::Scalar(1.0));

	// 转置得到垂直方向的Sobel核
	sobelKernelTrs = sobelKernel.t();

	// 卷积计算x和y方向的梯度
	imgDu = corealgmatlab::conv2(img, sobelKernel, CONVOLUTION_SAME);
	imgDv = corealgmatlab::conv2(img, sobelKernelTrs, CONVOLUTION_SAME);

	if (imgDu.size() != imgDv.size())return;

	// 将笛卡尔坐标系的梯度(dx,dy)转换为极坐标(幅值,角度)
	cartToPolar(imgDu, imgDv, imgWeight, imgAngle, false);
	
	// 将梯度角度归一化到 [0, π] 区间
	// 原因：角点的边缘方向是无向的，180度和0度表示同一方向
	for (int i = 0; i < imgDu.rows; i++)
	{
		for (int j = 0; j < imgDu.cols; j++)
		{
			dtype* dataAngle = imgAngle.ptr<dtype>(i);
			if (dataAngle[j] < 0)
				dataAngle[j] = dataAngle[j] + CV_PI;
			else if (dataAngle[j] > CV_PI)
				dataAngle[j] = dataAngle[j] - CV_PI;
		}
	}
	/*
	for (int i = 0; i < imgDu.rows; i++)
	{
		dtype* dataDv = imgDv.ptr<dtype>(i);
		dtype* dataDu = imgDu.ptr<dtype>(i);
		dtype* dataAngle = imgAngle.ptr<dtype>(i);
		dtype* dataWeight = imgWeight.ptr<dtype>(i);
		for (int j = 0; j < imgDu.cols; j++)
		{
			if (dataDu[j] > 0.000001)
			{
				dataAngle[j] = atan2((dtype)dataDv[j], (dtype)dataDu[j]);
				if (dataAngle[j] < 0)dataAngle[j] = dataAngle[j] + CV_PI;
				else if (dataAngle[j] > CV_PI)dataAngle[j] = dataAngle[j] - CV_PI;
			}
			dataWeight[j] = std::sqrt((dtype)dataDv[j] * (dtype)dataDv[j] + (dtype)dataDu[j] * (dtype)dataDu[j]);
		}
	}
	*/
}
//**************************非极大值抑制（NMS）*****************************//
// 功能：从角点响应图中提取局部最大值点，抑制非极大值点
// 原理：将图像划分为多个patch，每个patch内只保留响应最强的角点
// 参数：
//   inputCorners: 输入的角点响应图（每个像素的响应强度）
//   outputCorners: 输出的角点位置列表
//   patchSize: 检测patch的大小，控制相邻角点的最小间距
//   threshold: 角点响应的最小阈值，低于此值的点被过滤
//   margin: 图像边缘的安全边界，避免在边缘检测角点
//*************************************************************************//
void CornerDetAC::nonMaximumSuppression(Mat& inputCorners, vector<Point2f>& outputCorners, int patchSize, dtype threshold, int margin)
{
	if (inputCorners.size <= 0)
	{
		cout << "The imput mat is empty!" << endl; return;
	}
	
	// 以patchSize为步长滑动窗口，遍历整个图像
	for (int i = margin + patchSize; i <= inputCorners.cols - (margin + patchSize+1); i = i + patchSize + 1)
	{
		for (int j = margin + patchSize; j <= inputCorners.rows - (margin + patchSize+1); j = j + patchSize + 1)
		{
			// 在当前patch内寻找局部最大值
			dtype maxVal = inputCorners.ptr<dtype>(j)[i];
			int maxX = i; int maxY = j;
			for (int m = i; m <= i + patchSize ; m++)
			{
				for (int n = j; n <= j + patchSize ; n++)
				{
					dtype temp = inputCorners.ptr<dtype>(n)[m];
					if (temp > maxVal)
					{
						maxVal = temp; maxX = m; maxY = n;
					}
				}
			}
			
			// 如果局部最大值小于阈值，跳过该patch
			if (maxVal < threshold)continue;
			
			// 二次检验：在更大的邻域内确认是否为真正的局部最大值
			// 避免在patch边界处遗漏更强的响应点
			int flag = 0;
			for (int m = maxX - patchSize; m <= min(maxX + patchSize, inputCorners.cols - margin-1); m++)
			{
				for (int n = maxY - patchSize; n <= min(maxY + patchSize, inputCorners.rows - margin-1); n++)
				{
					if (inputCorners.ptr<dtype>(n)[m]>maxVal && (m<i || m>i + patchSize || n<j || n>j + patchSize))
					{
						flag = 1; break;
					}
				}
				if (flag)break;
			}
			if (flag)continue;
			
			// 确认为有效角点，加入输出列表
			outputCorners.push_back(Point(maxX, maxY));
			std::vector<dtype> e1(2, 0.0);
			std::vector<dtype> e2(2, 0.0);
			cornersEdge1.push_back(e1);
			cornersEdge2.push_back(e2);
		}
	}
}

int cmp(const pair<dtype, int> &a, const pair<dtype, int> &b)
{
	return a.first > b.first;
}

//find modes of smoothed histogram
void CornerDetAC::findModesMeanShift(vector<dtype> hist, vector<dtype> &hist_smoothed, vector<pair<dtype, int>> &modes, dtype sigma){
	//efficient mean - shift approximation by histogram smoothing
	//compute smoothed histogram
	bool allZeros = true;
	for (int i = 0; i < hist.size(); i++)
	{
		dtype sum = 0;
		for (int j = -(int)round(2 * sigma); j <= (int)round(2 * sigma); j++)
		{
			int idx = 0;
			idx = (i + j) % hist.size();
			sum = sum + hist[idx] * normpdf(j, 0, sigma);
		}
		hist_smoothed[i] = sum;
		if (abs(hist_smoothed[i] - hist_smoothed[0]) > 0.0001)allZeros = false;// check if at least one entry is non - zero
		//(otherwise mode finding may run infinitly)
	}
	if (allZeros)return;

	//mode finding
	for (int i = 0; i < hist.size(); i++)
	{
		int j = i;
		while (true)
		{
			float h0 = hist_smoothed[j];
			int j1 = (j + 1) % hist.size();
			int j2 = (j - 1) % hist.size();
			float h1 = hist_smoothed[j1];
			float h2 = hist_smoothed[j2];
			if (h1 >= h0 && h1 >= h2)
				j = j1;
			else if (h2 > h0 && h2 > h1)
				j = j2;
			else 
				break;
		}
		bool ys = true;
		if (modes.size() == 0)
		{
			ys = true;
		}
		else
		{
			for (int k = 0; k < modes.size(); k++)
			{
				if (modes[k].second == j)
				{
					ys = false;
					break;
				}
			}
		}
		if (ys == true)
		{
			modes.push_back(std::make_pair(hist_smoothed[j], j));
		}
	}
	std::sort(modes.begin(), modes.end(), cmp);
}

//**************************估计角点的边缘方向*****************************//
// 功能：通过梯度方向直方图估计角点处两条边缘的方向
// 原理：棋盘格角点由两条边缘组成，统计角点邻域内的梯度方向分布，
//      找到直方图的两个峰值即为两条边缘的方向
// 参数：
//   imgAngle: 角点邻域的梯度方向图
//   imgWeight: 角点邻域的梯度强度图
//   index: 当前处理的角点索引
//*************************************************************************//
void CornerDetAC::edgeOrientations(Mat imgAngle, Mat imgWeight, int index){
	// 直方图的bin数量（将0~π划分为32个区间）
	int binNum = 32;

	// 将角度和权重图转换为向量，便于统计
	if (imgAngle.size() != imgWeight.size())return;
	vector<dtype> vec_angle, vec_weight;
	for (int i = 0; i < imgAngle.cols; i++)
	{
		for (int j = 0; j < imgAngle.rows; j++)
		{
			// 将法向量角度转换为边缘方向角度（旋转90度）
			float angle = imgAngle.ptr<dtype>(j)[i] + CV_PI / 2;
			angle = angle > CV_PI ? (angle - CV_PI) : angle;
			vec_angle.push_back(angle);

			vec_weight.push_back(imgWeight.ptr<dtype>(j)[i]);
		}
	}

	// 创建加权的方向直方图
	dtype pin = (CV_PI / binNum);  // 每个bin的角度范围
	vector<dtype> angleHist(binNum, 0);
	for (int i = 0; i < vec_angle.size(); i++)
	{
		// 计算角度所属的bin，并累加梯度权重
		int bin = max(min((int)floor(vec_angle[i] / pin), binNum - 1), 0);
		angleHist[bin] = angleHist[bin] + vec_weight[i];
	}

	// 使用Mean-Shift算法平滑直方图并寻找峰值（模态）
	vector<dtype> hist_smoothed(angleHist);
	vector<std::pair<dtype, int> > modes;
	findModesMeanShift(angleHist, hist_smoothed, modes, 1);

	// 如果只有一个或没有峰值，说明不是有效的角点
	if (modes.size() <= 1)return;

	// 计算两个主要峰值对应的方向角度
	float fo[2];
	fo[0] = modes[0].second*pin;  // 第一条边缘方向
	fo[1] = modes[1].second*pin;  // 第二条边缘方向
	dtype deltaAngle = 0;
	
	// 按角度大小排序
	if (fo[0] > fo[1])
	{
		dtype t = fo[0];
		fo[0] = fo[1];
		fo[1] = t;
	}

	// 计算两条边缘的夹角
	deltaAngle = MIN(fo[1] - fo[0], fo[0] - fo[1] + (dtype)CV_PI);
	
	// 如果夹角太小（<17度），认为不是有效的角点
	if (deltaAngle <= 0.3)return;

	// 将角度转换为单位向量，存储两条边缘的方向
	cornersEdge1[index][0] = cos(fo[0]);
	cornersEdge1[index][1] = sin(fo[0]);
	cornersEdge2[index][0] = cos(fo[1]);
	cornersEdge2[index][1] = sin(fo[1]);
}

float CornerDetAC::norm2d(cv::Point2f o)
{
	return sqrt(o.x*o.x + o.y*o.y);
}
//**************************亚像素精化角点位置*****************************//
// 功能：对检测到的角点进行亚像素级精化，提高定位精度
// 原理：1) 先估计角点处两条边缘的方向
//      2) 根据梯度信息精化边缘方向
//      3) 根据两条边缘的交点精化角点位置
// 参数：
//   cornors: 输入输出的角点位置（会被精化）
//   imgDu, imgDv: 图像的x和y方向梯度
//   imgAngle, imgWeight: 梯度方向和强度
//   radius: 精化时使用的邻域半径
//*************************************************************************//
void CornerDetAC::refineCorners(vector<Point2f> &cornors, Mat imgDu, Mat imgDv, Mat imgAngle, Mat imgWeight, float radius){
	// 图像尺寸
	int width = imgDu.cols;
	int height = imgDu.rows;
	
	// 遍历所有候选角点
	for (int i = 0; i < cornors.size(); i++)
	{
		// 提取当前角点的位置
		int cu = cornors[i].x;
		int cv = cornors[i].y;
		
		// 提取角点邻域，用于估计边缘方向
		int startX, startY, ROIwidth, ROIheight;
		startX = MAX(cu - radius, (dtype)0);
		startY = MAX(cv - radius, (dtype)0);
		ROIwidth = MIN(cu + radius + 1, (dtype)width - 1) - startX;
		ROIheight = MIN(cv + radius + 1, (dtype)height - 1) - startY;

		Mat roiAngle, roiWeight;
		roiAngle = imgAngle(Rect(startX, startY, ROIwidth, ROIheight));
		roiWeight = imgWeight(Rect(startX, startY, ROIwidth, ROIheight));
		
		// 估计角点处两条边缘的初始方向
		edgeOrientations(roiAngle, roiWeight, i);

		// 如果边缘方向无效，跳过该角点
		if (cornersEdge1[i][0] == 0 && cornersEdge1[i][1] == 0 || cornersEdge2[i][0] == 0 && cornersEdge2[i][1] == 0)
			continue;

		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		//%%%%   边缘方向精化阶段    %%%%
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		// 构建两条边缘的结构张量（Structure Tensor）
		cv::Mat A1 = cv::Mat::zeros(cv::Size(2, 2), mtype);
		cv::Mat A2 = cv::Mat::zeros(cv::Size(2, 2), mtype);

		// 遍历邻域内的每个像素
		for (int u = startX; u < startX + ROIwidth; u++)
			for (int v = startY; v < startY + ROIheight; v++)
			{
				// 获取像素的梯度向量并归一化
				cv::Point2f o(imgDu.at<dtype>(v, u), imgDv.at<dtype>(v, u));
				float no = norm2d(o);
				if (no < 0.1)
					continue;
				o = o / no;
				
				// 精化第一条边缘的方向
				// 通过点积判断梯度是否垂直于边缘方向（内点检测）
				dtype t0 = abs(o.x*cornersEdge1[i][0] + o.y*cornersEdge1[i][1]);
				if (t0 < 0.25) // 如果梯度接近垂直于边缘，则为内点
				{
					// 累加梯度外积，构建结构张量 A1 = Σ[grad*grad^T]
					Mat addtion(1, 2, mtype);
					addtion.col(0).setTo(imgDu.at<dtype>(v, u));
					addtion.col(1).setTo(imgDv.at<dtype>(v, u));
					Mat addtionu = imgDu.at<dtype>(v, u)*addtion;
					Mat addtionv = imgDv.at<dtype>(v, u)*addtion;
					for (int j = 0; j < A1.cols; j++)
					{
						A1.at<dtype>(0, j) = A1.at<dtype>(0, j) + addtionu.at<dtype>(0, j);
						A1.at<dtype>(1, j) = A1.at<dtype>(1, j) + addtionv.at<dtype>(0, j);
					}
				}
				
				// 精化第二条边缘的方向（同样的过程）
				dtype t1 = abs(o.x*cornersEdge2[i][0] + o.y*cornersEdge2[i][1]);
				if (t1 < 0.25) // 内点检测
				{
					// 累加梯度外积，构建结构张量 A2
					Mat addtion(1, 2, mtype);
					addtion.col(0).setTo(imgDu.at<dtype>(v, u));
					addtion.col(1).setTo(imgDv.at<dtype>(v, u));
					Mat addtionu = imgDu.at<dtype>(v, u)*addtion;
					Mat addtionv = imgDv.at<dtype>(v, u)*addtion;
					for (int j = 0; j < A2.cols; j++)
					{
						A2.at<dtype>(0, j) = A2.at<dtype>(0, j) + addtionu.at<dtype>(0, j);
						A2.at<dtype>(1, j) = A2.at<dtype>(1, j) + addtionv.at<dtype>(0, j);
					}
				}
			}//end for
			
		// 通过特征值分解获得精化后的边缘方向
		// 结构张量的最小特征向量即为边缘方向
		cv::Mat v1, foo1;
		cv::Mat v2, foo2;
		cv::eigen(A1, v1, foo1);  // A1的特征分解
		cv::eigen(A2, v2, foo2);  // A2的特征分解
		
		// 取最小特征值对应的特征向量作为精化后的边缘方向
		cornersEdge1[i][0] = -foo1.at<dtype>(1, 0);
		cornersEdge1[i][1] = -foo1.at<dtype>(1, 1);
		cornersEdge2[i][0] = -foo2.at<dtype>(1, 0);
		cornersEdge2[i][1] = -foo2.at<dtype>(1, 1);

		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		//%%%%   角点位置精化阶段    %%%%
		//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		// 使用最小二乘法求解两条边缘的交点作为精化后的角点位置
		cv::Mat G = cv::Mat::zeros(cv::Size(2, 2), mtype);  // 法方程系数矩阵
		cv::Mat b = cv::Mat::zeros(cv::Size(1, 2), mtype);  // 法方程右端项
		
		for (int u = startX; u < startX + ROIwidth; u++)
			for (int v = startY; v < startY + ROIheight; v++)
			{
				// 获取像素的梯度向量并归一化
				cv::Point2f o(imgDu.at<dtype>(v, u), imgDv.at<dtype>(v, u));
				float no = norm2d(o);
				if (no < 0.1)
					continue;
				o = o / no;
				
				// 鲁棒的亚像素角点估计（不考虑中心像素）
				if (u != cu || v != cv)
				{
					// 计算像素相对角点的位置向量
					cv::Point2f w(u - cu, v - cv);
					
					// 计算位置向量在两条边缘方向上的投影
					float wvv1 = w.x*cornersEdge1[i][0] + w.y*cornersEdge1[i][1];
					float wvv2 = w.x*cornersEdge2[i][0] + w.y*cornersEdge2[i][1];

					// 计算投影向量
					cv::Point2f wv1(wvv1 * cornersEdge1[i][0], wvv1 * cornersEdge1[i][1]);
					cv::Point2f wv2(wvv2 * cornersEdge2[i][0], wvv2 * cornersEdge2[i][1]);
					
					// 计算像素到两条边缘的垂直距离
					cv::Point2f vd1(w.x - wv1.x, w.y - wv1.y);
					cv::Point2f vd2(w.x - wv2.x, w.y - wv2.y);
					dtype d1 = norm2d(vd1), d2 = norm2d(vd2);
					
					// 如果像素靠近某条边缘（距离<3像素）且梯度垂直于边缘
					if ((d1 < 3) && abs(o.x*cornersEdge1[i][0] + o.y*cornersEdge1[i][1]) < 0.25 \
						|| (d2 < 3) && abs(o.x*cornersEdge2[i][0] + o.y*cornersEdge2[i][1]) < 0.25)
					{
						// 将该像素纳入最小二乘系统
						// 构建线性方程: G*p = b，其中p是角点位置
						dtype du = imgDu.at<dtype>(v, u), dv = imgDv.at<dtype>(v, u);
						cv::Mat uvt = (Mat_<dtype>(2, 1) << u, v);
						cv::Mat H = (Mat_<dtype>(2, 2) << du*du, du*dv, dv*du, dv*dv);
						G = G + H;
						cv::Mat t = H*(uvt);
						b = b + t;
					}
				}	
			}//endfor
			
		// 检查系数矩阵G的秩，只有满秩时才能求解
		Mat s, u, v;
		SVD::compute(G, s, u, v);
		int rank = 0;
		for (int k = 0; k < s.rows; k++)
		{
			if (s.at<dtype>(k, 0) > 0.0001 || s.at<dtype>(k, 0) < -0.0001)// 非零奇异值
			{
				rank++;
			}
		}
		
		if (rank == 2)  // 系统满秩，可以求解
		{
			// 求解线性方程，得到精化后的角点位置
			cv::Mat mp = G.inv()*b;
			cv::Point2f  corner_pos_new(mp.at<dtype>(0, 0), mp.at<dtype>(1, 0));
			
			// 如果位置更新太大（>4像素），认为精化失败，标记为无效角点
			if (norm2d(cv::Point2f(corner_pos_new.x - cu, corner_pos_new.y - cv)) >= 4)
			{
				cornersEdge1[i][0] = 0;
				cornersEdge1[i][1] = 0;
				cornersEdge2[i][0] = 0;
				cornersEdge2[i][1] = 0;
			}
			else  // 更新角点位置
			{
				cornors[i].x = mp.at<dtype>(0, 0);
				cornors[i].y = mp.at<dtype>(1, 0);
			}
		}
		else  // 系统欠秩，无法求解，标记为无效角点
		{
			cornersEdge1[i][0] = 0;
			cornersEdge1[i][1] = 0;
			cornersEdge2[i][0] = 0;
			cornersEdge2[i][1] = 0;
		}
	}
}

//compute corner statistics
void CornerDetAC::cornerCorrelationScore(Mat img, Mat imgWeight, vector<Point2f> cornersEdge, float &score){
	//center
	int c[] = { imgWeight.cols / 2, imgWeight.cols / 2 };

	//compute gradient filter kernel(bandwith = 3 px)
	Mat img_filter = Mat::ones(imgWeight.size(), imgWeight.type());
	img_filter = img_filter*-1;
	for (int i = 0; i < imgWeight.cols; i++)
	{
		for (int j = 0; j < imgWeight.rows; j++)
		{
			Point2f p1 = Point2f(i - c[0], j - c[1]);
			Point2f p2 = Point2f(p1.x*cornersEdge[0].x*cornersEdge[0].x + p1.y*cornersEdge[0].x*cornersEdge[0].y,
				p1.x*cornersEdge[0].x*cornersEdge[0].y + p1.y*cornersEdge[0].y*cornersEdge[0].y);
			Point2f p3 = Point2f(p1.x*cornersEdge[1].x*cornersEdge[1].x + p1.y*cornersEdge[1].x*cornersEdge[1].y,
				p1.x*cornersEdge[1].x*cornersEdge[1].y + p1.y*cornersEdge[1].y*cornersEdge[1].y);
			float norm1 = sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
			float norm2 = sqrt((p1.x - p3.x)*(p1.x - p3.x) + (p1.y - p3.y)*(p1.y - p3.y));
			if (norm1 <= 1.5 || norm2 <= 1.5)
			{
				img_filter.ptr<dtype>(j)[i] = 1;
			}
		}
	}

	//normalize
	Mat mean, std, mean1, std1;
	meanStdDev(imgWeight, mean, std);
	meanStdDev(img_filter, mean1, std1);
	for (int i = 0; i < imgWeight.cols; i++)
	{
		for (int j = 0; j < imgWeight.rows; j++)
		{
			imgWeight.ptr<dtype>(j)[i] = (dtype)(imgWeight.ptr<dtype>(j)[i] - mean.ptr<double>(0)[0]) / (dtype)std.ptr<double>(0)[0];
			img_filter.ptr<dtype>(j)[i] = (dtype)(img_filter.ptr<dtype>(j)[i] - mean1.ptr<double>(0)[0]) / (dtype)std1.ptr<double>(0)[0];
		}
	}

	//convert into vectors
	vector<float> vec_filter, vec_weight;
	for (int i = 0; i < imgWeight.cols; i++)
	{
		for (int j = 0; j < imgWeight.rows; j++)
		{
			vec_filter.push_back(img_filter.ptr<dtype>(j)[i]);
			vec_weight.push_back(imgWeight.ptr<dtype>(j)[i]);
		}
	}

	//compute gradient score
	float sum = 0;
	for (int i = 0; i < vec_weight.size(); i++)
	{
		sum += vec_weight[i] * vec_filter[i];
	}
	sum = (dtype)sum / (dtype)(vec_weight.size() - 1);
	dtype score_gradient = sum >= 0 ? sum : 0;

	//create intensity filter kernel
	Mat kernelA, kernelB, kernelC, kernelD;
	createkernel(atan2(cornersEdge[0].y, cornersEdge[0].x), atan2(cornersEdge[1].y, cornersEdge[1].x), c[0], kernelA, kernelB, kernelC, kernelD);//1.1 �������ֺ�

	//checkerboard responses
	float a1, a2, b1, b2;
	a1 = kernelA.dot(img);
	a2 = kernelB.dot(img);
	b1 = kernelC.dot(img);
	b2 = kernelD.dot(img);

	float mu = (a1 + a2 + b1 + b2) / 4;

	float score_a = (a1 - mu) >= (a2 - mu) ? (a2 - mu) : (a1 - mu);
	float score_b = (mu - b1) >= (mu - b2) ? (mu - b2) : (mu - b1);
	float score_1 = score_a >= score_b ? score_b : score_a;

	score_b = (b1 - mu) >= (b2 - mu) ? (b2 - mu) : (b1 - mu);
	score_a = (mu - a1) >= (mu - a2) ? (mu - a2) : (mu - a1);
	float score_2 = score_a >= score_b ? score_b : score_a;

	float score_intensity = score_1 >= score_2 ? score_1 : score_2;
	score_intensity = score_intensity > 0.0 ? score_intensity : 0.0;

	score = score_gradient*score_intensity;
}
//score corners
void CornerDetAC::scoreCorners(Mat img, Mat imgAngle, Mat imgWeight, vector<Point2f> &cornors, vector<int> radius, vector<float> &score){
	//for all corners do
	for (int i = 0; i < cornors.size(); i++)
	{
		//corner location
		int u = cornors[i].x+0.5;
		int v = cornors[i].y+0.5;
		if (i == 278)
		{
			int aaa = 0;
		}
		//compute corner statistics @ radius 1
		vector<float> scores;
		for (int j = 0; j < radius.size(); j++)
		{
			scores.push_back(0);
			int r = radius[j];
			if (u > r&&u <= (img.cols - r - 1) && v>r && v <= (img.rows - r - 1))
			{
				int startX, startY, ROIwidth, ROIheight;
				startX = u - r;
				startY = v - r;
				ROIwidth = 2 * r + 1;
				ROIheight = 2 * r + 1;

				Mat sub_img = img(Rect(startX, startY, ROIwidth, ROIheight)).clone();
				Mat sub_imgWeight = imgWeight(Rect(startX, startY, ROIwidth, ROIheight)).clone();
				vector<Point2f> cornersEdge;
				cornersEdge.push_back(Point2f((float)cornersEdge1[i][0], (float)cornersEdge1[i][1]));
				cornersEdge.push_back(Point2f((float)cornersEdge2[i][0], (float)cornersEdge2[i][1]));
				cornerCorrelationScore(sub_img, sub_imgWeight, cornersEdge, scores[j]);
			}
		}
		//take highest score
		score.push_back(*max_element(begin(scores), end(scores)));
	}

}


//**************************主角点检测函数*****************************//
// 功能：检测棋盘格角点的完整流程
// 原理：基于论文 "Automatic Camera and Range Sensor Calibration using a single Shot"
//      1. 多尺度模板匹配检测候选角点
//      2. 非极大值抑制筛选角点
//      3. 亚像素精化角点位置和方向
//      4. 角点评分和过滤
// 参数：
//   Src: 输入图像
//   resultCornors: （未使用）
//   mcorners: 输出的角点结构体（包含位置、方向、分数）
//   scoreThreshold: 角点分数阈值
//   isrefine: 是否进行亚像素精化
//*************************************************************************//
void CornerDetAC::detectCorners(Mat &Src, vector<Point> &resultCornors, Corners& mcorners,  dtype scoreThreshold, bool isrefine)
{
	Mat gray, imageNorm;
	gray = Mat(Src.size(), CV_8U);

	// 转换为灰度图像
	if (Src.channels() == 3)
	{
		cvtColor(Src, gray, COLOR_BGR2GRAY);
	}
	else
	{
		gray = Src.clone();
	}

	// 高斯模糊，减少噪声影响
	cv::GaussianBlur(gray, gray, cv::Size(9,9), 1.5);

	// 归一化图像到[0,1]范围，便于后续处理
	normalize(gray, imageNorm, 0, 1, cv::NORM_MINMAX, mtype);

	// 初始化角点响应图（存储每个像素作为角点的强度）
	Mat imgCorners = Mat::zeros(imageNorm.size(), mtype);

	// 临时存储变量：用于存储4个扇形核的卷积结果
	Mat imgCornerA1(imageNorm.size(), mtype);  // 扇形核A的响应
	Mat imgCornerB1(imageNorm.size(), mtype);  // 扇形核B的响应
	Mat imgCornerC1(imageNorm.size(), mtype);  // 扇形核C的响应
	Mat imgCornerD1(imageNorm.size(), mtype);  // 扇形核D的响应

	Mat imgCornerA(imageNorm.size(), mtype);
	Mat imgCornerB(imageNorm.size(), mtype);
	Mat imgCorner1(imageNorm.size(), mtype);  // 情况1的角点响应（A和B为白，C和D为黑）
	Mat imgCorner2(imageNorm.size(), mtype);  // 情况2的角点响应（A和B为黑，C和D为白）
	Mat imgCornerMean(imageNorm.size(), mtype);  // 4个扇形区域的平均值

	double t = (double)getTickCount();

	//==================== 多尺度、多方向模板匹配 ====================//
	// 使用3个尺度（radius: 4, 8, 12），每个尺度2个方向（45度和90度）
	// 共6个模板配置（i=0~5，i/2对应尺度索引）
	for (int i = 0; i < 6; i++)
	{
		Mat kernelA1, kernelB1, kernelC1, kernelD1;
		// 创建当前尺度和方向的4个扇形核
		createkernel(templateProps[i].x, templateProps[i].y, radius[i / 2], kernelA1, kernelB1, kernelC1, kernelD1);

		// 使用4个扇形核对图像进行卷积，获得各象限的响应
#if 1
		imgCornerA1 = corealgmatlab::conv2(imageNorm, kernelA1, CONVOLUTION_SAME);
		imgCornerB1 = corealgmatlab::conv2(imageNorm, kernelB1, CONVOLUTION_SAME);
		imgCornerC1 = corealgmatlab::conv2(imageNorm, kernelC1, CONVOLUTION_SAME);
		imgCornerD1 = corealgmatlab::conv2(imageNorm, kernelD1, CONVOLUTION_SAME);
#else	
		filter2D(imageNorm, imgCornerA1, mtype, kernelA1);//a1
		filter2D(imageNorm, imgCornerB1, mtype, kernelB1);//a2
		filter2D(imageNorm, imgCornerC1, mtype, kernelC1);//b1
		filter2D(imageNorm, imgCornerD1, mtype, kernelD1);//b2
#endif	
		
		// 计算4个象限的平均响应值
		imgCornerMean = (imgCornerA1 + imgCornerB1 + imgCornerC1 + imgCornerD1) / 4.0;
		
		// 情况1: 对角象限A和B为亮色（白），C和D为暗色（黑）
		// 计算响应强度：min(A-mean, B-mean) 表示A、B都比平均亮
		//              min(mean-C, mean-D) 表示C、D都比平均暗
		getMin(imgCornerA1 - imgCornerMean, imgCornerB1 - imgCornerMean, imgCornerA);
		getMin(imgCornerMean - imgCornerC1, imgCornerMean - imgCornerD1, imgCornerB);
		getMin(imgCornerA, imgCornerB, imgCorner1);
		
		// 情况2: 对角象限C和D为亮色（白），A和B为暗色（黑）
		getMin(imgCornerMean - imgCornerA1, imgCornerMean - imgCornerB1, imgCornerA);
		getMin(imgCornerC1 - imgCornerMean, imgCornerD1 - imgCornerMean, imgCornerB);
		getMin(imgCornerA, imgCornerB, imgCorner2);

		// 更新总的角点响应图：取多个尺度和方向的最大响应
		getMax(imgCorners, imgCorner1, imgCorners);
		getMax(imgCorners, imgCorner2, imgCorners);
	}
#ifdef show_
	namedWindow("ROI");
	imshow("ROI", imgCorners); waitKey(50);  // 可视化角点响应图
#endif

	t = ((double)getTickCount() - t) / getTickFrequency();

	//==================== 非极大值抑制提取候选角点 ====================//
	// 参数说明：patchSize=3, threshold=0.025, margin=5
	nonMaximumSuppression(imgCorners, cornerPoints, 3, 0.025, 5);

	//==================== 后处理：精化和评分 ====================//
	
	// 计算图像的梯度信息（用于亚像素精化）
	Mat imageDu(gray.size(), mtype);
	Mat imageDv(gray.size(), mtype);
	Mat img_angle = cv::Mat::zeros(gray.size(), mtype);
	Mat img_weight = cv::Mat::zeros(gray.size(), mtype);
	getImageAngleAndWeight(imageNorm, imageDu, imageDv, img_angle, img_weight);
	
	// 亚像素精化（可选）
	if (isrefine == true)
	{
		// 精化角点位置和边缘方向（半径=10像素）
		refineCorners(cornerPoints, imageDu, imageDv, img_angle, img_weight, 10);
		
		// 移除精化失败的角点（边缘方向为零的角点）
		if (cornerPoints.size() > 0)
		{
			for (int i = 0; i < cornerPoints.size(); i++)
			{
				if (cornersEdge1[i][0] == 0 && cornersEdge1[i][0] == 0)
				{
					cornerPoints[i].x = 0; cornerPoints[i].y = 0;
				}
			}
		}
	}

	// 对所有候选角点进行评分
	vector<float> score;
	scoreCorners(imageNorm, img_angle, img_weight, cornerPoints, radius, score);

#ifdef show_
	namedWindow("src");
	imshow("src", Src); 
	waitKey(0);
#endif

	//==================== 根据分数阈值过滤角点 ====================//
	int nlen = cornerPoints.size();
	if (nlen > 0)
	{
		for (int i = 0; i < nlen;i++)
		{
			// 只保留分数高于阈值的角点
			if (score[i] > scoreThreshold)
			{
				mcorners.p.push_back(cornerPoints[i]);  // 角点位置
				mcorners.v1.push_back(cv::Vec2f(cornersEdge1[i][0], cornersEdge1[i][1]));  // 第一条边缘方向
				mcorners.v2.push_back(cv::Vec2f(cornersEdge2[i][0], cornersEdge2[i][1]));  // 第二条边缘方向
				mcorners.score.push_back(score[i]);  // 角点分数
			}
		}
	}

	//==================== 归一化边缘方向向量 ====================//
	// 确保边缘方向向量的一致性：
	// 1. v1的和为正
	// 2. v1和v2的夹角方向一致
	std::vector<cv::Vec2f> corners_n1(mcorners.p.size());
	for (int i = 0; i < corners_n1.size(); i++)
	{
		// 调整v1方向，使其坐标和为正
		if (mcorners.v1[i][0] + mcorners.v1[i][1] < 0.0)
		{
			mcorners.v1[i] = -mcorners.v1[i];
		}
		corners_n1[i] = mcorners.v1[i];
		
		// 调整v2方向，使v1和v2始终形成固定的旋转方向
		float flipflag = corners_n1[i][0] * mcorners.v2[i][0] + corners_n1[0][1] * mcorners.v2[i][1];
		if (flipflag > 0)
			flipflag = -1;
		else
			flipflag = 1;
		mcorners.v2[i] = flipflag * mcorners.v2[i];
	}
}

