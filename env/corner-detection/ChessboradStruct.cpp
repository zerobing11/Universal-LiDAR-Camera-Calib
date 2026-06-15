/*  Copyright 2017 onlyliu(997737609@qq.com).                             */
/*                                                                        */
/*  Automatic Camera and Range Sensor Calibration using a single Shot     */
/*  this project realize the papar: Automatic Camera and Range Sensor     */
/*  Calibration using a single Shot                                       */

#include "ChessboradStruct.h"
#include <fstream>  
#include <limits>
#include<numeric>

#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */

ChessboradStruct::ChessboradStruct()
{

}

ChessboradStruct::~ChessboradStruct()
{

}

inline float distv(cv::Vec2f a, cv::Vec2f b)
{
	return std::sqrt((a[0] - b[0])*(a[0] - b[0]) + (a[1] - b[1])*(a[1] - b[1]));
}

inline float mean_l(std::vector<float> &resultSet)
{
	double sum = std::accumulate(std::begin(resultSet), std::end(resultSet), 0.0);
	double mean = sum / resultSet.size(); //ŸùÖµ  
	return mean;
}

inline float stdev_l(std::vector<float> &resultSet, float &mean)
{
	double accum = 0.0;
	mean = mean_l(resultSet);
	std::for_each(std::begin(resultSet), std::end(resultSet), [&](const double d) {
		accum += (d - mean)*(d - mean);
	});
	double stdev = sqrt(accum / (resultSet.size() - 1)); //·œ²î 
	return stdev;
}

inline float stdevmean(std::vector<float> &resultSet)
{
	float stdvalue, meanvalue;

	stdvalue = stdev_l(resultSet, meanvalue);

	return stdvalue / meanvalue;
}

/**
 * @brief 在指定方向上寻找最近的未使用角点作为邻居
 * @param idx 当前角点的索引
 * @param v 搜索方向向量
 * @param chessboard 不断被填充的棋盘格矩阵，被次运行一次这个函数，这个棋盘格就会被填充一个数字
 * @param corners 所有检测到的角点信息
 * @param neighbor_idx [输出] 找到的邻居角点索引
 * @param min_dist [输出] 到邻居的距离
 * @return 成功返回1
 */
int ChessboradStruct::directionalNeighbor(int idx, cv::Vec2f v, cv::Mat chessboard, Corners& corners, int& neighbor_idx, float& min_dist)
{

#if 1
	// 初始化：将所有角点索引放入unused列表
	std::vector<int> unused(corners.p.size());
	for (int i = 0; i < unused.size(); i++)
	{
		unused[i] = i;
	}
	
	// 遍历棋盘格，标记已使用的角点为-1
	for (int i = 0; i < chessboard.rows; i++)
		for (int j = 0; j < chessboard.cols; j++)
		{
			int xy = chessboard.at<int>(i, j);
			if (xy >= 0)  // 该位置已分配角点
			{
				unused[xy] = -1;  // 标记为已使用
			}
		}

	// 从unused列表中移除已使用的角点（值为-1的元素）
	int nsize = unused.size();
	for (int i = 0; i < nsize;)
	{
		if (unused[i] < 0)  // 该角点已被使用
		{
			std::vector<int>::iterator iter = unused.begin() + i;
			unused.erase(iter);
			i = 0;  // 重新开始遍历
			nsize = unused.size();
			continue;
		}
		i++;
	}

	std::vector<float> dist_edge;   // 候选角点到初始起点的向量在v垂直方向上的投影
	std::vector<float> dist_point;  // 候选角点到初始起点的向量在v方向上的投影

	cv::Vec2f idxp = cv::Vec2f(corners.p[idx].x, corners.p[idx].y);  // 当前角点位置
	
	// 遍历所有未使用的角点
	for (int i = 0; i < unused.size(); i++)
	{
		int ind = unused[i];
		// diri：从当前角点指向候选角点的向量
		cv::Vec2f diri = cv::Vec2f(corners.p[ind].x, corners.p[ind].y) - idxp;
		
		// disti：diri在方向v上的投影长度（点乘）
		// 正值表示在v方向前方，负值表示在v方向后方，我们要的是沿v方向的，也就是正值
		float disti = diri[0] * v[0] + diri[1] * v[1];

		// de：diri垂直于v方向的分量
		// 计算方法：总向量 - 平行分量 = 垂直分量
		cv::Vec2f de = diri - disti*v;
		dist_edge.push_back(distv(de, cv::Vec2f(0, 0)));  // 垂直距离的模
		
		// 保存平行距离
		dist_point.push_back(disti);
	}
#else
	// list of neighboring elements, which are currently not in use
	std::vector<int> unused(corners.p.size());
	for (int i = 0; i < unused.size(); i++)
	{
		unused[i] = i;
	}
	for (int i = 0; i < chessboard.rows; i++)
		for (int j = 0; j < chessboard.cols; j++)
		{
			int xy = chessboard.at<int>(i, j);
			if (xy >= 0)
			{
				unused[xy] = -1;//flag the used idx
			}
		}

	std::vector<float> dist_edge;
	std::vector<float> dist_point;

	cv::Vec2f idxp = cv::Vec2f(corners.p[idx].x, corners.p[idx].y);
	// direction and distance to unused corners
	for (int i = 0; i < corners.p.size(); i++)
	{
		if (unused[i] == -1)
		{
			dist_point.push_back(std::numeric_limits<float>::max());
			dist_edge.push_back(0);
			continue;
		}
		cv::Vec2f diri = cv::Vec2f(corners.p[i].x, corners.p[i].y) - idxp;
		float disti = diri[0] * v[0] + diri[1] * v[1];

		cv::Vec2f de = diri - disti*v;
		dist_edge.push_back(distv(de, cv::Vec2f(0, 0)));
		// distances
		dist_point.push_back(disti);
	}

#endif

	//
	int min_idx = 0;
	min_dist = std::numeric_limits<float>::max();

	// 遍历所有候选角点，计算综合距离并找出最小值
	for (int i = 0; i < dist_point.size(); i++)
	{
		// 我们需要的是沿v方向的，因此点乘必须是正
		if (dist_point[i] > 0)
		{
			// 综合距离 = 平行距离 + 5 * 垂直距离
			// 对应中心点(1,1)来说，沿v方向的一共就三个点，(0,0)(0,1)(0,2)，权重5可以强烈惩罚偏离方向v的角点(0,0)(0,2)，确保选择的邻居尽可能在v方向上
			float m = dist_point[i] + 5 * dist_edge[i];
			if (m < min_dist)
			{
				min_dist = m;
				min_idx = i;
			}
		}
	}
	
	// 输出找到的邻居角点索引
	neighbor_idx = unused[min_idx];

	return 1;
}



//以指定角点为中心，利用其两个主方向向量v1和v2，在周围寻找8个邻居角点，构建一个3x3的初始棋盘格
cv::Mat ChessboradStruct::initChessboard(Corners& corners, int idx)
{
	if (corners.p.size() < 9)
	{
		logd("not enough corners!\n");
		chessboard.release();//return empty!
		return chessboard;
	}
	
	// 创建3x3矩阵，初始值全为-1,表示未分配角点
	chessboard = -1 * cv::Mat::ones(3, 3, CV_32S);
	
	// v1, v2是该角点的两个正交主方向向量（由角点检测算法计算得到）
	// v1通常对应棋盘格的横向方向，v2对应纵向方向
	cv::Vec2f v1 = corners.v1[idx];  // 第一主方向向量
	cv::Vec2f v2 = corners.v2[idx];  // 第二主方向向量
	chessboard.at<int>(1, 1) = idx;  // 将当前角点设为棋盘格中心(1,1)位置
	
	// 用于存储到邻居角点的距离
	std::vector<float> dist1(2);  // 存储沿v1方向（左右）的距离
	std::vector<float> dist2(6);  // 存储沿v2方向及对角线方向的距离

	// 沿着v1和v2两个主方向，寻找最近的四个邻居角点：
	// 沿 +v1 方向寻找右侧邻居 -> 位置(1,2)
	directionalNeighbor(idx, +1 * v1, chessboard, corners, chessboard.at<int>(1, 2), dist1[0]);
	// 沿 -v1 方向寻找左侧邻居 -> 位置(1,0)
	directionalNeighbor(idx, -1 * v1, chessboard, corners, chessboard.at<int>(1, 0), dist1[1]);
	// 沿 +v2 方向寻找下侧邻居 -> 位置(2,1)
	directionalNeighbor(idx, +1 * v2, chessboard, corners, chessboard.at<int>(2, 1), dist2[0]);
	// 沿 -v2 方向寻找上侧邻居 -> 位置(0,1)
	directionalNeighbor(idx, -1 * v2, chessboard, corners, chessboard.at<int>(0, 1), dist2[1]);

	// 沿着四个对角线方向，寻找四个角落的邻居 ：
	// 从左侧邻居(1,0)出发，沿 -v2 方向寻找左上角 -> 位置(0,0)
	directionalNeighbor(chessboard.at<int>(1, 0), -1 * v2, chessboard, corners, chessboard.at<int>(0, 0), dist2[2]);
	// 从左侧邻居(1,0)出发，沿 +v2 方向寻找左下角 -> 位置(2,0)
	directionalNeighbor(chessboard.at<int>(1, 0), +1 * v2, chessboard, corners, chessboard.at<int>(2, 0), dist2[3]);
	// 从右侧邻居(1,2)出发，沿 -v2 方向寻找右上角 -> 位置(0,2)
	directionalNeighbor(chessboard.at<int>(1, 2), -1 * v2, chessboard, corners, chessboard.at<int>(0, 2), dist2[4]);
	// 从右侧邻居(1,2)出发，沿 +v2 方向寻找右下角 -> 位置(2,2)
	directionalNeighbor(chessboard.at<int>(1, 2), +1 * v2, chessboard, corners, chessboard.at<int>(2, 2), dist2[5]);

	// 初始棋盘格的质量检查
	bool sigood = false;
	
	// 检查1：任何一个邻居的距离为负数（查找失败），则拒绝
	sigood = sigood || (dist1[0]<0) || (dist1[1]<0);
	sigood = sigood || (dist2[0]<0) || (dist2[1]<0) || (dist2[2]<0) || (dist2[3]<0) || (dist2[4]<0) || (dist2[5]<0);
	
	// 检查2：距离分布的CV（标准差/均值）过大，则拒绝
	// stdevmean(dist1)是来检查左右两个邻居的距离是否相近
	// stdevmean(dist2)检查上下及四个对角邻居的距离是否相近
	sigood = sigood || (stdevmean(dist1) > 0.3) || (stdevmean(dist2) > 0.3);

	// 如果不满足均匀性要求，返回空矩阵
	if (sigood == true)
	{
		chessboard.release();
		return chessboard;
	}
	
	// 所有检查通过，返回构建好的3x3棋盘格
	return chessboard;
}

/**
 * @brief 计算棋盘格的能量值
 * @param chessboard 棋盘格矩阵
 * @param corners 所有角点信息
 * @return 能量值E（越小越好，负值表示高质量）
 * 
 * 能量函数设计原理：
 * E = E_corners + λ × area × E_structure
 * 
 * 两个竞争目标：
 * 1. E_corners（负值）：鼓励更多角点，越多越好，这个很重要！！保证了角点阵越大就越越好
 * 2. E_structure（正值）：惩罚几何不规整，越规整越小
 * 
 * 高质量棋盘格特征：
 * - 角点数量多 → E_corners很负
 * - 几何规整（接近矩形网格）→ E_structure很小
 * - 最终E为负值且绝对值大 → 质量好
 */
float ChessboradStruct::chessboardEnergy(cv::Mat chessboard, Corners& corners)
{
	float lamda = m_lamda;  // 结构惩罚权重（由外部设置）
	
	// ========== 第一项：角点数量能量 ==========
	// 负值：角点越多，E_corners越负，能量越低（越好）
	// 例如：3×3=9个点 → E_corners=-9
	//       4×4=16个点 → E_corners=-16（更好）
	float E_corners = -1 * chessboard.size().area();
	
	// ========== 第二项：结构规整性能量 ==========
	// 衡量棋盘格的几何规整性（共线性检查）
	// 原理：对于理想的矩形网格，任意连续3个角点应该共线
	float E_structure = 0;
	
	// ---------- 检查所有行的共线性 ----------
	// 遍历每一行，检查其中每3个连续角点的共线程度
	for (int i = 0; i < chessboard.rows; i++)
		for (int j = 0; j < chessboard.cols-2; j++)  // 滑动窗口，每次取3个点
		{
			std::vector<cv::Vec2f> x;  // 存储3个连续角点的位置
			float E_structure0 = 0;
			
			// 提取第i行的第j、j+1、j+2列的3个角点
			// 示意图：x[0]---x[1]---x[2]
			for (int k = j; k <= j + 2; k++)
			{
				int n = chessboard.at<int>(i, k);  // 获取角点索引
				x.push_back(corners.p[n]);          // 获取角点坐标
			}
			
			// 共线性检查：计算x[1]偏离x[0]和x[2]连线的程度
			// 理想情况：x[1]应该是x[0]和x[2]的中点
			// 偏离向量 = x[0] + x[2] - 2*x[1]
			// 如果完美共线，偏离向量应该是(0,0)
			E_structure0 = distv(x[0] + x[2] - 2 * x[1], cv::Vec2f(0,0));
			
			// 归一化：除以x[0]到x[2]的距离
			// 这样E_structure0变成相对误差，不受尺度影响
			float tv = distv(x[0] - x[2], cv::Vec2f(0, 0));
			E_structure0 = E_structure0 / tv;
			
			// 保留最大的偏离值（最差情况）
			// 只要有一处严重不规整，整个棋盘格就被认为质量差
			if (E_structure < E_structure0)
				E_structure = E_structure0;
		}

	// ---------- 检查所有列的共线性 ----------
	// 与行检查完全对称，只是遍历方向不同
	for (int i = 0; i < chessboard.cols; i++)
		for (int j = 0; j < chessboard.rows-2; j++)  // 滑动窗口，每次取3个点
		{
			std::vector<cv::Vec2f> x;
			float E_structure0 = 0;
			
			// 提取第i列的第j、j+1、j+2行的3个角点
			for (int k = j; k <= j + 2; k++)
			{
				int n = chessboard.at<int>(k, i);  // 注意：这里k和i的顺序与行检查相反
				x.push_back(corners.p[n]);
			}
			
			// 同样的共线性检查
			E_structure0 = distv(x[0] + x[2] - 2 * x[1], cv::Vec2f(0, 0));
			float tv = distv(x[0] - x[2], cv::Vec2f(0, 0));
			E_structure0 = E_structure0 / tv;
			
			// 保留最大偏离
			if (E_structure < E_structure0)
				E_structure = E_structure0;
		}

	// ========== 综合能量计算 ==========
	// E = E_corners + λ × area × E_structure
	// 
	// 参数说明：
	// - λ (lamda)：结构惩罚权重，越大对规整性要求越严格
	// - area：棋盘格面积，大棋盘格应该有更高的规整性要求
	// - E_structure：归一化的最大偏离度
	// 
	// 能量解读：
	// E < -10：高质量棋盘格（在chessboardsFromCorners中的阈值）
	// E ≈ 0：  角点数量和结构惩罚抵消
	// E > 0：  低质量，结构太差
	float E = E_corners + lamda*chessboard.size().area()*E_structure;

	return E;
}

/**
 * @brief 基于3个已知点的位置，预测下一个点的位置（线性外推）
 * @param p1 第一组点（如棋盘格的倒数第3列/行）
 * @param p2 第二组点（如棋盘格的倒数第2列/行）
 * @param p3 第三组点（如棋盘格的最后一列/行）
 * @param pred [输出] 预测的下一组点位置
 * 
 * 算法原理：
 * 假设3个连续点呈现某种趋势（方向和间距），利用这个趋势外推下一个点
 * 
 * 示意图：
 *   p1 ----v1----> p2 ----v2----> p3 ----预测----> pred
 *   ●              ●              ●                 ●
 * 
 * 预测公式：
 * 1. 计算向量：v1 = p2-p1, v2 = p3-p2
 * 2. 预测角度：a2 = atan2(v2), a3 = 2*a2 - a1 (线性外推角度)，为啥不用平均嘞？
 * 3. 预测距离：s2 = |v2|, s3 = 2*s2 - s1 (线性外推距离)
 * 4. 最终位置：pred = p3 + 0.75*s3*方向向量
 *    (0.75系数：稍保守的预测，避免过度外推)
 */
void ChessboradStruct::predictCorners(std::vector<cv::Vec2f>& p1, std::vector<cv::Vec2f>& p2, 
	std::vector<cv::Vec2f>& p3, std::vector<cv::Vec2f>& pred)
{
	cv::Vec2f v1, v2;    // 相邻点之间的向量
	float a1, a2, a3;    // 向量的角度
	float s1, s2, s3;    // 向量的长度（距离）
	pred.resize(p1.size());
	
	// 对每一行/列的角点进行独立预测
	for (int i = 0; i < p1.size(); i++)
	{
		// 计算相邻点之间的向量
		v1 = p2[i] - p1[i];  // 从p1到p2的向量
		v2 = p3[i] - p2[i];  // 从p2到p3的向量
		
		// 预测角度（方向）
		a1 = atan2(v1[1], v1[0]);  // v1的角度
		a2 = atan2(v2[1], v2[0]);  // v2的角度（注意：原代码有bug，应该是v2）
		a3 = 2.0 * a2 - a1;         // 线性外推：保持角度变化趋势

		// 预测距离（尺度）
		s1 = distv(v1, cv::Vec2f(0, 0));  // v1的长度
		s2 = distv(v2, cv::Vec2f(0, 0));  // v2的长度
		s3 = 2 * s2 - s1;                  // 线性外推：保持距离变化趋势
		
		// 最终预测位置 = p3 + 预测向量
		// 0.75系数：稍保守，避免外推过远导致匹配失败
		pred[i] = p3[i] + 0.75*s3*cv::Vec2f(cos(a3), sin(a3));
	}
}

void ChessboradStruct::assignClosestCorners(std::vector<cv::Vec2f>&cand, std::vector<cv::Vec2f>&pred, std::vector<int> &idx)
{
	//return error if not enough candidates are available
	if (cand.size() < pred.size())
	{
		idx.resize(1);
		idx[0] = -1;
		return;
	}
	idx.resize(pred.size());

	//build distance matrix
	cv::Mat D = cv::Mat::zeros(cand.size(), pred.size(), CV_32FC1);
	float mind = FLT_MAX;
	for (int i = 0; i < D.cols; i++)//ÁÐÓÅÏÈ
	{
		cv::Vec2f delta;
		for (int j = 0; j < D.rows; j++)
		{
			delta = cand[j] - pred[i];
			float s = distv(delta, cv::Vec2f(0, 0));
			D.at<float>(j, i) = s;
			if (s < mind)
			{
				mind = s;
			}
		}
	}
	
	// search greedily for closest corners
	for (int k = 0; k < pred.size(); k++)
	{
		bool isbreak = false;
		for (int i = 0; i < D.rows; i++)
		{
			for (int j = 0; j < D.cols; j++)
			{
				if (fabs(D.at<float>(i, j) - mind) < 10e-10)
				{
					idx[j] = i;
					for (int m = 0; m < D.cols; m++)
					{
						D.at<float>(i, m) = FLT_MAX;
					}
					for (int m = 0; m < D.rows; m++)
					{
						D.at<float>(m,j) = FLT_MAX;
					}
					isbreak = true;
					break;
				}
			}
			if (isbreak == true)
				break;
		}
		mind = FLT_MAX;
		for (int i = 0; i < D.rows; i++)
		{
			for (int j = 0; j < D.cols; j++)
			{
				if (D.at<float>(i, j) < mind)
				{
					mind = D.at<float>(i, j);
				}
			}
		}
	}
}



/**
 * @brief 向指定方向扩展棋盘格（增加一行或一列）
 * @param chessboard 当前棋盘格矩阵
 * @param corners 所有检测到的角点信息
 * @param border_type 扩展方向：0=右, 1=下, 2=左, 3=上
 * @return 扩展后的棋盘格矩阵，失败则返回原棋盘格
 * 
 * 算法原理（以向右扩展为例）：
 * 1. 提取棋盘格最右边的3列角点：p1(倒数第3列), p2(倒数第2列), p3(最后一列)
 * 2. 利用这3列的位置关系，预测下一列的角点位置pred
 *    预测公式：pred = p3 + 0.75*(p3-p2的向量趋势)
 * 3. 在未使用的候选角点中，找到与pred最接近的角点
 * 4. 扩展棋盘格（增加一列），填入找到的角点索引
 * 
 * 扩展示意图（向右扩展）：
 *    p1   p2   p3   pred(新列)
 *    ●    ●    ●    ●  <- 基于这3个点预测新位置
 *    ●    ●    ●    ●
 *    ●    ●    ●    ●
 *   倒数  倒数  最后  扩展
 *   第3列 第2列  列   的列
 */
cv::Mat ChessboradStruct::growChessboard(cv::Mat chessboard, Corners& corners, int border_type)
{
	// 输入检查
	if (chessboard.empty() == true)
	{
		return chessboard;
	}
	
	std::vector<cv::Point2f> p = corners.p;  // 所有角点的位置
	
	// 与 directionalNeighbor 函数相同的逻辑，构建未使用角点列表
	std::vector<int> unused(p.size());
	for (int i = 0; i < unused.size(); i++)
	{
		unused[i] = i;
	}
	
	// 标记已被使用的角点
	for (int i = 0; i < chessboard.rows; i++)
		for (int j = 0; j < chessboard.cols; j++)
		{
			int xy = chessboard.at<int>(i, j);
			if (xy >= 0)
			{
				unused[xy] = -1;  // 已使用
			}
		}

	// 从列表中删除已使用的角点
	int nsize = unused.size();
	for (int i = 0; i < nsize; )
	{
		if (unused[i] < 0)
		{
			std::vector<int>::iterator iter = unused.begin() + i;
			unused.erase(iter);
			i = 0; 
			nsize = unused.size();
			continue;
		}
		i++;
	}

	// 构建候选角点位置列表
	std::vector<cv::Vec2f> cand;
	for (int i = 0; i < unused.size(); i++)
	{
		cand.push_back(corners.p[unused[i]]);
	}
	
	cv::Mat chesstemp;  // 扩展后的棋盘格

	switch (border_type)
	{
	case 0:  // 向右扩展（增加一列）
	{
		// 提取最右边3列的角点位置，用于预测下一列
		// 示意图：
		//   col-3  col-2  col-1  (新列)
		//    p1     p2     p3    pred
		//    ●      ●      ●   →  ●
		//    ●      ●      ●   →  ●
		//    ●      ●      ●   →  ●
		
		std::vector<cv::Vec2f> p1, p2, p3, pred;
		for (int row = 0; row < chessboard.rows; row++)
			for (int col = 0; col < chessboard.cols; col++)
			{
				if (col == chessboard.cols - 3)  // 倒数第3列
				{				
					int ij = chessboard.at<int>(row, col);
					p1.push_back(cv::Vec2f(p[ij]));
				}
				if (col == chessboard.cols - 2)  // 倒数第2列
				{
					int ij = chessboard.at<int>(row, col);
					p2.push_back(cv::Vec2f(p[ij]));
				}
				if (col == chessboard.cols - 1)  // 最后一列
				{
					int ij = chessboard.at<int>(row, col);
					p3.push_back(cv::Vec2f(p[ij]));
				}
			}
		
		// 基于p1,p2,p3的位置趋势，预测下一列每个角点的位置
		std::vector<int> idx;
		predictCorners(p1, p2, p3, pred);  // 预测pred位置
		
		// 在候选角点中，找到与pred最接近的角点
		assignClosestCorners(cand, pred, idx);
		if (idx[0] < 0)  // 匹配失败
		{
			return chessboard;  // 返回原棋盘格（扩展失败）
		}

		// 扩展棋盘格：在右边增加一列
		// cv::copyMakeBorder参数(top, bottom, left, right)
		cv::copyMakeBorder(chessboard, chesstemp, 0, 0, 0, 1, 0, 0);

		// 填充新列的角点索引
		for (int i = 0; i < chesstemp.rows; i++)
		{
			chesstemp.at<int>(i, chesstemp.cols - 1) = unused[idx[i]];  // 最右列
		}
		chessboard = chesstemp.clone();

		break;
	}
	case 1:  // 向下扩展（增加一行）
	{
		// 提取最下边3行的角点位置，用于预测下一行
		// 示意图：
		//   row-3: p1 → ● ● ● ● 
		//   row-2: p2 → ● ● ● ●
		//   row-1: p3 → ● ● ● ●
		//   (新行): pred → ● ● ● ● (要预测的)
		
		std::vector<cv::Vec2f> p1, p2, p3, pred;
		for (int row = 0; row < chessboard.rows; row++)
			for (int col = 0; col < chessboard.cols; col++)
			{
				if (row == chessboard.rows - 3)  // 倒数第3行
				{
					int ij = chessboard.at<int>(row, col);
					p1.push_back(cv::Vec2f(p[ij]));
				}
				if (row == chessboard.rows - 2)  // 倒数第2行
				{
					int ij = chessboard.at<int>(row, col);
					p2.push_back(cv::Vec2f(p[ij]));
				}
				if (row == chessboard.rows - 1)  // 最后一行
				{
					int ij = chessboard.at<int>(row, col);
					p3.push_back(cv::Vec2f(p[ij]));
				}
			}
		
		std::vector<int> idx;
		predictCorners(p1, p2, p3, pred);  // 预测下一行的位置
		assignClosestCorners(cand, pred, idx);
		if (idx[0] < 0)  // 匹配失败
		{
			return chessboard;
		}

		// 扩展棋盘格：在下边增加一行
		cv::copyMakeBorder(chessboard, chesstemp, 0, 1, 0, 0, 0, 0);
		
		// 填充新行的角点索引
		for (int i = 0; i < chesstemp.cols; i++)
		{
			chesstemp.at<int>(chesstemp.rows - 1, i) = unused[idx[i]];  // 最下行
		}
		chessboard = chesstemp.clone();

		break;
	}
	case 2:  // 向左扩展（增加一列）
	{
		// 提取最左边3列的角点位置，用于预测新的左侧列
		// 注意：这里是逆向预测，从col=2 → col=1 → col=0 → pred(新列)
		// 示意图：
		//  (新列)  col-0  col-1  col-2
		//   pred    p3     p2     p1
		//    ●   ←  ●      ●      ●
		//    ●   ←  ●      ●      ●
		//    ●   ←  ●      ●      ●
		
		std::vector<cv::Vec2f> p1, p2, p3, pred;
		for (int row = 0; row < chessboard.rows; row++)
			for (int col = 0; col < chessboard.cols; col++)
			{
				if (col == 2)  // 第3列（从左数）
				{
					int ij = chessboard.at<int>(row, col);
					p1.push_back(cv::Vec2f(p[ij]));
				}
				if (col == 1)  // 第2列
				{
					int ij = chessboard.at<int>(row, col);
					p2.push_back(cv::Vec2f(p[ij]));
				}
				if (col == 0)  // 第1列（最左）
				{
					int ij = chessboard.at<int>(row, col);
					p3.push_back(cv::Vec2f(p[ij]));
				}
			}
		
		std::vector<int> idx;
		predictCorners(p1, p2, p3, pred);  // 预测更左侧的位置
		assignClosestCorners(cand, pred, idx);
		if (idx[0] < 0)  // 匹配失败
		{
			return chessboard;
		}

		// 扩展棋盘格：在左边增加一列
		cv::copyMakeBorder(chessboard, chesstemp, 0, 0, 1, 0, 0, 0);
		
		// 填充新列的角点索引
		for (int i = 0; i < chesstemp.rows; i++)
		{
			chesstemp.at<int>(i, 0) = unused[idx[i]];  // 最左列
		}
		chessboard = chesstemp.clone();

		break;
	}
	case 3:  // 向上扩展（增加一行）
	{
		// 提取最上边3行的角点位置，用于预测新的上侧行
		// 注意：这里是逆向预测，从row=2 → row=1 → row=0 → pred(新行)
		// 示意图：
		//  (新行): pred → ● ● ● ● (要预测的)
		//          ↑
		//   row-0: p3 → ● ● ● ●
		//   row-1: p2 → ● ● ● ●
		//   row-2: p1 → ● ● ● ●
		
		std::vector<cv::Vec2f> p1, p2, p3, pred;
		for (int row = 0; row < chessboard.rows; row++)
			for (int col = 0; col < chessboard.cols; col++)
			{
				if (row == 2)  // 第3行（从上数）
				{
					int ij = chessboard.at<int>(row, col);
					p1.push_back(cv::Vec2f(p[ij]));
				}
				if (row == 1)  // 第2行
				{
					int ij = chessboard.at<int>(row, col);
					p2.push_back(cv::Vec2f(p[ij]));
				}
				if (row == 0)  // 第1行（最上）
				{
					int ij = chessboard.at<int>(row, col);
					p3.push_back(cv::Vec2f(p[ij]));
				}
			}
		
		std::vector<int> idx;
		predictCorners(p1, p2, p3, pred);  // 预测更上侧的位置
		assignClosestCorners(cand, pred, idx);
		if (idx[0] < 0)  // 匹配失败
		{
			return chessboard;
		}
		
		// 扩展棋盘格：在上边增加一行
		cv::copyMakeBorder(chessboard, chesstemp, 1, 0, 0, 0, 0, 0);
		
		// 填充新行的角点索引
		for (int i = 0; i < chesstemp.cols; i++)
		{
			chesstemp.at<int>(0, i) = unused[idx[i]];  // 最上行
		}
		chessboard = chesstemp.clone();
		break;
	}
	default:
		break;
	}
	
	return chessboard;  // 返回扩展后的棋盘格
}

/*下面这个函数是棋盘格检测的入口函数，相当于这个cpp文件的main函数，std::vector<cv::Mat>& chessboards是最后的输出,
那么，chessboards中chessboard的排序，以及chessboard中角点的排序是受什么所影响的?
首先chessboards中chessboard的是受Corners& corners中的角点排序强影响的，因为初始棋盘格增长就是按照着角点顺序一个一个遍历来的，
因此才会出现image_corner_test中，chessboard的顺序是左右下，这是因为角点存储顺序是有一定规律的，会先遍历到右板的中心点，然后左板中心点，最后底板中心点；
chessboard中角点排序：(0,0)是棋盘格的四个角随机，这个取决于角点的方向向量是什么样子的
*/
 void ChessboradStruct::chessboardsFromCorners( Corners& corners, std::vector<cv::Mat>& chessboards, float lamda, int target_corner_count)
 {
	 logd("Structure recovery:\n");
	 m_lamda = lamda;  //能量函数权重，越大越严格，要求的棋盘格越规整
	 
	 // 棋盘格点数约束，从外部传入
	 logd("Target corner count: %d\n", target_corner_count);
	 
	 
	 int valid_initial_count = 0;
	 int energy_pass_count = 0;
	 int final_quality_count = 0;
	 int added_chessboards = 0;
	 
	 // 遍历所有角点，尝试以每个角点为起点构建棋盘格
	 for (int i = 0; i < corners.p.size(); i++)
	 {
		 // initChessboard会尝试找到该角点的最近邻角点，构建3x3的初始棋盘格
		 cv::Mat csbd = initChessboard(corners, i);
		 if (csbd.empty() == true)
		 {
			 continue;
		 }
		 valid_initial_count++;
		 // 计算初始棋盘格的能量值 能量值越小表示棋盘格质量越好，能量值>0表示质量太差，跳过质量差的候选棋盘格
		 float E = chessboardEnergy(csbd, corners);
	 	//这个阈值很低了
		 if (E > 0){ continue; }  
		 energy_pass_count++;
		 cv::Mat chessboard = csbd.clone();
		 int s = 0; 
		 // 对初始棋盘格，进行扩展，直到无法进一步优化或达到目标点数
		 while (true)
		 {
			 s++;
			 // 计算棋盘格的能量值，第一次循环是3x3的初始棋盘格，后续就是拓展棋盘格了
			 float energy = chessboardEnergy(chessboard, corners);
			 // 检查是否已达到目标角点数，达到则退出扩展循环
			 int current_corner_count = chessboard.rows * chessboard.cols;
			 if (current_corner_count >= target_corner_count)
			 {
				 break;
			 }
 
			 // 没有到达目标点数，分别尝试向上、下、左、右四个方向扩展棋盘格，生成4个方向的扩展候选方案
			 std::vector<cv::Mat> proposal(4);
			 std::vector<float> p_energy(4);
			 
			 // 计算每个扩展方案的能量值，并检查点数约束
			 for (int j = 0; j < 4; j++)
			 {
				 proposal[j] = growChessboard(chessboard, corners, j);
				 
				 // 检查扩展后的角点数是否超过目标
				 if (!proposal[j].empty())
				 {
					 int proposal_corner_count = proposal[j].rows * proposal[j].cols;
					 //超过目标点数，设置为无效
					 if (proposal_corner_count > target_corner_count)
					 {
						 p_energy[j] = std::numeric_limits<float>::max();
					 }
					 //符合的就计算能量值
					 else
					 {
						 p_energy[j] = chessboardEnergy(proposal[j], corners);  // 计算该方案的能量值
					 }
				 }
				 //没有扩展成功，设置为无效
				 else
				 {
					 p_energy[j] = std::numeric_limits<float>::max();
				 }
			 }
			 
			 // 在符合点数要求的扩展方案中能量值最小的方案
			 float min_value = p_energy[0];
			 int min_idx = 0;
			 for (int i0 = 1; i0 < p_energy.size(); i0++)
			 {
				 if (min_value > p_energy[i0])
				 {
					 min_value = p_energy[i0];
					 min_idx = i0;
				 }
			 }
		 	
			 //判定棋盘格是否继续生长
			 cv::Mat chessboardt;  //扩展后的棋盘格
			 
			 // 最优扩展方案的能量 < 当前棋盘格的能量，继续while拓展
			 if (p_energy[min_idx] < energy)
			 {
				 chessboardt = proposal[min_idx];
				 chessboard = chessboardt.clone();// 更新当前棋盘格为扩展后的结果
			 }
			 else
			 {
				 // 所有扩展方案都没有改进，退出while循环，不再扩展，保留当前棋盘格
				 break;
			 }
		 }//end while

		 // 能量值<-10的棋盘格可以加入候选列表
		 float final_energy = chessboardEnergy(chessboard, corners);
		 
		 // 重新查一下角点数
		 int final_corner_count = chessboard.rows * chessboard.cols;
		 bool corner_count_match = (final_corner_count == target_corner_count);
		 
		 if (final_energy < -10 && corner_count_match)
		 {
			 final_quality_count++;
			 // 检查新发现的棋盘格是否与已存在的棋盘格有重叠
			 cv::Mat overlap = cv::Mat::zeros(cv::Size(2, chessboards.size()), CV_32FC1);
			 
			 // 遍历所有已存在的棋盘格，检查重叠情况
			 for (int j = 0; j < chessboards.size(); j++)
			 {
				 bool isbreak = false;
				 // 检查当前棋盘格与第j个已存在棋盘格是否有共同角点
				 for (int k = 0; k < chessboards[j].size().area(); k++)
				 {
					 int refv = chessboards[j].at<int>(k / chessboards[j].cols, k % chessboards[j].cols);
					 for (int l = 0; l < chessboard.size().area(); l++)
					 {
						 int isv = chessboard.at<int>(l / chessboard.cols, l % chessboard.cols);
						 if (refv == isv)  // 发现共同角点，表示有重叠
						 {
							 overlap.at<float>(j, 0) = 1.0;  // 标记有重叠
							 float s = chessboardEnergy(chessboards[j], corners);  // 计算已存在棋盘格的能量
							 overlap.at<float>(j, 1) = s;  // 保存能量值用于比较
							 isbreak = true;
							 break;
						 }
					 }
				 }
			 }//endfor
 
			 // 根据重叠情况决定是否添加新棋盘格
			 // 检查是否存在重叠
			 bool isoverlap = false;
			 for (int i0 = 0; i0 < overlap.rows; i0++)
			 {
				 if (overlap.empty() == false)
				 {
					 if (fabs(overlap.at<float>(i0, 0)) > 0.000001)  // 检查是否有重叠标记
					 {
						 isoverlap = true;
						 break;
					 }
				 }
			 }
			 
			 if (isoverlap == false)
			 {
				 // 情况1：无重叠，直接添加新棋盘格
				 chessboards.push_back(chessboard);
				 added_chessboards++;
			 }
			 else
			 {
				 // 情况2：存在重叠，需要比较质量并决定替换策略
				 bool flagpush = true;  // 是否应该添加新棋盘格
				 std::vector<bool> flagerase(overlap.rows);  // 标记哪些已存在棋盘格应该被删除
				 for (int m = 0; m < flagerase.size(); m++)
				 {
					 flagerase[m] = false;
				 }
				 
				 float ce = chessboardEnergy(chessboard, corners);  // 新棋盘格的能量值
				 
				 // 比较新棋盘格与重叠的已存在棋盘格的质量
				 for (int i1 = 0; i1 < overlap.rows; i1++)
				 {
					 if (fabs(overlap.at<float>(i1, 0)) > 0.0001)  // 如果存在重叠
					 {	
						 bool isb1 = overlap.at<float>(i1, 1) > ce;  // 比较能量值
 
						 // 使用整数比较避免浮点数精度问题
						 int a = int(overlap.at<float>(i1, 1) * 1000);
						 int b = int(ce * 1000);
 
						 bool isb2 = a > b;
						 if (isb1 != isb2)
							 printf("find bug!\n");  // 调试信息
 
						 if (isb2)  // 如果新棋盘格质量更好（能量值更小）
						 {	
							 flagerase[i1] = true;  // 标记删除已存在的棋盘格
						 }
						 else  // 如果已存在棋盘格质量更好
						 {
							 flagpush = false;  // 不添加新棋盘格
						 }
					 }
				 }//end for
 
				 // 执行替换操作
				 if (flagpush == true)  // 如果决定添加新棋盘格
				 {
					 // 删除质量较差的已存在棋盘格
					 for (int i1 = 0; i1 < chessboards.size();)
					 {
						 std::vector<cv::Mat>::iterator it = chessboards.begin() + i1;
						 std::vector<bool>::iterator it1 = flagerase.begin() + i1;
						 if (*it1 == true)  // 如果标记为删除
						 {
							 chessboards.erase(it);  // 删除棋盘格
							 flagerase.erase(it1);   // 删除标记
							 i1 = 0;  // 重新开始遍历（因为删除了元素）
						 }
						 i1++;
					 }
					 chessboards.push_back(chessboard);  // 添加新棋盘格
					 added_chessboards++;
				 }
			 }//endif
		 }//endif
	 }//end for 
	 
 }
 
// #define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
// void ChessboradStruct::drawchessboard(cv::Mat img, Corners& corners, std::vector<cv::Mat>& chessboards, char * title, int t_, cv::Rect rect)
// {
//         printf("end!\n");
//
// 	cv::RNG rng(0xFFFFFFFF);
// 	std::string s("If it's useful, please give a star ^-^.");
// 	std::string s1("https://github.com/onlyliucat\n");
// 	//std::cout<<BOLDBLUE<<s<<std::endl<<BOLDGREEN<<s1<<std::endl;//fyy
// 	cv::Mat disp = img.clone();
//
// 	if (disp.channels() < 3)
// 		cv::cvtColor(disp, disp, CV_GRAY2BGR);
// 	float scale = 0.3;
// 	int n = 8;
// 	if (img.rows < 2000 || img.cols < 2000)
// 	{
// 		scale = 1;
// 		n = 2;
// 	}
// 	for (int k = 0; k < chessboards.size(); k++)
// 	{
// 		cv::Scalar s(rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0));
// 		s = s * 255;
//
// 		for (int i = 0; i < chessboards[k].rows; i++)
// 			for (int j = 0; j < chessboards[k].cols; j++)
// 			{
// 				int d = chessboards[k].at<int>(i, j);
// 				//cv::circle(disp, cv::Point2f(corners.p[d].x + rect.x, corners.p[d].y + rect.y), n, s, n);
// 				cv::putText(disp,std::to_string(d),cv::Point2f(corners.p[d].x + rect.x, corners.p[d].y + rect.y),cv::FONT_HERSHEY_SIMPLEX,0.3,cv::Scalar(255,255,255),1,1);
// 			}
// 	}
// 	cv::Mat SmallMat;
// 	cv::resize(disp, SmallMat, cv::Size(), scale, scale);
// 	cv::namedWindow(title);
// 	cv::imshow(title, SmallMat);
// 	cv::waitKey(t_);
// }



