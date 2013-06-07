
#ifndef MATRIX_H_
#define MATRIX_H_

#include <math.h>
#include <vector>

using namespace std;

#define GetRealData(mat,index) ((mat->data[index-1]))
#define GetRealData2D(mat,x,y) ((mat->data[(x-1)*(mat->nWidth)+y-1]))
#define GetRealData3D(mat,x,y,z) ((mat->data[(z-1)*((mat->size[1])*(mat->size[0]))+(x-1)*(mat->size[1])+y-1]))




namespace FASTAI{
	namespace Util{
		namespace Math{
			typedef struct Matrix{\
				int dims;\
				int nWidth;\
				vector<int> size;\
				vector<double> data;\
				Matrix(){\
					size.resize(3);\
				}\
			} Matrix, *PMatrix;

			/**
			 *创建一个矩阵
			 *@param dims: 维数
			 *@param size: 每一维的大小数组
			 *@param data: 数据
			 */
			PMatrix CreateMatrix(int dims,vector<int>& size,vector<double>& data);
			PMatrix CreateMatrix2D(int row,int col);

			/**
			 * 二维矩阵乘法
			 * 分配一个内存矩阵，如果out为NULL
			 */
			PMatrix MatrixMul2D(PMatrix A,PMatrix B,PMatrix out = NULL);

			/**
			 * 矩阵绝对值差矩阵
			 * 分配一个内存矩阵，如果out为NULL
			 */
			PMatrix MatrixDiffAbs(PMatrix A,PMatrix B,PMatrix out = NULL);
			
			/**
			 * 矩阵中各位最大值
			 */
			double MaxOfMatrix(PMatrix A);
			
			/**
			 * 矩阵中各位最小值
			 */
			double MinOfMatrix(PMatrix A);

			/**
			 * 求矩阵的秩
			 */
			double Det(PMatrix A);

			/**
			 * 矩阵清零
			 */
			void ZeroMatrix(PMatrix A);

			/**
			 * 清理矩阵
			 */
			inline void CleanUpMatrix(PMatrix& A);

			/**
			 * 矩阵求转置
			 * 分配一个内存矩阵，如果out为NULL
			 */
			PMatrix MatrixTranspose(PMatrix A,PMatrix out = NULL);
			

			/**
			 * 矩阵求逆
			 * 分配一个内存矩阵，如果out为NULL
			 */
			PMatrix MatrixInv2D(PMatrix A,PMatrix out = NULL);
			
			/**
			 * 创建伴随矩阵
			 * 分配一个内存矩阵，如果out为NULL
			 */
			PMatrix CreateAdjMatrix2D(PMatrix A,int i,int j,PMatrix out = NULL);

			/**
			 * 输出低维矩阵的表示
			 */
			
			void DumpMatrix(vector<double>& data,int dims,vector<int>& arrays);
			void DumpMatrix(PMatrix A);

		}
	}
}

#endif