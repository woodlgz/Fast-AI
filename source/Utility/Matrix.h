
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
			 *����һ������
			 *@param dims: ά��
			 *@param size: ÿһά�Ĵ�С����
			 *@param data: ����
			 */
			PMatrix CreateMatrix(int dims,vector<int>& size,vector<double>& data);
			PMatrix CreateMatrix2D(int row,int col);

			/**
			 * ��ά����˷�
			 * ����һ���ڴ�������outΪNULL
			 */
			PMatrix MatrixMul2D(PMatrix A,PMatrix B,PMatrix out = NULL);

			/**
			 * �������ֵ�����
			 * ����һ���ڴ�������outΪNULL
			 */
			PMatrix MatrixDiffAbs(PMatrix A,PMatrix B,PMatrix out = NULL);
			
			/**
			 * �����и�λ���ֵ
			 */
			double MaxOfMatrix(PMatrix A);
			
			/**
			 * �����и�λ��Сֵ
			 */
			double MinOfMatrix(PMatrix A);

			/**
			 * ��������
			 */
			double Det(PMatrix A);

			/**
			 * ��������
			 */
			void ZeroMatrix(PMatrix A);

			/**
			 * �������
			 */
			inline void CleanUpMatrix(PMatrix& A);

			/**
			 * ������ת��
			 * ����һ���ڴ�������outΪNULL
			 */
			PMatrix MatrixTranspose(PMatrix A,PMatrix out = NULL);
			

			/**
			 * ��������
			 * ����һ���ڴ�������outΪNULL
			 */
			PMatrix MatrixInv2D(PMatrix A,PMatrix out = NULL);
			
			/**
			 * �����������
			 * ����һ���ڴ�������outΪNULL
			 */
			PMatrix CreateAdjMatrix2D(PMatrix A,int i,int j,PMatrix out = NULL);

			/**
			 * �����ά����ı�ʾ
			 */
			
			void DumpMatrix(vector<double>& data,int dims,vector<int>& arrays);
			void DumpMatrix(PMatrix A);

		}
	}
}

#endif