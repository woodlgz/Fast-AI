
#include "Matrix.h"
#include <stdlib.h>

namespace FASTAI{
	namespace Util{
		namespace Math{
			PMatrix CreateMatrix(int dims,vector<int>& size,vector<double>& data){
				PMatrix ret=new Matrix;
				ret->dims=dims;
				ret->size=size;
				ret->data=data;
				ret->nWidth=size[1];//only for 2D matrix
				return ret;
			}

			PMatrix CreateMatrix2D(int row,int col){
				PMatrix ret = new Matrix;
				ret->dims = 2;
				ret->size[0] = row;
				ret->size[1] = col;
				ret->nWidth = col;
				ret->data.resize(row*col);
				return ret;
			}

			void ZeroMatrix(PMatrix A){
				int n=A->dims;
				int cnt=1;
				for(int i=0;i<n;i++){
					cnt *= A->size[i];
				}
				for(int i=1;i<=cnt;i++){
					GetRealData(A,i)=0;
				}
				return;
			}

			PMatrix MatrixMul2D(PMatrix A,PMatrix B,PMatrix out){
				if(A==NULL || B==NULL)return NULL;
				if(A->size[1]!=B->size[0])return NULL;
				if(out == NULL){
					out=new Matrix;
					out->dims=2;
					out->size[0]=A->size[0];
					out->size[1]=B->size[1];
					out->data.resize((out->size[0])*(out->size[1]));
					out->nWidth=out->size[1];
				}//else out must be a matrix of structure that fits A.*B
				else 
					ZeroMatrix(out);
				for(int i=1;i<=out->size[0];i++){
					for(int j=1;j<=out->size[1];j++){
						for(int k=1;k<=A->size[1];k++){
							GetRealData2D(out,i,j)+=GetRealData2D(A,i,k)*GetRealData2D(B,k,j);
						}
					}
				}
				return out;
			}


			PMatrix MatrixDiffAbs(PMatrix A,PMatrix B,PMatrix out){
				if(A->dims!=B->dims)return NULL;
				if(out == NULL){
					out=new Matrix;
					out->dims=A->dims;
				}//else out must be an existed matrix of a right structure
				int cnt=1;
				for(int i=0;i<A->dims;i++){
					out->size[i]=A->size[i];
					cnt*=out->size[i];
				}
				for(int i=1;i<=cnt;i++){
					GetRealData(out,i)=fabs(GetRealData(A,i)-GetRealData(B,i));
				}
				return out;
			}

			double MaxOfMatrix(PMatrix A){
				int cnt=1;
				for(int i=0;i<A->dims;i++){
					cnt*=A->size[i];
				}
				double max=GetRealData(A,1);
				for(int i=1;i<=cnt;i++){
					if(max<GetRealData(A,i)){
						max=GetRealData(A,i);
					}
				}
				return max;
			}

			double MinOfMatrix(PMatrix A){
				int cnt=1;
				for(int i=0;i<A->dims;i++){
					cnt*=A->size[i];
				}
				double min=GetRealData(A,1);
				for(int i=1;i<=cnt;i++){
					if(min>GetRealData(A,i)){
						min=GetRealData(A,i);
					}
				}
				return min;
			}

			double Det(PMatrix A){
				double sum=0.0;
				PMatrix sub = new Matrix;
				if(A->size[0]==2&&A->size[1]==2){
					return (GetRealData2D(A,1,1)*GetRealData2D(A,2,2)-GetRealData2D(A,1,2)*GetRealData2D(A,2,1));
				}
				sub->dims = A->dims;
				sub->size[0] = A->size[0]-1;
				sub->size[1] = A->size[1]-1;
				sub->nWidth = A->nWidth-1;
				sub->data.resize(sub->size[0]*sub->size[1]);
				for(int i=1;i<=A->size[1];i++){
					sub=CreateAdjMatrix2D(A,1,i,sub);
					sum+=((1+i)%2==0?1.0:-1.0)*GetRealData2D(A,1,i)*Det(sub);
				}
				CleanUpMatrix(sub);
				return sum;
			}

			PMatrix MatrixTranspose(PMatrix A,PMatrix out){
				if(!A)return NULL;
				if(out==NULL){
					out=new Matrix;
					out->dims=A->dims;
					out->size[0]=A->size[1];
					out->size[1]=A->size[0];
					out->nWidth=out->size[1];
					out->data.resize(out->size[0]*out->size[1]);
				}//else out must be an existed matrix of a right structure
				for(int i=1;i<=A->size[0];i++){
					for(int j=1;j<=A->size[1];j++){
						GetRealData2D(out,j,i)=GetRealData2D(A,i,j);
					}
				}
				return out;
			}

			PMatrix MatrixInv2D(PMatrix A,PMatrix out){
				if(!A||A->dims!=2||A->size[0]!=A->size[1])return NULL;
				if(out == NULL){
					out=new Matrix;
					out->data.resize(A->size[0]*A->size[1]);
					out->size[0]=A->size[0];
					out->size[1]=A->size[1];
					out->dims=2;
					out->nWidth=out->size[1];
				}//else out must be an existed matrix of a right structure
				double detA=Det(A);
				PMatrix tmp = new Matrix;
				tmp->dims = 2;
				tmp->size[0] = A->size[0]-1;
				tmp->size[1] = A->size[1]-1;
				tmp->nWidth = A->nWidth-1;
				tmp->data.resize(tmp->size[0]*tmp->size[1]);
				int cnt=1;
				for(int i=1;i<=A->size[0];i++){
					for(int j=1;j<=A->size[1];j++){
						tmp=CreateAdjMatrix2D(A,i,j,tmp);
						//DumpMatrix<double>((double*)tmp->data,tmp->dims,tmp->size);
						GetRealData(out,cnt)=((i+j)%2==0?1.0:-1.0)*Det(tmp)/detA;
						cnt++;
					}
				}
				CleanUpMatrix(tmp);
				if(out->size[0]==2&&out->size[1]==2){
					double t=GetRealData2D(out,1,2);
					GetRealData2D(out,1,2)=GetRealData2D(out,2,1);
					GetRealData2D(out,2,1)=t;
				}
				tmp=MatrixTranspose(out);
				CleanUpMatrix(out);
				out = tmp;
				return out;
			}

			PMatrix CreateAdjMatrix2D(PMatrix A,int i,int j,PMatrix out){
				if(i<0||i>A->size[0]||j<0||j>A->size[1])return NULL;
				if(out == NULL){
					out=new Matrix;
					out->data.resize((A->size[0]-1)*(A->size[1]-1));
					out->dims=2;
					out->size[0]=A->size[0]-1;
					out->size[1]=A->size[1]-1;
					out->nWidth=out->size[1];
				}//else out must be an existed matrix of a right structure
				int cnt=1;
				for(int m=1;m<=A->size[0];m++){
					if(m==i)continue;
					for(int n=1;n<=A->size[1];n++){
						if(n==j)continue;
						GetRealData(out,cnt)=GetRealData2D(A,m,n);
						cnt++;
					}
				}
				return out;
			}

			void CleanUpMatrix(PMatrix& A){
				if(A!=NULL){
					delete A;
					A = NULL;
				}
			}

			void DumpMatrix(vector<double>& data,int dims,vector<int>& arrays){
				if(dims<=2){
					int x,y;
					for(x=1;x<=arrays[0];x++){
						for(y=1;y<=arrays[1];y++){
							printf("%f\t",data[(x-1)*arrays[1]+y-1]);
						}
						printf("\n");
					}
		
				}
			}

			void DumpMatrix(PMatrix A){
				if(A){
					DumpMatrix(A->data,A->dims,A->size);
				}
			}
		}
	}
}

