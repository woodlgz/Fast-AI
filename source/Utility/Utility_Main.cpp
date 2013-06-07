
#include <iostream>
#include "Util.h"

using namespace std;
using namespace FASTAI::Util::Common;
using namespace FASTAI::Util::Math;

int main(int argc,char** argv){
	//test random generator
	RandomFactory* rFactory = RandomFactory::getFactory();
	for(int i=0;i<100;i++){
		cout<<rFactory->getRandomDouble()<<endl;
	}
	//test matrix tool
	vector<int> sizeOfA(2);
	vector<double> dataOfA;
	double dataA[] = {10,-3  ,2,
					 2 , 4  ,-1,
					 1 , 2  ,5
					};
	sizeOfA.resize(2);
	sizeOfA[0] = 3;sizeOfA[1] = 3;
	dataOfA.resize(sizeof(dataA)/sizeof(double));
	for(int i=sizeof(dataA)/sizeof(double)-1;i>=0;i--){
		dataOfA[i] = dataA[i];
	}
	PMatrix A = CreateMatrix(2,sizeOfA,dataOfA);

	vector<int> sizeOfB;
	vector<double> dataOfB;
	double dataB[] = { 3 ,
					   20,
					  -12};
	sizeOfB.resize(2);
	sizeOfB[0] = 3; sizeOfB[1] = 1;
	dataOfB.resize(sizeof(dataB)/sizeof(double));
	for(int i=sizeof(dataB)/sizeof(double)-1;i>=0;i--){
		dataOfB[i] = dataB[i];
	}
	PMatrix B = CreateMatrix(2,sizeOfB,dataOfB);
	PMatrix Result = MatrixMul2D(A,B);
	cout<<"A.*B = "<<endl;
	DumpMatrix(Result);
	CleanUpMatrix(Result);
	PMatrix Inv = MatrixInv2D(A);
	cout<<"Inv A: "<<endl;
	DumpMatrix(Inv);
	Result = MatrixMul2D(A,Inv);
	cout<<"A.*Inv(A):"<<endl;
	DumpMatrix(Result);
	CleanUpMatrix(Result);
	CleanUpMatrix(Inv);
	CleanUpMatrix(A);
	CleanUpMatrix(B);
#ifdef _WIN32
	system("pause");
#endif
	return 0;
}