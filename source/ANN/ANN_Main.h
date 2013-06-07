

#ifndef ANN_MAIN_H__
#define ANN_MAIN_H__

#include <string>
#include "ANN.h"

using namespace std;
using namespace FASTAI::ANN;

typedef double (*TESTFUNC)(double);

class Test{
public:
	Test(){
		if(random == NULL)
			random  = RandomFactory::getFactory();
	}
	inline void setTestName(const char* testName){
		prefix = testName;
	}
	inline void setTestFunc(TESTFUNC func){
		testFunction = func;
	}
	int cmp(const void* a,const void* b){
		int _a = *(int*)a;
		int _b = *(int*)b;
		if(data[_a][0]<data[_b][0])return -1;
		else if(data[_a][0]>data[_b][0])return 1;
		else return 0;
	}
	inline double uniform(double floor,double ceil){
		return floor+random->getRandomDouble()*(ceil-floor);
	}

	inline double RandomNorm(double mu,double sigma,double floor,double ceil){
		double x,prob,y;
		do{
			x=uniform(floor,ceil);
			prob=1/sqrt(2*3.1415926*sigma)*exp(-1*(x-mu)*(x-mu)/(2*sigma*sigma));
			y=1.0*random->getRandom()/0xFFFFFFFF;
		}while(y>prob);
		return x;
	}
	int  partition(int* input,int size);
	void quicksort(int* input,int size);
	void trainRBF();
	void testTrainedRBF(const char* testSet);
	void trainedRBFLearningProcess();
	void XORClassifier();
	void trainXORClassifierLearningProcess();
public:
	static RandomFactory* random;
private:
	DataSet data;
	string prefix;
	TESTFUNC testFunction;
};


#endif