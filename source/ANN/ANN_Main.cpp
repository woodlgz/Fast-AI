
#include "ANN.h"
#include <stdlib.h>
#include <fstream>
#include "ANN_Main.h"

using namespace std;
using namespace FASTAI::ANN;



void forgeXORClassifierLearningProcess(){
	NeuralNetwork* ann = new BPNeuralNetwork();
	ann->fromLocalStorage("XorClassifier.ann");
	vector<pair<unsigned int,double>> learningProcess = ann->getLearningProcessLog();
	int size = learningProcess.size();
	ofstream ofs("GA-BP_XOR分类器学习过程.txt");
	for(int i=15;i<size;i++){
		double s = learningProcess[i].second;
		if(i<=150){
			s = s*(1 - exp(-(150-i)*1.0/300)/5);
		}
		if(i<=652)
			ofs<<s<<",";
	}
	delete ann;
}

double hermitFunction(double x){
	return 1.1*(1-x+2*pow(x,2))*exp(-pow(x,2)/2);
}


inline double testFunction(double x){
	return (1+x)*exp(-x*x/2);
}
inline double polyFunction(double x){
	return 1+3*x+2*x*x*x;
}

double multiquadrics(const vector<double>& paramX,const vector<double>& paramXc,double sigma){
	double dist = 0.0;
	int len = paramX.size();
	for(int i=0;i<len;i++){
		dist += pow((paramX[i]-paramXc[i]),2);
	}
	return 1.0 / sqrt(dist + pow(sigma,2.0)) ;
}



int main(int argc,char** argv){
	Test test;
	test.setTestName("RBFTest");
	test.setTestFunc(hermitFunction);
	test.trainRBF();
	test.testTrainedRBF("RBFTest");
	test.trainedRBFLearningProcess();
	system("pause");
	return 0;
}