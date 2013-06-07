
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <string>
#include <assert.h>
#include "ANN.h"


#include <boost/date_time/posix_time/posix_time_types.hpp>

using namespace FASTAI::ANN;


double NeuralNetwork::sigmoidFunction(double x){
	return 1.0 / (1.0 + exp(-x));
}

/** input param is the value of sigmoidFunction(x)*/
double NeuralNetwork::dSigmoidFunction(double y){
	return y * (1.0 - y);
}

/** Guass Radius Base Function */
double NeuralNetwork::guassRadiusBaseFunction(const vector<double>& paramX,const vector<double>& paramXc,double sigma){
	double dist = 0.0;
	assert(paramX.size()==paramXc.size() || AIException::assertFailed("Guass RBF Failed for invalid parameters"));
	int len = paramX.size();
	for(int i=0;i<len;i++){
		dist += pow((paramX[i]-paramXc[i]),2);
	}
	return exp(-1.0 * dist / pow(sigma,2) /2) ;
}


double NeuralNetwork::polyharmonicRadiusBaseFunction(const vector<double>& paramX,const vector<double>& paramXc,double sigma){
	double dist = 0.0;
	assert(paramX.size()==paramXc.size() || AIException::assertFailed("Poly RBF Failed for invalid parameters"));
	int len = paramX.size();
	for(int i=0;i<len;i++){
		dist += pow((paramX[i]-paramXc[i]),2);
	}
	double ans = sqrt(dist);
	cout<<dist<<" "<<ans<<endl;
	return ans;
}

void NeuralNetwork::loadTrainingSet(const char* path){
	m_TrainingSet.resize(m_InitSizeOfSet);
	m_TrainingSet.loadData(path);
}

void NeuralNetwork::logLearningProcess(vector<double>& error){
	static boost::posix_time::ptime initTime= boost::posix_time::microsec_clock::local_time();
	int  size = error.size();
	double errorSquare = 0.0;
	for(int i=0;i<size;i++){
		errorSquare = error[i]*error[i];
	}
	errorSquare /=2;
	boost::posix_time::ptime currentTime = boost::posix_time::microsec_clock::local_time();
	boost::posix_time::time_duration diff = currentTime - initTime;
	unsigned int diffMS = diff.total_milliseconds();
	pair<unsigned int,double>& last = m_LearningProcess.back();
	if(m_LearningProcess.size()>0&&last.first == diffMS)
		return;
	else 
		m_LearningProcess.push_back(make_pair(diffMS,errorSquare));
}