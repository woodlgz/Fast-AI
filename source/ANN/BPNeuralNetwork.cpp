
#include <stdlib.h>
#include <fstream>
#include "ANN.h"
using namespace FASTAI::ANN;

NeuralNetwork* BPNeuralNetwork::fromLocalStorage(const char* path){
	ifstream ifs(path);
	if(ifs.good()){
		boost::archive::text_iarchive ia(ifs);
		ia & (*this);
	}else
		throw AIException("can't serialize object.invalid path");
}
void BPNeuralNetwork::toLocalStorage(const char* path){
	ofstream ofs(path);
		if(ofs.good()){
			boost::archive::text_oarchive oa(ofs);
			oa & (*this);
	}else
		throw AIException("can't serialize object.invalid path");
}

void BPNeuralNetwork::init(){
	//initialize weights
	for(int i=0;i<m_NumLayers-1;i++){
		output[i][m_Nodes[i]] = 1.0; // bias
		int cnti = m_Nodes[i];
		int cntj = m_Nodes[i+1];
		for(int j=0;j<=cnti;j++){
			for(int k=0;k<cntj;k++)
				weight[i][j][k] = RANDOM_DOUBLE() * 4 -2;
		}
	}
}

void BPNeuralNetwork::doTraining(){
	int seq[1024];
	int time = m_TrainingSet.size()%1024==0?m_TrainingSet.size()>>10:1+(m_TrainingSet.size()>>10);
	for(m_CurrentIteration=0;m_CurrentIteration<m_MaxIterations;m_CurrentIteration++){
		int base = 0;
		for(int i=1;i<time;i++){
			for(int j=1023;j>=0;j--)
				seq[j] = j;
			for(int j=1023;j>=0;j--){
				swap(seq[RANDOM_INT()%(j+1)],seq[j]);
			}
			for(int j=0;j<1024;j++){
				setInput(m_TrainingSet[base+seq[j]]);
				setTarget(m_TrainingSet.getExpected(base+seq[j]));
				trainBP();
			}
			base+=1024;
		}
		int remain = m_TrainingSet.size()-base;
		for(int i=0;i<remain;i++){
			seq[i] = i;
		}
		for(int i=remain-1;i>=0;i--)
			swap(seq[RANDOM_INT()%(i+1)],seq[i]);
		for(int i=0;i<remain;i++){
			setInput(m_TrainingSet[base+seq[i]]);
			setTarget(m_TrainingSet.getExpected(base+seq[i]));
			trainBP();
		}
		if(m_IsLearningProcessRecorded){
			int s = m_Nodes[m_NumLayers-1];
			vector<double> outputError(s);
			int r = RANDOM_INT() % m_TrainingSet.size();
			setInput(m_TrainingSet[r]);
			setTarget(m_TrainingSet.getExpected(r));
			trainBP();
			for(int v=0;v<s;v++){
				outputError[v] = target[v] - output[m_NumLayers-1][v];
			}
			logLearningProcess(outputError);
		}
	}
}

void BPNeuralNetwork::trainBP(){
	//pass through the network
	for(int i=1;i<m_NumLayers;i++){ //current layer i
		int prev_layer_nodes = m_Nodes[i-1];
		int cur_layer_nodes = m_Nodes[i];
		for(int k=0;k<cur_layer_nodes;k++){
			double sum = 0.0;
			for(int j=0;j<=prev_layer_nodes;j++){
				sum+= weight[i-1][j][k] * output[i-1][j];
			}
			output[i][k] = ActivateFunction(sum);
		}
	}
	//back propagation
	double delta[2][MAX_NODE_PER_LAYER];
	double* curDelta = delta[0];
	double* nextDelta = delta[1];
	int cnto = m_Nodes[m_NumLayers-1];
	vector<double> outputError(cnto);
	for(int i=0;i<cnto;i++){
		outputError[i] = (target[i] - output[m_NumLayers-1][i]);
		curDelta[i] = DActivateFunction(output[m_NumLayers-1][i]) * outputError[i];
	}
	for(int i=m_NumLayers-1;i>0;i--){
		int cntj = m_Nodes[i-1];
		int cntk = m_Nodes[i];
		//figure out the equivalent delta of last layer
		for(int j=0;j<=cntj;j++){
			double error = 0.0;
			for(int k=0;k<cntk;k++){
				error += weight[i-1][j][k] * curDelta[k];
			}
			nextDelta[j] = DActivateFunction(output[i-1][j]) * error;
		}
		//modify the weight between layer i and layer i-1
		for(int k=0;k<cntk;k++){
			for(int j=0;j<=cntj;j++){
				weight[i-1][j][k] += m_etas[i-1] * curDelta[k] * output[i-1][j];
			}
		}
		swap(curDelta,nextDelta);
	}
}

void BPNeuralNetwork::pass(){
	for(int i=1;i<m_NumLayers;i++){ //current layer i
		int prev_layer_nodes = m_Nodes[i-1];
		int cur_layer_nodes = m_Nodes[i];
		for(int k=0;k<cur_layer_nodes;k++){
			double sum = 0.0;
			for(int j=0;j<=prev_layer_nodes;j++){
				sum+= weight[i-1][j][k] * output[i-1][j];
			}
			output[i][k] = ActivateFunction(sum);
		}
	}
}