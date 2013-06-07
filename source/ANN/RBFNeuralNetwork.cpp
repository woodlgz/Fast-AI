
#include "ANN.h"

#include <fstream>

using namespace FASTAI::ANN;


NeuralNetwork* RBFNeuralNetwork::fromLocalStorage(const char* path){
	ifstream ifs(path);
	if(ifs.good()){
		boost::archive::text_iarchive ia(ifs);
		ia & (*this);
	}else
		throw AIException("can't serialize object.invalid path");
}
void RBFNeuralNetwork::toLocalStorage(const char* path){
	ofstream ofs(path);
		if(ofs.good()){
			boost::archive::text_oarchive oa(ofs);
			oa & (*this);
	}else
		throw AIException("can't serialize object.invalid path");
}

void RBFNeuralNetwork::init(){
	if(!m_bUseGradientDesc && !m_bUseRegression) // for purpose of classification,use Gradient Desc Method
		m_bUseGradientDesc = true;
	if(m_bUseRegression){
		m_NumLayers  = 3;
		m_Nodes.resize(m_NumLayers);
		m_Nodes[0] = m_NumOfInputDim;
		m_Nodes[1] = m_NumOfBaseFunc;
		m_Nodes[2] = 1;
		m_NumOfOutputDim = 1;
		ActivateFunction = RBFNeuralNetwork::NoneActivateFunction;	//for regression,no activate function is used
	}else {
		m_NumLayers = 4;
		m_Nodes.resize(m_NumLayers);
		m_Nodes[0] = m_NumOfInputDim;
		m_Nodes[1] = m_NumOfBaseFunc;
		m_Nodes[2] = m_NumOfBaseFunc;
		m_Nodes[3] = m_NumOfOutputDim;
	}
	output.resize(m_NumLayers);
	for(int i=0;i<m_NumLayers;i++){
			output[i].resize(m_Nodes[i]);
	}
	m_Center.resize(m_NumOfBaseFunc);
	for(int i=0;i<m_NumOfBaseFunc;i++){
		m_Center[i].resize(m_NumOfInputDim);
	}
	m_sigma.resize(m_NumOfBaseFunc);
	//weight between input layer and RBF layer set to 1.0
	for(int i=0;i<m_NumOfInputDim;i++){
		for(int j=0;j<m_NumOfBaseFunc;j++){
			weight[0][i][j] = 1.0;
		}
	}
	//weight after RBF layer generated randomly
	for(int i=1;i<m_NumLayers-1;i++){
		int cnti = m_Nodes[i];
		int cnti_ = m_Nodes[i+1];
		for(int j=0;j<cnti;j++){
			for(int k=0;k<cnti_;k++){
				weight[i][j][k] = RANDOM_DOUBLE() * 4 -2;
			}
		}
	}
	m_etas.assign(m_NumLayers,DEFAULT_ETA);
	m_etas[1] = 0.01;
	m_CurrentIteration = 0;
}

void RBFNeuralNetwork::doTraining(){
	if(m_bUseGradientDesc) // use Gradient Descend Method
		return gradientTraining();
	else //use SOM Method
		somTraining();
}


void RBFNeuralNetwork::pass(){
	for(int i = 0;i<m_NumOfBaseFunc;i++){
		output[1][i] = RBF(output[0],m_Center[i],m_sigma[i]);
	}
	for(int i=1;i<m_NumLayers-1;i++){
		int cnti = m_Nodes[i];
		int cnti_ = m_Nodes[i+1];
		for(int j=0;j<cnti_;j++){
			double sum = 0.0;
			for(int k=0;k<cnti;k++){
				sum += output[i][k] * weight[i][k][j];
			}
			output[i+1][j] = ActivateFunction(sum);
		}
	}
}

//assume that the size of training set is not larger than 1024
void RBFNeuralNetwork::kmeans(){
	int size = m_TrainingSet.size();
	int sizeSeg = size / m_NumOfBaseFunc;
	assert(sizeSeg>=1 || AIException::assertFailed("too few training data"));
	int base = 0;
	//initialize the centers of RBF
	for(int i=0;i<m_NumOfBaseFunc-1;i++){
		m_Center[i] = m_TrainingSet[base + (RANDOM_INT() % sizeSeg)];
		base += sizeSeg;
	}
	sizeSeg = size - base;
	m_Center[m_NumOfBaseFunc-1] = m_TrainingSet[base + (RANDOM_INT() % sizeSeg)];
	vector<vector<int>> cluster(m_NumOfBaseFunc);
	bool done = false;
	while(!done){
		cluster.clear();
		for(int i=0;i<size;i++){
			double minDist = 1.0e30;
			int min = 0;
			for(int j=0;j<m_NumOfBaseFunc;j++){
				double dist = calcDist(m_TrainingSet[i],m_Center[j]);
				if(dist < minDist){
					minDist = dist;
					min = j;
				}
			}
			cluster[min].push_back(i);
		}
		//calcuate new center for all clusters
		done = true;
		vector<double> newCenter(m_NumOfInputDim);
		for(int i=0;i<m_NumOfBaseFunc;i++){
			for(int j=0;j<m_NumOfInputDim;j++)
				newCenter[j] = 0.0;
			int cluster_size = cluster[i].size();
			for(int j=0;j<cluster_size;j++){
				for(int k=0;k<m_NumOfInputDim;k++)
					newCenter[k]+=m_TrainingSet[cluster[i][j]][k];
			}
			for(int j=0;j<m_NumOfInputDim;j++){
				newCenter[j] /= cluster_size;
				if(fabs(m_Center[i][j]-newCenter[j]) > m_CenterError){
					done = false;
				}
				m_Center[i][j] = newCenter[j];
			}
		}
	}
}

void RBFNeuralNetwork::somTraining(){
	//use k-mean algorithm to handle cluster
	kmeans();
	//calcuate sigmas
	calcSigma();
	//calcuate Green matrix
	calcGreen();
	//calcuate pseudo inverse of Green matrix,and done with training
	PMatrix pi = pseudo_invert();
	PMatrix expected = CreateMatrix2D(m_Green->size[0],1);
	expected->data = target;
	PMatrix w = MatrixMul2D(pi,expected);
	for(int i=0;i<m_NumOfBaseFunc;i++){
		weight[m_NumLayers-2][i][0] = GetRealData2D(w,i+1,1);
	}
	CleanUpMatrix(pi);
	CleanUpMatrix(w);
	CleanUpMatrix(expected);
}

void RBFNeuralNetwork::gradientTraining(){
	int base = 0 ;
	int segSize = m_TrainingSet.size() / m_NumOfBaseFunc;
	//initialize weights,centers,sigma
	for(int i=0;i<m_NumOfBaseFunc;i++){
		int cnt = m_Nodes[2];
		for(int j=0;j<cnt;j++){
			weight[1][i][j] = RANDOM_DOUBLE() * 0.2 - 0.1;
		}
		segSize = i==m_NumOfBaseFunc-1?m_TrainingSet.size()-base:segSize;
		m_Center[i] = m_TrainingSet[base+(RANDOM_INT()%segSize)];
		m_sigma[i] = RANDOM_DOUBLE() * 0.2 + 0.1;
		base += segSize;
	}
	int numTestCase = m_NumOfBaseFunc * 3;
	int size  = m_TrainingSet.size();
	segSize = size / numTestCase;
	assert(segSize>=1 || AIException::assertFailed("too few training data"));
	m_CurrentIteration = 0;
	vector<int> seq(numTestCase);
	for(int i=0,base=0;i<numTestCase;i++,base+=segSize){
		segSize = i==numTestCase-1?m_TrainingSet.size()-base:segSize;
		seq[i] = (RANDOM_INT()%segSize) + base;
	}
	while(m_CurrentIteration<m_MaxIterations){
		//segSize = size / numTestCase;
		//assert(segSize>=1 || AIException::assertFailed("too few training data"));
		//vector<int> seq(numTestCase);
		vector<double> error(numTestCase);
		bool done = true;
		for(int i=0;i<numTestCase;i++){
		//for(int i=0,base=0;i<numTestCase;i++,base+=segSize){
			//segSize = i==numTestCase-1?m_TrainingSet.size()-base:segSize;
			//seq[i] = (RANDOM_INT()%segSize) + base;
			setInput(m_TrainingSet[seq[i]]);
			double expected = m_TrainingSet.getExpected(seq[i])[0];
			pass();
			error[i] = expected - output[m_NumLayers-1][0];
			if(fabs(error[i])>m_TargetError)
				done = false;
		}
		if(m_IsLearningProcessRecorded)
			logLearningProcess(error);
		if(done)
			break;
		for(int j = 0;j<m_NumOfBaseFunc;j++){
			double deltaWeight = 0.0;
			double deltaSigma = 0.0;
			vector<double> deltaCenter(m_NumOfInputDim);
			deltaCenter.assign(m_NumOfInputDim,0.0);
			/*if(fabs(m_sigma[j])<0.1){
				//cout<<m_sigma[j]<<endl;
				m_sigma[j] = 0.1;
			}*/
			for(int i=0;i<numTestCase;i++){
				double G =  error[i] * RBF(m_TrainingSet[seq[i]],m_Center[j],m_sigma[j]);
				deltaWeight += G;
				deltaSigma  += G * calcDist(m_TrainingSet[seq[i]],m_Center[j]);
				for(int k=0;k<m_NumOfInputDim;k++){
					deltaCenter[k] += (m_TrainingSet[seq[i]][k] - m_Center[j][k]) * G;
				}
			}
			deltaWeight = m_etas[1] * deltaWeight;
			deltaSigma = m_etas[1] * deltaSigma * weight[1][j][0] / pow(m_sigma[j],3);
			double T = m_etas[1] * weight[1][j][0] / pow(m_sigma[j],2);
			for(int k=0;k<m_NumOfInputDim;k++){
				deltaCenter[k] = T * deltaCenter[k] ;
				m_Center[j][k] += deltaCenter[k];
			}
			weight[1][j][0] += deltaWeight;
			m_sigma[j] += deltaSigma;
		}
		m_CurrentIteration++;
	}
	
}

PMatrix RBFNeuralNetwork::pseudo_invert(){
	PMatrix trans = MatrixTranspose(m_Green);
	PMatrix tmp = MatrixMul2D(trans,m_Green);
	PMatrix inv = MatrixInv2D(tmp);
	CleanUpMatrix(tmp);
	tmp = MatrixMul2D(inv,trans);
	CleanUpMatrix(inv);
	CleanUpMatrix(trans);
	return tmp;
}

void RBFNeuralNetwork::calcSigma(){
	for(int i=0;i<m_NumOfBaseFunc;i++){
		double dist = 1e30;
		for(int j=0;j<m_NumOfBaseFunc;j++){
			if(i==j)
				continue;
			double tmp = calcDist(m_Center[i],m_Center[j]);
			if(tmp<dist)
				dist = tmp;
		}
		m_sigma[i] = sqrt(dist);
	}
}

void RBFNeuralNetwork::calcGreen(){
	//num: the number of sample vectors for regression
	int num = m_NumOfBaseFunc * 3;
	int segSize = m_TrainingSet.size() / num;
	int base = 0;
	vector<int> seq(num);
	for(int i=0;i<num;i++)
		seq[i] = i;
	for(int i=num-1;i>=0;i--)
		swap(seq[RANDOM_INT()%(i+1)],seq[i]);
	m_Green = CreateMatrix2D(num,m_NumOfBaseFunc);
	target.resize(num);
	for(int i=1;i<=num;i++){
		int rnd = seq[i-1]*segSize+(RANDOM_INT()%segSize);
		vector<double>& data = m_TrainingSet[rnd];
		target[i-1] = m_TrainingSet.getExpected(rnd)[0];
		for(int j=1;j<=m_NumOfBaseFunc;j++){
			GetRealData2D(m_Green,i,j) = RBF(data,m_Center[j-1],m_sigma[j-1]);
		}
	}
}

//note: return Dist^2
double RBFNeuralNetwork::calcDist(const vector<double>& v1,const vector<double>& v2){
	assert(v1.size() == v2.size() || AIException::assertFailed("calcDist invalid parameters"));
	int len = v1.size();
	double dist = 0.0;
	for(int i=0;i<len;i++){
		dist += pow(v1[i]-v2[i],2);
	}
	return dist;
}