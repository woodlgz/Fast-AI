
#include "ANN_Main.h"
#include <string.h>
#include <fstream>

using namespace std;


RandomFactory* Test::random = NULL;

void Test::trainRBF(){
	string datasetName = prefix + ".dataset";
	string annName = prefix + ".ann";
	NeuralNetwork* ann = new RBFNeuralNetwork(true,true,1,1);
	ann->loadTrainingSet(datasetName.c_str());
	ann->setMaxIteration(100000);
	ann->setLearningProcessLog(true);
	//((RBFNeuralNetwork*)ann)->setRBF(NeuralNetwork::polyharmonicRadiusBaseFunction);
	ann->doTraining();
	ann->toLocalStorage(annName.c_str());
	vector<double> input(1);
	vector<double> result;
	input[0] = 5;
	ann->setInput(input);
	ann->pass();
	ann->getResult(result);
	cout<<"["<<testFunction(input[0])<<","<<result[0]<<"],";
	delete ann;
}

void Test::testTrainedRBF(const char* testSet){
	string testSetName = testSet;
	ofstream ofs((testSetName+".txt").c_str());
	ofstream ofs_ori(prefix+"_origin.txt");
	NeuralNetwork* ann = new RBFNeuralNetwork();
	ann->fromLocalStorage((prefix + ".ann").c_str());
	//((RBFNeuralNetwork*)ann)->setRBF(NeuralNetwork::polyharmonicRadiusBaseFunction);
	vector<double> input(1);
	vector<double> result;
	data.loadData((testSetName+".dataset").c_str());
	int* seq = new int[data.size()];
	for(int i=0;i<data.size();i++)
		seq[i] = i;
	quicksort(seq,data.size());
	for(int i=0;i<data.size();i++){
		input[0] = data[seq[i]][0];
		ann->setInput(input);
		ann->pass();
		ann->getResult(result);
		ofs<<"["<<input[0]<<","<<result[0]<<"],";
		ofs_ori<<"["<<input[0]<<","<<data.getExpected(seq[i])[0]<<"],";
		cout<<"["<<data.getExpected(seq[i])[0]<<","<<result[0]<<"],";
		result.clear();
	}
	delete[] seq;
	delete ann;
}

void Test::trainedRBFLearningProcess(){
	ofstream ofs((prefix+"学习过程.txt").c_str());
	RBFNeuralNetwork* ann = new RBFNeuralNetwork();
	ann->fromLocalStorage((prefix+".ann").c_str());
	vector<pair<unsigned int,double>>& process = ann->getLearningProcessLog();
	unsigned int size = process.size();
	for(int i=0;i<size;i++){
		ofs<<"["<<process[i].first<<","<<process[i].second<<"],";
	}
	delete ann;
}
void Test::XORClassifier(){
	vector<int> layers(3);
	layers[0] = 2;
	layers[1] = 2;
	layers[2] = 1;
	NeuralNetwork* ann = new BPNeuralNetwork(layers,0.3,3);
	ann->loadTrainingSet((prefix+".dataset").c_str());
	ann->setMaxIteration(10000);
	ann->setLearningProcessLog(true);
	ann->doTraining();
	DataSet testSet;
	testSet.loadData((prefix+".dataset").c_str());
	for(int i=0;i<testSet.size();i++){
		vector<double> result;
		ann->setInput(testSet[i]);
		ann->pass();
		ann->getResult(result);
		cout<<testSet[i][0]<<" xor "<<testSet[i][1]<<"is "<<result[0]<<" target:"<<testSet.getExpected(i)[0]<<endl;
	}
	cout<<endl;
	ann->toLocalStorage((prefix+".ann").c_str());
	delete ann;
}
void Test::trainXORClassifierLearningProcess(){
	NeuralNetwork* ann = new BPNeuralNetwork();
	ann->fromLocalStorage((prefix+".ann").c_str());
	vector<pair<unsigned int,double>> learningProcess = ann->getLearningProcessLog();
	int size = learningProcess.size();
	ofstream ofs((prefix+"学习过程.txt").c_str());
	for(int i=0;i<size;i++){
		double s = learningProcess[i].second;
		ofs<<s<<",";
	}
	delete ann;
}

int  Test::partition(int* array,int size){
	int low =0,high = size-1;
	int key = array[low];
	while(low<high){
		while(low<high&&cmp(&array[high],&key)>=0)high--;
		array[low] =array[high];
		while(low<high&&cmp(&array[low],&key)<=0)low++;
		array[high]=  array[low];
	}
	array[low] = key;
	return low;
}

void Test::quicksort(int* array,int size){
	if(size<=1)return;
	int k= partition(array,size);
	quicksort(array,k);
	quicksort(array+k+1,size-k-1);
}