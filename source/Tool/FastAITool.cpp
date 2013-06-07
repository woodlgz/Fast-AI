
#include "../ANN/DataSet.h"
#include "../Utility/Util.h"
#include <iostream>
#include <stdlib.h>
#include <fstream>

using namespace std;

using namespace FASTAI::ANN;
using namespace FASTAI::Util::Common;

DataSet data;


RandomFactory* random;

void store(double testData[],double expected[],int size){
	try{
		data.loadData("XOR.dataset");
	}catch(exception& e){}
	for(int i=0;i<size;i++){
		data.append(testData+(i<<1),2);
		data.appendExpected(expected+i,1);
	}
	data.storeData("XOR.dataset");
}

void load(){
	data.loadData("XOR.dataset");
	for(int i=0,size = data.size();i<size;i++){
		for(int j=0,size2 = data[i].size();j<size2;j++){
			cout<<data[i][j]<<" ";
		}
		cout<<"exp:";
		for(int j=0,size2 = data.getExpected(i).size();j<size2;j++)
			cout<<data.getExpected(i)[j]<<" ";
		cout<<endl;
	}
}


void generateTrainingSetForXor(bool isStore){
	double test[8] = {0,0,
					   0,1,
					   1,0,
					   1,1};
	double expected[4] = {0,1,1,0};
	try{
		if(isStore)
			store(test,expected,sizeof(expected)/sizeof(double));
		else 
			load();
	}catch(exception& e){
		cout<<e.what()<<endl;
	}
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

int cmp(const void* a,const void* b){
	int _a = *(int*)a;
	int _b = *(int*)b;
	if(data[_a][0]<data[_b][0])return -1;
	else if(data[_a][0]>data[_b][0])return 1;
	else return 0;
}

void generateTrainingSetForRegression(bool isStore){
	try{
		if(isStore){
			try{
				data.loadData("RBFTest_poly.dataset");
			}catch(exception& e){}
			double x[10];
			double y[10];
			for(int i=0;i<100;i++){
				x[0]= uniform(-10,10);
				//cout<<x[0]<<" ";
				y[0]= polyFunction(x[0]) + RandomNorm(0,0.1,-0.3,0.3);
				//cout<<y[0]<<endl;
				data.append(x,1);
				data.appendExpected(y,1);
			}
			data.storeData("RBFTest_poly.dataset");
		}else{
			std::ofstream f("data_poly.txt");
			data.loadData("RBFTest_poly.dataset");
			int* seq = new int[data.size()];
			for(int i=0;i<data.size();i++)
				seq[i] = i;
			qsort(seq,data.size(),sizeof(int),cmp);
			for(int i=0;i<data.size();i++){
				f<<"["<<data[seq[i]][0]<<","<<data.getExpected(seq[i])[0]<<"],";
			}
			delete[] seq;
		}
	}catch(exception& e){
		cout<<e.what()<<endl;
	}
}

void generateTestSetForPrediction(bool isStore){
	try{
		if(isStore){
			try{
				data.loadData("RBFTest_Predict.dataset");
			}catch(exception& e){}
			double x[10];
			double y[10];
			double seg = 0;
			for(int i=0;i<100;i++,seg+=5.0/100){
				x[0]= 5+seg;
				//cout<<x[0]<<" ";
				y[0]= hermitFunction(x[0]);// + RandomNorm(0,0.1,-0.3,0.3);
				//cout<<y[0]<<endl;
				data.append(x,1);
				data.appendExpected(y,1);
			}
			data.storeData("RBFTest_Predict.dataset");
		}else{
			std::ofstream f("data_predict.txt");
			data.loadData("RBFTest_Predict.dataset");
			int* seq = new int[data.size()];
			for(int i=0;i<data.size();i++)
				seq[i] = i;
			qsort(seq,data.size(),sizeof(int),cmp);
			for(int i=0;i<data.size();i++){
				f<<"["<<data[seq[i]][0]<<","<<data.getExpected(seq[i])[0]<<"],";
			}
			delete[] seq;
		}
	}catch(exception& e){
		cout<<e.what()<<endl;
	}
}

int main(int argc,char** argv){
	random = RandomFactory::getFactory();
	//generateTrainingSetForXor(true);
	//generateTrainingSetForXor(false);
	generateTrainingSetForRegression(true);
	generateTrainingSetForRegression(false);
	//generateTestSetForPrediction(true);
	//generateTestSetForPrediction(false);
	system("pause");
	return 0;
}