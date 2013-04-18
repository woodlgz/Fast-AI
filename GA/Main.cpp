/*
 * Main.cpp
 *
 *  Created on: Apr 15, 2013
 *      Author: woodlgz
 */

#include <iostream>
#include "GA.h"

using namespace std;
using namespace FASTAI::GA;

/**
 * demo of using GA Template
 * steps:
 * 	1.derive the GeneticPhase class and Env Class
 * 	2.call GA::Solve
 */


class MyGeneticPhase : public GeneticPhase{
public:
	MyGeneticPhase():GeneticPhase(LEN){
		init();
	}
	~MyGeneticPhase(){
		cleanup();
	}

	void* read(){
		int* result = new int[1];
		*result = 0;
		for(int i=0;i<m_Len;i++){
			*result += m_Coding[i]<<i;
		}
		return result;
	}
private:
	void init(){
		time_t t;
		m_Coding = new unsigned char[m_Len];
		for(int i=0;i<m_Len;i++){
			t = time(NULL);
			srand(t);
			m_Coding[i] = rand()%2;
		}
	}
	void cleanup(){
		if(m_Coding)
			delete[] m_Coding;
	}
	void crossing(GeneticPhase* phase){
		MyGeneticPhase* _phase = static_cast<MyGeneticPhase*>(phase);
		time_t t = time(NULL);
		srand(t);
		int i = rand()%_phase->m_Len;
		swap(m_Coding[i],_phase->m_Coding[i]);
	}
	void mutate(){
		time_t t = time(NULL);
		srand(t);
		int i = rand()%m_Len;
		t = time(NULL);
		srand(t);
		m_Coding[i] = rand()%2==0?~m_Coding[i]:m_Coding[i];
	}

public :
	static const int LEN  = 6;
};



class MyEnv : public Env{
public:
	MyEnv(float cRate, float mRate, GFactory* factory):
		Env(cRate,mRate,factory){
	}
private:
	void judge(){
		m_avg = 0;
		for(int i=0;i<m_PSize;i++){
			int len = m_Population[i]->getLen();
			int sum = 0;
			for(int j=0;j<len;j++){
				sum += m_Population[i]->getCodeAt(j)<<j;
			}
			m_Score[i] = sum;
			m_avg += sum;
		}
		m_avg = m_avg / m_PSize;
	}

	float judge(int i){
		return m_Score[i]*1.0 / m_avg;
	}
private:
	int m_avg;
};


int main(int argc,char** argv){
	GFactory* factory = GeneticFactory<MyGeneticPhase>::getFactory();
	Env* env = new MyEnv(0.02,0.01,factory);
	GeneticPhase* answer = Solve(env);
	int* result = (int*)answer->read();
	cout<<"answer:"<<*result<<endl;
	delete[] result;
	delete[] factory;
	delete[] env;
	return 0;
}
