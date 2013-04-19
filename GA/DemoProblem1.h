/*
 * DemoProblem1.h
 *
 *  Created on: Apr 18, 2013
 *      Author: woodlgz
 */

#ifndef DEMOPROBLEM1_H_
#define DEMOPROBLEM1_H_

/**
 * demo of using GA Template
 * steps:
 * 	1.derive the GeneticPhase class and Env Class
 * 	2.call GA::Solve
 */

#include "GA.h"

using namespace std;
using namespace FASTAI::GA;

class Demo1GeneticPhase : public GeneticPhase{
public:
	Demo1GeneticPhase():GeneticPhase(LEN){
		init();
	}
	virtual ~Demo1GeneticPhase(){
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
			srand(rand()%t);
			m_Coding[i] = rand()%2;
		}
	}
	void cleanup(){
		if(m_Coding)
			delete[] m_Coding;
	}
	void crossing(GeneticPhase* phase){
		Demo1GeneticPhase* _phase = static_cast<Demo1GeneticPhase*>(phase);
		time_t t = time(NULL);
		srand(rand()%t);
		int i = rand()%_phase->m_Len;
		swap(m_Coding[i],_phase->m_Coding[i]);
	}
	void mutate(){
		time_t t = time(NULL);
		srand(rand()%t);
		int i = rand()%m_Len;
		t = time(NULL);
		srand(rand()%t);
		m_Coding[i] = rand()%2==0?(~(0x01&m_Coding[i]))&0x01:m_Coding[i];
	}

public :
	static const int LEN  = 6;
};



class Demo1Env : public Env{
public:
	Demo1Env(float cRate, float mRate, GFactory* factory):
		Env(cRate,mRate,factory){
	}
private:
	void judge(){
		m_ScoreMax = 0.0;
		for(int i=0;i<m_PSize;i++){
			int len = m_Population[i]->getLen();
			int sum = 0;
			for(int j=0;j<len;j++){
				sum += m_Population[i]->getCodeAt(j)<<j;
			}
			m_Score[i] = sum;
			if(m_ScoreMax<m_Score[i])
				m_ScoreMax = m_Score[i];
		}
	}

	float judge(int i){
		return m_Score[i] / m_ScoreMax;
	}
private:
	float m_ScoreMax;
};

#endif /* DEMOPROBLEM1_H_ */
