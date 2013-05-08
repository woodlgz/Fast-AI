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
	~Demo1GeneticPhase(){
		cleanup();
	}

	void* read(){
		m_Answer = calcValueOfCode();
		cout<<"max value:"<<m_Answer<<endl;
		return &m_Answer;
	}

	int calcValueOfCode(){
		m_Answer = 0;
		for(int i=0;i<m_Len;i++){
			m_Answer+= m_Coding[i]<<i;
		}
		return m_Answer;
	}

private:
	void init(){
		time_t t;
		m_Coding = new int[m_Len];
		for(int i=0;i<m_Len;i++){
			m_Coding[i] = GENERATE_RANDOM()%2;
		}
	}
	void cleanup(){
		if(m_Coding){
			delete[] m_Coding;
			m_Coding = NULL;
		}
	}
	void crossing(GeneticPhase* phase){
		Demo1GeneticPhase* _phase = static_cast<Demo1GeneticPhase*>(phase);
		int i = GENERATE_RANDOM()%_phase->m_Len;
		swap(m_Coding[i],_phase->m_Coding[i]);
	}
	void mutate(){
		int i = GENERATE_RANDOM()%m_Len;
		m_Coding[i] = GENERATE_RANDOM()%2==0?(~(0x01&m_Coding[i]))&0x01:m_Coding[i];
	}
	void reConstruct(){
		bool valid = false;
		for(int i=0;i<m_Len;i++){
			m_Coding[i] = GENERATE_RANDOM()%2;
			if(!valid && m_Coding[i] == 1)
				valid = true;
		}
		if(!valid){
			m_Coding[GENERATE_RANDOM()%m_Len] = 1;
		}
	}
public :
	static const int LEN  = 6;
};



class Demo1Env : public Env{
public:
	Demo1Env(GFactory* factory,float cRate, float mRate, int age):
		Env(factory,cRate,mRate,age){
	}
private:
	void judge(){
		m_ScoreMax = 0.0;
		for(int i=0;i<m_PSize;i++){
			int len = m_Population[i]->getLen();
			int sum = 0;
			for(int j=0;j<len;j++){
				sum += *((int*)(m_Population[i]->getCodeAt(j)))<<j;
			}
			m_Score[i] = sum;
			if(m_ScoreMax<m_Score[i])
				m_ScoreMax = m_Score[i];
		}
	}

	float judge(int i){
		if(m_ScoreMax<1e-38&&m_ScoreMax>-1e-38)
				return 0.0;
		return m_Score[i] / m_ScoreMax;
	}
};

#endif /* DEMOPROBLEM1_H_ */
