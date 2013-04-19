/*
 * DemoProblem2.h
 *
 *  Created on: Apr 19, 2013
 *      Author: woodlgz
 */

#ifndef DEMOPROBLEM2_H_
#define DEMOPROBLEM2_H_

/**
 * demo of solving Backpack Problem using GA engine
 */

#include "GA.h"

using namespace std;
using namespace FASTAI::GA;

class Demo2GeneticPhase : public GeneticPhase{
public:
	Demo2GeneticPhase():GeneticPhase(LEN){
		init();
	}
	virtual ~Demo2GeneticPhase(){
		cleanup();
	}

	void* read(){
		return NULL;
	}
private:
	void init(){

	}
	void cleanup(){

	}
	void crossing(GeneticPhase* phase){

	}
	void mutate(){

	}

public :
	static int LEN;
};

int Demo2GeneticPhase::LEN = 10;


class Demo1Env : public Env{
public:
	Demo1Env(float cRate, float mRate, GFactory* factory):
		Env(cRate,mRate,factory){
	}
private:
	void judge(){

	}

	float judge(int i){
		return 0.0;
	}

};

#endif /* DEMOPROBLEM2_H_ */
