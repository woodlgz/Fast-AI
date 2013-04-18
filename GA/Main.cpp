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
	void init(){
	}
private:
	void crossing(GeneticPhase* phase){
	}
	void mutate(){
	}

};



class MyEnv : public Env{
private:
	float judge(int i){
		return 0.0;
	}
};


int main(int argc,char** argv){
	GFactory* factory = GeneticFactory<MyGeneticPhase>::getFactory();
	return 0;
}
