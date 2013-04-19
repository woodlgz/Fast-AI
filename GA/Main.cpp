/*
 * Main.cpp
 *
 *  Created on: Apr 15, 2013
 *      Author: woodlgz
 */

#include <iostream>
#include <assert.h>
#include "DemoProblem1.h"

using namespace std;
using namespace FASTAI::GA;




int main(int argc,char** argv){
	time_t t;
	t = time(NULL);
	cout<<"start:"<<t<<endl;
	GFactory* factory = GeneticFactory<Demo1GeneticPhase>::getFactory();
	assert(factory && "can't create GFactory");
	Env* env = new Demo1Env(0.02,0.01,factory);
	GeneticPhase* answer = Solve(env,5,10);
	int* result = (int*)answer->read();
	cout<<"answer:"<<*result<<endl;
	delete[] result;
	delete factory;
	delete env;
	t = time(NULL);
	cout<<"end:"<<t<<endl;
	return 0;
}
