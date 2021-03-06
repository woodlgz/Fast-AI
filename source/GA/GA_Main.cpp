/*
 * GA_Main.cpp
 *
 *  Created on: Apr 15, 2013
 *      Author: woodlgz
 */

#include <iostream>
#include <assert.h>

#include "GA_DemoProblem1.h"
#include "GA_DemoProblem2.h"
#include "GA_DemoProblem3.h"

#if defined _WIN32

#include <stdlib.h>

#endif

using namespace std;
using namespace FASTAI::GA;

void Demo1(){
	GFactory* factory = GeneticFactory<Demo1GeneticPhase>::getFactory();
	assert(factory && "can't create GFactory");
	Env* env = new Demo1Env(factory,0.02,0.01,100);
	GeneticPhase* answer = Solve(env,100);
	answer->read();
	delete factory;
	delete env;
}

/**
 * Demo2 Test Data:
 *
  (1)
   	unsigned int len = 16;
   	unsigned int itemCnt[17]		=	{0,2,3,4,3,2,1,10,2,3,12,9,5,6,16,17,18};	//first element is just place holder
   	unsigned int value[17]		=	{0,2,5,4,3,6,7, 3,5,2,2 ,2,8,3, 1,2 , 1};
   	unsigned int weight[17]		=	{0,2,3,3,4,5,6, 4,5,4,2 ,1,6,3, 2,1 , 3};
   	unsigned int packSize = 20;
   	unsigned int packItemCnt = 7;
   	unsigned int solution[17];
  (2)
	unsigned int len = 20;
	unsigned int itemCnt[21]	=	{0,5,3,2,9,3,1,9,3,4,12,6,4,3,8,17,18, 4,1,9,3};	//first element is just place holder
	unsigned int value[21]		=	{0,2,5,4,3,6,7, 3,5,2,2 ,3,2,3, 1,3 ,1,9,6,2,7};
	unsigned int weight[21]		=	{0,2,3,3,4,5,4, 4,5,4,2 ,3,5,3, 2,3 ,3,8,4,3,5};
	unsigned int packSize = 30;
	unsigned int packItemCnt = 10;
	unsigned int solution[21];
 */

void Demo2(){
	unsigned int len = 15;
	unsigned int itemCnt[16]	=	{0,5,3,2,3,1,9,3,12,6,4,3,17,18, 4,9};	//first element is just place holder
	unsigned int value[16]		=	{0,2,5,4,6,7, 3,5,2,3,2,3,3 ,1,9,2};
	unsigned int weight[16]		=	{0,2,3,3,5,4, 4,5,2,3,5,3,3 ,3,8,3};
	unsigned int packSize = 20;
	unsigned int packItemCnt = 10;
	unsigned int solution[16];
	Demo2GeneticPhase::Problem_Init(packSize,packItemCnt,len,itemCnt,value,weight);
	cout<<"normal solution"<<endl;
	cout<<"max value:"<<Demo2GeneticPhase::DPSolution()<<endl;
	Demo2GeneticPhase::readDPSolution(solution);
	for(int i=1;i<=len;i++){
		cout<<"take "<<solution[i]<<" items of type "<<i<<",weights:"<<weight[i]*solution[i]<<",values:"<<value[i]*solution[i]<<endl;
	}
	cout<<"GA solution"<<endl;
	GFactory* factory = GeneticFactory<Demo2GeneticPhase>::getFactory();
	Env* env = new Demo2Env(factory,0.02,0.01,1000);	// the longer the age is ,the more likely it is to get the right answer
	GeneticPhase* answer = Solve(env,100);
	answer->read();
	delete factory;
	delete env;
}


void Demo3(){
	int Graph[][Demo3GeneticPhase::MAX_VETEX+1] = {{0},{0,0,3,1,2,7},{0,5,0,1,3,3},{0,6,4,0,4,2},{0,3,2,5,0,3},{0,2,4,1,3,0}};//顶点为5的完全图
	//int Graph[][Demo3GeneticPhase::MAX_VETEX+1] = {{0},{0,0,1,2},{0,1,0,1},{0,1,2,0}};
	Demo3GeneticPhase::ProblemInit(Graph,5);
	cout<<"normal solution:"<<endl;
	Demo3GeneticPhase::DPSolver(1);
	cout<<"GA solution:"<<endl;
	GFactory* factory = GeneticFactory<Demo3GeneticPhase>::getFactory();
	Env* env = new Demo3Env(factory,0.02,0.01,100);
	GeneticPhase* answer = Solve(env,100);
	answer->read();
	delete factory;
	delete env;
}



int main(int argc,char** argv){
	try{
		cout<<"=====================Demo No.1 Max Value Problem:"<<endl;
		Demo1();
		cout<<"=====================Demo No.2 BackPack Problem:"<<endl;
		Demo2();
		cout<<"=====================Demo No.3 TSP Problem:"<<endl;
		Demo3();
	}catch(exception& e){
		cout<<e.what()<<endl;
	}
#if defined _WIN32
	system("pause");
#endif
	return 0;
}

