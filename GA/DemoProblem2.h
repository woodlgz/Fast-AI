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
 * problem description:
 *	背包限定容纳重量为PackSize，限定物品数量为PackItemCnt.背包客在探险过程中找到LEN种物品，每种物品标记为i,0<=i<LEN.
 *	假设:
 *	1.每种物品的价值记为Value[i]，重量为Weight[i]，每种物品的数量记为ItemCnt[i];
 *	2.0<Value[i]<=10,0<Weight[i]<=10,0<ItemCnt<=100
 *	那么背包客所能收集到物品的最大价值为多少？
  *  设X[i]为第i种物品收入背包的数量,问题转化为约束问题的求解：
 *	  约束条件：
 *		（1）sum(X[i])<=PackItemCnt
 *		（2）sum(X[i]*Weight[i])<=PackSize
 *		（3）0<=X[i]<=ItemCnt[i]
 *	  求max = sum(X[i]*Value[i]);
 */


#include <stdio.h>
#include "GA.h"

using namespace std;
using namespace FASTAI::GA;

typedef unsigned int uint;

class Demo2GeneticPhase : public GeneticPhase{
public:
	Demo2GeneticPhase():GeneticPhase(Len){
		init();
	}
	 ~Demo2GeneticPhase(){
		cleanup();
	}

	void* read(){
		cout<<"max value:"<<m_Answer<<endl;
		for(int i=1;i<=m_Len;i++){
			cout<<"take "<<m_Coding[i]<<" items of type "<<i<<",weights:"<<Weight[i]*m_Coding[i]<<",values:"<<Value[i]*m_Coding[i]<<endl;
		}
		return &m_Answer;
	}

	int calcValueOfCode(){
		m_Answer = 0;
		for(int i=1;i<=m_Len;i++){
			m_Answer+= m_Coding[i]*Value[i];
		}
		return m_Answer;
	}

	inline void* getCodeAt(int i){
				return &m_Coding[i];
	}
	GeneticPhase& operator = (GeneticPhase& phase){
			Demo2GeneticPhase* _phase = static_cast<Demo2GeneticPhase*>(&phase);
			memcpy((void*)m_Coding,(void*)(_phase->m_Coding),(m_Len+1)*sizeof(int));
			this->m_Answer = _phase->calcValueOfCode();
			return *this;
	}
private:
	void init(){
		m_Coding = new int[m_Len+1];
		m_RandomAux = new int[m_Len+1];
		int mPackSize = Demo2GeneticPhase::PackSize;
		int mPackItemCnt = Demo2GeneticPhase::PackItemCnt;
		for(int i=1;i<=m_Len;i++)
			m_RandomAux[i] = i;
		for(int i=m_Len;i>=1;i--){
			srand(rand()%time(NULL));
			swap(m_RandomAux[(rand()%i)+1],m_RandomAux[i]);
		}
		for(int i=1;i<=m_Len;i++){
			int tmp = min(min(ItemCnt[m_RandomAux[i]],mPackSize / Weight[m_RandomAux[i]]),(uint)mPackItemCnt);
			srand(rand()%time(NULL));
			m_Coding[m_RandomAux[i]] = rand()%(tmp+1);
			mPackItemCnt -= m_Coding[m_RandomAux[i]];
			mPackSize -= m_Coding[m_RandomAux[i]]*Weight[m_RandomAux[i]];
		}
	}
	void cleanup(){
		if(m_Coding){
			delete[] m_Coding;
			m_Coding = NULL;
		}
		if(m_RandomAux){
			delete[] m_RandomAux;
			m_RandomAux = NULL;
		}
	}
	void crossing(GeneticPhase* phase){
		Demo2GeneticPhase* _phase = static_cast<Demo2GeneticPhase*>(phase);
		time_t t;
		t = time(NULL);
		srand(rand()%t);
		int rInt = (rand()%m_Len)+1;
		swap(m_Coding[rInt],_phase->m_Coding[rInt]);
	}

	void mutate(){
		time_t t;
		t = time(NULL);
		srand(rand()%t);
		int rInt = (rand()%m_Len)+1;
		t =  time(NULL);
		srand(rand()%t);
		m_Coding[rInt] = rand()%(ItemCnt[rInt]+1); //this may lead to frequent reproduction
	}

	void reConstruct(){
		int mPackSize = Demo2GeneticPhase::PackSize;
		int mPackItemCnt = Demo2GeneticPhase::PackItemCnt;
		for(int i=1;i<=m_Len;i++)
			m_RandomAux[i] = i;
		for(int i=m_Len;i>=1;i--){
				srand(rand()%time(NULL));
				swap(m_RandomAux[(rand()%i)+1],m_RandomAux[i]);
		}
		for(int i=1;i<=m_Len;i++){
			int tmp = min(min(ItemCnt[m_RandomAux[i]],mPackSize / Weight[m_RandomAux[i]]),(uint)mPackItemCnt);
			srand(rand()%time(NULL));
			m_Coding[m_RandomAux[i]] = rand()%(tmp+1);
			mPackItemCnt -= m_Coding[m_RandomAux[i]];
			mPackSize -= m_Coding[m_RandomAux[i]]*Weight[m_RandomAux[i]];
		}
	}

public :
	static void Problem_Init(){
		scanf("%u %u %u",&PackSize,&PackItemCnt,&Len);
		assert(Len<=MAX_LEN && PackSize <= MAX_PACKSIZE && PackItemCnt <= MAX_ITEMCNT);
		for(int i=1;i<=Len;i++){
			scanf("%u %u %u",&ItemCnt[i],&Value[i],&Weight[i]);
		}
	}
	static void Problem_Init(uint _PackSize, uint _PackItemCnt, uint _Len,
				uint _ItemCnt[], uint _Value[], uint _Weight[]){
		assert(_Len<=MAX_LEN && _PackSize <= MAX_PACKSIZE && _PackItemCnt <= MAX_ITEMCNT);
		Len = _Len;
		PackSize = _PackSize;
		PackItemCnt = _PackItemCnt;
		for(int i=1;i<=Len;i++){
			ItemCnt[i] = _ItemCnt[i];
			Value[i] = _Value[i];
			Weight[i] = _Weight[i];
		}
	}
	static uint DPSolution(){
		for(int i=0;i<=Len;i++){
			for(int j=0;j<=PackSize;j++){
				for(int k=0;k<=MAX_ITEMCNT;k++){
					dp[i][j][k] = 0;
					Operation[i][j][k] = 0;
				}
			}
		}
		for(int i=1;i<=Len;i++){
			int w = 0 , v = 0;
			for(int m = 0;m<=ItemCnt[i];m++){
				for(int j=PackSize-w;j>=0;j--){
					for(int k = PackItemCnt-m;k>=0;k--){
						uint tmp = dp[i-1][j+w][k+m]+v;
						if(tmp>=dp[i][j][k]){
							Operation[i][j][k] = m;
						}
						dp[i][j][k] = max(tmp,dp[i][j][k]);
					}
				}
				w+=Weight[i];
				v+=Value[i];
			}
		}
		return dp[Len][0][0];
	}

	static inline void readDPSolution(uint Op[]){
		int m = 0;
		for(int i=Len,j=0,k=0;i>=1;i--){
			Op[i] = Operation[i][j][k];
			m = Op[i];
			j+=m*Weight[i];
			k+=m;
		}
	}
public :
	friend class Demo2Env;
	static const int MAX_LEN  = 100;
	static const int MAX_PACKSIZE = 100;
	static const int MAX_ITEMCNT = 100;
private :
	static uint dp[MAX_LEN+1][MAX_PACKSIZE+1][MAX_ITEMCNT+1];
	static uint Operation[MAX_LEN+1][MAX_PACKSIZE+1][MAX_ITEMCNT+1];
	static uint Value[MAX_LEN+1];
	static uint Weight[MAX_LEN+1];
	static uint ItemCnt[MAX_LEN+1];
	static int Len;
	static int PackSize;
	static int PackItemCnt;
private:
	int* m_RandomAux;
};

uint Demo2GeneticPhase::dp[MAX_LEN+1][MAX_PACKSIZE+1][MAX_ITEMCNT+1];
uint Demo2GeneticPhase::Operation[MAX_LEN+1][MAX_PACKSIZE+1][MAX_ITEMCNT+1];
uint Demo2GeneticPhase::Value[MAX_LEN+1];
uint Demo2GeneticPhase::Weight[MAX_LEN+1];
uint Demo2GeneticPhase::ItemCnt[MAX_LEN+1];
int Demo2GeneticPhase::Len;
int Demo2GeneticPhase::PackSize;
int Demo2GeneticPhase::PackItemCnt;

class Demo2Env : public Env{
public:
	Demo2Env(GFactory* factory,float cRate, float mRate, int age):
		Env(factory,cRate,mRate,age){
	}
private:
	void judge(){
		m_ScoreMax = 0.0;
		for(int i=0;i<m_PSize;i++){
			int len  = m_Population[i]->getLen();
			int mPackSize = Demo2GeneticPhase::PackSize;
			int mPackItemCnt = Demo2GeneticPhase::PackItemCnt;
			m_Score[i] = 0.0;
			for(int j=1;j<=len&&mPackSize>=0&&mPackItemCnt>=0;j++){
				int x = *(int*)(m_Population[i]->getCodeAt(j));
				m_Score[i] += x * Demo2GeneticPhase::Value[j];
				mPackSize -= x * Demo2GeneticPhase::Weight[j];
				mPackItemCnt -= x;
			}
			if(mPackSize<0 || mPackItemCnt<0){
				m_Score[i] = 0.0;
				continue;
			}
			if(m_ScoreMax<m_Score[i])
				m_ScoreMax = m_Score[i];
		}
	}

	// make sure the output score is between (0.0,1.0]
	float judge(int i){
		if(m_ScoreMax<1e-38&&m_ScoreMax>-1e-38)
			return 0.0;
		return m_Score[i] / m_ScoreMax;
	}
};

#endif /* DEMOPROBLEM2_H_ */
