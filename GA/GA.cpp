/*
 * GA.cpp
 *
 *  Created on: Apr 15, 2013
 *      Author: woodlgz
 */

#include "GA.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>

using namespace std;
	

namespace FASTAI{
	namespace GA{

		void Env::evaluate(){
			float accumScore = 0.0;
			m_Max = 0;
			m_Min = 0;
			judge();
			for(int i=0;i<m_PSize;i++){
				m_Score[i] = judge(i);
				if(m_Score[i]<=1e-38&&m_Score[i]>=-1e-38){ // replace it,so that population is all possible surrivor
					if(i<=0){ // i == 0
						m_Population[i]->reConstruct(); // make sure the first element is a possible surrivor
						m_Population[i]->resetAnswer();
						judge();
						m_Score[i] = judge(i);
					}else {
						int s = reproduction(0,i-1);
						*m_Population[i] = *m_Population[s];
						m_Score[i] = m_Score[s];
					}
				}
				m_ScoreAux[i] = m_Score[i] + accumScore;
				accumScore = m_ScoreAux[i];
				if(m_Score[m_Max]<m_Score[i])
					m_Max = i;
				if(m_Score[m_Min]>m_Score[i])
					m_Min = i;
			}
			if(m_PSize>0){
				m_ScoreAvg = m_ScoreAux[m_PSize-1] / m_PSize;
			}
			if(!m_HistoryBest->isBetterThan(m_Population[m_Max]))
				*m_HistoryBest = *m_Population[m_Max];		// copy the best one
		}

		//note : if accumScore == 0.0 or m_Score has elements that equal to 0.0,the result can be unreliable
		bool Env::reproduction(){
			if(m_Age*1.0/m_AgeMax<=0.2){	// try to trigger a leap in the late period
				m_Population[m_Min]->reConstruct();
				m_Population[m_Min]->resetAnswer();
				return true;
			}
			float accumScore = m_ScoreAux[m_PSize-1];
			unsigned int iScore = (unsigned int)(accumScore * Env::BASE);
			float score = (GENERATE_RANDOM()%(iScore+1))*1.0 / Env::BASE;
			int lb = 0, ub = m_PSize;
			while(lb<=ub){
				int mid = lb+((ub-lb)>>1);
				if(m_ScoreAux[mid]<score)
					lb = mid+1;
				else ub = mid-1;
			}
			if(lb == -1 || lb == m_Min)
				return false;
			*(m_Population[m_Min]) = *(m_Population[lb]);	// genetic copy
			m_Population[m_Min]->mutate();				// try to make some change,so that it may be better than ever
			m_Population[m_Min]->resetAnswer();

			// m_Min and m_ScoreAux has been dirty;m_Max remained clean
			return true;
		}

		//note : if accumScore == 0.0 or m_Score has elements that equal to 0.0,the result can be unreliable
		int Env::reproduction(int begin, int end){
			if(end<begin)
				return -1;
			float accumScore = m_ScoreAux[end];
			unsigned int iScore = (unsigned int)(accumScore * Env::BASE);
			float score = (GENERATE_RANDOM()%(iScore+1))*1.0 / Env::BASE;
			int lb = begin, ub = end;
			while(lb<=ub){
				int mid = lb+((ub-lb)>>1);
				if(m_ScoreAux[mid]<score)
					lb = mid+1;
				else ub = mid-1;
			}
			return lb;
		}

		bool Env::exchage(){
			int rInt = (GENERATE_RANDOM()%(Env::BASE))+1;
			if(rInt<=m_CRate){	//bingo
				rInt = GENERATE_RANDOM()%(m_PSize);
				swap(m_Population[rInt],m_Population[m_PSize-1]);
				rInt = GENERATE_RANDOM()%(m_PSize-1);
				m_Population[rInt]->crossing(m_Population[m_PSize-1]);
				m_Population[rInt]->resetAnswer();
				//m_Max may change
				return true;
			}
			return false;
		}

		bool Env::mutate(){
			int rInt = (GENERATE_RANDOM()%(Env::BASE))+1;
			if(rInt<=m_MRate){	//bingo
				rInt = GENERATE_RANDOM()%(m_PSize);
				m_Population[rInt]->mutate();
				m_Population[rInt]->resetAnswer();
				return true;
			}
			return false;
		}

		void Env::initPopulation(int size){
			if(size<=0 || m_Factory == NULL)return;
			m_Population = new GeneticPhase*[size];
			for(int i =0;i<size;i++){
				m_Population[i] = m_Factory->newInstance();
			}
			m_PSize = size;
			m_Score = new float[size];
			m_ScoreAux = new float[size];
			memset(m_Score,0,size);
			memset(m_ScoreAux,0,size);
			m_HistoryBest = m_Factory->newInstance();
		}

		GeneticPhase* Solve(Env* env, int pSize){
			int answer = 0;
			if(!env)
				return NULL;
			env->initPopulation(pSize);
			while(!env->isEndOfWorld()){
				env->evaluate();
				int tmp = env->bestFit()->getAnswer();
				if(answer!=tmp){
					answer = tmp;
					cout<<"answer at age "<<env->getAge()<<" is "<<answer<<endl;
				}
				env->reproduction();
				env->exchage();
				env->mutate();
			}
			env->evaluate();	//upate score
			return env->getHistoryBest();
		}
	};
};
