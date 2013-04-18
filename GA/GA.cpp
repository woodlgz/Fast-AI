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
				m_ScoreAux[i] = m_Score[i] + accumScore;
				accumScore = m_ScoreAux[i];
				if(m_Score[m_Max]<m_Score[i])
					m_Max = i;
				if(m_Score[m_Min]>m_Score[i])
					m_Min = i;
			}
		}

		bool Env::reproduction(){
			float accumScore = m_ScoreAux[m_PSize-1];
			int iScore = int(accumScore * Env::BASE);
			time_t t = time(NULL);
			srand(t);
			float score = (rand()%(iScore+1))*1.0 / Env::BASE;
			int lb = 0, ub = m_PSize;
			while(lb<=ub){
				int mid = lb+((ub-lb)>>1);
				if(m_ScoreAux[mid]<=score)
					lb = mid+1;
				else ub = mid-1;
			}
			if(ub == -1 || ub == m_Min)
				return false;
			*(m_Population[m_Min]) = *(m_Population[ub]);	// genetic copy
			// m_Min and m_ScoreAux has been dirty
			return true;
		}

		bool Env::exchage(){
			time_t t = time(NULL);
			srand(t);
			int rInt = rand()%(Env::BASE+1);
			if(rInt<=m_CRate){	//bingo
				t = time(NULL);
				srand(t);
				rInt = rand()%(m_PSize);
				swap(m_Population[rInt],m_Population[m_PSize-1]);
				t = time(NULL);
				srand(t);
				rInt = rand()%(m_PSize-1);
				m_Population[rInt]->crossing(m_Population[m_PSize-1]);
				return true;
			}
			return false;
		}

		bool Env::mutate(){
			time_t t = time(NULL);
			srand(t);
			int rInt = rand()%(Env::BASE+1);
			if(rInt<=m_MRate){	//bingo
				t = time(NULL);
				srand(t);
				rInt = rand()%(m_PSize);
				m_Population[rInt]->mutate();
				return true;
			}
			return false;
		}

		GeneticPhase* Solve(Env* env, int pSize, int max_time){
			if(!env)
				return NULL;
			env->initPopulation(pSize);
			while(max_time>0){
				env->evaluate();
				env->reproduction();
				env->exchage();
				env->mutate();
				max_time--;
			}
			env->evaluate();	//upate score
			return env->bestFit();
		}
	};
};
