/*
 * GA.h
 *
 *  Created on: Apr 15, 2013
 *      Author: woodlgz
 */

#ifndef GA_H_
#define GA_H_

#include <stdlib.h>
#include <string.h>
#include "AIException.h"

using namespace std;

namespace FASTAI{
	namespace GA{

		class GeneticPhase;
		class Possibility;
		class Env;

		/**
		 * genetic information
		 * the implementation of its subclass differs from problem to problem
		 */
		class GeneticPhase{
		public:
			GeneticPhase(){
				m_Coding = NULL;
				m_Len = 0;
			}
			virtual ~GeneticPhase(){
				if(m_Coding!=NULL)
					delete[] m_Coding;
			}
			inline void setCoding(char* coding,int len){
				m_Coding = coding;
				m_Len = len;
			}
		protected:
			virtual void crossing(GeneticPhase* phase) = 0;
			virtual void mutate() = 0;
		public:
			friend class Env;
		private:
			char* m_Coding;
			int m_Len;
		};

		/**
		 * environment in which the population grows
		 */
		class Env{
		public:
			Env(float cRate,float mRate){
				m_CRate = cRate * BASE;
				m_MRate = mRate * BASE;
				m_Population = NULL;
				m_Score = NULL;
				m_PSize = 0;
			}

			virtual ~Env(){
				if(m_Score!=NULL)
					delete[] m_Score;
				if(m_ScoreAux!=NULL)
					delete[] m_ScoreAux;
			}



			/**
			 * evaluate all element in population.
			 * generate score that defines the fitness of this element.
			 * the higher the score is ,the more fitness it takes on.
			 * score values between [0.0,1.0]
			 * */
			virtual void evaluate();

			/**
			 * replace the worst one with a better one
			 */
			virtual bool reproduction();

			/**
			 * take two elements in according to CRate for genetic information exchange
			 */
			virtual bool exchage();

			/**
			 * take one element in according to MRate for genetic mutation
			 */
			virtual bool mutate();

			/**
			 * return the index of the element that fit best in population
			 */
			inline int bestFit(){
				return m_Max;
			}

			/**
			 * return the index of the element that has the least fitness
			 */
			inline int leastFit(){
				return m_Min;
			}

			/**
			 * set the cross rate for genetic information exchange
			 */
			inline void setCRate(float cRate){
				m_CRate = int(cRate * BASE);
			}

			/**
			 * set the mutation rate for genetic mutation
			 */
			inline void setMRate(float mRate){
				m_MRate = int(mRate * BASE);
			}

			inline int getCRate(){
				return m_CRate;
			}

			inline int getMRate(){
				return m_MRate;
			}

			/**
			 * set the initial population
			 */
			inline void setPopulation(GeneticPhase* population,int size){
				if(size<=0)return;
				m_Population = population;
				m_PSize = size;
				m_Score = new float[size];
				m_ScoreAux = new float[size];
				memset(m_Score,0,size);
			}

			/**
			 * pick one element from the population
			 */
			inline GeneticPhase* getElement(int i){
				if(m_PSize>0 && i<m_PSize){
					return m_Population[i];
				}
				return NULL;
			}
		protected:
			/**
			 * calculate the score for element
			 * @param: index for the element in population
			 */
			virtual float judge(int i) = 0;

		public:
			const static int BASE = 10000;
		protected:
			int m_CRate;
			int m_MRate;
			int m_PSize;
			int m_Max;
			int m_Min;
			float* m_Score;
			float* m_ScoreAux;						//auxilary array
			GeneticPhase* m_Population;
		};

		/**
		 * using GA to solve the problem defined by environment
		 */
		GeneticPhase* Solve(Env* env);

	}
};

#endif /* GA_H_ */
