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
			}

			/**
			 * evaluate all element in population.
			 * generate score that defines the fitness of this element.
			 * the higher the score is ,the more fitness it takes on.
			 * score values between [0,100]
			 * */
			virtual void evaluate() = 0;

			/**
			 * return the index of the element that fit best in population
			 */
			virtual int bestFit();

			/**
			 * replace the worst one with a better one
			 */
			virtual void reproduction();

			/**
			 * take two elements in according to CRate for genetic information exchange
			 */
			virtual void exchage();

			/**
			 * take one element in according to MRate for genetic mutation
			 */
			virtual void mutate();

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
				m_Population = pupulation;
				m_PSize = size;
				m_Score = new int[size];
				memset(m_Score,0,size);
			}

			/**
			 * pick one element from the population
			 */
			inline GeneticPhase* getElement(int i){
				if(m_Size>0 && i<m_Size){
					return m_Population[i];
				}
				return NULL;
			}

		public:
			const static int BASE = 10000;
		protected:
			int m_CRate;
			int m_MRate;
			int m_PSize;
			int* m_Score;
			GeneticPhase* m_Population;
		};

		/**
		 * using GA to solve the problem defined by environment
		 */
		GeneticPhase* Solve(Env* env);

	}
};

#endif /* GA_H_ */
