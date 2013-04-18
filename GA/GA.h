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
#include <algorithm>
#include "AIException.h"

using namespace std;

namespace FASTAI{
	namespace GA{

		const int MAX_AGE = 100 ;
		const int POPULATION_DEFAULT_SIZE = 100;

		class GeneticPhase;
		class Env;


		/**
		 * genetic information
		 * the implementation of its subclass differs from problem to problem
		 */
		class GeneticPhase{
		public:
			GeneticPhase(int len):m_Len(len),m_Coding(NULL){
			}
			virtual ~GeneticPhase(){
			}

			/**
			 * set the coding
			 */
			inline void setCoding(unsigned char* coding,int len){
				m_Coding = coding;
				m_Len = len;
			}
			/**
			 * get len of coding
			 */
			inline int getLen(){
				return m_Len;
			}
			/*
			 * get code at position i in coding
			 */
			inline unsigned char getCodeAt(int i){
				return m_Coding[i];
			}

			/**
			 * genetic phase copy
			 */
			virtual GeneticPhase& operator = (GeneticPhase& phase){
					memcpy((void*)m_Coding,(void*)(phase.m_Coding),m_Len);
					return *this;
			}

			/**
			 *	genetic phase copy
			 */
			virtual GeneticPhase& copy(GeneticPhase& phase){
					return this->operator=(phase);
			}
			/**
			 * decode the coding,returns a more readable object
			 */
			virtual void* read() = 0;

		protected:
			/**
			 * initialize the genetic code
			 */
			virtual void init() = 0;

			/**
			 * clean up
			 */
			virtual void cleanup() = 0;

			/**
			 * handle cross exchange between genetic phases
			 */
			virtual void crossing(GeneticPhase* phase) = 0;

			/**
			 * handle the genetic mutation
			 */
			virtual void mutate() = 0;
		public:
			friend class Env;
		protected:
			int m_Len;
			unsigned char* m_Coding;
		};

		/**
		 * GeneticFactory that generate new element of population
		 * subClass should implement newInstance method,and call GeneticPhase::init
		 * subClass should be Singleton
		 */
		class _GeneticFactory{
		protected:
			_GeneticFactory(){
			}
		public:
			virtual GeneticPhase* newInstance() = 0;
		};

		typedef _GeneticFactory GFactory;

		template<class F>
		class GeneticFactory : public GFactory{
		private:
			GeneticFactory(){
				Factory = this;
			}
		public:
			static GFactory* getFactory(){
				if(Factory != NULL)
					Factory = new GeneticFactory<F>();
				return Factory;
			}
			inline GeneticPhase* newInstance(){
				return new F();
			}
		public:
			static GFactory* Factory;
		};

		template<class F>
		GFactory* GeneticFactory<F>::Factory = NULL;

		/**
		 * environment in which the population grows
		 */
		class Env{
		public:
			Env(float cRate,float mRate){
				m_CRate = cRate * BASE;
				m_MRate = mRate * BASE;
				m_Factory = NULL;
				m_Population = NULL;
				m_Score = NULL;
				m_PSize = 0;
			}
			Env(float cRate,float mRate,GFactory* factory){
				m_CRate = cRate * BASE;
				m_MRate = mRate * BASE;
				m_Factory = factory;
				m_Population = NULL;
				m_Score = NULL;
				m_PSize = 0;
			}

			virtual ~Env(){
				for(int i=0;i<m_PSize;i++){
					if(m_Population[i]!=NULL)
						delete m_Population[i];
				}
				delete[] m_Population;
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
			 * return the the element that fit best in population
			 */
			inline GeneticPhase* bestFit(){
				return m_Population[m_Max];
			}

			/**
			 * return the element that has the least fitness
			 */
			inline GeneticPhase* leastFit(){
				//recalculate the minimun,fot it may be dirty
				for(int i=0;i<m_PSize;i++){
					if(m_Score[m_Min]>m_Score[i])
						m_Min = i;
				}
				return m_Population[m_Min];
			}

			/**
			 * set factory that generate new element in the population
			 */
			inline void setGeneticFactory(GFactory* factory){
				m_Factory = factory;
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

			inline void setPopulationSize(int size){
				m_PSize = size;
			}

			/**
			 * initialize the population
			 */
			inline void initPopulation(int size){
				if(size<=0 || m_Factory == NULL)return;
				m_Population = new GeneticPhase*[size];
				for(int i =0;i<size;i++){
					m_Population[i] = m_Factory->newInstance();
				}
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
			/*
			 * evaluate the whole population.
			 * it can provide some information for judge(i).
			 * called before judge(i) in evaluate().
			 * overwrite this if needed.
			 */
			virtual void judge() {

			}

			/**
			 * calculate the score for element,called in evaluate()
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
			GeneticPhase** m_Population;
			GFactory* m_Factory;
		};

		/**
		 * using GA to solve the problem defined by environment
		 * @param env : the specified environment
		 * @param pSize : the population size
		 * @param max_time : the max evolution time.
		 */
		GeneticPhase* Solve(Env* env, int pSize = POPULATION_DEFAULT_SIZE, int max_time = MAX_AGE);


	}
};

#endif /* GA_H_ */
