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
#include <time.h>
#include "AIException.h"
using namespace std;

namespace FASTAI{
	namespace GA{

		const int POPULATION_DEFAULT_SIZE = 100;

		class GeneticPhase;
		class Env;


		/**
		 * genetic information
		 * the implementation of its subclass differs from problem to problem
		 */
		class GeneticPhase{
		public:
			GeneticPhase(int len):m_Len(len),m_Coding(NULL),m_Answer(-1){
			}
			virtual ~GeneticPhase(){
			}

			/**
			 * set the coding
			 */
			inline virtual void setCoding(void* coding,int len){
				m_Coding = (int*)coding;
				m_Len = len;
			}
			/**
			 * get len of coding
			 */
			inline int getLen(){
				return m_Len;
			}
			/*
			 * get code at position i in coding.
			 * in concern with different type of coding when overwrite m_Coding,return a pointer
			 */
			inline virtual void* getCodeAt(int i){
				return &m_Coding[i];
			}

			/**
			 * genetic phase copy
			 */
			virtual GeneticPhase& operator = (GeneticPhase& phase){
					memcpy((void*)m_Coding,(void*)(phase.m_Coding),m_Len*sizeof(int));
					m_Answer = phase.getAnswer();
					return *this;
			}

			/**
			 *	genetic phase copy
			 */
			virtual GeneticPhase& copy(GeneticPhase& phase){
					return this->operator=(phase);
			}
			/**
			 * translate the code into human-readable message
			 * returns a object if detailed information is needed.
			 * the returned object should be a member variable
			 */
			virtual void* read() = 0;

			/**
			 * calculate the code and normalize it as a number no less than zero
			 */
			virtual int calcValueOfCode() = 0;

			/**
			 * get answer denoted by a number,usually this will be the max value or the min value
			 */
			int getAnswer(){
				if(m_Answer < 0)
					m_Answer = calcValueOfCode();
				return m_Answer;
			}

			void resetAnswer(){
				m_Answer = -1;
			}
		protected:
			/**
			 * initialize the genetic code.
			 * it is advised that the initial seq should be a possible surrivor (scores higher than 0.0)
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

			/**
			 * reconstruct the phase so that it is a possible surrivor,which means it should scores higher than 0.0
			 */
			virtual void reConstruct() = 0;
		public:
			friend class Env;
		protected:
			int m_Len;
			int* m_Coding;
			int m_Answer;
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
				if(Factory == NULL)
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
			Env(float cRate = DEFAULT_CROSSRATE,float mRate = DEFAULT_MUTATERATE, int age = MAX_AGE){
				m_CRate = cRate * BASE;
				m_MRate = mRate * BASE;
				m_Factory = NULL;
				m_Population = NULL;
				m_Score = NULL;
				m_ScoreAux = NULL;
				m_PSize = 0;
				m_ScoreAvg = 0.0;
				m_ScoreMax = 0.0;
				m_Age = m_AgeMax = age;
				m_HistoryBest = NULL;
			}
			Env(GFactory* factory,float cRate = DEFAULT_CROSSRATE, float mRate = DEFAULT_MUTATERATE,
					int age = MAX_AGE){
				m_CRate = cRate * BASE;
				m_MRate = mRate * BASE;
				m_Factory = factory;
				m_Population = NULL;
				m_Score = NULL;
				m_ScoreAux = NULL;
				m_PSize = 0;
				m_ScoreAvg = 0.0;
				m_ScoreMax = 0.0;
				m_Age = m_AgeMax = age;
				m_HistoryBest = NULL;
			}

			virtual ~Env(){
				for(int i=0;i<m_PSize;i++){
					if(m_Population[i]!=NULL)
						delete m_Population[i];
				}
				if(m_Population!=NULL)
					delete[] m_Population;
				if(m_Score!=NULL)
					delete[] m_Score;
				if(m_ScoreAux!=NULL)
					delete[] m_ScoreAux;
				if(m_HistoryBest!=NULL)
					delete m_HistoryBest;
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
			 * return the the element that fit best in population in current age
			 */
			inline GeneticPhase* bestFit(){
				return m_Population[m_Max];
			}

			/**
			 * return the element that has the least fitness in current age
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
			 * return the age of now
			 */
			inline int getAge(){
				return m_AgeMax - m_Age;
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

			inline bool isEndOfWorld(){
				return m_Age-- <= 0;
			}
			/**
			 * initialize the population
			 */
			void initPopulation(int size);

			/**
			 * pick one element from the population
			 */
			inline GeneticPhase* getElement(int i){
				if(m_PSize>0 && i<m_PSize){
					return m_Population[i];
				}
				return NULL;
			}

			inline GeneticPhase* getHistoryBest(){
				return m_HistoryBest;
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
		private:

			/**
			 *	conduct reproduction in subset of population
			 *  @param begin : begin of index of subset
			 *  @param end	 : end of index of subset
			 *  @return value: the index of selected element in reproduction
			 */
			int reproduction(int begin, int end);

		public:
			const static int BASE = 10000;
			const static int MAX_AGE = 10000;
			const static float DEFAULT_CROSSRATE = 0.02;
			const static float DEFAULT_MUTATERATE = 0.01;
		protected:
			int m_CRate;
			int m_MRate;
			int m_PSize;
			int m_Max;
			int m_Min;
			int m_AgeMax;
			int m_Age;
			float* m_Score;
			float* m_ScoreAux;						//auxilary array
			float  m_ScoreAvg;						//average score , may be needed when judging an element
			float  m_ScoreMax;						//max score , may be needed when judging an element
			GeneticPhase** m_Population;
			GeneticPhase* m_HistoryBest;
			GFactory* m_Factory;
		};

		/**
		 * using GA to solve the problem defined by environment
		 * @param env : the specified environment
		 * @param pSize : the population size
		 * @param max_time : the max evolution time.
		 */
		GeneticPhase* Solve(Env* env, int pSize = POPULATION_DEFAULT_SIZE);


	}
};

#endif /* GA_H_ */
