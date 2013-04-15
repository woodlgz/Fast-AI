/*
 * GA.h
 *
 *  Created on: Apr 15, 2013
 *      Author: woodlgz
 */

#ifndef GA_H_
#define GA_H_

#include <vector>
#include "AIException.h"

using namespace std;

namespace FASTAI{
	namespace GA{

		class Population;
		class GeneticPhase;
		class Possibility;
		class Env;

		class GeneticPhase{
		public:
			GeneticPhase(){
			}
			virtual ~GeneticPhase(){
			}
		protected:
			virtual void crossing(GeneticPhase* phase) = 0;
			virtual void mutate() = 0;
		public:
			friend class Env;
		};

		class Population{
		public:
			Population(){
			}

			Population(int size,GeneticPhase* phases){
				for(int i=0;i<size;i++)
					m_phases.push_back(&phases[i]);
				m_maxSize = size<<1;
			}

			inline GeneticPhase* get(int i){
				return m_phases[i];
			}

			inline void set(int i,GeneticPhase* phase){
				m_phases[i]= phase;
			}

			inline void add(GeneticPhase* phase){
				if(m_phases.size()<m_maxSize)
					m_phases.push_back(phase);
			}

			int bestFit();

		private:
			int m_maxSize;
			vector<GeneticPhase*> m_phases;

		};

		class Possibility{

		};

		class Env{
		public:
			Env(float cRate,float mRate){
				m_cRate = cRate * BASE;
				m_mRate = mRate * BASE;
			}
			virtual int evaluate(GeneticPhase* phase) = 0;

			inline void setCRate(float cRate){
				m_cRate = int(cRate * BASE);
			}
			inline void setMRate(float mRate){
				m_mRate = int(mRate * BASE);
			}
			inline int getCRate(){
				return m_cRate;
			}
			inline int getMRate(){
				return m_mRate;
			}
			inline void setPopulation(Population* po){
				m_Po = po;
			}

		public:
			const static int BASE = 10000;
		protected:
			int m_cRate;
			int m_mRate;
			Population* m_Po;
		};

		void Solve(Population* po,Env* env);

	}
};

#endif /* GA_H_ */
