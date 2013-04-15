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

		void solve(Population* po,Env* env){

		}

		class Population{
		public:
			Population(){
			}
			Population(int size,GeneticPhase* phases){
				for(int i=0;i<size;i++)
					m_phases.insert(&phases[i]);
			}
			GeneticPhase* get(int i){
				return m_phases[i];
			}
			void set(int i,GeneticPhase* phase){
				m_phases[i]= phase;
			}
			void add(GeneticPhase* phase){
				m_phases.insert(phase);
			}
		private:
			vector<GeneticPhase*> m_phases;
		};

		class GeneticPhase{
		public:
			GeneticPhase(){
			}
			virtual ~GenticPhase(){
			}
			void setEnvironMent(Env* env){
				m_env = env;
			}
		protected:
			virtual void crossing(GeneticPhase* phase) = 0;
			virtual void mutate() = 0;
		private:
			Env* m_env;
			friend class Env;
		};

		class Possibility{

		};

		class Env{
		public:
			Env(float cRate,float mRate){
				m_cRate = cRate * BASE;
				m_mRate = mRate * BASE;
			}
			virtual bool isFit(GeneticPhase* phase) = 0;

			void setCRate(float cRate){
				m_cRate = int(cRate * BASE);
			}
			void setMRate(float mRate){
				m_mRate = int(mRate * BASE);
			}
			int getCRate(){
				return m_cRate;
			}
			int getMRate(){
				return m_mRate;
			}
		public:
			const static int BASE = 10000;
		protected:
			int m_cRate;
			int m_mRate;
		};

	}
};

#endif /* GA_H_ */
