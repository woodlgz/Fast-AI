/*
 * GA.cpp
 *
 *  Created on: Apr 15, 2013
 *      Author: woodlgz
 */

#include "GA.h"
#include <stdlib.h>

namespace FASTAI{
	namespace GA{

		int Env::bestFit(){
			int max = 0;
			for(int i=0;i<this->m_PSize;i++){
				if(m_Score[max]<m_Score[i])
					max = i;
			}
			return max;
		}

		void Env::reproduction(){

		}

		void Env::exchage(){

		}

		void Env::mutate(){

		}

		GeneticPhase* Solve(Env* env){
			return NULL;
		}
	};
};
