
#ifndef HYBRID_H_
#define HYBRID_H_

#include "GA/GA.h"
#include "ANN/ANN.h"
#include "Utility/Util.h"

using namespace FASTAI::GA;
using namespace FASTAI::ANN;

namespace FASTAI{
	namespace HYBRID{
		typedef enum{
			GA_BP,
			GA_RBF,
			BP_GA
		} HYBRID_STATEGY;
	};
}

#endif