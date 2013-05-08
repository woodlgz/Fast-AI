/*
 * Util.cpp
 *
 *  Created on: Apr 17, 2013
 *      Author: woodlgz
 */

#include "Util.h"
#include <assert.h>

namespace FASTAI{
	namespace Util{
		namespace Common{

			
			RandomFactory RandomFactory::SELF;

			bool RandomFactory::initFactory(){
#if defined _WIN32
			 BOOL result=	CryptAcquireContext(
								&m_hCryptProv,
								NULL,
								NULL,
								PROV_RSA_FULL,
								0);
			 assert(result == TRUE);
#endif
				return false;
			}

			void RandomFactory::cleanUp(){
#if defined _WIN32
				CryptReleaseContext(m_hCryptProv,0);
#endif
			}

			unsigned int RandomFactory::getRandom(){
#if defined _WIN32
				if(!CryptGenRandom(m_hCryptProv,sizeof(unsigned int),(BYTE*)&m_Int)){
					srand(rand()%time(NULL));
					m_Int = rand();
				}
#else
				srand(rand()%time(NULL));
				m_Int = rand();
#endif
				return m_Int;
			}

		};
	};
};
