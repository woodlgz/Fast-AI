/*
 * util.h
 *
 *  Created on: Apr 17, 2013
 *      Author: woodlgz
 */

#ifndef UTIL_H_
#define UTIL_H_

using namespace std;

#include <time.h>
#include <stdlib.h>
#if defined _WIN32
#pragma comment(lib,"crypt32.lib")
#include <windows.h>
#include <wincrypt.h>
#endif

namespace FASTAI{
	namespace Util{
		namespace Common{

			class RandomFactory{
			private:
				RandomFactory(){
					initFactory();
				}
				~RandomFactory(){
					cleanUp();
				}
			public:
				bool initFactory();
				void cleanUp();
				unsigned int getRandom();
			public:
				static RandomFactory* getFactory(){
					return &SELF;
				}
			private:
				static RandomFactory SELF;
			private:
#if defined _WIN32
				HCRYPTPROV  m_hCryptProv;
#endif
				unsigned int  m_Int;
			};


		}
	};
};

#endif /* UTIL_H_ */
