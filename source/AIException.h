/*
 * AIException.h
 *
 *  Created on: Apr 15, 2013
 *      Author: woodlgz
 */

#ifndef AIEXCEPTION_H_
#define AIEXCEPTION_H_

namespace FASTAI{

class AIException{
public:
	static bool assertFailed(const char* message){
		throw AIException(message);
		return false;
	}
public:
	AIException():message("Unkown Exception Caused by Unkown Exception"){
	}
	AIException(const char* str):message(str){
	}
	const char* getMessage(){
		return message;
	}
private:
	const char* message;
};


};

#endif /* AIEXCEPTION_H_ */
