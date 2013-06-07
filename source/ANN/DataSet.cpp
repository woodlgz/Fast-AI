
#include "DataSet.h"
#include <fstream>
using namespace FASTAI::ANN;


void DataSet::loadData(const char* path){
	ifstream ifs(path);
	if(!ifs.good())
		throw exception("can't open path");
	boost::archive::text_iarchive ia(ifs);
	ia & (*this);
}

void DataSet::storeData(const char* path){
	ofstream ofs(path);
	if(!ofs.good())
		throw exception("can't open path");
	boost::archive::text_oarchive oa(ofs);
	oa & (*this);
}