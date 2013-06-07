
#ifndef DATASET_H_
#define DATASET_H_

#include <vector>

#include <boost/archive/tmpdir.hpp>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <boost/serialization/vector.hpp>


using namespace std;

namespace FASTAI{
	namespace ANN{
		class DataSet{
			friend class boost::serialization::access;
			template<class Archive>
			void serialize(Archive& ar,const unsigned int version){
				ar & datasource & expectedData;
			}
		public:
			DataSet(){};
			DataSet(const char* path);
			virtual ~DataSet(){};
			virtual void loadData(const char* path);
			virtual void storeData(const char* path);
			inline vector<double>& operator[](int i){
				return datasource[i];
			}
			inline vector<double>& getExpected(int i){
				return expectedData[i];
			}
			inline int size(){
				return datasource.size();
			}
			inline void resize(int size){
				datasource.resize(size);
			}
			inline void append(vector<double>& v){
				datasource.push_back(v);
			}
			inline void append(double v[],int size){
				vector<double> vec(size);
				for(int i=0;i<size;i++)
					vec[i] = v[i];
				datasource.push_back(vec);
			}
			inline void appendExpected(vector<double>& v){
				expectedData.push_back(v);
			}
			inline void appendExpected(double v[],int size){
				vector<double> vec(size);
				for(int i=0;i<size;i++)
					vec[i] = v[i];
				expectedData.push_back(vec);
			}
		private:
			vector<vector<double>> datasource;
			vector<vector<double>> expectedData;
		};
	};
};

#endif //endof DATASET_H_