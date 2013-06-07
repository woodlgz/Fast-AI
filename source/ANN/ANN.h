
#ifndef ANN_H_
#define ANN_H_

#include <vector>
#include <utility>
#include <assert.h>
#include "../Utility/Util.h"
#include "../AIException.h"
#include "DataSet.h"


#include <boost/archive/tmpdir.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/assume_abstract.hpp>

using namespace std;
using namespace FASTAI::Util::Common;
using namespace FASTAI::Util::Math;

#define RANDOM_DOUBLE()	(m_Random->getRandomDouble())
#define RANDOM_INT()	(m_Random->getRandom())

namespace FASTAI{
	namespace ANN{
		
		const int DEFAULT_LAYERS = 2;
		const int MAX_LAYERS = 10;
		const int MAX_NODE_PER_LAYER = 20;
		const int MAX_TRAINING_TIMES = 1000000;
		const int TRAINING_SET_INIT_SIZE = 1024;
		const int DEFAULT_RBF_DIM = 10;
		const double DEFAULT_ETA = 0.1;

		typedef double (*ACTIVATE_FUNC)(double x);
		typedef double (*DIFF_ACTIVATE_FUNC)(double x);
		typedef double (*RADIUS_BASE_FUNC)(const vector<double>& paramX,const vector<double>& paramXc , double sigma);

		class NeuralNetwork{
			friend class boost::serialization::access;
			template<class Archive>
			void serialize(Archive& ar,const unsigned int version){
				ar & m_InitSizeOfSet;
				ar & m_NumLayers;
				ar &  m_MaxIterations;
				ar & m_CurrentIteration;
				ar & weight;
				ar & output;
				ar & target;
				ar & m_Nodes;
				ar & m_etas;
				ar & m_TrainingSet;
				ar & m_IsLearningProcessRecorded;
				ar & m_LearningProcess;
			}
		public:
			static double sigmoidFunction(double x);
			static double dSigmoidFunction(double y);
			static double guassRadiusBaseFunction(const vector<double>& param,const vector<double>& paramXc,double sigma);
			static double polyharmonicRadiusBaseFunction(const vector<double>& param,const vector<double>& paramXc,double sigma);
		public:
			NeuralNetwork(){
				m_Random = Util::Common::RandomFactory::getFactory();
				ActivateFunction = sigmoidFunction;
				DActivateFunction = dSigmoidFunction;
				m_InitSizeOfSet = TRAINING_SET_INIT_SIZE;
				m_IsLearningProcessRecorded = false;
			}
			NeuralNetwork(vector<int>& nodes,vector<double>& etas,int layers = DEFAULT_LAYERS,int times = MAX_TRAINING_TIMES):
				m_NumLayers(layers),
				m_Nodes(nodes),
				m_etas(etas),
				m_MaxIterations(times),
				m_CurrentIteration(0)
			{
				assert(layers<=MAX_LAYERS&&
					   layers>=2&&
					   layers==m_Nodes.size()&&
					   layers==m_etas.size()+1&&
					   "invalid parmeter in NeuralNetwork Constructor"
					   );
				m_Random = Util::Common::RandomFactory::getFactory();
				ActivateFunction = sigmoidFunction;
				DActivateFunction = dSigmoidFunction;
				m_InitSizeOfSet = TRAINING_SET_INIT_SIZE;
				m_IsLearningProcessRecorded = false;
				output.resize(m_NumLayers);
				for(int i=0;i<m_NumLayers;i++){
					output[i].resize(m_Nodes[i]+1);
				}
			}
			NeuralNetwork(vector<int>& nodes,double staticEta = DEFAULT_ETA,int layers = DEFAULT_LAYERS,int times = MAX_TRAINING_TIMES):
				m_NumLayers(layers),
				m_Nodes(nodes),
				m_MaxIterations(times),
				m_CurrentIteration(0)
			{
				assert(layers<=MAX_LAYERS&&
					   layers>=2&&
					   layers==m_Nodes.size()&&
					   "invalid parmeter in NeuralNetwork Constructor"
					   );
				m_Random = Util::Common::RandomFactory::getFactory();
				ActivateFunction = sigmoidFunction;
				DActivateFunction = dSigmoidFunction;
				m_InitSizeOfSet = TRAINING_SET_INIT_SIZE;
				m_IsLearningProcessRecorded = false;
				for(int i=1;i<m_NumLayers;i++){
					m_etas.push_back(staticEta);
				}
				output.resize(m_NumLayers);
				for(int i=0;i<m_NumLayers;i++){
					output[i].resize(m_Nodes[i]+1);
				}
			}
			virtual ~NeuralNetwork(){
			}
			
			virtual NeuralNetwork* NeuralNetwork::fromLocalStorage(const char* path) = 0;
			virtual void NeuralNetwork::toLocalStorage(const char* path) = 0;
			inline void setMaxIteration(unsigned int times){
				m_MaxIterations = times;
			}
			inline void setInitTrainingSize(int size){
				m_InitSizeOfSet = size;
			}
			inline void setActivationFunction(ACTIVATE_FUNC func,DIFF_ACTIVATE_FUNC dfunc){
				ActivateFunction = func;
				DActivateFunction = dfunc;
			}
			inline void setInput(vector<double>& dataInput){
				assert(dataInput.size()==m_Nodes[0]);
				output[0] = dataInput;
			}
			inline void setTarget(vector<double>& dataOutput){
				assert(dataOutput.size()==m_Nodes[m_Nodes.size()-1]);
				target = dataOutput;
			}
			inline void getResult(vector<double>& out){
				int cnt = m_Nodes[m_NumLayers-1];
				for(int i=0;i<cnt;i++){
					out.push_back(output[m_NumLayers-1][i]);
				}
			}
			inline int getNumOfLayers(){
				return m_NumLayers;
			}
			inline vector<int>& getNodesOfLayers(){
				return m_Nodes;
			}
			inline void setNodesOfLayers(vector<int>& nodes){
				m_Nodes = nodes;
			}
			inline vector<double>& getEtaOfLayers(){
				return m_etas;
			}
			inline void setEtaOfLayers(vector<double>& etas){
				m_etas = etas;
			}
			inline void setLearningProcessLog(bool b){
				m_IsLearningProcessRecorded = b;
			}
			inline vector<pair<unsigned int,double>>& getLearningProcessLog(){
				return m_LearningProcess;
			}
			/**
			 * record the learning process of the network
			 *
			 */
			void logLearningProcess(vector<double>& error);
			/**
			 * load training set into memory,
			 *if size of training set is too huge,you should split the data file into several training set
			 * and train the network seperately
			 * this method should be called before doTraining
			 */
			void loadTrainingSet(const char* path);

			/**
			 * train the neural network
			 */
			virtual void doTraining() = 0;
			/**
			 * let input pass through the network
			 */
			virtual void pass() = 0; 
		protected:
			/**
			 * initialize the neural network setups
			 */
			virtual void init(){};
			
		protected:
			/** determine if the learning process should be recorded*/
			bool m_IsLearningProcessRecorded;
			/** initial size of training set*/
			int m_InitSizeOfSet;
			/** layers of neural network*/
			int m_NumLayers;
			/**max training times*/
			unsigned int m_MaxIterations;
			/**current training iteration*/
			unsigned int m_CurrentIteration;
			/** weights between neurons,weight[i][j][k] for weight between node j at layer i and node k at layer i+1*/
			double weight[MAX_LAYERS][MAX_NODE_PER_LAYER][MAX_NODE_PER_LAYER];
			/** output of neurons of every layer*/
			vector<vector<double>> output;
			/** expected output of the network */
			vector<double> target;
			/** random generator*/
			Util::Common::RandomFactory* m_Random;
			/** activation function*/
			ACTIVATE_FUNC ActivateFunction;
			/** one order diffentiation of activation function,for convience,daf is wrapped as daf(af(x))*/
			DIFF_ACTIVATE_FUNC DActivateFunction;
			/**node count of per layer*/
			vector<int> m_Nodes;
			/**learning rate per layers*/
			vector<double> m_etas;
			/**training set*/
			DataSet m_TrainingSet;
			/**learning process*/
			vector<pair<unsigned int,double>> m_LearningProcess;
		};

		BOOST_SERIALIZATION_ASSUME_ABSTRACT(NeuralNetwork)

		class BPNeuralNetwork : public NeuralNetwork{
			friend class boost::serialization::access;
			template<class Archive>
			void serialize(Archive& ar,const unsigned int version){
				ar & boost::serialization::base_object<NeuralNetwork>(*this);
			}
		public:
			BPNeuralNetwork(){}
			BPNeuralNetwork(vector<int>& nodes,vector<double>& etas,int layers = DEFAULT_LAYERS):NeuralNetwork(nodes,etas,layers){
				init();
			}
			BPNeuralNetwork(vector<int>& nodes,double eta,int layers = DEFAULT_LAYERS):NeuralNetwork(nodes,eta,layers){
				init();
			}
			virtual ~BPNeuralNetwork(){
				cleanup();
			}
			NeuralNetwork* fromLocalStorage(const char* path);
			void toLocalStorage(const char* path);
			virtual void doTraining();
			virtual void pass();
		protected:
			virtual void init();
			virtual void cleanup(){};
			virtual void trainBP();
		};
		
		//Generalized RBF Neural Network
		//使用广义RBF神经网络进行回归，或类似核方法的线性分类
		//广义RBF的两种实现策略:
		//(1)采用自组织聚类方法确定RBF中心,然后计算Green矩阵的伪逆
		//(2)采用梯度下降法，计算参数的增量
		class RBFNeuralNetwork : public NeuralNetwork{
			friend class boost::serialization::access;
			template<class Archive>
			void serialize(Archive& ar,const unsigned int version){
				ar & boost::serialization::base_object<NeuralNetwork>(*this);
				ar & m_bUseRegression;
				ar & m_bUseGradientDesc;
				ar & m_NumOfBaseFunc;
				ar & m_NumOfInputDim;
				ar & m_NumOfOutputDim;
				ar & m_TargetError;
				ar & m_CenterError;
				ar & m_Center;
				ar & m_sigma;
			}
		public :
			inline static double NoneActivateFunction(double x){
				return x;
			}
		public:
			RBFNeuralNetwork():m_bUseGradientDesc(false),m_bUseRegression(true),
				RBF(NeuralNetwork::guassRadiusBaseFunction),m_NumOfBaseFunc(DEFAULT_RBF_DIM),
				m_NumOfInputDim(1),m_NumOfOutputDim(1),m_TargetError(0.1),m_CenterError(0.5),m_Green(NULL)
			{
				init();
			}
			RBFNeuralNetwork(bool useGD,bool useRegress,int numInputDim,int numOutputDim):m_bUseGradientDesc(useGD),m_bUseRegression(useRegress),
				RBF(NeuralNetwork::guassRadiusBaseFunction),m_NumOfBaseFunc(DEFAULT_RBF_DIM),
				m_NumOfInputDim(numInputDim),m_NumOfOutputDim(numOutputDim),m_TargetError(0.1),m_CenterError(0.5),m_Green(NULL)
			{
				init();
			}
			virtual ~RBFNeuralNetwork(){
				cleanup();
			}
			NeuralNetwork* fromLocalStorage(const char* path);
			void toLocalStorage(const char* path);

			inline void setTargetError(double targetError){
				m_TargetError = targetError;	
			}
			inline void setCenterError(double centerError){
				m_CenterError = centerError;
			}
			inline void setRBF(RADIUS_BASE_FUNC func){
				RBF = func;
			}
			virtual void doTraining();
			virtual void pass();
		protected:
			virtual void init();
			virtual void cleanup(){
				CleanUpMatrix(m_Green);
			};
			virtual void kmeans();
			virtual void gradientTraining();
			virtual void somTraining();
			void calcGreen();
			void calcSigma();
			double calcDist(const vector<double>& v1,const vector<double>& v2);
			PMatrix pseudo_invert();
		protected:
			/**Radius Base Function*/
			RADIUS_BASE_FUNC RBF;
			/**determin usage for this network,true for regression,false for classification*/
			bool m_bUseRegression;
			/**determin implementation method for this network,true for gradient desc,else for SOM method*/
			bool m_bUseGradientDesc;
			/**number of centers for Base Function,default to be 10;this should be higher than dim of input vector*/
			int m_NumOfBaseFunc;
			/**dimension of input vector*/
			int m_NumOfInputDim;
			/**dimension of output vector*/
			int m_NumOfOutputDim;
			/**expected error,used in Gradient Descend Method for regression usage*/
			double m_TargetError;
			/**in k-means,final center is calculated until the error is less than m_CenterError*/
			double m_CenterError;
			/** Green Matrix*/
			PMatrix m_Green;
			/** the center of RBF,m_Center[i] stand for center vector in training set*/
			vector<vector<double>> m_Center;
			/**extra prameter for RBFs*/
			vector<double> m_sigma;
		};

	};
};

#endif	//end of ANN_H_