/*
 * DemoProblem3.h
 *
 *  Created on: Apr 19, 2013
 *      Author: woodlgz
 */

/*
 * demo of solving TSP.
 * problem description:
 * ��������n���̵꣬ÿ���̵���һ�ֲ�ͬ��Ʒ������һ���ɹ�ԱҪ�ɹ���n����Ʒ��
 * ��ɹ�Ա��ֻ����ÿ���̵�һ���ҷ���ԭ����������СȨͨ·��
 * ���⼴��ȡn������Ĵ�Ȩ����ͼ�µ���С���ܶٻ�·��
 * Ϊ�����⣬�ٶ�����ͼ����ȫͼ�����������ǲ���ʽ���������ǿ���ʹ��Christofides 3/2�����㷨���
 * �����ȫͼ��TSP���⻹����ʹ�ö�̬�滮���⣬����ʹ�÷�֧���޷�
 * ��ʹ���Ŵ��㷨����⣬��ô����ͼΪ��ȫͼ���Ƿ��������ǲ���ʽ������ֻҪ���Ա�֤ͼ�ǹ��ܶ�ͼ��
 * һ�����ƽ��ǿ��Ի�õ�
 */

#ifndef DEMOPROBLEM3_H_
#define DEMOPROBLEM3_H_

#include <iostream>
#include "GA.h"

using namespace std;
using namespace FASTAI::GA;

class Demo3GeneticPhase : public GeneticPhase{
public:
	static const int MAX_VETEX = 30;
private :
	static int VETEX;
	static int Graph[MAX_VETEX+1][MAX_VETEX+1];
public:
	Demo3GeneticPhase():GeneticPhase(Demo3GeneticPhase::VETEX+1){
		init();
	}
	~Demo3GeneticPhase(){
		cleanup();
	}

	void* read(){
		cout<<"min:"<<m_Answer<<endl;
		for(int i=0;i<VETEX;i++)
			cout<<m_Coding[i]<<"->";
		cout<<m_Coding[VETEX]<<endl;
		return NULL;
	}

	int calcValueOfCode(){
		m_Answer = 0;
		for(int i=0;i<VETEX;i++)
			m_Answer += Graph[m_Coding[i]][m_Coding[i+1]];
		return m_Answer;
	}

	inline void* getCodeAt(int i){
				return &m_Coding[i];
	}
	GeneticPhase& operator = (GeneticPhase& phase){
		Demo3GeneticPhase* _phase = static_cast<Demo3GeneticPhase*>(&phase);
		memcpy((void*)m_Coding,(void*)(_phase->m_Coding),m_Len*sizeof(int));
		this->m_Answer = _phase->calcValueOfCode();
		return *this;
	}
	bool Demo3GeneticPhase::isBetterThan(GeneticPhase* phase){
		return getAnswer()<phase->getAnswer();
	}
private:
	/** �Ի���Ƭ�ν��г�ʼ��*/
	void init(){
		m_Coding = new int[VETEX+1];
		reConstruct();
	}
	/** �������Ƭ��*/
	void cleanup(){
		if(m_Coding){
			delete[] m_Coding;
			m_Coding = NULL;
		}
	}
	/** ��Ի���Ƭ�ε��ӽ�����*/
	void crossing(GeneticPhase* phase){
		Demo3GeneticPhase* _phase = static_cast<Demo3GeneticPhase*>(phase);
		int rInt = GENERATE_RANDOM()%(VETEX+1);
		int tmp = m_Coding[rInt];
		for(int i=0;i<=VETEX;i++){
			m_Coding[i] = m_Coding[i] == tmp?_phase->m_Coding[rInt]:(m_Coding[i]==_phase->m_Coding[rInt]?tmp:m_Coding[i]);
		}
		int tmp2 = _phase->m_Coding[rInt];
		for(int i=0;i<=VETEX;i++){
			_phase->m_Coding[i] = _phase->m_Coding[i]==tmp2?tmp:(_phase->m_Coding[i]==tmp?tmp2:_phase->m_Coding[i]);
		}
	}

	/** ��Ի���Ƭ�εı��촦��*/
	void mutate(){
		int rInt = (GENERATE_RANDOM()%(m_Len-2))+1;
		int rInt2 = (GENERATE_RANDOM()%(m_Len-2))+1;
		swap(m_Coding[rInt],m_Coding[rInt2]);
	}

	/** �߼�����,���¹������֤�����Ϊ���ܴ��ĸ���*/
	void reConstruct(){
		for(int i=1;i<=VETEX;i++)
			m_Coding[i] = i;
		for(int i=VETEX;i>0;i--)
			swap(m_Coding[(GENERATE_RANDOM()%VETEX)+1],m_Coding[(GENERATE_RANDOM()%VETEX)+1]);
		m_Coding[0] = m_Coding[VETEX]; 
	}

public:
	//Ϊ�˽��г���ⷨ��GA���ĶԱȣ����峣��ⷨ�ľ�̬solver
	static void ProblemInit(int G[][Demo3GeneticPhase::MAX_VETEX+1],int vetex){
		//��ʼ�����⣬����ʼ����������ڲ��̵��ͼ��Ϊ�˷��㴦��ٶ����̵�ı�Ŵ�1��ʼ
		//�ڱ�Demo��ͼ�����ݽṹʹ�ö�ά�����ʾ��ʽ��������Ҫ���Ը����ڽӱ�
		//Graph[i][j]��ʾ�̵�i���̵�j��·����Ȩֵ����ȨֵС��0��ʾi��j������
		for(int i=1;i<=vetex;i++){
			for(int j=1;j<=vetex;j++){
				Graph[i][j] = G[i][j];
				//printf("G[%d][%d]:%d\t",i,j,G[i][j]);
			}
		}
		VETEX = vetex;
	}
	//��ȫͼ�ж�
	static bool isCompleteGraph(){
		int cnt[MAX_VETEX+1];
		memset(cnt,0,sizeof(cnt));
		for(int i=1;i<=VETEX;i++)
			for(int j=1;j<=VETEX;j++)
				if(Graph[i][j]>0)
					cnt[i]++;
		for(int i=1;i<=VETEX;i++)
			if(cnt[i]!=VETEX-1)
				return false;
		return true;
	}
	//�Ƿ��������ǲ���ʽ�����ж�
	static bool triangleSuffice(){
		for(int i=1;i<=VETEX;i++){
			for(int j=1;j<=VETEX;j++){
				for(int k=1;k<=VETEX;k++){
					if(Graph[i][j]>0&&Graph[i][k]>0&&Graph[j][k]>0)
						if(Graph[i][j]+Graph[j][k]<Graph[i][k])
							return false;
				}
			}
		}
		return true;
	}

	// ������ȫͼ��DP�ⷨ
	// @param ������Դ��
	static void DPSolver(int start){
		bool subset[MAX_VETEX+1];	//ʣ���Ӽ�
		int path[MAX_VETEX+1];		//���Ż�·
		memset(subset,true,sizeof(subset));
		subset[start] = false;
		int m = _DPSolver(start,start,subset,path,0);
		cout<<"min:"<<m<<endl;
		cout<<path[0];
		for(int i=1;i<=VETEX;i++){
			cout<<"->"<<path[i];
		}
		cout<<endl;
	}
	// ��֧���޽ⷨ,���ڸ���
	static void AStarSolver(){
	}
	// �����������ǲ���ʽ��ͼ��Christrofides���ƽⷨ,���ڸ���
	static void ChristrofidesSolver(){
	}
private:
	/*@param source: �����Դ��
	 *@param start:	 ����һ��ѡ�еı������չ�ڵ�
	 *@param subset: ����չ�ڵ��Ӽ�
	 *@param path:   ��¼�ӱ��㿪ʼͨ����ͨ�Ӽ����ڵ������Ŀ���·��
	 *@param depth:  ��ǰ���
	 */
	static int _DPSolver(int source,int start,bool subset[],int path[],int depth){
		int m = 0x1FFFFFFF;
		int tmppath[MAX_VETEX+1];
		if(depth == VETEX-1)
			subset[source] = true;
		if(depth == VETEX)
			m = 0;
		tmppath[depth] = start;
		path[depth] = start;
		for(int i=1;i<=VETEX;i++){
			if(subset[i]){
				subset[i] = false;
				int ans = _DPSolver(source,i,subset,tmppath,depth+1) + Graph[start][i];
				if(m>ans){
					for(int j=depth+1;j<=VETEX;j++)
						path[j]=tmppath[j];
					m = ans;
				}
				subset[i] = true;
			}
		}
		if(depth == VETEX-1)
			subset[source] = false;
		return m;
	}
};


int Demo3GeneticPhase::VETEX = 6;
int Demo3GeneticPhase::Graph[Demo3GeneticPhase::MAX_VETEX+1][Demo3GeneticPhase::MAX_VETEX+1];

class Demo3Env : public Env{
public:
	Demo3Env(GFactory* factory,float cRate, float mRate, int age):
		Env(factory,cRate,mRate,age){
	}
private:
	void judge(){
		m_ScoreMax = 0.0;
		for(int i=0;i<m_PSize;i++){
			m_Score[i] = 100.0 / m_Population[i]->calcValueOfCode();	//��ȨֵԽ�ͷ���Խ��
			if(m_ScoreMax<m_Score[i])
				m_ScoreMax = m_Score[i];
		}
	}

	float judge(int i){
		if(m_ScoreMax<1e-38&&m_ScoreMax>-1e-38)
			return 0.0;
		return m_Score[i] / m_ScoreMax;
	}
	
};

#endif