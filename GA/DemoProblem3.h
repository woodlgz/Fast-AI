/*
 * DemoProblem3.h
 *
 *  Created on: Apr 19, 2013
 *      Author: woodlgz
 */

/*
 * demo of solving TSP.
 * problem description:
 * 城市内有n个商店，每个商店卖一种不同物品，现在一个采购员要采购这n种物品，
 * 求采购员在只访问每个商店一次且返回原点的情况下最小权通路。
 * 问题即求取n个顶点的带权无向图下的最小哈密顿回路。
 * 为简化问题，假定无向图是完全图而且满足三角不等式条件，于是可以使用Christofides 3/2近似算法求解
 * 解决完全图的TSP问题还可以使用动态规划来解，或者使用分支界限法
 * 若使用遗传算法来求解，那么无论图为完全图或是否满足三角不等式条件，只要可以保证图是哈密顿图，
 * 一个近似解是可以获得的
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
	/** 对基因片段进行初始化*/
	void init(){
		m_Coding = new int[VETEX+1];
		reConstruct();
	}
	/** 清理基因片段*/
	void cleanup(){
		if(m_Coding){
			delete[] m_Coding;
			m_Coding = NULL;
		}
	}
	/** 针对基因片段的杂交处理*/
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

	/** 针对基因片段的变异处理*/
	void mutate(){
		int rInt = (GENERATE_RANDOM()%(m_Len-2))+1;
		int rInt2 = (GENERATE_RANDOM()%(m_Len-2))+1;
		swap(m_Coding[rInt],m_Coding[rInt2]);
	}

	/** 高级变异,重新构造基因保证基因成为可能存活的个体*/
	void reConstruct(){
		for(int i=1;i<=VETEX;i++)
			m_Coding[i] = i;
		for(int i=VETEX;i>0;i--)
			swap(m_Coding[(GENERATE_RANDOM()%VETEX)+1],m_Coding[(GENERATE_RANDOM()%VETEX)+1]);
		m_Coding[0] = m_Coding[VETEX]; 
	}

public:
	//为了进行常规解法和GA求解的对比，定义常规解法的静态solver
	static void ProblemInit(int G[][Demo3GeneticPhase::MAX_VETEX+1],int vetex){
		//初始化问题，即初始化代表城市内部商店的图，为了方便处理假定各商店的编号从1开始
		//在本Demo中图的数据结构使用二维数组表示方式，如有需要可以改用邻接表
		//Graph[i][j]表示商店i到商店j的路径的权值，若权值小于0表示i和j不相邻
		for(int i=1;i<=vetex;i++){
			for(int j=1;j<=vetex;j++){
				Graph[i][j] = G[i][j];
				//printf("G[%d][%d]:%d\t",i,j,G[i][j]);
			}
		}
		VETEX = vetex;
	}
	//完全图判断
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
	//是否满足三角不等式条件判断
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

	// 对于完全图的DP解法
	// @param 出发的源点
	static void DPSolver(int start){
		bool subset[MAX_VETEX+1];	//剩余子集
		int path[MAX_VETEX+1];		//最优回路
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
	// 分支界限解法,先挖个坑
	static void AStarSolver(){
	}
	// 对于满足三角不等式的图的Christrofides近似解法,先挖个坑
	static void ChristrofidesSolver(){
	}
private:
	/*@param source: 最初的源点
	 *@param start:	 由上一层选中的本层的扩展节点
	 *@param subset: 带扩展节点子集
	 *@param path:   记录从本层开始通过联通子集各节点获得最短目标的路径
	 *@param depth:  当前深度
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
			m_Score[i] = 100.0 / m_Population[i]->calcValueOfCode();	//总权值越低分数越高
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