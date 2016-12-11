#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <fstream>
#include <array>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include <time.h>
#define MAX 10000
using namespace std;
string filename="largeData.csv";
string testfile="largeData.csv";
const int attrcount=5;
const int rowcount=28000;
const int testrows=28000;
int hashcount[attrcount];
string data[rowcount][attrcount];
string testData[rowcount][attrcount];
string countlabels[MAX][attrcount];
float counts[MAX][attrcount];
float probabilities[MAX][attrcount];
int entrycounters[attrcount];
int classcount=2;
string classes[2]={"yes","no"};
void readFileToData()
{

    char temp[1024];
    strcpy(temp,filename.c_str());
    std::ifstream file(temp);

        for(int row = 0; row < rowcount; ++row)
        {
        std::string line;
        std::getline(file, line);
        if ( !file.good() )
            break;

        std::stringstream iss(line);

        for (int col = 0; col < attrcount; ++col)
        {

            std::string val;
            std::getline(iss, val, ',');
            if ( !iss.good() )
                break;

            std::stringstream convertor(val);
            convertor >> data[row][col];
        }
        }
    
}

void readTestFileToData()
{

    char temp[1024];
    strcpy(temp,testfile.c_str());
    std::ifstream file(temp);

        for(int row = 0; row < rowcount; ++row)
        {
        std::string line;
        std::getline(file, line);
        if ( !file.good() )
            break;

        std::stringstream iss(line);

        for (int col = 0; col < attrcount; ++col)
        {

            std::string val;
            std::getline(iss, val, ',');
            if ( !iss.good() )
                break;

            std::stringstream convertor(val);
            convertor >> testData[row][col];
        }
        }
    
}

void printData()
{
    for(int row=0;row<rowcount;row++)
    {    
        cout<<"\n";
        for(int col=0;col<attrcount;col++)
        {
            cout<<data[row][col]<<"\t";
        }

    }
    cout<<"\n";
}

void printTestData()
{
    for(int row=0;row<rowcount;row++)
    {    
        cout<<"\n";
        for(int col=0;col<attrcount;col++)
        {
            cout<<testData[row][col]<<"\t";
        }

    }
    cout<<"\n";
}
int isInLearningModel(string key,int attr)
{
    for(int i=0;i<entrycounters[attr];i++)
    {
        if(key==countlabels[i][attr])
            return i;
    }
    return -1;
}

void insertInLearningModel(string key, int attr)
{
    int index=isInLearningModel(key,attr);
    if(index>=0)
    {
        counts[index][attr]=counts[index][attr]+1;
    }
    else
    {
        countlabels[entrycounters[attr]][attr]=key;    
	counts[entrycounters[attr]][attr]=1;
        entrycounters[attr]=entrycounters[attr]+1;
    }
}

void split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
}


std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}
void calculateProbabilities(){
	for(int i=0;i<attrcount;i++)
    {
        for(int j=0;j<entrycounters[i];j++)
        {
		if(i==attrcount-1)
		{
			probabilities[j][i]=(float)counts[j][i]/(float)rowcount;
			continue;
		}
            	probabilities[j][i]=(float)counts[j][i]/(float)(counts[isInLearningModel(split(countlabels[j][i],',')[1]+","+split(countlabels[j][i],',')[1],attrcount-1)][attrcount-1]);
        }
    }

}
void learn()
{
    for(int row=0;row<rowcount;row++)
    {    
        for(int col=0;col<attrcount;col++)
        {
	    if(data[row][col]=="")
		continue;
            insertInLearningModel(data[row][col]+","+data[row][attrcount-1],col);
        }
    }
//add barrier
    calculateProbabilities();
}
int getComboIndex(string combo,int attrind)
{
	for(int i=0;i<entrycounters[attrind];i++)
	{
		if(countlabels[i][attrind]==combo)
			return i;
	}
	return -1;
}

int getClassIndex(string classname)
{
	for(int i=0;i<classcount;i++)
	{
		if(classname==classes[i])
			return i;
	}
	return -1;
}
string maxProbClass(double arr[])
{
	double max=-1;
	int maxind=-1;
	for(int i=0;i<classcount;i++)
	{
		if(arr[i]>max)
		{		
			max=arr[i];
			maxind=i;
		}
	}
	return classes[maxind];
}
void test_entire()
{
	double 	probs[classcount];
	double acc=0;
	for(int row=0;row<testrows;row++)
	{
		//initiallize class prob array to 1
		for(int i=0;i<classcount;i++)
		{
			probs[i]=0;
		}
		for(int col=0;col<attrcount;col++)
		{
			for(int classind=0;classind<classcount;classind++)
			{
				if(probs[classind]==0)
					probs[classind]=probabilities[col][getComboIndex(testData[row][col]+","+classes[classind],col)];	
				else
					probs[classind]=probs[classind]*probabilities[col][getComboIndex(testData[row][col]+","+classes[classind],col)];	
			}			

		}
		string maxClass=maxProbClass(probs);
		if(testData[row][attrcount-1]==maxClass)
			acc++;

			for(int col=0;col<attrcount;col++)
	        {
			    if(testData[row][col]=="")
				continue;
		            insertInLearningModel(testData[row][col]+","+maxClass,col);
	        }
		calculateProbabilities();
	}	

}

int main()
{
    double runtime;
    readFileToData();    
    runtime = omp_get_wtime();
    learn();
    runtime = omp_get_wtime() - runtime;
    cout<< "Learning runs in "  << runtime << " seconds\n";
    readTestFileToData();
     runtime = omp_get_wtime();
    test_entire();
    runtime = omp_get_wtime() - runtime;
    cout<< "Testing runs in "  << runtime << " seconds\n";
    return 0;
}
