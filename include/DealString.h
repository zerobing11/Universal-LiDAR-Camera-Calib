#ifndef LIBDEALSTRING_H_
#define LIBDEALSTRING_H_
#include <string>
#include <string>
#include <boost/iterator/iterator_concepts.hpp>
#include <vector>
#include<iostream>
using namespace std;

string substr_start_end(const string content,const string start,string end="",int spos=0,int epos=0)
{
  int n_end,n_start;
  string result;
  if(start=="")
    n_start = 0;
  else
    n_start=content.find(start)+spos;

  if(end=="")
    n_end = content.size();
  else
    n_end=content.find(end)+epos;

  result = content.substr(n_start,n_end-n_start);
  return result;
}

string substr_start_from(const std::string content,const std::string from,const std::string start,std::string end,int spos=0,int epos=0)
{
  int n_end,n_start,n_from;
  string result;
  if(start=="")
    n_start = 0;
  else
    n_from = content.find(from);
    n_start=content.find(start,n_from)+spos;

  if(end=="")
    n_end = content.size();
  else
    n_end=content.find(end)+epos;

  result = content.substr(n_start,n_end-n_start);
  return result;
  
}

string substr_start_from(const std::string content,const int from,const std::string start,std::string end,int spos=0,int epos=0)
{
  int n_end,n_start;
  string result;
  if(start=="")
    n_start = 0;
  else
    n_start=content.find(start,from)+spos;

  if(end=="")
    n_end = content.size();
  else
    n_end=content.find(end)+epos;

  result = content.substr(n_start,n_end-n_start);
  return result;
  
}


vector<float> read_sapce_string(const std::string content,string s,int start)
{
  vector<float> result;
  float d_num = 0;
  int l = content.length();
  string line = content.substr(start+1,l-start);//左闭右开函数
  while(line.find(s)!=std::string::npos)
  {
    int locate = line.find(s);
    string s_num= line.substr(0,locate);
    d_num = std :: stof(s_num);
    result.push_back(d_num);
    l = line.length();
    line = line.substr(locate+1,l-locate-1);
  }
  d_num = std :: stof(line);
  result.push_back(d_num);
  return result;
}

vector<double> read_sapce_string(const std::string content,string s,int start,int end)
{
  vector<double> result;
  double d_num = 0;
  int l=0;
  string line = content.substr(start,end-start);//左闭右开函数
  //cout<<line<<endl;
  while(line.find(s)!=std::string::npos)
  {
    int locate = line.find(s);
    string s_num= line.substr(0,locate);
    d_num = atof(s_num.c_str());
    result.push_back(d_num);
    l = line.length();
    line = line.substr(locate+1,l-locate-1);
  }
  d_num = atof(line.c_str());
  //cout<<d_num<<endl;
  result.push_back(d_num);
  return result;
}

//定位字符s重复出现次数的在content中的位置
int locate_repeat(const std::string content,string s,int num)
{
  int locate=0;
  int glboal_locate = 0;
  int times = num;
  string s_content = content;
  while(times!=0)
  {
    locate = s_content.find(s);
    glboal_locate = glboal_locate + locate;
    //cout<<glboal_locate<<endl;
    times--;
    s_content  =  s_content.substr(locate+1, s_content.length()-locate-1);
    //cout<<s_content<<endl;
  }
  return glboal_locate+num-1;
}


vector<string> read_format(string content,string s)
{
  int l=0;
  vector<string> result;
  while(content.find(s)!=content.npos)
  {
    int locate = content.find(s);
    string s_num= content.substr(0,locate);
    l = content.length();
    content = content.substr(locate+1,l-locate-1);
    result.push_back(s_num);
  }
  result.push_back(content);
  return result;
}


#endif
