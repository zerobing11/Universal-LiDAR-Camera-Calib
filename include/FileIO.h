#ifndef _FILEIO_
#define _FILEIO_
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
//用于删除目录
#include <dirent.h>  
//用于新建目录
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#include <boost/concept_check.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>//使用到了四元数数据结构

#include <opencv/cv.h>    
#include <opencv/highgui.h>    

#include "DealString.h"

using namespace std;
using namespace cv;
namespace FileIO
{
  inline string GetParentDir(const string& path)
  {
    size_t pos = path.find_last_of('/');
    if(pos == string::npos)
    {
      return string();
    }
    if(pos == 0)
    {
      return string("/");
    }
    return path.substr(0, pos);
  }

  inline bool CreateDirectories(const string& dir_path)
  {
    if(dir_path.empty())
    {
      return true;
    }
    struct stat st;
    if(stat(dir_path.c_str(), &st) == 0)
    {
      return S_ISDIR(st.st_mode);
    }
    string current;
    current.reserve(dir_path.size());
    for(size_t i = 0; i < dir_path.size(); ++i)
    {
      char c = dir_path[i];
      current.push_back(c);
      if(c == '/' && current.size() > 1)
      {
        if(stat(current.c_str(), &st) != 0)
        {
          if(mkdir(current.c_str(), 0755) != 0 && errno != EEXIST)
          {
            return false;
          }
        }
      }
    }
    if(stat(current.c_str(), &st) != 0)
    {
      if(mkdir(current.c_str(), 0755) != 0 && errno != EEXIST)
      {
        return false;
      }
    }
    return true;
  }
  vector<vector<float>> ReadTxt2Float(string path,bool print = false)
  {
      vector<vector<float>> res;
      ifstream f;
      f.open(path.c_str());
      char c[8000];
      string content;
      while(!f.eof())
      {
	  f.getline(c,8000);
	  content = c;
	  
	  // 去除前后空格
	  size_t start = content.find_first_not_of(" \t\r\n");
	  size_t end = content.find_last_not_of(" \t\r\n");
	  if(start == string::npos) // 空行或只包含空格
	  {
	    continue;
	  }
	  content = content.substr(start, end - start + 1);
	  
	  if(content!="")
	  {
	      vector<string> p;
	      p = read_format(content," ");
	      vector<float> tmp;
	      for(int i=0;i<p.size();i++)
	      {
		// 跳过空字符串，避免stof转换错误
		if(p[i].empty() || p[i] == "" || p[i] == " ")
		{
		  continue;
		}
		try
		{
		  tmp.push_back(std::stof(p[i]));
		}
		catch(const std::invalid_argument& e)
		{
		  // cout<<"[Warning] ReadTxt2Float: Invalid number format in "<<path<<": '"<<p[i]<<"', skipping."<<endl;
		}
	      }
	      if(!tmp.empty())  // 只添加非空的行
	      {
		res.push_back(tmp);
	      }
	  }
      }
      f.close();
      //是否需要打印
      if(print)
      {
	 for(int i=0;i<res.size();i++)
	{
	  string content = "";
	  for(int j=0;j<res[i].size();j++)
	  {
	    content = content + std::to_string(res[i][j])+" ";
	  }
	  cout<<content<<endl;
	}
      }
      return res;
  }
  
  vector<string> ReadTxt2String(string path,bool print = false)
  {
      vector<string> res;
      ifstream f;
      f.open(path.c_str());
      char c[4000];
      string content;
      while(!f.eof())
      {
	f.getline(c,4000);
	content = c;
	if(content!="")
	  res.push_back(content);
      }
      f.close();
      //是否需要打印
      if(print)
      {
	 for(int i=0;i<res.size();i++)
	{
	  cout<<res[i]<<endl;
	}
      }
      return res;
  }
  
  void WriteSting2Txt(string path,vector<string> content)
  {
    ofstream f_out;
    f_out.open(path.c_str());
    if(!f_out){
      cout << "Unable to open otfile";
      //exit(1); 
    }
    for(int i=0;i<content.size();i++)
    {
      f_out<<content[i]<<endl;
    }
    f_out.close();
    cout<<"Finish Writting "<<path<<endl;
  }
  
  //输入/root/fyy/a.jpg 返回 .jpg 和 /root/fyy/
  vector<string> ProcessPath(string path,bool print = false)
  {
    vector<string> res;
    vector<string> temp = read_format(path,"/");
    int offset = 0;
    if(temp[temp.size()-1].find(".")!=string::npos)
    {
      res.push_back(substr_start_end(temp[temp.size()-1],"."));
      offset = 1;
    }
    else
    {
      res.push_back("");
      offset = 0;
    }
    string c = "";
    for(int i=0;i<temp.size()-offset;i++)
    {
      c = c + temp[i]+"/";
    }
    res.push_back(c);
    
    if(print)
    {
      cout<<res[0]<<endl;
      cout<<res[1]<<endl;
    }
    return res;
  }
  
  map<string,vector<float>> read_config(string path)
  {
    map<string,vector<float>> res;
    ifstream f;
    f.open(path.c_str());
    char buffer[5000];
    while((!f.eof())&&(f.good()))
    {
	  f.getline(buffer,5000);
	  string content = buffer;
	  
	  // 去除前后空格
	  size_t start = content.find_first_not_of(" \t\r\n");
	  size_t end = content.find_last_not_of(" \t\r\n");
	  if(start == string::npos) // 空行或只包含空格
	  {
	    continue;
	  }
	  content = content.substr(start, end - start + 1);
	  
	  vector<string> messages;
	  if(content!="")
	  {
	    messages = read_format(content," ");
	    if(messages.empty() || messages[0].empty())
	    {
	      continue;
	    }
	    string name = messages[0];
	    cout<<name<<" ";
	    vector<float> data;
	    for(int i=1;i<messages.size();i++)
	    {
	      // 跳过空字符串，避免stof转换错误
	      if(messages[i].empty() || messages[i] == "" || messages[i] == " ")
	      {
	        continue;
	      }
	      try
	      {
	        data.push_back(std::stof(messages[i]));
	        cout<<std::stof(messages[i])<<" ";
	      }
	      catch(const std::invalid_argument& e)
	      {
	        // cout<<"[Warning] Invalid number format: '"<<messages[i]<<"', skipping."<<endl;
	      }
	    }
	    cout<<endl;
	    res.insert(pair<string,vector<float>>(name,data));
	  }
	  
    }
    f.close();
    return res;
  }
  
  //使用#表示有注释了
  map<string,vector<float>> read_config_withcomments(string path)
  {
    map<string,vector<float>> res;
    ifstream f;
    f.open(path.c_str());
    char buffer[5000];
    while((!f.eof())&&(f.good()))
    {
	  f.getline(buffer,5000);
	  string content = buffer;
	  
	  // 去除前后空格
	  size_t start = content.find_first_not_of(" \t\r\n");
	  size_t end = content.find_last_not_of(" \t\r\n");
	  if(start == string::npos) // 空行或只包含空格
	  {
	    continue;
	  }
	  content = content.substr(start, end - start + 1);
	  
	  // 跳过注释行
	  if(content[0] == '#')
	  {
	    continue;
	  }
	  
	  vector<string> messages;
	  if(content!="")
	  {
	    messages = read_format(content," ");
	    if(messages.empty() || messages[0].empty())
	    {
	      continue;
	    }
	    string name = messages[0];
	    cout<<name<<" ";
	    vector<float> data;
	    for(int i=1;i<messages.size();i++)
	    {
	      // 跳过空字符串和注释，避免stof转换错误
	      if(messages[i].empty() || messages[i] == "" || messages[i] == " " || messages[i][0] == '#')
	      {
	        break;  // 遇到注释符号，后面的内容都不处理
	      }
	      try
	      {
	        data.push_back(std::stof(messages[i]));
	        cout<<std::stof(messages[i])<<" ";
	      }
	      catch(const std::invalid_argument& e)
	      {
	        // cout<<"[Warning] Invalid number format: '"<<messages[i]<<"', skipping."<<endl;
	      }
	    }
	    cout<<endl;
	    res.insert(pair<string,vector<float>>(name,data));
	  }
	  
    }
    f.close();
    return res;
  }

  void WriteSenserParameters(Eigen::Matrix3d Rcl,Eigen::Vector3d tcl,cv::Mat intrincMatrix,cv::Mat distParameter,string save_path)
  {
    vector<string> calibraion_res;
    calibraion_res.push_back( to_string(Rcl(0,0))+" "+to_string(Rcl(0,1))+" "+to_string(Rcl(0,2)) );
    calibraion_res.push_back( to_string(Rcl(1,0))+" "+to_string(Rcl(1,1))+" "+to_string(Rcl(1,2)) );
    calibraion_res.push_back( to_string(Rcl(2,0))+" "+to_string(Rcl(2,1))+" "+to_string(Rcl(2,2)) );
    calibraion_res.push_back( to_string(tcl[0])+" "+to_string(tcl[1])+" "+to_string(tcl[2]) );
    calibraion_res.push_back( to_string(intrincMatrix.at<double>(0,0))+" "+to_string(intrincMatrix.at<double>(0,1))+" "+to_string(intrincMatrix.at<double>(0,2)) );
    calibraion_res.push_back( to_string(intrincMatrix.at<double>(1,0))+" "+to_string(intrincMatrix.at<double>(1,1))+" "+to_string(intrincMatrix.at<double>(1,2)) );
    calibraion_res.push_back( to_string(intrincMatrix.at<double>(2,0))+" "+to_string(intrincMatrix.at<double>(2,1))+" "+to_string(intrincMatrix.at<double>(2,2)) );
    calibraion_res.push_back(to_string(distParameter.at<double>(0,0))+" "+
                to_string(distParameter.at<double>(0,1))+" "+
                to_string(distParameter.at<double>(0,2))+" "+
                to_string(distParameter.at<double>(0,3))+" "+
                to_string(distParameter.at<double>(0,4)));
    FileIO::WriteSting2Txt(save_path,calibraion_res);
  }
}
#endif