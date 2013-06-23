
#include <Rcpp.h>
#include <iostream>
#include <string>
#include <boost/regex.hpp>
#include <boost/algorithm/string.hpp>
#include <vector>

using namespace Rcpp;
using namespace std;
using namespace boost;

std::string remove_numbers(std::string input)
{
boost::regex digit("\\d");
std::string fmt = "";
std::string output = boost::regex_replace(input,digit,fmt);
return output;
}

std::string remove_whitespace(std::string input)
{
boost::regex white("\\s+");
std::string fmt = " ";
std::string output = boost::regex_replace(input,white,fmt);
return output;
}

std::string remove_punctuation(std::string input)
{
boost::regex punct("[[:punct:]]");
std::string fmt = " ";
std::string output = boost::regex_replace(input,punct,fmt);
return output;
}

vector<std::string> isolate_words(std::string input)
{
vector<std::string> output_vector;
boost::split(output_vector,input,boost::is_any_of("\t "));
return output_vector;
}

vector<std::string> eliminate_empty_words(vector<std::string> input)
{
vector<std::string> output_vector;
int length = input.size();
for (int i=0;i<length;i++)
{
if (input[i]=="\0") cout << "eine leerzeile";
else
  {output_vector.push_back(input[i]);
	cout << "input" << input[i];
	}
}
return output_vector;
}

// [[Rcpp::export]]
void print_vector(vector<std::string> input)
{
for( std::vector<string>::iterator i = input.begin(); i != input.end(); ++i)
	    Rcout << *i << ' ';
}