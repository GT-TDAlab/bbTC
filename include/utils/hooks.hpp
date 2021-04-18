#ifndef HOOKS_H_
#define HOOKS_H_

#include <iostream>
#include <chrono>
#include <unordered_map>
#include <string>
#include <iomanip>
#include <sstream>
#include <vector>


// Global timer variable
static std::chrono::steady_clock::time_point hooks_start ;
static std::chrono::steady_clock::time_point hooks_end ;

static std::vector< std::string > Fields;

void HooksAddField(std::string field) {
  Fields.push_back(field);
}
void HooksRegionBegin(std::string name)
{
    // Add region name to the result string
    HooksAddField(name);

    // Start the timer
    hooks_start = std::chrono::steady_clock::now();;
}

double HooksRegionEnd()
{
  // Stop the timer
  auto hooks_end = std::chrono::steady_clock::now();
  // Add time elapsed to result string
  double time_ms = std::chrono::duration_cast<std::chrono::microseconds>
                        (hooks_end - hooks_start).count()/1000000.;

  std::stringstream stream;
  stream << std::fixed << std::setprecision(4) << time_ms;
  HooksAddField(stream.str());
  return time_ms;
}

void PrintFields() {
  for(unsigned int i=0; i<Fields.size(); i+=2) {
    std::cout << Fields[i] << " : " << Fields[i+1] << std::endl;
  }
  // for(auto s : Fields) {
    // std::cout << s << std::endl;
  // }
}

void PrintFieldsToJson() {

}

#endif