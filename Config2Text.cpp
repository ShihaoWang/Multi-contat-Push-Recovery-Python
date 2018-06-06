// g++ -Wall -O3 -o Config2Text Config2Text.cpp
#include <iostream>
#include <fstream>
using namespace std;

int main()
{
    string dataArray[36];
    ifstream input_file;
    input_file.open("robot_angle_init.config");
    if(input_file.is_open())
    {
        char temp;
        input_file>>temp;
        input_file>>temp;
        for(int i = 0; i<36; i++)
        {
            input_file>>dataArray[i];
        }
    }
    input_file.close();
    // After the loading, this program will rewrite them into a column fashion
    ofstream output_file;
    output_file.open("robot_angle_init.txt", std::ofstream::out);
    for (int i = 0; i < 36; i++)
    {
        output_file<<dataArray[i]<<endl;
    }
    output_file.close();
    return 0;
}
