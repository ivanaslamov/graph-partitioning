#include <iostream>
#include <fstream>
#include <cmath>
#include <limits>
#include <climits>
#include <vector>
#include <map>
#include <algorithm>

#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <limits>
#include <climits>

#include "refine.h"
#include "agglomerate.h"

using namespace std;

int main(int argc,char** argv)
{
	double capacity = 10;

	// ==== Pre Processing ====

	std::ifstream is("network.dat", std::ifstream::binary);

	int min_node = INT_MAX;
	int max_node = INT_MIN;

	vector<double> edges;

	if (is)
	{
		int so,si;
		double e;

		while(is)
		{
			is >> so >> si >> e;

			edges.push_back(so);
			edges.push_back(si);
			edges.push_back(e);
			
			max_node = max_node > so ? max_node : so;
			max_node = max_node > si ? max_node : si;

			min_node = min_node < so ? min_node : so;
			min_node = min_node < si ? min_node : si;
		}

		is.close();
	}

	// Create a matrix for the return argument

	// get pointer to output matrix
	double* ms = (double*) malloc(sizeof(double) * (max_node-1)*2);

	double* parts = (double*) malloc(sizeof(double) * (max_node));

	// Do the actual computations in a subroutine
	int len = agglomerate(ms,&edges[0],edges.size()/3,3,0,max_node,capacity);

	refine(parts,ms,len,&edges[0],edges.size()/3,3,0,max_node,capacity);

	free(parts);

	free(ms);

    return 0;
}