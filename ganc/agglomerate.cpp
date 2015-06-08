
#include "mex.h"

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

using namespace std;


// based on the code from http://kukuruku.co/hub/cpp/avl-trees
// and randomized binary search trees http://habrahabr.ru/post/145388/
// and random shuffle http://stackoverflow.com/questions/18993998/iterating-over-the-array-in-random-order


struct community
{
	int    partition;
  double cost;
};

// ==== RANDOM BINARY TREE FUNCTIONS ====

struct node
{
  // tree variables
  int key;
  int size;

  node* left;
  node* right;

  // heap variables 
  int heap_partition_out;
  int heap_partition_in;
  double heap_maximum;

  // load
	community* c;

  node(int k, community* load)
  {
    key = k;
    left = right = 0;
    size = 1;
    
    heap_partition_out = -1;
    heap_partition_in = -1;
    heap_maximum = -1;

    c = load;
  }
};

int getsize(node* p) // обертка для поля size, работает с пустыми деревьями (t=NULL)
{
    if( !p ) return 0; 
    return p->size; 
}

void fixsize(node* p) // установление корректного размера дерева
{
    p->size = getsize(p->left)+getsize(p->right)+1; 
}

node* rotateright(node* p) // правый поворот вокруг узла p
{
    node* q = p->left; 
    if( !q ) return p; 
    p->left = q->right; 
    q->right = p; 
    q->size = p->size; 
    fixsize(p); 
    return q; 
}

node* rotateleft(node* q) // левый поворот вокруг узла q
{
    node* p = q->right;
    if( !p ) return q;
    q->right = p->left;
    p->left = q;
    p->size = q->size;
    fixsize(q);
    return p;
}

node* insertroot(node* p, int k, community* c) // вставка нового узла с ключом k в корень дерева p 
{
    if( !p ) return new node(k, c); 
    if( k<p->key ) 
    {
        p->left = insertroot(p->left, k, c); 
        return rotateright(p); 
    }
    else 
    {
        p->right = insertroot(p->right, k, c);
        return rotateleft(p);
    }
}

node* insert(node* p, int k, community* c) // insert k key in a tree with p root
{
    if( !p ) return new node(k, c);
    if( rand()%(p->size+1)==0 ) 
        return insertroot(p,k,c); 
    if( p->key>k ) 
        p->left = insert(p->left,k,c); 
    else
        p->right = insert(p->right,k,c); 
    fixsize(p); 
    return p;
}

node* join(node* p, node* q) // join two trees
{
    if( !p ) return q;
    if( !q ) return p;
    if( rand()%(p->size+q->size) < p->size ) 
    {
      p->right = join(p->right,q); 
      fixsize(p);

  		p->heap_partition_out = -1;
  		p->heap_partition_in = -1;
  		p->heap_maximum = -1;

      return p; 
    }
    else 
    {
      q->left = join(p,q->left); 
      fixsize(q); 

  		p->heap_partition_out = -1;
  		p->heap_partition_in = -1;
  		p->heap_maximum = -1;

      return q; 
    }
}

node* remove(node* p, int k) // removing first node with the same key
{
    if( !p ) return p;

	p->heap_partition_out = -1;
	p->heap_partition_in = -1;
	p->heap_maximum = -1;

    if( p->key==k ) 
    {
        node* q = join(p->left,p->right); 
        delete p; 
        return q; 
    }
    else if( k<p->key )
        p->left = remove(p->left,k); 
    else 
        p->right = remove(p->right,k); 
    return p; 
}

// debug

int clean(node* p)
{
    if( !p ) return 0;

    clean(p->left);

    clean(p->right);

    delete p;

    return 0;
}

int check(node* p, int k)
{
    if( !p || k==0 ) return 0;

    printf("key size: %d %d\n",p->key,p->size);

    check(p->left,k-1);

    check(p->right,k-1);

    return 0;
}

// ==== GREEDY AGGLOMERATION ====

void dirty(node* p, int k)
{
  if( !p ) return;

  p->heap_partition_out = -1;
  p->heap_partition_in = -1;
  p->heap_maximum = -1;

  if( k==p->key )
  {
    return;
  }
  
  if( k<p->key ) 
    dirty(p->left,k); 
  else
    dirty(p->right,k);
}

double update_heap(node* p)
{
  if( !p ) return -INFINITY;

  if( p->heap_partition_out > -1 ) return p->heap_maximum;

  double left_cost  = update_heap(p->left);
  double right_cost = update_heap(p->right);

  if(p->c->cost > left_cost && p->c->cost > right_cost)
  {
    p->heap_partition_out = p->key;
    p->heap_partition_in  = p->c->partition;
    p->heap_maximum       = p->c->cost;
  }
  else if( left_cost > right_cost )
  {
    p->heap_partition_out = p->left->heap_partition_out;
    p->heap_partition_in  = p->left->heap_partition_in;
    p->heap_maximum       = p->left->heap_maximum;
  }
  else
  {
    p->heap_partition_out = p->right->heap_partition_out;
    p->heap_partition_in  = p->right->heap_partition_in;
    p->heap_maximum       = p->right->heap_maximum;
  }

  return p->heap_maximum;
}

int agglomerate( double *mp, double *np, double *cp, double *w, int m, int n, int w_order, int max_node )
{
  std::vector< std::map<int, double> > edges(max_node+1);

  // reading edges

  for(int i=0;i<m;i++)
  {
    int so = (int) w_order ? w[i] : w[n*i];
    int si = (int) w_order ? w[i+m] : w[n*i+1];
    double e = w_order ? w[i+2*m] : w[n*i+2];

    if( so != si )
    {
      edges[so].insert( std::pair<int,double>(si,e) );
      edges[si].insert( std::pair<int,double>(so,e) );  
    }
  }

  // init variables

  std::vector<double> degree(max_node+1);

  std::vector<double> inner(max_node+1);

  for(int i=1;i<edges.size();i++)
  {
    double weight = 0; 

    for (std::map<int,double>::iterator it=edges[i].begin(); it!=edges[i].end(); ++it)
    {
      weight += it->second;
    }

    degree[i] = weight;
    inner[i] = 0;
  }

  for(int i=0;i<max_node-1;i++)
  {
  	np[i] = 0;
  }

  // heap construction

  std::vector<int> indexes(max_node+1);

  for(int i=0;i<max_node+1;i++)
  {
    indexes[i] = i;
  }

  // shuffle connections

  random_shuffle(indexes.begin(),indexes.end());

  // initialize tree

  node* tree = NULL;

  std::vector< std::map<int, double> > communities(max_node+1);

  std::vector<community> connections(max_node+1);

  for(int i=0;i<indexes.size();i++)
  {
  	connections[ indexes[i] ].partition = -1;
  	connections[ indexes[i] ].cost      = -INFINITY;
    	
    if( edges[ indexes[i] ].size() > 0 )
    {
		// insert elements into the tree

    	for (std::map<int,double>::iterator it=edges[ indexes[i] ].begin(); it!=edges[ indexes[i] ].end(); ++it)
  		{
  			double dcost = 2*it->second / ( degree[ indexes[i] ] + degree[ it->first ] );

  			communities[ indexes[i] ][it->first] = dcost;

  			if( dcost > connections[ indexes[i] ].cost || connections[ indexes[i] ].partition == -1 )
  			{
  				connections[ indexes[i] ].partition = it->first;
		    	connections[ indexes[i] ].cost      = dcost;
  			}
  		}

      if(connections[ indexes[i] ].partition != -1)
        tree = insert(tree, indexes[i], &connections[ indexes[i] ]);
    }
  }

  // initialize heap

  update_heap(tree);

clock_t t1,t2;
t1=clock();

  int k;
  double nassoc = 0;

  for(k=0;k<max_node-1 && tree;k++)
  {
    int m = max( tree->heap_partition_in , tree->heap_partition_out );
    int n = min( tree->heap_partition_in , tree->heap_partition_out );

    if( tree->heap_maximum == -INFINITY)
      break;

    if( k % 10000 == 0)
      cout << k << " agglomerate " << m << " " << n << endl;

  	nassoc += tree->heap_maximum;

  	double edge = edges[m].find(n)->second;

  	tree = remove(tree,m);

  	edges[ n ].erase (m);
    edges[ m ].erase (n);

    // merge edges

  	for (std::map<int,double>::iterator it=edges[n].begin(); it!=edges[n].end(); ++it)
    {
  		if( connections[ it->first ].partition == m || connections[ it->first ].partition == n )
  		{
  			connections[ it->first ].partition = -1;
  			dirty(tree,it->first);
  		}
  	}

  	communities[n].erase( m );

    for (std::map<int,double>::iterator it=edges[m].begin(); it!=edges[m].end(); ++it)
    {
    	if( connections[ it->first ].partition == m || connections[ it->first ].partition == n )
    	{
    		connections[ it->first ].partition = -1;
    		dirty(tree,it->first);
    	}

    	edges[it->first].erase( m );

    	std::map<int,double>::iterator edge = edges[n].find( it->first );

    	if( edge == edges[n].end() )
    	{
    		double value = it->second;
    		edges[n][it->first] = value;
    		edges[it->first][n] = value;
    	}
    	else
    	{
    		double value = it->second + edge->second;
    		edges[n][it->first] = value;
    		edges[it->first][n] = value;
    	}

    	communities[it->first].erase( m );
    }

    // update variables

    inner[n] = inner[n] + inner[m] + 2*edge;

    degree[n] = degree[n] + degree[m];


	connections[n].partition = -1;
	connections[n].cost      = -INFINITY;

	for (std::map<int,double>::iterator it=edges[n].begin(); it!=edges[n].end(); ++it)
	{
		double dcost = (inner[ it->first ] + inner[ n ] + 2*it->second)/(degree[ it->first ] + degree[ n ]) - inner[ it->first ]/degree[ it->first ] - inner[ n ]/degree[ n ];

		communities[n][it->first] = dcost;
		communities[it->first][n] = dcost;

		// update heap nodes with new maximum if needed

		if( connections[ it->first ].partition == -1 )
		{

			// scan
			for (std::map<int,double>::iterator itt=communities[it->first].begin(); itt!=communities[it->first].end(); ++itt)
			{
				if( connections[ it->first ].cost < itt->second || connections[ it->first ].partition == -1 )
				{
					connections[ it->first ].partition = itt->first;
			    	connections[ it->first ].cost      = itt->second;
				}
			}
		}
		else if( connections[ it->first ].cost < dcost )
		{
			connections[ it->first ].partition = n;
			connections[ it->first ].cost = dcost;

			dirty(tree,it->first);
		}

		if( connections[ n ].cost < dcost || connections[ n ].partition == -1 )
		{
			connections[ n ].partition = it->first;
	    	connections[ n ].cost      = dcost;
		}

		if(connections[it->first].partition == -1)
		{
			tree = remove(tree,it->first);
		}
	}

    if(connections[n].partition == -1)
  	{
  		tree = remove(tree,n);
  	}
    else
    {
      dirty(tree,n);
    }

	// record curvature
	mp[2*k]   = m;
	mp[2*k+1] = n;

	// record updates
	np[k] = nassoc;

    update_heap(tree);
  }

  for(int i=k; i<max_node-1; i++)
  {
	np[i] = np[i-1];
  }

  // record curvature

  cp[0] = 0 + 2*np[0] - np[1];

  for (int i=1; i<max_node-2; i++) {
    cp[i] = -np[i-1] + 2*np[i] - np[i+1];
  }

  cp[max_node-2] = 0;
  
t2=clock();
float diff ((float)t2-(float)t1);
cout << "time " << diff << " " << k << endl;

  return k;
}

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[] )
{
  // Check for proper number of arguments
  if (nrhs < 1) { 
    mexErrMsgTxt("GANC ERROR: Input matrix with edges listed is required."); 
  }
  else if (nlhs != 4) {
    mexErrMsgTxt("GANC ERROR: Output should be four variables."); 
  }

  // ==== Read Input ====

  double *w;
  unsigned int m,n;
  
  // edges
  m = mxGetM(prhs[0]);
  n = mxGetN(prhs[0]);
  w = mxGetPr(prhs[0]);

  // ==== Pre Processing ====

  // find number of nodes (i.e. the maximum)

  int min_node = INT_MAX;
  int max_node = INT_MIN;

  for(int i=0; i<m; i++)
  {
      int so = (int) w[i];
      int si = (int) w[i+m];

      max_node = max_node > so ? max_node : so;
      max_node = max_node > si ? max_node : si;

      min_node = min_node < so ? min_node : so;
      min_node = min_node < si ? min_node : si;
  }

  // Create a matrix for the return argument

  // get pointer to output matrix
  double* ms = (double*) malloc(sizeof(double) * (max_node-1)*2);// mxGetPr(plhs[0]);
  double* ns = (double*) malloc(sizeof(double) * (max_node-1));// mxGetPr(plhs[1]);
  double* cs = (double*) malloc(sizeof(double) * (max_node-1));// mxGetPr(plhs[2]);

  // Do the actual computations in a subroutine
  int len = agglomerate(ms,ns,cs,w,m,n,1,max_node);

  //for(int i=0;i<1000;i++)
  //{
	//		agglomerate(ms,ns,cs,w,m,n,1,max_node);
  //}
  
  plhs[0] = mxCreateDoubleMatrix(len, 2, mxREAL); 
  plhs[1] = mxCreateDoubleMatrix(len, 1, mxREAL); 
  plhs[2] = mxCreateDoubleMatrix(len, 1, mxREAL);
  plhs[3] = mxCreateDoubleScalar(max_node);

  double* mp = mxGetPr(plhs[0]);
  double* np = mxGetPr(plhs[1]);
  double* cp = mxGetPr(plhs[2]);

  for(int i=0; i<len; i++)
  {
    // merges
    mp[i]     = ms[2*i];
    mp[i+len] = ms[2*i+1];

    // norm. cut updates
    np[i] = ns[i];

    // curvature
    cp[i] = cs[i];
  }

  free(ms);
  free(ns);
  free(cs);

  return;
}
