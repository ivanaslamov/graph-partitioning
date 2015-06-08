
// #include "mex.h"

#include <iostream>
#include <cmath>
#include <limits>
#include <climits>
#include <vector>
#include <map>
#include <list>
#include <set>
#include <algorithm>

#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <limits>
#include <climits>

#include "refine.h"

using namespace std;


// based on the code from http://kukuruku.co/hub/cpp/avl-trees
// and randomized binary search trees http://habrahabr.ru/post/145388/
// and random shuffle http://stackoverflow.com/questions/18993998/iterating-over-the-array-in-random-order


struct refine_community
{
	int    partition;
  int    node;
  double cost;
};

// ==== RANDOM BINARY TREE FUNCTIONS ====

struct node_theap
{
  // tree variables
  int key;
  int size;

  node_theap* left;
  node_theap* right;

  // heap variables 
  int heap_partition_out;
  int heap_partition_in;
  int heap_partition_node;
  double heap_maximum;

  // load
	refine_community* c;

  node_theap(int k, refine_community* load)
  {
    key = k;
    left = right = 0;
    size = 1;
    
    heap_partition_out = -1;
    heap_partition_in = -1;
    heap_partition_node = -1;
    heap_maximum = -1;

    c = load;
  }
};

int getsize_theap(node_theap* p) // wrapper for the size, can handle empty trees
{
  if( !p ) return 0; 
  return p->size; 
}

void fixsize_theap(node_theap* p) // set tree size
{
  p->size = getsize_theap(p->left)+getsize_theap(p->right)+1; 
}

node_theap* rotateright_theap(node_theap* p) // right turn around p
{
  node_theap* q = p->left;
  if( !q ) return p;
  p->left = q->right;
  q->right = p;
  q->size = p->size;
  fixsize_theap(p);
  return q;
}

node_theap* rotateleft_theap(node_theap* q) // left turn around p
{
  node_theap* p = q->right;
  if( !p ) return q;
  q->right = p->left;
  p->left = q;
  p->size = q->size;
  fixsize_theap(q);
  return p;
}

node_theap* insertroot_theap(node_theap* p, int k, refine_community* c) // insert new node at the top
{
    if( !p ) return new node_theap(k, c); 
    if( k<p->key ) 
    {
        p->left = insertroot_theap(p->left, k, c); 
        return rotateright_theap(p); 
    }
    else 
    {
        p->right = insertroot_theap(p->right, k, c);
        return rotateleft_theap(p);
    }
}

node_theap* insert_theap(node_theap* p, int k, refine_community* c) // insert k key in a tree with p root
{
    if( !p ) return new node_theap(k, c);
    if( rand()%(p->size+1)==0 )
        return insertroot_theap(p,k,c);
    if( p->key>k )
        p->left = insert_theap(p->left,k,c);
    else
        p->right = insert_theap(p->right,k,c);
    fixsize_theap(p);
    return p;
}

node_theap* join_theap(node_theap* p, node_theap* q) // join two trees
{
    if( !p ) return q;
    if( !q ) return p;
    if( rand()%(p->size+q->size) < p->size )
    {
      p->right = join_theap(p->right,q); 
      fixsize_theap(p);

  		p->heap_partition_out = -1;
  		p->heap_partition_in = -1;
      p->heap_partition_node = -1;
  		p->heap_maximum = -1;

      return p; 
    }
    else 
    {
      q->left = join_theap(p,q->left); 
      fixsize_theap(q); 

  		p->heap_partition_out = -1;
  		p->heap_partition_in = -1;
      p->heap_partition_node = -1;
  		p->heap_maximum = -1;

      return q; 
    }
}

node_theap* remove_theap(node_theap* p, int k) // removing first node with the same key
{
  if( !p ) return p;

	p->heap_partition_out = -1;
	p->heap_partition_in = -1;
  p->heap_partition_node = -1;
	p->heap_maximum = -1;

  if( p->key==k ) 
  {
      node_theap* q = join_theap(p->left,p->right); 
      delete p; 
      return q; 
  }
  else if( k<p->key )
      p->left = remove_theap(p->left,k); 
  else 
      p->right = remove_theap(p->right,k); 

  return p; 
}

// ==== GREEDY AGGLOMERATION ====

void dirty_theap(node_theap* p, int k)
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
    dirty_theap(p->left,k); 
  else
    dirty_theap(p->right,k);
}

double update_theap(node_theap* p)
{
  if( !p ) return -INFINITY;

  if( p->heap_partition_out > -1 ) return p->heap_maximum;
  
  double left_cost  = update_theap(p->left);
  double right_cost = update_theap(p->right);

  if(p->c->cost > 0 && p->c->cost > left_cost && p->c->cost > right_cost)
  {
    p->heap_partition_out  = p->key;
    p->heap_partition_in   = p->c->partition;
    p->heap_partition_node = p->c->node;
    p->heap_maximum        = p->c->cost;
  }
  else if( left_cost > 0 && left_cost > right_cost )
  {
    p->heap_partition_out  = p->left->heap_partition_out;
    p->heap_partition_in   = p->left->heap_partition_in;
    p->heap_partition_node = p->left->heap_partition_node;
    p->heap_maximum        = p->left->heap_maximum;
  }
  else if( right_cost > 0 )
  {
    p->heap_partition_out  = p->right->heap_partition_out;
    p->heap_partition_in   = p->right->heap_partition_in;
    p->heap_partition_node = p->right->heap_partition_node;
    p->heap_maximum        = p->right->heap_maximum;
  }

  return p->heap_maximum;
}

int refine( double *assignment, double *merges, int merges_rows, double *w, int m, int n, int w_order, int max_node, double capacity)
{
  // variables

  double epsilon = 1e-10;

  double total = 0; // total degree of a graph

  node_theap* tree = NULL;

  vector< map<int,double> > edges(max_node+1);

  vector<double> vertex_degree(max_node+1);

  vector<refine_community> connections(max_node+1);

  vector<double> degree(max_node+1);
  vector<double> inner(max_node+1);
  vector<double> weights(max_node+1);
  
  vector< map<int,set<int> > > partitions(max_node+1);
  vector< map<int,double> > edge_part(max_node+1);

  vector< map<int, pair<int,double> > > communities(max_node+1);

  vector<int> parts(max_node+1);
  
  // edges

  {
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
  }
  
  // vertex degrees

  {
    for(int i=0;i<edges.size();i++)
    {
      double weight = 0; 

      for (std::map<int,double>::iterator it=edges[i].begin(); it!=edges[i].end(); ++it)
      {
        weight += it->second;
      }

      vertex_degree[i] = weight;
      total += weight;
    }
  }
  
  // merges

  {
    vector< set<int> > verticies(max_node+1);

    for(int i=0;i<max_node+1;i++)
    {
      verticies[i].insert(i);
      weights[i] = 1;
    }

    for(int i=0;i<merges_rows;i++)
    {
      // w_order need to load merges from MATLAB

      int x = (int) w_order ? merges[i] : merges[2*i];
      int y = (int) w_order ? merges[i+merges_rows] : merges[2*i+1];

      verticies[y].insert(verticies[x].begin(), verticies[x].end() );
      verticies[x].clear();

      weights[y] += weights[x];
      weights[x] = 0;
    }

    // assignment

    for(int i=0;i<max_node+1;i++)
      if( verticies[i].size() > 0 )
        for (set<int>::iterator it=verticies[i].begin(); it!=verticies[i].end(); ++it)
          parts[ *it ] = (double) i;
  }

  // state variables

  {
    for(int i=0;i<max_node+1;i++)
    {
      degree[i] = 0;
      inner[i]  = 0;
    }

    for(int i=0;i<max_node+1;i++)
    {
      int partition = (int) parts[i];

      // init edge to home partition to zero
      edge_part[i][ partition ] = 0;

      if( edges[i].size() > 0 )
      {
        for (map<int, double>::iterator it=edges[i].begin(); it!=edges[i].end(); ++it)
        {
          if( partition != parts[ it->first ] )
            partitions[ partition ][ parts[ it->first ] ].insert(i);

          degree[ partition ] += it->second;

          if( partition == parts[ it->first ] )
            inner[ partition ] += it->second;

          map<int, double>::iterator itt = edge_part[i].find( parts[ it->first ] );

          if( itt == edge_part[i].end() )
            edge_part[i][ parts[ it->first ] ] = it->second;
          else
          {
            double value = it->second + itt->second;
            edge_part[i][ parts[ it->first ] ] = value;
          }
        }
      }
    }
  }

  // building an initial tree

  for(int i=0;i<partitions.size();i++)
  {
    for(map<int, set<int> >::iterator it=partitions[i].begin(); it!=partitions[i].end(); ++it)
    {
      int j = it->first;
      
      int node = -1;
      double max_cost = -INFINITY;

      if(i != j)
      {
        for (set<int>::iterator itt=it->second.begin(); itt!=it->second.end(); ++itt)
        {
          // node joes from j to i

          int u = *itt;

          double merged_score = edge_part[u][j];

          double remain_score = abs(degree[i] - vertex_degree[u]) < epsilon ? 0 : (-edge_part[u][i]);

          double cost = merged_score + remain_score;

          if( (max_cost < cost || node == -1) && (weights[j] + 1) <= capacity )
          {
            node = u;
            max_cost = cost;
          }
        }
      }

      if(node != -1)
        communities[i][j] = make_pair(node,max_cost);
    }
  }

  // heap construction

  {
    std::vector<int> indexes(max_node+1);

    for(int i=0;i<max_node+1;i++)
    {
      indexes[i] = i;
    }

    // shuffle connections

    random_shuffle(indexes.begin(),indexes.end());

    // initialize tree

    for(int i=0;i<indexes.size();i++)
    {
      connections[ indexes[i] ].partition = -1;
      connections[ indexes[i] ].node = -1;
      connections[ indexes[i] ].cost = -INFINITY;

      if( communities[ indexes[i] ].size() > 0 )
      {
        // insert elements into the tree
        
        for (map<int, pair<int,double> >::iterator it=communities[ indexes[i] ].begin(); it!=communities[ indexes[i] ].end(); ++it)
        {
          double dcost = it->second.second;

          if( dcost > connections[ indexes[i] ].cost || connections[ indexes[i] ].partition == -1 || connections[ indexes[i] ].node == -1 )
          {
            connections[ indexes[i] ].partition = it->first;
            connections[ indexes[i] ].node      = it->second.first;
            connections[ indexes[i] ].cost      = dcost;
          }
        }
      }

      tree = insert_theap(tree, indexes[i], &connections[ indexes[i] ]);
    }
  }

  // initialize heap

  update_theap(tree);

clock_t t1,t2;
t1=clock();

  int k;
  double nassoc = 0;

  for(k=0;tree;k++)
  {
    int m = tree->heap_partition_out;
    int n = tree->heap_partition_in;
    int v = tree->heap_partition_node;
    double c = tree->heap_maximum;

    if( c < epsilon )
      break;

    nassoc += tree->heap_maximum;

    parts[v] = (double) n;

    degree[m] -= vertex_degree[v];
    degree[n] += vertex_degree[v];

    inner[m] -= 2*edge_part[v][m];
    inner[n] += 2*edge_part[v][n];

    weights[m] -= 1;
    weights[n] += 1;

    // update edge_part
    // update partitions

    for (map<int, double>::iterator it=edges[v].begin(); it!=edges[v].end(); ++it)
    {
      double edge = edge_part[it->first][m];

      if( abs(edge - it->second) < epsilon )
      {
        edge_part[it->first].erase(m);
      }
      else
      {
        double value = edge - it->second;
        edge_part[it->first][m] = value;
      }

      map<int, double>::iterator itt = edge_part[it->first].find(n);

      if( itt == edge_part[it->first].end() )
      {
        edge_part[it->first][n] = it->second;
      }
      else
      {
        double value = it->second + itt->second;
        edge_part[it->first][n] = value;
      }

      // update

      partitions[m][parts[it->first]].erase(v);
      partitions[parts[it->first]][m].erase(it->first);

      if( n != parts[it->first] )
      {
        partitions[n][parts[it->first]].insert(v);
        partitions[parts[it->first]][n].insert(it->first);
      }
    }

    connections[m].partition = -1;
    connections[m].node = -1;
    connections[m].cost = -INFINITY;

    communities[m].clear();

    for (map<int, set<int> >::iterator it=partitions[m].begin(); it!=partitions[m].end(); ++it)
    {
      int i = m;
      int j = it->first;

      double node = -1;
      double max_cost = -INFINITY;

      for (set<int>::iterator itt=it->second.begin(); itt!=it->second.end(); ++itt)
      {
        // u goes from m to j

        int u = *itt;
      
        double merged_score = edge_part[u][j];

        double remain_score = abs(degree[i] - vertex_degree[u]) < epsilon ? 0 : (-edge_part[u][i]);

        double cost = merged_score + remain_score;

        if( (max_cost < cost || node == -1) && (weights[j] + 1) <= capacity )
        {
          node = u;
          max_cost = cost;
        }
      }

      if( node == -1 )
        communities[i].erase(j);
      else
        communities[i][j] = make_pair(node,max_cost);

      if( connections[m].cost < max_cost )
      {
        connections[m].partition = it->first;
        connections[m].node = node;
        connections[m].cost = max_cost;
      }

      i = it->first;
      j = m;

      node = -1;
      max_cost = -INFINITY;

      for (set<int>::iterator itt=partitions[ it->first ][m].begin(); itt!=partitions[ it->first ][m].end(); ++itt)
      {
        // u goes from i to m

        int u = *itt;

        double merged_score = edge_part[u][j];

        double remain_score = abs(degree[i] - vertex_degree[u]) < epsilon ? 0 : (-edge_part[u][i]);

        double cost = merged_score + remain_score;

        if( (max_cost < cost || node == -1) && (weights[j] + 1) <= capacity )
        {
          node = u;
          max_cost = cost;
        }
      }
      
      if( node == -1)
        communities[it->first].erase(m);
      else
        communities[it->first][m] = make_pair(node,max_cost);

      if( connections[it->first].cost < max_cost )
      {
        connections[it->first].partition = m;
        connections[it->first].node = node;
        connections[it->first].cost = max_cost;

        dirty_theap(tree,it->first);
      }
      else if( connections[it->first].partition == m )
      {
        connections[it->first].partition = -1;
        connections[it->first].node = -1;
        connections[it->first].cost = -1;

        for(map<int, pair<int,double> >::iterator itt=communities[ it->first ].begin(); itt!=communities[ it->first ].end(); ++itt)
        {
          if( connections[ it->first ].cost < itt->second.second || connections[ it->first ].partition == -1 || connections[ it->first ].node == -1 )
          {
            connections[ it->first ].partition = itt->first;
            connections[ it->first ].node      = itt->second.first;
            connections[ it->first ].cost      = itt->second.second;
          }
        }

        dirty_theap(tree,it->first);
      }
    }

    if(weights[m] < epsilon)
    {
      // check partition size
      tree = remove_theap(tree,m);
    }
    else
      dirty_theap(tree,m);

    connections[n].partition = -1;
    connections[n].node = -1;
    connections[n].cost = -INFINITY;

    communities[n].clear();

    for (map<int, set<int> >::iterator it=partitions[n].begin(); it!=partitions[n].end(); ++it)
    {
      int i = n;
      int j = it->first;

      double node = -1;
      double max_cost = -INFINITY;

      for (set<int>::iterator itt=it->second.begin(); itt!=it->second.end(); ++itt)
      {
        // u goes from n to j

        int u = *itt;

        double merged_score = edge_part[u][j];

        double remain_score = abs(degree[i] - vertex_degree[u]) < epsilon ? 0 : (-edge_part[u][i]);

        double cost = merged_score + remain_score;

        if( (max_cost < cost || node == -1) && (weights[j] + 1) <= capacity )
        {
          node = u;
          max_cost = cost;
        }
      }

      if( node == -1 )
        communities[i].erase(j);
      else
        communities[i][j] = make_pair(node,max_cost);

      if( connections[n].cost < max_cost || connections[n].partition == -1 || connections[n].node == -1 )
      {
        connections[n].partition = it->first;
        connections[n].node = node;
        connections[n].cost = max_cost;
      }

      i = it->first;
      j = n;

      node = -1;
      max_cost = -INFINITY;

      for (set<int>::iterator itt=partitions[ it->first ][n].begin(); itt!=partitions[ it->first ][n].end(); ++itt)
      {
        int u = *itt;

          double merged_score = edge_part[u][j];

          double remain_score = abs(degree[i] - vertex_degree[u]) < epsilon ? 0 : (-edge_part[u][i]);

          double cost = merged_score + remain_score;

        if( (max_cost < cost || node == -1) && (weights[j] + 1) <= capacity )
        {
          node = u;
          max_cost = cost;
        }
      }

      if( node == -1)
        communities[it->first].erase(n);
      else
        communities[it->first][n] = make_pair(node,max_cost);

      if( connections[it->first].cost < max_cost || connections[it->first].partition == -1 || connections[it->first].node == -1 )
      {
        connections[it->first].partition = n;
        connections[it->first].node = node;
        connections[it->first].cost = max_cost;

        dirty_theap(tree,it->first);
      }
      else if( connections[it->first].partition == n )
      {
        connections[it->first].partition = -1;
        connections[it->first].node = -1;
        connections[it->first].cost = -1;

        for(map<int, pair<int,double> >::iterator itt=communities[ it->first ].begin(); itt!=communities[ it->first ].end(); ++itt)
        {
          if( connections[ it->first ].cost < itt->second.second || connections[ it->first ].partition == -1 || connections[ it->first ].node == -1 )
          {
            connections[ it->first ].partition = itt->first;
            connections[ it->first ].node      = itt->second.first;
            connections[ it->first ].cost      = itt->second.second;
          }
        }

        dirty_theap(tree,it->first);
      }
    }

    dirty_theap(tree,n);

    update_theap(tree);
  }
  
t2=clock();
float diff ((float)t2-(float)t1);
cout << "time " << diff << " " << k << endl;

  for(int i=1;i<=max_node;i++)
  {
    assignment[i-1] = parts[i];
  }

  return 0;
}

/*
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[] ) 
{
  // Check for proper number of arguments

  if (nrhs < 1) { 
    mexErrMsgTxt("GANC ERROR: Input matrix with edges listed is required."); 
  }

  // ==== Read Input ====

  // edges

  double *w;
  unsigned int m,n;
  
  m = mxGetM(prhs[0]);
  n = mxGetN(prhs[0]);
  w = mxGetPr(prhs[0]);

  // merges

  double *merges;
  unsigned int merges_m;

  merges_m = mxGetM(prhs[1]);
  merges   = mxGetPr(prhs[1]);

  // refine after cutstep

  double capacity = (double) mxGetScalar(prhs[2]);

  // ==== Pre Processing ====

  // find number of nodes (i.e. the maximum)

  int min_node = INT_MAX;
  int max_node = INT_MIN;

  for(int i=0;i<m;i++)
  {
    int so = (int) w[i];
    int si = (int) w[i+m];

    max_node = max_node > so ? max_node : so;
    max_node = max_node > si ? max_node : si;

    min_node = min_node < so ? min_node : so;
    min_node = min_node < si ? min_node : si;
  }

  if (min_node != 1)
  { 
    mexErrMsgTxt("GANC ERROR: lowest vertex index != 1"); 
  }

  plhs[0] = mxCreateDoubleMatrix(max_node, 1, mxREAL); 
  
  double* parts = mxGetPr(plhs[0]);

  // Do the actual computations in a subroutine

  refine(parts,merges,merges_m,w,m,n,1,max_node,capacity);
}
*/
