function P = ganc(A,n)

	[merges,scores,curvature,maxnode] = agglomerate(A);

	if (nargin == 1)
		[value,n] = max(curvature);
	else
		n = maxnode - n;
	endif

	P = refine(A,merges,n);

endfunction