function P = scnc(W,capacity)

	A = agglomerate(W,capacity);

	P = refine(W,A,capacity);

endfunction