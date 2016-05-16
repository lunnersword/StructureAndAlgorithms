//
//  stringMatch.c
//  StructureAndAlgorithms
//
//  Created by lunner on 8/17/15.
//
//

#include "stringMatch.h"
#include <string.h>
#include <stdlib.h>

/*
 Algorithm		Preprocessing time			Matching time
 Naive				0					O((n-m+1)m)
 Rabin-Karp			ø(m)					O((n-m+1)m)
 Finite automaton		O(m|sum symbol|		ø(n)
 Knuth-Morris-Pratt		ø(m)					ø(n)
*/

Naive-String-Matcher(T, P) {
	n = T.length
	m = P.length
	for s in 0...n-m {
		if (P[1...m] == T[s+1...s+m]) {
			print "Pattern occurs with shift " s 
		}
	}
}
//C code
//#include <string.h>
int *naive_string_matcher(const char* str, const char *pattern) {
	int n, m, i;
	n = strlen(str);
	m = strlen(pattern);
	for (i = 0; i < n-m+1; i++) {
		if (strncmp(str+i, pattern, len2) == 0) {
			printf("Pattern occurs with index %d", i);
		}
	}
}


//KMP
//π[q] = max{k: k<q and Pk ] Pq}
Compute-Prefix-Function(P) {
	m = p.length
	let π[1...m] be a new array
	π[1] = 0
	k = 0
	for q in 2...m {
		while (k > 0 and P[k+1] != P[q]) {
			k = π[k]
		}
		if P[k+1] == P[q] {
			k = k+1
		}
		π[q] = k
	}
}

KMP-Matcher(T, P) {
	n = T.length
	m = P.length
	π = Compute-Prefix-Function(P)
	q = 0
	for i in 1...n {
		while (q>0 and P[q+1] != T[i]) {
			q = π[q]
		}
		if P[q+1] == T[i]
			q = q+1
		if q==m 
			print "Pattern occurs with shift" i-m
			q = π[q]
	}
}

int *pre_KMP(const char* str) {
	int len = strlen(str);
	int *next = (int *)malloc(sizeof(int)*(len));
	int k,q;
	next[0] = -1;
	//π[1] = 0;
	k = -1;
	for (q = 1; q < len; q++) {
		while (k>-1 && str[k+1] != str[q]) {
			k = next[k];
		}
		if (str[k+1] == str[q])
			k++;
		next[q] = k;
	}
	return next;
}

int *KMP_matcher(const char *str, const char *pattern) {
	int n = strlen(str);
	int m = strlen(pattern);
	int *next = pre_KMP(str);
	int *paired = (int*)malloc(sizeof(int)*(n+1));
	int *result = NULL;
	int i,q=-1, count = 0;
	for (i=0; i<n; i++) {
		while (q>-1 && pattern[q+1] != str[i]) {
			q = next[q];
		}
		if (pattern[q+1] == str[i]) {
			q++;
		}
		if (q==m-1) {
			paired[count++] = i-m+1;
			q = next[q];
		}
	}
	paired[count++] = -1;
	if (count > 0 && count < n+1) {
		result = (int*)malloc(count * sizeof(int));
		memcpy(result, paired, sizeof(int) * count);
		free(paired);
		return result;
		
	}
	free(paired);
	return result;
	
}
