//
//  main.c
//  StructureAndAlgorithms
//
//  Created by lunner on 8/21/15.
//
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stringMatch.h"
#include "Sort.h"


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

int main(int argc, int **argv) {
	char *str = "I fuck you deuckaddadadswafdasadasaadddaaddaaddaaddaadda";
	char *pattern = "adda";
	int *indices, *p;
	indices = KMP_matcher(str, pattern);
	if (indices) {
		p = indices;
		while (*p != -1) {
			printf("%d ", *p++);
		}
		free(indices);
	} else {
		printf("No matchs\n");
	}
	return 0;
	
}

