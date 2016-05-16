///
//  Sort.swift
//  StructureAndAlgorithms
//
//  Created by lunner on 7/24/15.
//
//
// insert sort is an efficient algorithm for sorting a small number of elements
for j = 2 to A.length {
	key = A[j]
	i = j - 1
	while i > 0 and A[i]>key {
		A[i+1] = A[i]
		i = i - 1 
	}
	A[i+1]=key
}//O(n^2)

// insert-sort recursive
insert_sort_recursive(A, n) {
	insert_sort_recursive(A, n-1)
	i = n - 1
	key = A[n]
	while(i>0 && key < A[i]) {
		A[i+1] = A[i]
		i--
	}
	A[i+1] = key
}
//Binary search
Here's the presudocode:

BINARY-SEARCH(A, v):
	low = 1
	high = A.length

	while low <= high
		mid = (low + high) / 2
		if A[mid] == v
			return mid
		if A[mid] < v
			low = mid + 1
		else
			high = mid - 1

	return NIL
The argument is fairly straightforward and I will make it brief:

T(n+1)=T(n/2)+c
This is the recurrence shown in the chapter text, and we know, that this is logarithmic time.

C code
// Indices in the C code are different
int binary_search(int A[], int length, int v) {
	int low  = 0;
	int high = length;
	
	int mid;
	while (low < high) {
		mid = (low + high) / 2;
		
		if (A[mid] == v)
			return mid;
		else if (A[mid] < v)
			low = mid + 1;
		else
			high = mid;
	}
	
	return -1;
}//O(lgn) worst case
//end binary search
//Binary insertion sort

for j = 2 to A.length {
	key = A[j]
	low = 1, high = j-1 
	while(low <= high) {
		m = (low+high) / 2 
		if key < A[m] {
			high = m - 1
		} else {
			low = m + 1
		}
	}
	for (i = j-1; i >= high + 1; i--) {
		A[i+1] = A[i]
	}
	A[high+1] = key
}

//★ Describe a Θ(nlgn)-time algorithm that, given a set S of n integers and another integer x, determines whether or not there exists two elements of S whose sum is exactly x.

//merge(A, p, q, r)

merge(A, p, q, r) {
	n1 = q-p+1 
	n2 = r - q
	let L[1...n1+1] and R[1...n2+1] be new arrays
	for i = 1 to n1{
		L[i] = A[p+i-1]
	}
	for j = 1 to n2 {
		R[j] = A[q+j]
	}
	L[n1+1] = ∞
	R[n2+1] = ∞
	i = 1
	j = 1 
	for k = p to r {
		if L[i] <= R[j] {
			A[k] = L[i]
			i++
		} else {
			A[k] = R[j]
			j++
		}
	}
}

merge-sort(A, p, r) {
	if p < r {
		q = (p+r) / 2 
		merge-sort(A, p, q)
		merge-sort(A, q+1, r)
		merge(A, p, q, r)
	}
}//nlgn
//Problems 2.1
//Consider a modification to merge sort in which n=k sublists of length k are sorted using insertion sort and then merged using the standard merging mechanism, where k is a value to be determined.
//3. The largest value of k
//The largest value is k=lgn. If we substitute, we get:

//Θ(nlgn+nlgnlgn)=Θ(nlgn)
//If k=f(n)>lgn, the complexity will be Θ(nf(n)), which is larger running time than merge sort.
C code
#include <stdlib.h>
#include <string.h>

#define INSERTION_SORT_TRESHOLD 20
#define SELECTION_SORT_TRESHOLD 15

void merge(int A[], int p, int q, int r) {
	int i, j, k;
	
	int n1 = q - p + 1;
	int n2 = r - q;
	
#ifdef MERGE_HEAP_ALLOCATION
	int *L = calloc(n1, sizeof(int));
	int *R = calloc(n2, sizeof(int));
#else
	int L[n1];
	int R[n2];
#endif
	
	memcpy(L, A + p, n1 * sizeof(int));
	memcpy(R, A + q + 1, n2 * sizeof(int));
	
	for(i = 0, j = 0, k = p; k <= r; k++) {
		if (i == n1) {
			A[k] = R[j++];
		} else if (j == n2) {
			A[k] = L[i++];
		} else if (L[i] <= R[j]) {
			A[k] = L[i++];
		} else {
			A[k] = R[j++];
		}
	}
	
#ifdef MERGE_HEAP_ALLOCATION
	free(L);
	free(R);
#endif
}

void merge_sort(int A[], int p, int r) {
	if (p < r) {
		int q = (p + r) / 2;
		merge_sort(A, p, q);
		merge_sort(A, q + 1, r);
		merge(A, p, q, r);
	}
}


void insertion_sort(int A[], int p, int r) {
	int i, j, key;
	for (j = p + 1; j <= r; j++) {
		key = A[j];
		i = j - 1;
		while (i >= p && A[i] > key) {
			A[i + 1] = A[i];
			i = i - 1;
		}
		A[i + 1] = key;
	}
}

void selection_sort(int A[], int p, int r) {
	int min, temp;
	for (int i = p; i < r; i++) {
		min = i;
		for (int j = i + 1; j <= r; j++)
			if (A[j] < A[min])
				min = j;
		temp = A[i];
		A[i] = A[min];
		A[min] = temp;
	}
}

void mixed_sort_insertion(int A[], int p, int r) {
	if (p >= r) return;
	
	if (r - p < INSERTION_SORT_TRESHOLD) {
		insertion_sort(A, p, r);
	} else {
		int q = (p + r) / 2;
		mixed_sort_insertion(A, p, q);
		mixed_sort_insertion(A, q + 1, r);
		merge(A, p, q, r);
	}
}

void mixed_sort_selection(int A[], int p, int r) {
	if (p >= r) return;
	
	if (r - p < SELECTION_SORT_TRESHOLD) {
		selection_sort(A, p, r);
	} else {
		int q = (p + r) / 2;
		mixed_sort_selection(A, p, q);
		mixed_sort_selection(A, q + 1, r);
		merge(A, p, q, r);
	}
}
//Python code
from itertools import repeat

def insertion_sort(A, p, r):
	for j in range(p + 1, r + 1):
	key = A[j]
	i = j - 1
	while i >= p and A[i] > key:
		A[i + 1] = A[i]
		i = i - 1
	A[i + 1] = key

def merge(A, p, q, r):
	n1 = q - p + 1
	n2 = r - q

	L = list(repeat(None, n1))
	R = list(repeat(None, n2))

	for i in range(n1):
		L[i] = A[p + i]

	for j in range(n2):
		R[j] = A[q + j + 1]

	i = 0
	j = 0
	for k in range(p, r + 1):
		if i == n1:
			A[k] = R[j]
			j += 1
		elif j == n2:
			A[k] = L[i]
			i += 1
		elif L[i] <= R[j]:
			A[k] = L[i]
			i += 1
		else:
			A[k] = R[j]
			j += 1

def merge_sort(A, p, r):
	if p < r:
		q = int((p + r) / 2)
		merge_sort(A, p, q)
		merge_sort(A, q + 1, r)
		merge(A, p, q, r)

def mixed_sort(A, p, r):
	if p >= r: return

	if r - p < 20:
		insertion_sort(A, p, r)
	else:
		q = int((p + r) / 2)
		mixed_sort(A, p, q)
		mixed_sort(A, q + 1, r)
		merge(A, p, q, r)
//Give an algorithm that determines the number of inversions in any permutation on n elements in ‚.n lg n/ worst-case time. If i < j and A[i] > A[j], the pair (i, j) is an inversion of A. 
MERGE-SORT(A, p, r):
	if p < r
		inversions = 0
		q = (p + r) / 2
		inversions += merge_sort(A, p, q)
		inversions += merge_sort(A, q + 1, r)
		inversions += merge(A, p, q, r)
		return inversions
	else
		return 0

MERGE(A, p, q, r)
	n1 = q - p + 1
	n2 = r - q
	let L[1..n₁] and R[1..n₂] be new arrays
	for i = 1 to n₁
		L[i] = A[p + i - 1]
	for j = 1 to n₂
		R[j] = A[q + j]
	i = 1
	j = 1
	for k = p to r
		if i > n₁
			A[k] = R[j]
			j = j + 1
		else if j > n₂
			A[k] = L[i]
			i = i + 1
		else if L[i] ≤ R[j]
			A[k] = L[i]
			i = i + 1
		else
			A[k] = R[j]
			j = j + 1
			inversions += n₁ - i
	return inversions
//C code
#include <stdio.h>

int merge(int A[], int p, int q, int r) {
	int i, j, k, inversions = 0;
	
	int n1 = q - p + 1;
	int n2 = r - q;
	
	int L[n1];
	int R[n2];
	
	for (i = 0; i < n1; i++) L[i] = A[p + i];
	for (j = 0; j < n2; j++) R[j] = A[q + j + 1];
	
	for(i = 0, j = 0, k = p; k <= r; k++) {
		if (i == n1) {
			A[k] = R[j++];
		} else if (j == n2) {
			A[k] = L[i++];
		} else if (L[i] <= R[j]) {
			A[k] = L[i++];
		} else {
			A[k] = R[j++];
			inversions += n1 - i;
		}
	}
	
	return inversions;
}

int merge_sort(int A[], int p, int r) {
	if (p < r) {
		int inversions = 0;
		int q = (p + r) / 2;
		inversions += merge_sort(A, p, q);
		inversions += merge_sort(A, q + 1, r);
		inversions += merge(A, p, q, r);
		return inversions;
	} else {
		return 0;
	}
}


//bubble sort
for i = 1 to A.length - 1 {
	for j = A.length downto i+1 {
		if A[j] < A[j-1] {
			exchange A[j] with A[j-1]
		}
	}
}

void bubble_sort(int a[], int n) {
	for(i = n-1; change=true; i>=1&&change; --i) {
		change = false
		for(j = 0;j<i;++j) {
			if(a[j]>a[j+1]) {
				a[j] <-->a[j+1]
				change = true
			}
		}
	}
}
//x-1 < [x]_ <= [x]- < x+1
//for any integer n,
[n/2]_ + [n/2]- = n 

a mod n = a - n[a/n]_ 
0 ≤ a mod n < n 
(a mod n) = (b mod n) if and only if n is a divisor of b - a.

void bigNumberAdd(char *a, char *b, char *sum, int m, int n) {
	int jin = 0;
	int s = 0;
	int k = max(m,n);
	sum[0] = 0;
	for(int i = m-1, int j = n-1; i >= 0 && j >= 0, i--, j--) {
		s = (a[i] - '0') + (b[j] - '0') + jin;
		sum[k--] = s % 10;
		jin = s / 10;
	}
	while(jin&&i>=0) {
		s = a[i] + jin;
		sum[k--] = s % 10;
		jin = s / 10;
	}
}

// max subarray sum 
find_max_crossing_subarray(A, low, mid, high) {
	leftSum = -∞
	sum = 0
	for i = mid downto low {
		sum = sum + A[i]
		if sum > leftSum {
			leftSum = sum
			maxLeft = i 
		}
	}
	rightSum = -∞
	sum = 0
	for j = mid + 1 to high {
		sum = sum + A[j]
		if sum > rightSum {
			rightSum = sum
			maxRight = j 
		}
	}
	return (max_left, max_right, left_sum+right_sum)
}

find_maximum_subarray(A, low, high) {
	if high == low {
		return (low, high, A[low]) //base case
	}
	else {
		mid = (low+high)/2 
		(left_low, left_high, left_sum) = find_maximum_subarray(A, low, mid)
		(right_low, right_high, right_sum) = find_maximum_subarray(A, mid+1, high)
		(cross_low, cross_high, cross_sum) = find_max_crossing_subarray(A, low, mid, high)
		if left_sum >= right_sum && left_sum >= cross_sum {
			return (left_low, left_high, left_sum)
		}
		else if right_sum>=left_sum && right_sum >= cross_sum {
			return (right_low, right_high, right_sum)
		} else {
			return (cross_low, cross_high, cross_sum)
		}
	}
}
//Exercise 4.1.3
#include <limits.h>

#define CROSSOVER_POINT 37

// A struct to represent the tuple

typedef struct {
	unsigned left;
	unsigned right;
	int sum;
} max_subarray;

// The brute force approach

max_subarray find_maximum_subarray_brute(int A[], unsigned low, unsigned high) {
	max_subarray result = {0, 0, INT_MIN};
	
	for (int i = low; i < high; i++) {
		int current_sum = 0;
		for (int j = i; j < high; j++) {
			current_sum += A[j];
			if (result.sum < current_sum) {
				result.left = i;
				result.right = j + 1;
				result.sum = current_sum;
			}
		}
	}
	
	return result;
}

// The divide-and-conquer solution

max_subarray find_max_crossing_subarray(int A[], unsigned low, unsigned mid, unsigned high) {
	max_subarray result = {-1, -1, 0};
	int sum = 0,
	left_sum = INT_MIN,
	right_sum = INT_MIN;
	
	for (int i = mid - 1; i >= (int) low; i--) {
		sum += A[i];
		if (sum > left_sum) {
			left_sum = sum;
			result.left = i;
		}
	}
	
	sum = 0;
	
	for (int j = mid; j < high; j++) {
		sum += A[j];
		if (sum > right_sum) {
			right_sum = sum;
			result.right = j + 1;
		}
	}
	
	result.sum = left_sum + right_sum;
	return result;
}

max_subarray find_maximum_subarray(int A[], unsigned low, unsigned high) {
	if (high == low + 1) {
		max_subarray result = {low, high, A[low]};
		return result;
	} else {
		unsigned mid = (low + high) / 2;
		max_subarray left = find_maximum_subarray(A, low, mid);
		max_subarray right = find_maximum_subarray(A, mid, high);
		max_subarray cross = find_max_crossing_subarray(A, low, mid, high);
		
		if (left.sum >= right.sum && left.sum >= cross.sum) {
			return left;
		} else if (right.sum >= left.sum && right.sum >= cross.sum) {
			return right;
		} else {
			return cross;
		}
	}
}

// The mixed algorithm

max_subarray find_maximum_subarray_mixed(int A[], unsigned low, unsigned high) {
	if (high - low < CROSSOVER_POINT) {
		return find_maximum_subarray_brute(A, low, high);
	} else {
		unsigned mid = (low + high) / 2;
		max_subarray left = find_maximum_subarray_mixed(A, low, mid);
		max_subarray right = find_maximum_subarray_mixed(A, mid, high);
		max_subarray cross = find_max_crossing_subarray(A, low, mid, high);
		
		if (left.sum >= right.sum && left.sum >= cross.sum) {
			return left;
		} else if (right.sum >= left.sum && right.sum >= cross.sum) {
			return right;
		} else {
			return cross;
		}
	}
}
//end exercise 4.1.3
//Exercise 4.1.5



// matrix multiply
squareMatrixMultiply(A, B) {
	n = A.rows
	let C be a new n*n matrix
		for i in 1...n {
		for j in 1...n {
			Cij = 0
			for k in 1...n {
				Cij = Cij + Aik * Bkj
			}
		}
	}
	return C
}

squareMatrixMultiplyRecursive(A, B) {
	n = A.rows
	let C be a new n*n matrix
	if n==1 {
		C11 = A11*B11
	}
	else {
		partition A, B, and C as in equations (4.9)
		C11 = squareMatrixMultiply(A11, B11) + 
		squareMatrixMultiply(A12, B21)
		C12 = squareMatrixMultiply(A11, B12) + 
		squareMatrixMultiply(A12, B22)
		C21 = squareMatrixMultiply(A21, B11) +
		squareMatrixMultiply(A22, B21)
		C22 = squareMatrixMultiply(A21, B12) +
		squareMatrixMultiply(A22, B22)
	}
}

//heap sort 
parent(i)
	return i/2 
left(i)
	return 2*i 
right(i)
	return 2*i+1 
max_heapify(A, i) {
//除i外， i的左右子树都已是大堆
	l = left(i)
	r = right(i)
	if l<=A.heap_size and A[l] > A[i]
		largest = l
	else 
		largest = i 
	if r <= A.heap_size and A[r] > A[largest]
		largest = r 
	if largest != i 
		exchange A[i] with A[largest]
		max_heapify(A, largest)
}
//The code for MAX-HEAPIFY is quite efficient in terms of constant factors, except possibly for the recursive call in line 10, which might cause some compilers to produce inefficient code. Write an efficient MAX-HEAPIFY that uses an iterative control construct (a loop) instead of recursion.

//As always, the most fun was converting from 1- to 0-based indexing.

//C code
#define PARENT(i) ((i - 1) / 2)
#define LEFT(i)   (2 * i + 1)
#define RIGHT(i)  (2 * i + 2)

typedef struct {
	int *nodes;
	int length;
	int heap_size;
} heap;

void max_heapify(heap A, int i) {
	int left, right, largest, temp;
	
	while(1) {
		left  = LEFT(i);
		right = RIGHT(i);
		
		if (left < A.heap_size && A.nodes[left] > A.nodes[i])
			largest = left;
		else
			largest = i;
		
		if (right < A.heap_size && A.nodes[right] > A.nodes[largest])
			largest = right;
		
		if (largest == i)
			return;
		
		temp = A.nodes[i];
		A.nodes[i] = A.nodes[largest];
		A.nodes[largest] = temp;
		
		i = largest;
	}
}

//max_heapify no recursion
max_heapify(A, s) {
	rc = A[s]
	for(j = left(s); j <= A.heap_size; j*=2) {
		if (j<A.heap_size && A[j] < A[j+1]) j++
		if !(rc<A[j]) break
		A[s] = A[j]
		s = j 
	}
	A[s] = rc
}

build_max_heap(A) {
	A.heap_size = A.length
	for i = A.length/2 downto 1 {
		max_heapify(A, i)
	}
}// linear time

heapsort(A) {
	build_max_heap(A)//O(n)
	for i = A.length downto 2 {//n-1
		exchange A[1] with A[i]
		A.heap_size = A.heap_size-1 
		max_heapify(A, 1)//O(lgn)
	}
}//O(nlgn)


//priority queue  A min-priority queue can be used in an event-driven simulator. The items in the queue are events to be simulated, each with an associated time of occurrence that serves as its key.
heap_maximum(A)	
	return A[1]

heap_extract_max(A) {
	if A.heap_size < 1 
		error "heap underflow"
	max = A[1]
	A[1] = A[A.heap_size]
	A.heap_size--
	max_heapify(A, 1)
	return max
}//O(lgn)
heap_delete(A, i):
	A[i] = A[A.heap_size]
	A.heap_size -= 1
	max_heapify(A, i)
//O(lgn)

heap_increase_key(A, i, key) {
	if key < A[i]
		error "new key is smaller than current key"
	A[i] = key
	while i > 1 and A[parent(i)] < A[i]
		exchange A[i] with A[Parent(i)]
		i = parent(i)
}//O(lgn)

max_heap_insert(A, key) {
	A.heap_size = A.heap_size+1 
	A[A.heap_size] = -∞
	heap_increase_key(A, A.heap_size, key)
}

build_max_heap'(A) {
	A.heap_size = 1 
	for i = 2 to A.length
		max_heap_insert(A, A[i])
}

//d-ary heaps 
//A d-ary heaps is like a binary heap, but non-leaf nodes have d children instead of 2 children.
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define PARENT(i,d) ((i - 1) / d)
#define CHILD(i,c,d) (d * i + c + 1)

typedef struct {
	int *elements;
	int d;
	int heap_size;
	int length;
} heap_t;

void max_heapify(heap_t *heap, int i) {
	int largest = i;
	
	for (int k = 0; k < heap->d; k++) {
		int child = CHILD(i, k, heap->d);
		if (child < heap->heap_size && heap->elements[child] > heap->elements[largest])
			largest = child;
	}
	
	if (largest != i) {
		int tmp = heap->elements[i];
		heap->elements[i] = heap->elements[largest];
		heap->elements[largest] = tmp;
		
		max_heapify(heap, largest);
	}
}

int extract_max(heap_t *heap) {
	int max = heap->elements[0];
	heap->elements[0] = heap->elements[heap->heap_size - 1];
	heap->heap_size--;
	max_heapify(heap, 0);
	return max;
};

void increase_key(heap_t *heap, int i, int key) {
	if (key < heap->elements[i]) {
		exit(0);
		fprintf(stderr, "new key is smaller than current key\n");
	}
	
	while (i > 0 && heap->elements[PARENT(i,heap->d)] < key) {
		heap->elements[i] = heap->elements[PARENT(i,heap->d)];
		i = PARENT(i,heap->d);
	}
	
	heap->elements[i] = key;
}

void insert(heap_t *heap, int key) {
	heap->heap_size++;
	heap->elements[heap->heap_size - 1] = INT_MIN;
	increase_key(heap, heap->heap_size - 1, key);
}
//end d-ary heaps
//Young tableaus
//An m×n Young tableau is an m×n matrix such that the entries of each row are in sorted order from left to right and the entries of each column are in sorted order from top to bottom. Some of the entries of a Young tableau may be ∞, which we treat as nonexistent elements. Thus, a Young tableau can be used to hold r≤mn finite numbers.
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>

typedef struct {
	int i;
	int j;
} cell;

typedef struct {
	int *elements;
	int m;
	int n;
} tableau_t;

cell up(cell c) {
	cell result = {c.i - 1, c.j};
	return result;
}

cell down(cell c) {
	cell result = {c.i + 1, c.j};
	return result;
}

cell left(cell c) {
	cell result = {c.i, c.j - 1};
	return result;
}

cell right(cell c) {
	cell result = {c.i, c.j + 1};
	return result;
}

cell make_cell(int i, int j) {
	cell result = {i, j};
	return result;
}

bool within(tableau_t *tableau, cell c) {
	return (c.i >= 0 && c.j >= 0 && c.i < tableau->m && c.j < tableau->n);
}

int get(tableau_t *tableau, cell c) {
	int index = c.i * tableau->n + c.j;
	return tableau->elements[index];
}

void set(tableau_t *tableau, cell c, int value) {
	int index = c.i * tableau->n + c.j;
	tableau->elements[index] = value;
}

void init_empty_tableau(tableau_t *tableau) {
	for (int i = 0; i < tableau->m * tableau-> n; i++) {
		tableau->elements[i] = INT_MAX;
	}
}

int extract_min(tableau_t *tableau) {
	int min, new;
	cell current = {0, 0},
	next;
	
	new = INT_MAX;
	min = get(tableau, current);
	
	set(tableau, current, INT_MAX);
	
	while (true) {
		int smallest;
		cell d = down(current);
		cell r = right(current);
		
		if (within(tableau, d) && get(tableau, d) < new) {
			next = d;
			smallest = get(tableau, next);
		} else {
			smallest = new;
		}
		
		if (within(tableau, r) && get(tableau, r) < smallest) {
			next = r;
			smallest = get(tableau, next);
		}
		
		if (new == smallest) {
			set(tableau, current, new);
			break;
		}
		
		set(tableau, current, smallest);
		current = next;
	}
	
	return min;
}

void insert(tableau_t *tableau, int key) {
	cell current = make_cell(tableau->m - 1, tableau->n - 1),
	next;
	
	if (get(tableau, current) != INT_MAX) {
		fprintf(stderr, "tableau is full\n");
		exit(0);
	}
	
	while (true) {
		int largest;
		cell u = up(current);
		cell l = left(current);
		
		if (within(tableau, u) && get(tableau, u) > key) {
			next = u;
			largest = get(tableau, next);
		} else {
			largest = key;
		}
		
		if (within(tableau, l) && get(tableau, l) > largest) {
			next = l;
			largest = get(tableau, next);
		}
		
		if (key == largest) {
			set(tableau, current, key);
			break;
		}
		
		set(tableau, current, largest);
		current = next;
	}
}

void sort(int *array, int size_sqrt) {
	int elements[size_sqrt * size_sqrt];
	tableau_t tableau = {elements, size_sqrt, size_sqrt};
	
	init_empty_tableau(&tableau);
	
	for (int i = 0; i < size_sqrt * size_sqrt; i++) {
		insert(&tableau, array[i]);
	}
	
	for (int i = 0; i < size_sqrt * size_sqrt; i++) {
		int next = extract_min(&tableau);
		array[i] = next;
	}
}

bool find(tableau_t *tableau, int key) {
	cell c = {tableau->m - 1, 0};
	
	while (within(tableau, c)) {
		int value = get(tableau, c);
		
		if (value == key) {
			return true;
		} else if (value > key) {
			c = up(c);
		} else {
			c = right(c);
		}
	}
	
	return false;
}
//end young tableaus
//quick sort 
//quick sort like merge sort, applies the divide-and-conquer paradigm
/*
Divide: Partition (rearrange) the array A[p...r] into two (possibly empty) subarrays A[p...q-1] and A[q+1...r] such that each element of A[p...q-1] is less than or equal to A[q] , which is, in turn, less than each element of A[q+1...r]. Compute the index q as part of this partitioning procedure.
Conquer: Sort the two subarrays A[p...q-1]andA[q+1...r] by recursive calls to quicksort.
QUICKSORT(A; p; r)
1 ifp<r
2 q = PARTITION(A,p,r)
3 QUICKSORT(A, p, q-1)
4 QUICKSORT(A, q+1, r)
*/
partition(A, p, r) {
//rearrange the array A in place
	x = A[r]
	i = p - 1 
	for j = p...r-1 {
		if A[j] <= x {
			i++
			exchange A[i] with A[j]
		}
	}
	exchange A[i+1] with A[r]
	return i+1 
}
randomized_paritition(A, p, r) {
	i = random(P, r)
	exchange A[r] with A[i]
	return partition(A, p, r)
}
randomized_quicksort(A, p, r) {
	if p < r {
		q = randomized_paritition(A, p, r)
		randomized_quicksort(A, p, q-1)
		randomized_quicksort(A, q+1, r)
	}
}



//We can improve the running time of quicksort in practice by taking advantage of the fast running time of insertion sort when its input is “nearly” sorted. Upon calling quicksort on a subarray with fewer than k elements, let it simply return without sorting the subarray. After the top-level call to quicksort returns, run insertion sort on the entire array to finish the sorting process. Argue that this sorting algorithm runs in O(nk+nlg(n/k)) expected time. How should we pick k, both in theory and in practice?
#define K 550
int partition(int[], int, int);
void limited_quicksort(int[], int, int, int);
void insertion_sort(int[], int, int);

void quicksort(int A[], int p, int r) {
	if (p < r - 1) {
	int q = partition(A, p, r);
		quicksort(A, p, q);
		quicksort(A, q + 1, r);
	}
}

void modified_quicksort(int A[], int p, int r) {
	limited_quicksort(A, p, r, K);
	insertion_sort(A, p, r);
}

void limited_quicksort(int A[], int p, int r, int treshold) {
	if (r - p > treshold) {
		int q = partition(A, p, r);
		limited_quicksort(A, p, q, treshold);
		limited_quicksort(A, q + 1, r, treshold);
	}
}

int partition(int A[], int p, int r) {
	int x, i, j, tmp;

	x = A[r - 1];
	i = p;

	for (j = p; j < r - 1; j++) {
		if (A[j] <= x) {
			tmp = A[i];
			A[i] = A[j];
			A[j] = tmp;
			i++;
		}
	}

	tmp = A[i];
	A[i] = A[r - 1];
	A[r - 1] = tmp;

	return i;
}

void insertion_sort(int A[], int p, int r) {
	int i, j, key;

	for (j = p + 1; j < r; j++) {
		key = A[j];
		for (i = j - 1; i >= p && A[i] > key; i--) {
			A[i + 1] = A[i];
		}
		A[i + 1] = key;
	}
}



hoare_partition(A, p, r) {
	x = A[p]
	i = p - 1 
	j = r + 1 
	while True {
		repeat
			j = j-1 
		until A[j] <= x
		repeat
			i = i+1 
		until A[i] >= x
		if i < j 
			exchange A[i] with A[j]
		else 
			return j 
	}
}
//Code
#include <stdbool.h>

int hoare_partition(int A[], int p, int r) {
	int x = A[p],
	i = p - 1,
	j = r,
	tmp;
	
	while(true) {
		do { j--; } while (!(A[j] <= x));
		do { i++; } while (!(A[i] >= x));
		
		if (i < j) {
			tmp = A[i]; A[i] = A[j]; A[j] = tmp;
		} else {
			return j;
		}
	}
}

void quicksort(int A[], int p, int r) {
	if (p < r - 1) {
		int q = hoare_partition(A, p, r);
		quicksort(A, p, q + 1);
		quicksort(A, q + 1, r);
	}
}

tail_recursive_quicksort(A, p, r) {
	while p < r {
		q = partition(A, p, r)
		tail_recursive_quicksort(A, p, q-1)
		p = q+1 
	}
}

//problem 7.2
#include <stdlib.h>

#define EXCHANGE(a, b) tmp = a; a = b; b = tmp;

typedef struct {
	int q;
	int t;
} pivot_t;

pivot_t partition(int[], int, int);
pivot_t randomized_partition(int[], int, int);

void quicksort(int A[], int p, int r) {
	if (p < r - 1) {
		pivot_t pivot = randomized_partition(A, p, r);
		quicksort(A, p, pivot.q);
		quicksort(A, pivot.t, r);
	}
}

pivot_t randomized_partition(int A[], int p, int r) {
	int i = rand() % (r - p) + p,
	tmp;
	
	EXCHANGE(A[i], A[r-1]);
	
	return partition(A, p, r);
}

pivot_t partition(int A[], int p, int r) {
	int x = A[r - 1],
	q = p,
	t,
	tmp;
	
	for (int i = p; i < r - 1; i++) {
		if (A[i] < x) {
			EXCHANGE(A[q], A[i]);
			q++;
		}
	}
	
	for (t = q; t < r && A[t] == x; t++);
	
	for (int i = r - 1; i >= t; i--) {
		if (A[i] == x) {
			EXCHANGE(A[t], A[i]);
			t++;
		}
	}
	
	pivot_t result = {q, t};
	return result;
}
//end problem 7.2
///C code
//We are always doing a tail-recursive call on the second partition. We can modify the algorithm to do the tail recursion on the larger partition. That way, we'll consume less stack.
//C code
#include <stdio.h>

int partition(int[], int, int);

static int stack_depth = 0;
static int max_stack_depth = 0;

void reset_stack_depth_counter();
void increment_stack_depth();
void decrement_stack_depth();

void tail_recursive_quicksort(int A[], int p, int r) {
	increment_stack_depth();
	
	while (p < r - 1) {
		int q = partition(A, p, r);
		
		if (q < (p + r) / 2) {
			tail_recursive_quicksort(A, p, q);
			p = q;
		} else {
			tail_recursive_quicksort(A, q + 1, r);
			r = q;
		}
	}
	
	decrement_stack_depth();
}

int partition(int A[], int p, int r) {
	int x, i, j, tmp;
	
	x = A[r - 1];
	i = p;
	
	for (j = p; j < r - 1; j++) {
		if (A[j] <= x) {
			tmp = A[i]; A[i] = A[j]; A[j] = tmp;
			i++;
		}
	}
	
	tmp = A[i]; A[i] = A[r - 1]; A[r - 1] = tmp;
	
	return i;
}

void increment_stack_depth() {
	stack_depth++;
	if (max_stack_depth < stack_depth) {
		max_stack_depth = stack_depth;
	}
}

void decrement_stack_depth() {
	stack_depth--;
}

void reset_stack_depth_counter() {
	max_stack_depth = 0;
	stack_depth = 0;
}


//problem 7.6
//fuzzy sorting of intervals

#include <stdbool.h>
#include <stdlib.h>

typedef struct {
	int left;
	int right;
} interval;

bool intersects(interval a, interval b) { return a.left <= b.right && b.left <= a.right; }
bool before(interval a, interval b)     { return a.right < b.left; }
bool after(interval a, interval b)      { return a.left > b.right; }

#define EXCHANGE(a, b) tmp = a; a = b; b = tmp;

interval partition(interval A[], int p, int r) {
	int pick, s, t, i;
	interval intersection, tmp;
	
	// Pick a random interval as a pivot
	pick = p + rand() % (r - p);
	EXCHANGE(A[pick], A[r-1]);
	intersection = A[r-1];
	
	// Find an intersection of the pivot and other intervals
	for (i = p; i < r - 1; i++) {
		if (intersects(intersection, A[i])) {
			if (A[i].left > intersection.left)
				intersection.left = A[i].left;
			if (A[i].right < intersection.right)
				intersection.right = A[i].right;
		}
	}
	
	// Classic partition around the intersection
	for (i = s = p; i < r - 1; i++) {
		if (before(A[i], intersection)) {
			EXCHANGE(A[i], A[s]);
			s++;
		}
	}
	EXCHANGE(A[r-1], A[s]);
	
	// Group intervals including the intersection
	for (t = s + 1, i = r - 1; t <= i;) {
		if (intersects(A[i], intersection)) {
			EXCHANGE(A[t], A[i]);
			t++;
		} else {
			i--;
		}
	}
	
	return (interval) {s, t};
}

void fuzzy_sort(interval array[], int p, int r) {
	if (p < r - 1) {
		interval pivot = partition(array, p, r);
		fuzzy_sort(array, p, pivot.left);
		fuzzy_sort(array, pivot.right, r);
	}
}
//end problem 7.6	 
	 
// Counting sort
	 counting_sort(A, B, k) {
		 let C[0...k] be a new array 
		 for i in 0...k {
			 C[i] = 0
		 }
		 for j in 1...A.length {
			 C[A[j]] = C[A[j]] + 1
		 }
		 //C[i] now contains the number of elements equal to i.
		 for i in 1...k {
			 C[i] = C[i] + C[i-1]
			 //C[i] now contains the number of elements less than or equal to i.
		 }
		 for j in A.length...1 {
			 B[C[A[j]]] = A[j]
			 C[A[j]] = C[A[j]] - 1
		 }
	 }//O(n+k)
	 //Describe an algorithm that, given n integers in the range 0 to k, preprocesses its input and then answers any query about how many of the n integers fall into a range [a..b] in O(1) time. Your algorithm should use Θ(n+k) preprocessing time.
	 
	// This is not even challenging.
	 
	// We just take the part of COUNTING-SORT that builds up the array C. Whenever we want to count the number of integers in [a..b], we take C[b] - C[a-1] (where C[-1] = 0). This yields the number of integers in the given range.
	//Θ(n+k)地求一组数据中属于[a,b]的个数

	
	//Radix sort
	 //Given n d-digit numbers in which each digit can take on up to k possible values, RADIX-SORT correctly sorts these numbers in Θ(d(n+k)) time if the stable sort it uses takes Θ(n+k) time.
	 radix_sort(A, d) {
		 for i in 1...d {
			 use a stable sort to sort array A on digit i 
		 }
	 }

	 //Which of the following sorting algorithms are stable: insertion sort, merge sort, heapsort, and quicksort? Give a simple scheme that makes any sorting algorithm stable. How much additional time and space does your scheme entail?
	 
	 //Stable: Insertion sort, merge sort
	 
	// Not stable: Heapsort, quicksort
	 
	// We can make any algorithm stable by mapping the array to an array of pairs, where the first element in each pair is the original element and the second is its index. Then we sort lexicographically. This scheme takes additional Θ(n) space.
	
	 //Show how to sort n integers in the range 0 to n3−1 in O(n) time.
	 
	 //We use radix sort. In this case, we have 2-digit numbers in base n. This makes RADIX-SORT to be Θ(2(n+n))=Θ(4n)=Θ(n).


	//Bucket sort
	 bucket_sort(A) {
		 let B[0...n-1] be a new array 
		 n = A.length
		 for i in 0...n-1 {
			 make B[i] an empty list
		 }
		 for i in 1...n {
 			inset A[i] into list B[int(nA[i])]
		 }
		 for i = 0 to n-1 {
			 sort list B[i] with insertion sort 
		 }
		 concatenate the lists B[0],B[1],...,B[n-1] together in order
	 }
	 
//problem 8-2 Sorting in place in linear time
#include <stdbool.h>
	 
	 typedef struct {
		 int key;
		 int value;
	 } item;
	 
	 static item tmp;
	 
#define EXCHANGE(a, b) tmp = a; a = b; b = tmp;
	 
	 void stable_linear_sort(item *A, int size) {
		 int zero = 0,
		 one  = 0;
		 item copy[size];
		 
		 for (int i = 0; i < size; i++) {
			 if (A[i].key == 0) {
				 one++;
			 }
		 }
		 
		 for (int i = 0; i < size; i++) {
			 if (A[i].key == 0) {
				 copy[zero] = A[i];
				 zero++;
			 } else {
				 copy[one] = A[i];
				 one++;
			 }
		 }
		 
		 for (int i = 0; i < size; i++) {
			 A[i] = copy[i];
		 }
	 }
	 
	 void linear_in_place_sort(item *A, int size) {
		 int left = -1,
		 right = size;
		 
		 while (true) {
			 do { left++;  } while (A[left].key  == 0);
			 do { right--; } while (A[right].key == 1);
			 
			 if (left > right) {
				 return;
			 }
			 
			 EXCHANGE(A[left], A[right]);
		 }
	 }
	 
	 void stable_in_place_sort(item *A, int size) {
		 for (int i = size; i > 0; i--) {
			 for (int j = 0; j < i; j++) {
				 if (A[j].key > A[j + 1].key) {
					 EXCHANGE(A[j], A[j+1]);
				 }
			 }
		 }
	 }
	 
	 void in_place_counting_sort(item *A, int size, int range) {
		 int counts[range + 1];
		 int positions[range + 1];
		 
		 for (int i = 0; i <= range; i++) {
			 counts[i] = 0;
		 }
		 
		 for (int i = 0; i < size; i++) {
			 counts[A[i].key]++;
		 }
		 
		 for (int i = 2; i <= range; i++) {
			 counts[i] += counts[i-1];
		 }
		 
		 for (int i = 0; i <= range; i++) {
			 positions[i] = counts[i];
		 }
		 
		 int i = 0;
		 while (i < size) {
			 int key = A[i].key;
			 bool placed = (positions[key - 1] <= i && i < positions[key]);
			 
			 if (placed) {
				 i++;
			 } else {
				 EXCHANGE(A[i], A[counts[key] - 1]);
				 counts[key]--;
			 }
		 }
	 }
	 
//problem 8.3
include <math.h>
#include <string.h>

#define MAX_LENGTH 10

// --- Structs and typedefs ---------------------------------------------------

// In order to simplify everything, both numbers and strings are meshed in a
// single union called key_t. The key does not know whether it is a number or a
// string - the handling code already knows it instead.

union key_t {
	int number;
	char string[MAX_LENGTH + 1];
};

typedef struct {
	union key_t key;
	int value;
} item;

typedef int (*key_f)(item, int);
typedef int (*dimension_f)(item);
typedef int (*compare_f)(item, item);

// --- Prototypes -------------------------------------------------------------

// Various sorting functinos

void partition(item *A, int size, int digits, int *groups, dimension_f dimension);
void radix_sort(item *A, int left, int right, int digits, key_f key);
void counting_sort(item *A, int left, int right, int dimension, key_f key, int key_index);

// Functions to work on numbers

int item_nth_digit(item i, int d);
int item_digits(item i);

// Functions to work on strings

int item_string_length(item i);
int item_nth_char(item i, int d);

// --- The solutions ----------------------------------------------------------

void sort_numbers(item *A, int size, int max_digits) {
	int groups[max_digits + 1];
	
	partition(A, size, max_digits, groups, item_digits);
	
	for (int i = 1; i < max_digits + 1; i++) {
		radix_sort(A, groups[i - 1], groups[i], i, item_nth_digit);
	}
}

void sort_strings(item *A, int size, int max_length) {
	int groups[max_length + 1];
	
	partition(A, size, max_length, groups, item_string_length);
	
	for (int len = max_length; len > 0; len--) {
		counting_sort(A, groups[len - 1], size, 26, item_nth_char, len - 1);
	}
}

// --- Auxiliary sorting functions --------------------------------------------

// Performs counting sort on a dimension (number of digits or string length)
// and populates a table (groups) with the position of each dimension.

void partition(item *A, int size, int max_dimension, int *groups, dimension_f dimension) {
	int counts[max_dimension + 1];
	item temp[size];
	
	for (int i = 0; i < max_dimension + 1; i++) { groups[i] = 0; }
	for (int i = 0; i < size;              i++) { groups[dimension(A[i])]++; }
	for (int i = 1; i < max_dimension + 1; i++) { groups[i] += groups[i - 1]; }
	for (int i = 0; i < max_dimension + 1; i++) { counts[i] = groups[i]; }
	for (int i = 0; i < size;              i++) { temp[i] = A[i]; }
	
	for (int i = size - 1; i >= 0; i--) {
		int d = dimension(temp[i]);
		int count = counts[d];
		
		A[count - 1] = temp[i];
		counts[d]--;
	}
}

// A simple radix sort

void radix_sort(item *A, int left, int right, int digits, key_f key) {
	for (int i = 0; i < digits; i++) {
		counting_sort(A, left, right, 10, key, i);
	}
}

// A slightly generalized counting sort

void counting_sort(item *A, int left, int right, int dimension, key_f key, int key_index) {
	int size = right - left;
	int counts[dimension];
	item temp[size];
	
	for (int i = 0;    i < dimension; i++) { counts[i] = 0; }
	for (int i = left; i < right;     i++) { counts[key(A[i], key_index)]++; }
	for (int i = 1;    i < dimension; i++) { counts[i] += counts[i - 1]; }
	for (int i = 0;    i < size;      i++) { temp[i] = A[left + i]; }
	
	for (int i = size - 1; i >= 0; i--) {
		int n = key(temp[i], key_index);
		int count = counts[n];
		
		A[left + count - 1] = temp[i];
		counts[n]--;
	}
}

// --- Key handling -----------------------------------------------------------

int count_digits(int n) {
	if (n == 0) {
		return 1;
	} else {
		return (int) log10(n) + 1;
	}
}

int nth_digit(int n, int d) {
	int magnitude = (int) pow(10, d);
	
	return (n / magnitude) % 10;
}

int item_nth_digit(item i, int d) {
	return nth_digit(i.key.number, d);
}

int item_digits(item i) {
	return count_digits(i.key.number);
}

int item_string_length(item a) {
	return strlen(a.key.string);
}

int item_nth_char(item a, int n) {
	return a.key.string[n] - 'a';
}
//end problem 8.3
//problem 8.4
#include <stdlib.h>

typedef int jug;

static int tmp;
#define EXCHANGE(a, b) {tmp = a; a = b; b = tmp;}

int cmp(jug red, jug blue);

void quadratic_pair(jug *red, jug *blue, int size) {
	for (int i = 0; i < size; i++) {
		for (int j = i; j < size; j++) {
			if (cmp(red[i], blue[j]) == 0) {
				EXCHANGE(blue[i], blue[j]);
				break;
			}
		}
	}
}

int partition(jug *red, jug *blue, int p, int q);

void quick_pair(jug *red, jug *blue, int p, int r) {
	if (p < r - 1) {
		int q = partition(red, blue, p, r);
		quick_pair(red, blue, p, q);
		quick_pair(red, blue, q + 1, r);
	}
}

int partition(jug *red, jug *blue, int p, int q) {
	int pivot, i;
	jug red_pivot, blue_pivot;
	
	// Pick a red pivot
	i = rand() % (q - p) + p;
	EXCHANGE(red[i], red[q - 1]);
	red_pivot = red[q - 1];
	
	// Find the blue pivot and put it in final position
	// NOTE: This look can be folded in the next one to minimize the number of
	// comparisons, but I will keep it here for clarity
	for (int i = p; i < q; i++) {
		if (cmp(red_pivot, blue[i]) == 0) {
			EXCHANGE(blue[i], blue[q - 1]);
			break;
		}
	}
	
	// Partition the blue jugs around the red pivot
	pivot = p;
	for (int i = p; i < q - 1; i++) {
		if (cmp(red_pivot, blue[i]) > 0) {
			EXCHANGE(blue[pivot], blue[i]);
			pivot++;
		}
	}
	
	// Put the blue pivot in place
	EXCHANGE(blue[pivot], blue[q-1]);
	blue_pivot = blue[pivot];
	
	// Partition the red jugs around the blue pivot
	int j = p;
	for (int i = p; i < q - 1; i++) {
		if (cmp(red[i], blue_pivot) < 0) {
			EXCHANGE(red[j], red[i]);
			j++;
		}
	}
	
	// Put the red pivot in place
	EXCHANGE(red[q - 1], red[j]);
	
	// Return the pivot index
	return pivot;
}

int cmp(jug red, jug blue) {
	return red - blue;
}
//end problem 8.4

	 //problem 8-5
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
	 
	 typedef struct {
		 int value;
		 int s;
	 } item;
	 
	 typedef struct {
		 item *elements;
		 int length;
		 int heap_size;
	 } heap_t;
	 
	 typedef struct {
		 int size;
		 int k;
		 int exhausted;
		 int *next_indices;
	 } sort_state_t;
	 
	 void merge_sort(int A[], int p, int r, int k, int s);
	 void min_heap_insert(heap_t *heap, item key);
	 int state_took_column(sort_state_t *state, int index);
	 item min_heap_push_pop(heap_t *heap, item new);
	 item heap_minimum(heap_t *heap);
	 item heap_extract_min(heap_t *heap);
	 
/*
 * Average soting is performed by just merge-sorting each column. That was
 * easy. Modifying merge sort was hard.
 */
	 
	 void k_sort(int *numbers, int size, int k) {
		 for (int i = 0; i < k; i++) {
			 merge_sort(numbers, 0, size, k, i);
		 }
	 }
	 
/*
 * Sorting a k-sorted array. We need to keep track of which column produced
 * the minumum element in the heap and this resulted in quite the tricky C
 * code. I don't think this is a good practice, but still, that's the best I'm
 * willing to make it right now.
 */
	 
	 void merge_k_sorted(int *numbers, int size, int k) {
		 int copy[size];
		 
		 item heap_elements[k];
		 heap_t heap = {heap_elements, k, 0};
		 
		 int next_indices[k];
		 sort_state_t state = {size, k, 0, next_indices};
		 
		 memcpy(copy, numbers, size * sizeof(int));
		 
		 for (int i = 0; i < k; i++) {
			 item new = {copy[i], i};
			 min_heap_insert(&heap, new);
			 next_indices[i] = i + k;
		 }
		 
		 for (int i = 0; i < size; i++) {
			 item min = heap_minimum(&heap);
			 numbers[i] = min.value;
			 
			 int next = state_took_column(&state, min.s);
			 
			 if (next != -1) {
				 min_heap_push_pop(&heap, (item) {copy[next], next % k});
			 } else {
				 heap_extract_min(&heap);
			 }
		 }
	 }
	 
	 int state_took_column(sort_state_t *state, int index) {
		 int size = state->size,
		 k = state->k,
		 s = index,
		 *next_indices = state->next_indices;
		 
		 if (next_indices[s] >= size) {
			 while (state->exhausted < k && next_indices[state->exhausted] >= state->size) {
				 state->exhausted++;
			 }
			 
			 if (state->exhausted == k) {
				 return -1;
			 }
			 
			 int next = next_indices[state->exhausted];
			 next_indices[state->exhausted] += k;
			 return next;
		 } else {
			 int next = next_indices[s];
			 next_indices[s] += k;
			 return s;
		 }
	 }
	 
/*
 * This is the merge sort from Chapter 2, modified to look only at indices
 * congruent to k modulo s. There are two very ugly and long macroses that
 * perform this unpleasant job. There's probably a nicer way to do the
 * calculation, but modular arithmetic has always been my Achilles' heel.
 */
	 
#define FIRST(index, k, s) ((index) + (s) - (index) % (k) + ((index) % (k) <= (s) ? 0 : (k)))
	 //(index+s)-index%k+ (index%k <= s ? 0 : k)
#define COUNT(a, b, k, s) (((b) - (a)) / (k) + ((((s) - (a) % (k)) + (k)) % (k) < ((b) - (a)) % (k) ? 1 : 0)) //(b-a)/k + (((s-a%k) +k) %k)<(b-a)%k ? 1:0
	 
	 void merge(int A[], int p, int q, int r, int k, int s) {
		 int i, j, l;
		 
		 int n1 = COUNT(p, q, k, s);
		 int n2 = COUNT(q, r, k, s);
		 
		 int L[n1];
		 int R[n2];
		 
		 for (i = FIRST(p, k, s), j = 0; i < q; j++, i += k) L[j] = A[i];
		 for (i = FIRST(q, k, s), j = 0; i < r; j++, i += k) R[j] = A[i];
		 
		 for(i = 0, j = 0, l = FIRST(p, k, s); l < r; l += k) {
			 if (i == n1) {
				 A[l] = R[j++];
			 } else if (j == n2) {
				 A[l] = L[i++];
			 } else if (L[i] <= R[j]) {
				 A[l] = L[i++];
			 } else {
				 A[l] = R[j++];
			 }
		 }
	 }
	 
	 void merge_sort(int A[], int p, int r, int k, int s) {
		 if (COUNT(p, r, k, s) > 1) {
			 int q = (p + r) / 2;
			 merge_sort(A, p, q, k, s);
			 merge_sort(A, q, r, k, s);
			 merge(A, p, q, r, k, s);
		 }
	 }
	 
/*
 * Finally, the min heap from exercise 6.5-3, modified to store items instead
 * of ints. When I first wrote it, I made an error in the implementation and
 * that sent me in a hour-long debugging session. C is fun.
 *
 * Also, there is a new heap operation (min_heap_push_pop) that is a faster
 * than heap_extract_min and then min_heap_insert.
 */
	 
#define PARENT(i) ((i - 1) / 2)
#define LEFT(i)   (2 * i + 1)
#define RIGHT(i)  (2 * i + 2)
	 
	 item heap_minimum(heap_t *heap) {
		 return heap->elements[0];
	 }
	 
	 void min_heapify(heap_t *heap, int i) {
		 int left  = LEFT(i),
		 right = RIGHT(i),
		 smallest;
		 
		 if (left < heap->heap_size && heap->elements[left].value < heap->elements[i].value) {
			 smallest = left;
		 } else {
			 smallest = i;
		 }
		 
		 if (right < heap->heap_size && heap->elements[right].value < heap->elements[smallest].value) {
			 smallest = right;
		 }
		 
		 if (smallest != i) {
			 item tmp = heap->elements[i];
			 heap->elements[i] = heap->elements[smallest];
			 heap->elements[smallest] = tmp;
			 
			 min_heapify(heap, smallest);
		 }
	 }
	 
	 item heap_extract_min(heap_t *heap) {
		 if (heap->heap_size == 0) {
			 fprintf(stderr, "heap underflow");
			 exit(0);
		 }
		 
		 item min = heap->elements[0];
		 heap->elements[0] = heap->elements[heap->heap_size - 1];
		 heap->heap_size--;
		 min_heapify(heap, 0);
		 
		 return min;
	 }
	 
	 void heap_decrease_key(heap_t *heap, int i, item key) {
		 if (key.value > heap->elements[i].value) {
			 fprintf(stderr, "new key is larger than current key");
			 exit(0);
		 }
		 
		 heap->elements[i].value = key.value;
		 while (i > 0 && heap->elements[PARENT(i)].value > heap->elements[i].value) {
			 item tmp = heap->elements[PARENT(i)];
			 heap->elements[PARENT(i)] = heap->elements[i];
			 heap->elements[i] = tmp;
			 i = PARENT(i);
		 }
	 }
	 
	 void min_heap_insert(heap_t *heap, item key) {
		 if (heap->length == heap->heap_size) {
			 fprintf(stderr, "heap overflow");
			 exit(0);
		 }
		 
		 heap->elements[heap->heap_size].value = INT_MAX;
		 heap->elements[heap->heap_size].s = key.s;
		 heap->heap_size++;
		 heap_decrease_key(heap, heap->heap_size - 1, key);
	 }
	 
	 item min_heap_push_pop(heap_t *heap, item new) {
		 item result = heap->elements[0];
		 heap->elements[0] = new;
		 min_heapify(heap, 0);
		 return result;
	 }
//end problem 8-5
	//problem 8.7
	//columnsort
#include <pthread.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
	 
#define STDLIB_SORT qsort
	 
	 typedef unsigned int number;
	 
	 typedef struct {
		 size_t start;
		 size_t size;
	 } column_t;
	 
	 typedef void column_sorter(number *, column_t *, int);
	 
	 void check_dimensions(size_t r, size_t s);
	 
/**
 * The basic column sort implementation. It does a copy of the array for steps
 * 3 and 5. It also does not sort the half-columns in the beginning and the
 * end, since that is not necessary for the correctness of the algorithm.
 */
	 
	 void columnsort(number *A, size_t r, size_t s, column_sorter sort_columns) {
		 size_t size = r * s;
		 number *copy;
		 column_t columns[s];
		 
		 check_dimensions(r, s);
		 
		 copy = calloc(size, sizeof(number));
		 
		 for (size_t i = 0; i < s; i++) {
			 columns[i] = (column_t) {i * r, r};
		 }
		 
		 sort_columns(A, columns, s);
		 
		 for (size_t i = 0; i < size; i++) {
			 copy[(i % s) * r + i / s] = A[i];
		 }
		 
		 sort_columns(copy, columns, s);
		 
		 for (size_t i = 0; i < size; i++) {
			 A[i] = copy[(i % s) * r + i / s];
		 }
		 
		 sort_columns(A, columns, s);
		 
		 for (size_t i = 0; i < s - 1; i++) {
			 columns[i] = (column_t) {i * r + r / 2, r};
		 }
		 
		 sort_columns(A, columns, s - 1);
		 
		 free(copy);
	 }
	 
/*
 * A function that compares numbers, to be passed to the stdlib sort.
 */
	 
	 int compare(const void *a, const void *b) {
		 number *first  = (number *) a;
		 number *second = (number *) b;
		 
		 if (*first == *second) {
			 return 0;
		 } else if (*first > *second) {
			 return 1;
		 } else {
			 return -1;
		 }
	 }
	 
/*
 * Verified the dimensions of the passed array.
 */
	 
	 void check_dimensions(size_t r, size_t s) {
		 if (r % 2) {
			 fprintf(stderr, "r must be even\n");
			 exit(0);
		 }
		 
		 if (r % s) {
			 fprintf(stderr, "s must divide r\n");
			 exit(0);
		 }
		 
		 if (r < 2 * s * s) {
			 fprintf(stderr, "r must be grater than 2s²\n");
			 exit(0);
		 }
	 }
	 
/*
 * A utility function to call with the array and a column.
 */
	 
	 void sort(number *A, column_t column) {
		 STDLIB_SORT(A + column.start, column.size, sizeof(number), compare);
	 }
	 
/*
 * Sequential sorting of columns
 */
	 
	 void sequential_sort_columns(number *numbers, column_t *columns, int size) {
		 for (int i = 0; i < size; i++) {
			 sort(numbers, columns[i]);
		 }
	 }
	 
/*
 * Parallel sorting of columns. This implementation is a bit naïve - it can
 * reuse existing threads instead of spawning new ones every time. Furthermore,
 * I never explored using locking mechanisms instead of joining the threads.
 */
	 
	 typedef struct {
		 number *numbers;
		 column_t column;
	 } job_t;
	 
	 void *sort_job(void *pjob) {
		 job_t *job = (job_t *) pjob;
		 sort(job->numbers, job->column);
		 return NULL;
	 }
	 
	 void threaded_sort_columns(number *numbers, column_t *columns, int size) {
		 void *status;
		 pthread_t threads[size];
		 job_t jobs[size];
		 pthread_attr_t attr;
		 
		 pthread_attr_init(&attr);
		 pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
		 
		 for (int i = 0; i < size; i++) {
			 jobs[i] = (job_t) {numbers, columns[i]};
			 pthread_create(&threads[i], &attr, sort_job, &jobs[i]);
		 }
		 
		 for (int i = 0; i < size; i++) {
			 pthread_join(threads[i], &status);
		 }
	 }
	 //end columnsort
	
//end problem 8.7

// 9.1 Minimum and maximum
minimum(A) {
	min = A[1]
	for i in 2...A.length {
		if min > A[i]{
			min = A[i]
		}
	}
	return min
}// upper bound of n-1 comparisons

//9.2 selection in expected linear time
//The following code for RANDOMIZED-SELECT returns the ith smallest element of the array A[p...r].
randomized_select(A, p, r, i) {
	if p==r {
		return A[p]
	}
	q = randomized_partition(A, p,r)
	k = q-p+1
	if i == k { //the pivot value is the answer
		return A[q]
	}
	else if i < k {
		return randomized_select(A,p,q-1,i)
	}
	else {
		return randomized_select(A, q+1, r, i-k)
	}
}
//iterative verson of randomized_select
C code
#include <stdlib.h>

static int tmp;
#define EXCHANGE(a, b) { tmp = a; a = b; b = tmp; }

int randomized_partition(int *A, int p, int r);

int randomized_select(int *A, int p, int r, int i) {
	while (p < r - 1) {
		int q = randomized_partition(A, p, r);
		int k = q - p;
		
		if (i == k) {
			return A[q];
		} else if (i < k) {
			r = q;
		} else {
			p = q + 1;
			i = i - k - 1;
		}
	}
	
	return A[p];
}

int partition(int *A, int p, int r) {
	int x, i, j;
	
	x = A[r - 1];
	i = p;
	
	for (j = p; j < r - 1; j++) {
		if (A[j] < x) {
			EXCHANGE(A[i], A[j]);
			i++;
		}
	}
	
	EXCHANGE(A[i], A[r - 1]);
	
	return i;
}

int randomized_partition(int *A, int p, int r) {
	int pivot = rand() % (r - p) + p;
	EXCHANGE(A[pivot], A[r - 1]);
	return partition(A, p, r);
}
//select
{
def select(items, n):
	med     = median(items)
	smaller = [item for item in items if item < med]
	larger  = [item for item in items if item > med]

	if len(smaller) == n:
		return med
	elif len(smaller) > n:
		return select(smaller, n)
	else:
		return select(list(larger), n - len(smaller) - 1)

def median(items):
	def median_index(n):
		if n % 2:
			return n // 2
		else:
			return n // 2 - 1

	def partition(items, element):
		i = 0

		for j in range(len(items) - 1):
			if items[j] == element:
				items[j], items[-1] = items[-1], items[j]

			if items[j] < element:
				items[i], items[j] = items[j], items[i]
				i += 1

		items[i], items[-1] = items[-1], items[i]

		return i

	def select(items, n):
		if len(items) <= 1:
			return items[0]

		medians = []

		for i in range(0, len(items), 5):
			group = sorted(items[i:i + 5])
			items[i:i + 5] = group
			median = group[median_index(len(group))]
			medians.append(median)

		pivot = select(medians, median_index(len(medians)))
		index = partition(items, pivot)

		if n == index:
			return items[index]
		elif n < index:
			return select(items[:index], n)
		else:
			return select(items[index + 1:], n - index - 1)

	return select(items[:], median_index(len(items)))
}

//9.3.6
/*
Exercise 9.3.6
The kth quantiles of an n-element set are k−1 order statistics that divide the sorted set into k equal-sized sets (to within 1). Give an O(nlgk)-time algorithm to list the kth quantiles of a set.

If k=1 we return an empty list.
If k is even, we find the median, partition around it, solve two similar subproblems of size ⌊n/2⌋ and return their solutions plus the median.
If k is odd, we find the ⌊k/2⌋ and ⌈k/2⌉ boundaries and the we reduce to two subproblems, each with size less than n/2. The worst case recurrence is:
T(n,k)=2T(⌊n/2⌋,k/2)+O(n)
Which is the desired bound ­ O(nlgk).*/

//9.3.7
//9.3.8
//9.3.9