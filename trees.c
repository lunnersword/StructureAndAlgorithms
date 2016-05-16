//
//  trees.c
//  StructureAndAlgorithms
//
//  Created by lunner on 8/25/15.
//
//

#include <stdio.h>


// MARK: Binary Search Tree
//二叉排序树 又叫 二叉查找树
/*
 let x be a node in a binary search tree. If y is node in the left subtree of x, then y.key < x.key. If y is a node in the right subtree of x, then y.key ≥ x.key
*/

typedef struct BiNode {
	ElemType data;
	struct BiNode *left, *right;
}BiNode, *BiTree;

inorder_tree_walk(x, visit) {
	if x!= nil {
		inorder_tree_walk(x.left)
		visit(x)
		inorder_tree_walk(x.right)
	}
}

tree_search(x, k) {
	if x==nil or k==x.key {
		return x
	}
	if k < x.key 
		return tree_search(x.left, k)
	else return tree_search(x.right, k)
}

iterative_tree_search(x, k) {
	while (x!=nil and k!=x.key) {
		if k<x.key
			x = x.left
		else
			x = x.right
			
	}
	return x
}

tree_minimum(x) {
	while x.left != nil {
		x = x.left
	}
	return x
}

tree_maximum(x) {
	while x.right != nil 
		x = x.right
	return x
}

tree_successor(x) {
	if x.right != nil
		return tree_minimum(x.right)
	y = x.p
	while y != nil and x==y.right 
		x = y
		y = y.p
	return y
}//the node with the smallest key greater than x.key

tree_insert(T, z) {
	y = nil
	x = T.root
	while x != nil {
		y = x
		if z.key < x.key
			x = x.left
		else x = x.right
	}
	z.p = y
	if y == nil
		T.root = z //tree T was empty
	else if z.key < y.key
		y.left = z
	else
		y.right = z
}

transplant(T, u, v) {
	if u.p == nil
		T.root = v
	else if u == u.p.left
		u.p.left = v
	else
		u.p.right = v 
	if v != nil
		v.p = u.p
}

tree_delete(T, z) {
	if z.left == nil
		transplant(T, z, z.right)
	else if z.right == nil
		transplant(T, z, z.left)
	else 
		y = tree_minimum(z.right)
		if y.p != z {
			transplant(T, y, y.right) //y must have no left child
			y.right = z.right
			y.right.p = y
		}
		transplant(T, z, y)
		y.left = z.left
		y.left.p = y
}

// MARK: Balanced Binary Tree or Height_Balanced Tree AVL
//任意节点左右子树的depth之差的绝对值不超过1
// MARK: red-black tree
/*
 A red-black tree is a binary search tree with one extra bit of storage per node: its color, which can be either RED or BLACK. By constraining the node colors on any simple path from the root to a leaf, red-black trees ensure that no such path is more than twice as long as any other, so that the tree is approximately balanced.
 
 A red-black tree is a binary tree that satisfies the following red-black properties:
 1. Every node is either red or black.
 2. The root is black.
 3. Every leaf (NIL) is black.
 4. If a node is red, then both its children are black.
 5. For each node, all simple paths from the node to descendant leaves contain the same number of black nodes.
 
 A red-black tree with n internal nodes has height at most 2*lg(n+1)
 
*/

LEFT-ROTATE(T, x) {
	y = x.right
	x.right = y.left 
	if y.left != T.nil {
		y.left.p = x
	}
	y.p = x.p 
	if (x.p == T.nil) {
		T.root = y
	}
	else if x = x.p.left {
		x.p.left = y
	}
	else {
		x.p.right = y
	}
	y.left = x
	x.p = y
}
//is symmetric to LEFT_ROTATE
RIGHT-ROTATE(T,y) {
	x = y.left
	y.left = x.right
	if x.right != T.nil {
		x.right.p = y 
	}
	x.p = y.p 
	if (y.p == T.nil) {
		T.root = y 
	} else if y = y.p.right {
		y.p.right = x
	} else {
		y.p.left = x
	}
	x.right = y
	y.p = x
}

RB-INSERT(T, z) {
	y = T.nil
	x = T.root
	while (x != T.nil) {
		y = x
		if (z.key < x.key) {
			x = x.left
		} else {
			x = x.right
		}
	}
	z.p = y
	if (y==T.nil) {
		T.root = z
	}
	else if z.key < y.key {
		y.left = z
	} else {
		y.right = z
	}
	z.left = T.nil
	z.right = T.nil
	z.color = RED
	RB-INSERT-FIXUP(T, z)
}//O(lg(n))

RB-INSERT-FIXUP(T, z) {
	while (z.p.color == RED) {
		if z.p == z.p.p.left {
			y = z.p.p.right
			if (y.color == RED) {
    			z.p.color = BLACK	//case 1
				y.color = BLACK		//case 1
				z.p.p.color = RED	//case 1
				z = z.p.p 			//case 1
			} else if (z == z.p.right){
				z = z.p 			//case 2
				LEFT-ROTATE(T, z)	//case 2
			} else {
				z.p.color = BLACK 		//case 3
				z.p.p.color = RED 		//case 3
				RIGHT-ROTATE(T, z.p.p) 	//case 3
			}
		} else {
			//same as then clause with "right" and "left" exchanged
		}
		
	}
	T.root.color = BLACK
}

RB-TRANSPLANT(T, u, v) {
	if u.p == T.nil {
		T.root = v
	} else if u == u.p.left {
		u.p.left = v 
	} else {
		u.p.right = v 
	}
	v.p = u.p
}

RB-DELETE(T, z) {
	y = z
	y_original_color = y.color 
	if z.left == T.nil {
		x = z.right 
		RB-TRANSPLANT(T, z, z.right)
	} else if z.right == T.nil {
		x = z.left
		RB-TRANSPLANT(T, z, z.left)
	} else y = tree_minimum(z.right) {
		y_original_color = y.color
		x = y.right
		if (y.p == z) {
			x.p = y 
		} else {
			RB-TRANSPLANT(T, y, y.right)
			y.right = z.right
			y.right.p = y 
		}
		RB-TRANSPLANT(T, z, y)
		y.left = z.left
		y.left.p = y 
		y.color = z.color
	}
	if y_original_color == BLACK {
		RB-DELETE-FIXUP(T, x)
	}
}

RB-DELETE-FIXUP(T, x) {
	while (x != T.root and x.color == BLACK) {
		if (x == x.p.left) {
			w = x.p.right
			if (w.color == RED) {
    			w.color = BLACK				//case 1
				x.p.color = RED				//case 1
				LEFT-ROTATE(T, x.p)			//case 1
				w = x.p.right				//case 1
			}
			if (w.left.color == BLACK and w.right.color == BLACK) {
				w.color = RED				//case 2
				x = x.p 					//case 2
			} else if (w.right.color == BLACK) {
				w.left.color = BLACK		//case 3
				w.color = RED				//case 3
				RIGHT-ROTATE(T, w)			//case 3
				w = x.p.right				//case 3
			} else {
				w.color = x.p.color				//case 4
				x.p.color = BLACK				//case 4
				w.right.color = BLACK			//case 4
				LEFT-ROTATE(T, x.p)				//case 4
				x = T.root						//case 4
			}
		} else {
			//same as then clause with "right" and "left" exchanged
		}
		x.color = BLACK
	}
}
