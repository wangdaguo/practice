package main

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

func main() {
	//node4 := &TreeNode{
	//	Val:   4,
	//	Left:  nil,
	//	Right: nil,
	//}
	//node2 := &TreeNode{
	//	Val:   2,
	//	Left:  node4,
	//	Right: nil,
	//}
	//node5 := &TreeNode{
	//	Val:   5,
	//	Left:  nil,
	//	Right: nil,
	//}
	//node3 := &TreeNode{
	//	Val:   3,
	//	Left:  nil,
	//	Right: node5,
	//}
	//root := &TreeNode{
	//	Val:   1,
	//	Left:  node2,
	//	Right: node3,
	//}
	//r := preorderTraversal(root)
	//r := inorderTraversal(root)
	//r := postorderTraversal(root)
	//r := levelOrder(root)

	//node5 := &TreeNode{
	//	Val:   3,
	//	Left:  nil,
	//	Right: nil,
	//}
	//node4 := &TreeNode{
	//	Val:   7,
	//	Left:  nil,
	//	Right: nil,
	//}
	//node3 := &TreeNode{
	//	Val:   6,
	//	Left:  node5,
	//	Right: node4,
	//}
	//node2 := &TreeNode{
	//	Val:   4,
	//	Left:  nil,
	//	Right: nil,
	//}
	//node1 := &TreeNode{
	//	Val:   5,
	//	Left:  node2,
	//	Right: node3,
	//}
	//r := hasPathSum(node1, 3)
	//r := sumNumbers(node1)
	//r := minDepthTest(node1)
	//r := rightSideView1(node1)
	//r := isValidBST1(node1)
	//bst := Constructor(node1)
	//bst.Next()
	//r := bst.Next()
	//r := math.Pow(float64(2), float64(2))
	//inorder := []int{9,3,15,20,7}
	//postorder := []int{9,15,7,20,3}
	//r := buildTree2(inorder, postorder)
	//var r int
	//HeightOfBinaryTree(node1, &r)
	//r := TwoNodeDistance(node1, 1, 4)
	//r := postorderTraversal1(node1)
	//r := isValidBST(node1)
	//fmt.Println(r)

	node7 := &TreeNode{
		Val:   7,
		Left:  nil,
		Right: nil,
	}
	node4 := &TreeNode{
		Val:   4,
		Left:  nil,
		Right: nil,
	}
	node2 := &TreeNode{
		Val:   2,
		Left:  node7,
		Right: node4,
	}
	node6 := &TreeNode{
		Val:   6,
		Left:  nil,
		Right: nil,
	}
	node5 := &TreeNode{
		Val:   5,
		Left:  node6,
		Right: node2,
	}
	node1 := &TreeNode{
		Val:   1,
		Left:  nil,
		Right: nil,
	}
	root := &TreeNode{
		Val:   3,
		Left:  node5,
		Right: node1,
	}
	r := lowestCommonAncestor22(root, node4, node6)
	fmt.Println(r)
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

/*
*
236. 二叉树的最近公共祖先
https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/
*/
func lowestCommonAncestor22(root, p, q *TreeNode) *TreeNode {
	l1, r1, l2, r2 := make([]*TreeNode, 0), []*TreeNode{}, make([]*TreeNode, 0), []*TreeNode{}
	DFSFind(root, p, l1, r1)
	DFSFind(root, q, l2, r2)
	fmt.Println(r1, r2)
	return nil
}

func DFSFind(root, node *TreeNode, list, r []*TreeNode) {
	if root == nil {
		return
	}
	if root == node {
		list = append(list, node)
		r = append(r, list...)
		return
	}
	list = append(list, root)
	if root.Left != nil {
		DFSFind(root.Left, node, list, r)
	}
	if root.Right != nil {
		DFSFind(root.Right, node, list, r)
	}
	list = list[:len(list)-1]
	return
}

/*
144. 二叉树的前序遍历
https://leetcode-cn.com/problems/binary-tree-preorder-traversal/
*/

//func preorderTraversal(root *TreeNode) []int {
//	if root == nil {
//		return []int{}
//	}
//	var r []int
//	helper(root, &r)
//	return r
//}
//
//func helper(root *TreeNode, r *[]int)  {
//	if root == nil {
//		return
//	}
//	*r = append(*r, root.Val)
//	helper(root.Left, r)
//	helper(root.Right, r)
//}

func preorderTraversal(root *TreeNode) []int {
	if root == nil {
		return []int{}
	}
	var stack []*TreeNode
	var r []int
	stack = append(stack, root)
	for len(stack) > 0 {
		node := stack[len(stack)-1]
		stack = stack[0 : len(stack)-1]
		r = append(r, node.Val)
		if node.Right != nil {
			stack = append(stack, node.Right)
		}
		if node.Left != nil {
			stack = append(stack, node.Left)
		}
	}
	return r
}

/*
94. 二叉树的中序遍历
https://leetcode-cn.com/problems/binary-tree-inorder-traversal/
*/
func inorderTraversal(root *TreeNode) []int {
	if root == nil {
		return []int{}
	}
	var stack []*TreeNode
	var r []int
	curNode := root
	for curNode != nil || len(stack) > 0 {
		for curNode != nil {
			stack = append(stack, curNode)
			curNode = curNode.Left
		}
		curNode = stack[len(stack)-1]
		stack = stack[0 : len(stack)-1]
		r = append(r, curNode.Val)
		curNode = curNode.Right
	}
	return r
}

/*
145. 二叉树的后序遍历
https://leetcode-cn.com/problems/binary-tree-postorder-traversal/
*/
func postorderTraversal(root *TreeNode) []int {
	if root == nil {
		return []int{}
	}
	var stack []*TreeNode
	var r []int
	curNode := root
	mp := make(map[*TreeNode]int)
	for curNode != nil || len(stack) > 0 {
		for curNode != nil {
			stack = append(stack, curNode)
			curNode = curNode.Left
		}
		curNode = stack[len(stack)-1]
		mp[curNode]++
		if curNode.Right != nil && mp[curNode] == 1 {
			curNode = curNode.Right
		} else {
			stack = stack[:len(stack)-1]
			r = append(r, curNode.Val)
			curNode = nil
		}
	}
	return r
}

func postorderTraversal1(root *TreeNode) []int {
	if root == nil {
		return []int{}
	}
	var stack []*TreeNode
	var pre *TreeNode
	var r []int
	node := root
	for node != nil || len(stack) > 0 {
		for node != nil {
			stack = append(stack, node)
			node = node.Left
		}
		peekNode := stack[len(stack)-1]
		if peekNode.Right == nil || peekNode.Right == pre {
			r = append(r, peekNode.Val)
			pre = peekNode
			stack = stack[:len(stack)-1]
		} else {
			node = peekNode.Right
		}
	}
	return r
}

/*
102. 二叉树的层次遍历
https://leetcode-cn.com/problems/binary-tree-level-order-traversal/
*/
func levelOrder(root *TreeNode) [][]int {
	if root == nil {
		return [][]int{}
	}
	var queue []*TreeNode
	var r [][]int
	var tmpNode []*TreeNode
	var tmp []int
	queue = append(queue, root)
	for len(queue) > 0 || len(tmpNode) > 0 {
		if len(queue) == 0 {
			queue = append(queue, tmpNode...)
			tmpNode = []*TreeNode{}
			r = append(r, tmp)
			tmp = []int{}
		}
		node := queue[0]
		tmp = append(tmp, node.Val)
		queue = queue[1:]
		if node.Left != nil {
			tmpNode = append(tmpNode, node.Left)
		}
		if node.Right != nil {
			tmpNode = append(tmpNode, node.Right)
		}
	}
	if len(tmp) > 0 {
		r = append(r, tmp)
	}
	return r
}

/*
100. 相同的树
https://leetcode-cn.com/problems/same-tree/
*/
func isSameTree(p *TreeNode, q *TreeNode) bool {
	if p == nil && q == nil {
		return true
	}
	if p != nil && q == nil || (p == nil && q != nil) {
		return false
	}
	if p.Val != q.Val {
		return false
	}
	return isSameTree(p.Left, q.Left) && isSameTree(p.Right, q.Right)
}

/*
101. 对称二叉树
https://leetcode-cn.com/problems/symmetric-tree/
*/
func isSymmetric(root *TreeNode) bool {
	if root == nil {
		return true
	}
	return isSymmetricImpl(root.Left, root.Right)
}
func isSymmetricImpl(left, right *TreeNode) bool {
	if left == nil && right == nil {
		return true
	} else if left == nil || right == nil {
		return false
	}
	if left.Val != right.Val {
		return false
	}
	return isSymmetricImpl(left.Left, right.Right) && isSymmetricImpl(left.Right, right.Left)
}

/*
226. 翻转二叉树
https://leetcode-cn.com/problems/invert-binary-tree/
*/
func invertTree(root *TreeNode) *TreeNode {
	if root == nil || (root.Left == nil && root.Right == nil) {
		return root
	}
	left := invertTree(root.Left)
	right := invertTree(root.Right)
	root.Left = right
	root.Right = left
	return root
}

func invertTree1(root *TreeNode) *TreeNode {
	if root == nil || (root.Left == nil && root.Right == nil) {
		return root
	}
	var queue []*TreeNode
	queue = append(queue, root)
	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]
		tmp := node.Left
		node.Left = node.Right
		node.Right = tmp
		if node.Left != nil {
			queue = append(queue, node.Left)
		}
		if node.Right != nil {
			queue = append(queue, node.Right)
		}
	}
	return root
}

/*
257. 二叉树的所有路径
https://leetcode-cn.com/problems/binary-tree-paths/
*/
//func binaryTreePaths(root *TreeNode) []string {
//	if root == nil {
//		return []string{}
//	}
//	var r []string
//	var strSlice []string
//	binaryTreePathsImpl(root, &strSlice, &r)
//	return r
//}
//func binaryTreePathsImpl(root *TreeNode, strSlice *[]string, r *[]string) {
//	*strSlice = append(*strSlice, strconv.Itoa(root.Val))
//	if root.Left == nil && root.Right == nil {
//		*r = append(*r, strings.Join(*strSlice, "->"))
//		*strSlice = (*strSlice)[0:len(*strSlice)-1]
//		return
//	}
//	if root.Left != nil {
//		binaryTreePathsImpl(root.Left, strSlice, r)
//	}
//	if root.Right != nil {
//		binaryTreePathsImpl(root.Right, strSlice, r)
//	}
//	*strSlice = (*strSlice)[0:len(*strSlice)-1]
//}

func binaryTreePaths(root *TreeNode) []string {
	var r []string
	binaryTreePathsImpl(root, "", &r)
	return r
}
func binaryTreePathsImpl(root *TreeNode, str string, r *[]string) {
	if root == nil {
		return
	}
	str += strconv.Itoa(root.Val)
	if root.Left == nil && root.Right == nil {
		*r = append(*r, str)
	} else {
		str += "->"
		binaryTreePathsImpl(root.Left, str, r)
		binaryTreePathsImpl(root.Right, str, r)
	}
}

/*
112. 路径总和
https://leetcode-cn.com/problems/path-sum/
*/
func hasPathSum(root *TreeNode, sum int) bool {
	if root == nil {
		return false
	}
	return hasPathSumImpl(root, sum)
}
func hasPathSumImpl(root *TreeNode, sum int) bool {
	if root == nil {
		return false
	}
	if root.Left == nil && root.Right == nil && sum-root.Val == 0 {
		return true
	}
	sum -= root.Val
	return hasPathSumImpl(root.Left, sum) || hasPathSumImpl(root.Right, sum)
}

/*
113. 路径总和 II
https://leetcode-cn.com/problems/path-sum-ii/
*/
func pathSum(root *TreeNode, sum int) [][]int {
	if root == nil {
		return [][]int{}
	}
	var r [][]int
	var curSlice []int
	pathSumImpl(root, &curSlice, sum, &r)
	return r
}
func pathSumImpl(root *TreeNode, curSlice *[]int, sum int, r *[][]int) {
	*curSlice = append(*curSlice, root.Val)
	if root.Left == nil && root.Right == nil && sumSlice(*curSlice) == sum {
		tmp := make([]int, len(*curSlice))
		copy(tmp, *curSlice)
		*r = append(*r, tmp)
		*curSlice = (*curSlice)[0 : len(*curSlice)-1]
		return
	}
	if root.Left != nil {
		pathSumImpl(root.Left, curSlice, sum, r)
	}
	if root.Right != nil {
		pathSumImpl(root.Right, curSlice, sum, r)
	}
	*curSlice = (*curSlice)[0 : len(*curSlice)-1]
}
func sumSlice(s []int) int {
	var r int
	for _, v := range s {
		r += v
	}
	return r
}

/*
129. 求根到叶子节点数字之和
https://leetcode-cn.com/problems/sum-root-to-leaf-numbers/
*/
func sumNumbers(root *TreeNode) int {
	if root == nil {
		return 0
	}
	var pathSlice []int
	treePath(root, 0, &pathSlice)
	return sumPath(pathSlice)
}
func treePath(root *TreeNode, pathVal int, pathSlice *[]int) {
	pathVal = pathVal*10 + root.Val
	if root.Left == nil && root.Right == nil {
		*pathSlice = append(*pathSlice, pathVal)
		return
	}
	if root.Left != nil {
		treePath(root.Left, pathVal, pathSlice)
	}
	if root.Right != nil {
		treePath(root.Right, pathVal, pathSlice)
	}
}
func sumPath(path []int) int {
	if len(path) < 1 {
		return 0
	}
	var r int
	for _, v := range path {
		r += v
	}
	return r
}

/*
111. 二叉树的最小深度
https://leetcode-cn.com/problems/minimum-depth-of-binary-tree/
*/
func minDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	if root.Left == nil && root.Right == nil {
		return 1
	}
	minDeep := int(^uint(0) >> 1)
	if root.Left != nil {
		minDeep = min(minDepth(root.Left)+1, minDeep)
	}
	if root.Right != nil {
		minDeep = min(minDepth(root.Right)+1, minDeep)
	}
	return minDeep
}
func min1(x, y int) int {
	if x > y {
		return y
	}
	return x
}

func min(x, y int) int {
	if x > y {
		return y
	}
	return x
}

func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

func minDepthTest(root *TreeNode) int {
	if root == nil {
		return 0
	}
	if root.Left == nil && root.Right == nil {
		return 1
	}
	var queue []*TreeNode
	queue = append(queue, root)
	mp := make(map[*TreeNode]int)
	mp[root] = 1
	minDeep := 0
	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]
		minDeep = mp[node]
		if node.Left == nil && node.Right == nil {
			break
		}
		if node.Left != nil {
			queue = append(queue, node.Left)
			mp[node.Left] = minDeep + 1
		}
		if node.Right != nil {
			queue = append(queue, node.Right)
			mp[node.Right] = minDeep + 1
		}
	}
	return minDeep
}

/*
104. 二叉树的最大深度
https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/
*/
func maxDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	if root.Left == nil && root.Right == nil {
		return 1
	}
	maxDeep := 0
	if root.Left != nil {
		maxDeep = max(maxDepth(root.Left)+1, maxDeep)
	}
	if root.Right != nil {
		maxDeep = max(maxDepth(root.Right)+1, maxDeep)
	}
	return maxDeep
}

/*
110. 平衡二叉树
https://leetcode-cn.com/problems/balanced-binary-tree/
*/
func isBalanced(root *TreeNode) bool {
	if root == nil {
		return true
	}
	if !isBalanced(root.Left) {
		return false
	}
	if !isBalanced(root.Right) {
		return false
	}
	if math.Abs(float64(deep(root.Left))-float64(deep(root.Right))) > 1 {
		return false
	}
	return true
}

func deep(root *TreeNode) int {
	if root == nil {
		return 0
	}
	if root.Left == nil && root.Right == nil {
		return 1
	}
	leftHeight := deep(root.Left) + 1
	rightHeight := deep(root.Right) + 1
	return max(leftHeight, rightHeight)
}

/*
337. 打家劫舍 III
https://leetcode-cn.com/problems/house-robber-iii/
*/
func rob1(root *TreeNode) int {
	r := robImpl(root)
	return max(r[0], r[1])
}

func robImpl(root *TreeNode) []int {
	if root == nil {
		return []int{0, 0}
	}
	left := robImpl(root.Left)
	right := robImpl(root.Right)
	noSelected := max(left[0], left[1]) + max(right[0], right[1])
	selected := root.Val + left[0] + right[0]
	return []int{noSelected, selected}
}

/*
107. 二叉树的层次遍历 II
https://leetcode-cn.com/problemset/all/?search=107
*/
func levelOrderBottom(root *TreeNode) [][]int {
	if root == nil {
		return [][]int{}
	}
	var queue, tmpQueue []*TreeNode
	queue = append(queue, root)
	var r [][]int
	var tmpVal []int
	for len(queue) > 0 || len(tmpQueue) > 0 {
		if len(queue) == 0 {
			r = append([][]int{tmpVal}, r...)
			queue = append(queue, tmpQueue...)
			tmpQueue = []*TreeNode{}
			tmpVal = []int{}
		}
		node := queue[0]
		queue = queue[1:]
		tmpVal = append(tmpVal, node.Val)
		if node.Left != nil {
			tmpQueue = append(tmpQueue, node.Left)
		}
		if node.Right != nil {
			tmpQueue = append(tmpQueue, node.Right)
		}
	}
	if len(tmpVal) > 0 {
		r = append([][]int{tmpVal}, r...)
	}
	return r
}

/*
103. 二叉树的锯齿形层次遍历
https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/
*/
func zigzagLevelOrder(root *TreeNode) [][]int {
	if root == nil {
		return [][]int{}
	}
	var queue1, queue2 []*TreeNode
	var tmp []int
	var r [][]int
	queue1 = append(queue1, root)
	for len(queue1) > 0 || len(queue2) > 0 {
		if len(queue1) > 0 {
			for len(queue1) > 0 {
				node := queue1[0]
				queue1 = queue1[1:]
				tmp = append(tmp, node.Val)
				if node.Left != nil {
					queue2 = append(queue2, node.Left)
				}
				if node.Right != nil {
					queue2 = append(queue2, node.Right)
				}
			}
		} else {
			for len(queue2) > 0 {
				node := queue2[0]
				queue2 = queue2[1:]
				tmp = append([]int{node.Val}, tmp...)
				if node.Left != nil {
					queue1 = append(queue1, node.Left)
				}
				if node.Right != nil {
					queue1 = append(queue1, node.Right)
				}
			}
		}
		r = append(r, tmp)
		tmp = []int{}
	}
	return r
}

/*
199. 二叉树的右视图
https://leetcode-cn.com/problems/binary-tree-right-side-view/
*/
func rightSideView(root *TreeNode) []int {
	if root == nil {
		return []int{}
	}
	var queue []*TreeNode
	var r []int
	queue = append(queue, root)
	for len(queue) > 0 {
		size := len(queue)
		for i := 0; i < size; i++ {
			node := queue[0]
			queue = queue[1:]
			if i == size-1 {
				r = append(r, node.Val)
			}
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
	}
	return r
}

func rightSideView1(root *TreeNode) []int {
	if root == nil {
		return []int{}
	}
	var queue []*TreeNode
	var depthQueue []int
	queue = append(queue, root)
	depthQueue = append(depthQueue, 0)
	distinctMap := make(map[int]int)
	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]
		depth := depthQueue[0]
		depthQueue = depthQueue[1:]
		if _, ok := distinctMap[depth]; !ok {
			distinctMap[depth] = node.Val
		}
		if node.Right != nil {
			queue = append(queue, node.Right)
			depthQueue = append(depthQueue, depth+1)
		}
		if node.Left != nil {
			queue = append(queue, node.Left)
			depthQueue = append(depthQueue, depth+1)
		}
	}
	r := make([]int, len(distinctMap))
	for k, v := range distinctMap {
		r[k] = v
	}
	return r
}

/*
98. 验证二叉搜索树
https://leetcode-cn.com/problems/validate-binary-search-tree/
*/
func isValidBST(root *TreeNode) bool {
	return helper(root, 0, 0, true, true)
}

func helper(root *TreeNode, lower, upper int, isLowerFirst, isRightFirst bool) bool {
	if root == nil {
		return true
	}
	val := root.Val
	if !isLowerFirst && val <= lower {
		return false
	}
	if !isRightFirst && val >= upper {
		return false
	}
	if !helper(root.Left, lower, val, isLowerFirst, false) {
		return false
	}
	if !helper(root.Right, val, upper, false, isRightFirst) {
		return false
	}
	return true
}

func isValidBST1(root *TreeNode) bool {
	if root == nil {
		return true
	}
	var stack []*TreeNode
	node := root
	val := ^int(^uint(0) >> 1)
	for len(stack) > 0 || node != nil {
		for node != nil {
			stack = append(stack, node)
			node = node.Left
		}
		node = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if node.Val <= val {
			return false
		}
		val = node.Val
		node = node.Right
	}
	return true
}

/*
235. 二叉搜索树的最近公共祖先
https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-search-tree/
*/
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	if p.Val > root.Val && q.Val > root.Val {
		return lowestCommonAncestor(root.Right, p, q)
	} else if p.Val < root.Val && q.Val < root.Val {
		return lowestCommonAncestor(root.Left, p, q)
	} else {
		return root
	}
}

/*
236. 二叉树的最近公共祖先
https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/
*/
func lowestCommonAncestor1(root, p, q *TreeNode) *TreeNode {
	parent := make(map[*TreeNode]*TreeNode)
	parent[root] = nil
	var stack []*TreeNode
	stack = append(stack, root)
	for true {
		node := stack[len(stack)-1]
		stack = stack[0 : len(stack)-1]
		if node.Left != nil {
			parent[node.Left] = node
			stack = append(stack, node.Left)
		}
		if node.Right != nil {
			parent[node.Right] = node
			stack = append(stack, node.Right)

		}
		_, ok1 := parent[p]
		_, ok2 := parent[q]
		if ok1 && ok2 {
			break
		}
	}
	set := make(map[*TreeNode]int)
	for p != nil {
		set[p] = 1
		p = parent[p]
	}
	for true {
		if _, ok := set[q]; !ok {
			q = parent[q]
		} else {
			break
		}
	}
	return q
}

/*
108. 将有序数组转换为二叉搜索树
https://leetcode-cn.com/problems/convert-sorted-array-to-binary-search-tree/submissions/
*/
func sortedArrayToBST(nums []int) *TreeNode {
	if len(nums) == 0 {
		return nil
	}
	mid := len(nums) / 2
	left := nums[:mid]
	right := nums[mid+1:]
	node := &TreeNode{nums[mid], sortedArrayToBST(left), sortedArrayToBST(right)}
	return node
}

/*
109. 有序链表转换二叉搜索树
https://leetcode-cn.com/problems/convert-sorted-list-to-binary-search-tree/
*/
type ListNode struct {
	Val  int
	Next *ListNode
}

func sortedListToBST(head *ListNode) *TreeNode {
	if head == nil {
		return nil
	}
	midNode := findMiddleElement(head)
	treeNode := &TreeNode{Val: midNode.Val}
	if head == midNode {
		return treeNode
	}
	treeNode.Left = sortedListToBST(head)
	treeNode.Right = sortedListToBST(midNode.Next)
	return treeNode
}

func findMiddleElement(head *ListNode) *ListNode {
	var prePtr *ListNode
	slowPtr, fastPtr := head, head
	for fastPtr != nil && fastPtr.Next != nil {
		prePtr = slowPtr
		slowPtr = slowPtr.Next
		fastPtr = fastPtr.Next.Next
	}
	if prePtr != nil {
		prePtr.Next = nil
	}
	return slowPtr
}

/*
173. 二叉搜索树迭代器
https://leetcode-cn.com/problems/binary-search-tree-iterator/
*/
type BSTIterator struct {
	Stack []*TreeNode
}

func Constructor(root *TreeNode) BSTIterator {
	bst := BSTIterator{
		Stack: []*TreeNode{},
	}
	bst.push(root)
	return bst
}
func (this *BSTIterator) push(node *TreeNode) {
	for node != nil {
		this.Stack = append(this.Stack, node)
		node = node.Left
	}
}

/** @return the next smallest number */
func (this *BSTIterator) Next() int {
	node := this.Stack[len(this.Stack)-1]
	this.Stack = this.Stack[:len(this.Stack)-1]
	if node.Right != nil {
		this.push(node.Right)
	}
	return node.Val
}

/** @return whether we have a next smallest number */
func (this *BSTIterator) HasNext() bool {
	return len(this.Stack) > 0
}

/*
230. 二叉搜索树中第K小的元素
https://leetcode-cn.com/problems/kth-smallest-element-in-a-bst/
*/
func kthSmallest(root *TreeNode, k int) int {
	if root == nil {
		return 0
	}
	var stack []*TreeNode
	node := root
	for len(stack) > 0 || node != nil {
		for node != nil {
			stack = append(stack, node)
			node = node.Left
		}
		node = stack[len(stack)-1]
		k--
		if k == 0 {
			return node.Val
		}
		stack = stack[:len(stack)-1]
		node = node.Right
	}
	return 0
}

/*
114. 二叉树展开为链表
https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/
*/

func flatten(root *TreeNode) {
	if root == nil {
		return
	}
	for root != nil {
		if root.Left == nil {
			root = root.Right
		} else {
			pre := root.Left
			for pre.Right != nil {
				pre = pre.Right
			}
			pre.Right = root.Right
			root.Right = root.Left
			root.Left = nil
			root = root.Right
		}
	}
}

func flatten1(root *TreeNode) {
	if root == nil {
		return
	}
	var stack []*TreeNode
	stack = append(stack, root)
	var pre *TreeNode
	for len(stack) > 0 {
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if pre != nil {
			pre.Right = node
			pre.Left = nil
		}
		if node.Right != nil {
			stack = append(stack, node.Right)
		}
		if node.Left != nil {
			stack = append(stack, node.Left)
		}
		pre = node
	}
}

/*
222. 完全二叉树的节点个数
https://leetcode-cn.com/problems/count-complete-tree-nodes/
*/
func countNodes(root *TreeNode) int {
	if root == nil {
		return 0
	}
	d := computeDepth(root)
	if d == 0 {
		return 1
	}
	left, right := 1, int(math.Pow(float64(2), float64(d)))-1
	var pivot int
	for left <= right {
		pivot = left + (right-left)/2
		if exists(pivot, d, root) {
			left = pivot + 1
		} else {
			right = pivot - 1
		}
	}
	return int(math.Pow(float64(2), float64(d))-1) + left
}
func computeDepth(root *TreeNode) int {
	var r int
	for root.Left != nil {
		root = root.Left
		r++
	}
	return r
}
func exists(idx, d int, node *TreeNode) bool {
	left, right := 0, int(math.Pow(float64(2), float64(d)))-1
	var pivot int
	for i := 0; i < d; i++ {
		pivot = left + (right-left)/2
		if idx > pivot {
			node = node.Right
			left = pivot + 1
		} else {
			node = node.Left
			right = pivot
		}
	}
	return node != nil
}

/*
105. 从前序与中序遍历序列构造二叉树
前序遍历 preorder = [3,9,20,15,7]
中序遍历 inorder = [9,3,15,20,7]
https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/
*/
func buildTree(preorder []int, inorder []int) *TreeNode {
	if len(preorder) < 1 {
		return nil
	}
	mp := make(map[int]int)
	for k, v := range inorder {
		mp[v] = k
	}
	root := &TreeNode{Val: preorder[0]}
	i := mp[preorder[0]] //1
	root.Left = buildTree(preorder[1:i+1], inorder[:i])
	root.Right = buildTree(preorder[i+1:], inorder[i+1:])
	return root
}

/*
前序遍历 preorder = [3,9,20,15,7]
中序遍历 inorder = [9,3,15,20,7]
*/
func buildTree1(preorder []int, inorder []int) *TreeNode {
	lens := len(preorder)
	if lens == 0 {
		return nil
	}
	return build(preorder, inorder, 0, lens-1, 0, lens-1)
}
func build(preorder, inorder []int, pl, pr, il, ir int) *TreeNode {
	if pl > pr || il > ir {
		return nil
	}
	root := &TreeNode{}
	root.Val = preorder[pl]
	i := index(inorder, root.Val) //1
	llen := i - il                //1
	//rlen:=ir-i
	root.Left = build(preorder, inorder, pl+1, pl+llen, il, i-1)
	root.Right = build(preorder, inorder, pl+llen+1, pr, i+1, ir)
	return root
}
func index1(nums []int, elem int) int {
	for i := 0; i < len(nums); i++ {
		if nums[i] == elem {
			return i
		}
	}
	return -1
}

/*
106. 从中序与后序遍历序列构造二叉树
中序遍历 inorder = [9,3,15,20,7]
后序遍历 postorder = [9,15,7,20,3]
https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/
*/
func buildTree2(inorder []int, postorder []int) *TreeNode {
	if len(inorder) < 1 {
		return nil
	}
	return buildImpl(postorder, inorder, 0, len(postorder)-1, 0, len(inorder)-1)
}
func buildImpl(postorder []int, inorder []int, pl, pr, il, ir int) *TreeNode {
	if ir < il || pr < pl {
		return nil
	}
	val := postorder[pr]
	root := &TreeNode{Val: val}
	i := index(inorder, val)
	llen := i - il
	root.Left = buildImpl(postorder, inorder, pl, pl+llen-1, il, i-1)
	root.Right = buildImpl(postorder, inorder, pl+llen, pr-1, i+1, ir)
	return root
}
func index(nums []int, elem int) int {
	for k, v := range nums {
		if v == elem {
			return k
		}
	}
	return -1
}

/*
96. 不同的二叉搜索树
https://leetcode-cn.com/problems/unique-binary-search-trees/
*/
func numTrees(n int) int {
	if n == 1 {
		return 1
	}
	dp := make([]int, n+1)
	dp[0] = 1
	dp[1] = 1
	for i := 2; i <= n; i++ {
		for j := 1; j <= i; j++ {
			dp[i] += dp[j-1] * dp[i-j]
		}
	}
	return dp[n]
}

/*
95. 不同的二叉搜索树 II
https://leetcode-cn.com/problems/unique-binary-search-trees-ii/
*/
func generateTrees(n int) []*TreeNode {
	if n == 0 {
		return []*TreeNode{}
	}
	return generateTrees1(1, n)
}
func generateTrees1(start, end int) []*TreeNode {
	var r []*TreeNode
	if start > end {
		r = append(r, nil)
		return r
	}
	for i := start; i <= end; i++ {
		leftTrees := generateTrees1(start, i-1)
		rightTrees := generateTrees1(i+1, end)
		for _, left := range leftTrees {
			for _, right := range rightTrees {
				cur := &TreeNode{Val: i}
				cur.Left = left
				cur.Right = right
				r = append(r, cur)
			}
		}
	}
	return r
}

/*
331. 验证二叉树的前序序列化
https://leetcode-cn.com/problems/verify-preorder-serialization-of-a-binary-tree/
*/
func isValidSerialization(preorder string) bool {
	if len(preorder) < 1 {
		return true
	}
	arr := strings.Split(preorder, ",")
	num := 1
	for _, v := range arr {
		if num == 0 {
			return false
		}
		if v != "#" {
			num++
		} else {
			num--
			if num < 0 {
				return false
			}
		}
	}
	if num == 0 {
		return true
	}
	return false
}
func isValidSerialization1(preorder string) bool {
	if len(preorder) < 1 {
		return true
	}
	arr := strings.Split(preorder, ",")
	var stack []string
	for i := 0; i < len(arr); i++ {
		stack = append(stack, arr[i])
		for len(stack) >= 3 && stack[len(stack)-1] == "#" && stack[len(stack)-2] == "#" {
			stack = stack[:len(stack)-2]
			if stack[len(stack)-1] == "#" {
				return false
			}
			stack = append(stack, "#")
		}
	}
	if len(stack) == 1 && stack[0] == "#" {
		return true
	}
	return false
}

func HeightOfBinaryTree(root *TreeNode, maxDistance *int) int {
	if root == nil {
		return -1
	}
	leftHigh := HeightOfBinaryTree(root.Left, maxDistance) + 1
	rightHigh := HeightOfBinaryTree(root.Right, maxDistance) + 1
	distance := leftHigh + rightHigh
	if distance > *maxDistance {
		*maxDistance = distance
	}
	if leftHigh > rightHigh {
		return leftHigh
	}
	return rightHigh
}

func TwoNodeDistance(root *TreeNode, num1, num2 int) int {
	if root == nil {
		return 0
	}
	var stack []*TreeNode
	stack = append(stack, root)
	for len(stack) > 0 {
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if node.Val == num1 || node.Val == num2 {
			return max(high(node.Left, num1, num2), high(node.Right, num1, num2))
		}
		if high(root.Left, num1, num2) > 0 && high(root.Right, num1, num2) > 0 {
			return high(root.Left, num1, num2) + high(root.Right, num1, num2)
		}
		if node.Left != nil {
			stack = append(stack, node.Left)
		}
		if node.Right != nil {
			stack = append(stack, node.Right)
		}
	}
	return 0
}

func high(root *TreeNode, num1, num2 int) int {
	if root == nil {
		return 0
	}
	var stack []*TreeNode
	stack = append(stack, root)
	var level int
	for len(stack) > 0 {
		level++
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if node.Val == num1 || node.Val == num2 {
			return level
		}
		if node.Left != nil {
			stack = append(stack, node.Left)
		}
		if node.Right != nil {
			stack = append(stack, node.Right)
		}
	}
	return 0
}

func isValidBST2(root *TreeNode) bool {
	return compareNodeVal(root, math.MinInt64, math.MaxInt64)
}

func compareNodeVal(root *TreeNode, lower, upper int) bool {
	if root == nil {
		return true
	}
	if root.Val <= lower || root.Val >= upper {
		return false
	}
	return compareNodeVal(root.Left, lower, root.Val) && compareNodeVal(root.Right, root.Val, upper)
}
