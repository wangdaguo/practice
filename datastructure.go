package main

import (
	"container/heap"
	"fmt"
	"sort"
)

func main() {
	//r := merge1([][]int{{1,3},{2,6},{8,10},{15,18}})
	//isConnected := [][]int{
	//	{1,1,0},
	//	{1,1,0},
	//	{0,0,1},
	//}
	//r := findCircleNumW(isConnected)
	//r := findDisappearedNumbers([]int{1,1})
	//r := [][]int{
	//	{1,2,3},
	//	{4,5,6},
	//	{7,8,9},
	//}
	//rotate(r)
	//r := maxChunksToSorted([]int{1,0,2,3,4})
	//r := minPathSum3([][]int{{1,2,3}, {4,5,6}})
	//r := dailyTemperatures([]int{73,74,75,71,69,72,76,73})

	//node2 := &ListNode{
	//	Val: 2,
	//	Next: nil,
	//}
	//node3 := &ListNode{
	//	Val: 3,
	//	Next: node2,
	//}
	//node4 := &ListNode{
	//	Val: 4,
	//	Next: node3,
	//}
	//node1 := &ListNode{
	//	Val: 1,
	//	Next: node4,
	//}
	//r := heapSort(node1)

	//node3 := &ListNode{
	//	Val: -1,
	//	Next: nil,
	//}
	//
	//var node2 *ListNode
	//
	//node1 := &ListNode{
	//	Val: 2,
	//	Next: nil,
	//}
	//r := mergeKLists([]*ListNode{node1, node2, node3})
	r := maxSlidingWindow1([]int{1,3,-1,-3,5,3,6,7}, 3)
	fmt.Println(r)
}

/**
56. 合并区间
https://leetcode.cn/problems/merge-intervals/description/
 */
func merge1(intervals [][]int) [][]int {
	if len(intervals) < 1 {
		return intervals
	}
	sort.Slice(intervals, func(i, j int) bool {
		if intervals[i][0] < intervals[j][0] {
			return true
		}
		return false
	})
	r := make([][]int, 0)
	pre := intervals[0]
	for i:=1; i<len(intervals); i++ {
		// pre 的后一个的开始元素可以与pre合并
		if intervals[i][0] <= pre[1] {
			if intervals[i][0] < pre[0] {
				pre[0] = intervals[i][0]
			}
			if intervals[i][1] > pre[1] {
				pre[1] = intervals[i][1]
			}
		} else {
			r = append(r, pre)
			pre = intervals[i]
		}
	}
	r = append(r, pre)
	return r
}

func findCircleNumW(isConnected [][]int) int {
	visited := make([]bool, len(isConnected))
	var r int
	for i:=0 ;i<len(isConnected); i++ {
		if !visited[i] {
			visited[i] = true
			findCircleNumB(isConnected, visited, i)
			r ++
		}
	}
	return r
}

func findCircleNumB(isConnected [][]int, visited []bool, i int) {
	for j:=0; j<len(isConnected[0]); j ++ {
		if isConnected[i][j] == 1 && !visited[j] {
			visited[j] = true
			findCircleNumB(isConnected, visited, j)
		}
	}
}

/**
448. 找到所有数组中消失的数字
https://leetcode.cn/problems/find-all-numbers-disappeared-in-an-array/description/
 */
func findDisappearedNumbers(nums []int) (ans []int) {
	n := len(nums)
	for _, v := range nums {
		v = (v-1) % n
		nums[v] += n
	}
	fmt.Println(nums)
	r := []int{}
	for i, v := range  nums {
		if v <= n {
			r = append(r, i+1)
		}
	}
	return r
}

/**
48. 旋转图像
https://leetcode.cn/problems/rotate-image/
 */
func rotate(matrix [][]int)  {
	n := len(matrix)
	for i := 0; i < n/2; i++ {
		for j := 0; j < (n+1)/2; j++ {
			/**
				i = 1, j = 0, n = 4
				1,0 => 0,2 | 0,2 => 2,3  | 2,3 => 3,1  | 3,1 => 1,0
			 */
			//matrix[j][len(matrix)-i-1] = matrix[i][j]  // 0,2 = 1,0
			//matrix[len(matrix)-i-1][len(matrix)-j-1] = matrix[j][len(matrix)-i-1] // 2,3 = 0,2
			//matrix[len(matrix)-j-1][i] = matrix[len(matrix)-i-1][len(matrix)-j-1] // 3,1 = 2,3
			//matrix[i][j] = matrix[len(matrix)-j-1][i] // 1,0 = 3,1
			matrix[i][j], matrix[j][len(matrix)-i-1], matrix[len(matrix)-i-1][len(matrix)-j-1], matrix[len(matrix)-j-1][i] =
				matrix[len(matrix)-j-1][i], matrix[i][j], matrix[j][len(matrix)-i-1], matrix[len(matrix)-i-1][len(matrix)-j-1]
		}
	}
}

/**
240. 搜索二维矩阵 II
https://leetcode.cn/problems/search-a-2d-matrix-ii/
 */
func searchMatrix(matrix [][]int, target int) bool {
	if len(matrix) < 1 {
		return false
	}
	i, j := 0, len(matrix[0])
	for i<len(matrix) || j>=0 {
		if j<0 || i>len(matrix)-1 {
			return false
		}
		if matrix[i][j] > target {
			j --
		} else if matrix[i][j] < target {
			i ++
		} else {
			return true
		}
	}
	return false
}

/**
769. 最多能完成排序的块
https://leetcode.cn/problems/max-chunks-to-make-sorted/
 */
func maxChunksToSorted(arr []int) int {
	r, max := 0, -1
	for k, v := range arr {
		if v > max {
			max = v
		}
		if max == k {
			r ++
			max = -1
		}
	}
	return r
}

/**
64. 最小路径和
https://leetcode.cn/problems/minimum-path-sum/
 */
func minPathSum2(grid [][]int) int {
	dp := make([][]int, len(grid))
	for i:=0; i<len(grid); i++ {
		dp[i] = make([]int, len(grid[i]))
	}
	for i:=0; i<len(grid); i++ {
		for j:=0; j<len(grid[0]); j++ {
			if i == 0 && j == 0 {
				dp[i][j] = grid[i][j]
			} else if i == 0 { // 第一列
				dp[i][j] = grid[i][j] + dp[i][j-1]
			} else if j == 0 { // 第一行
				dp[i][j] = grid[i][j] + dp[i-1][j]
			} else {
				dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
			}
		}
	}
	return dp[len(grid)-1][len(grid[0])-1]
}

func minPathSum3(grid [][]int) int {
	dp := make([]int, len(grid[0]))
	for i:=0; i<len(grid); i++ {
		for j := 0; j < len(grid[0]); j++ {
			if i==0 && j==0 {
				dp[j] = grid[i][j]
			} else if i == 0 {
				dp[j] = dp[j-1] + grid[i][j]
			} else if j == 0 {
				dp[j] = dp[j] + grid[i][j]
			} else {
				dp[j] = min(dp[j], dp[j-1])+grid[i][j]
			}
		}
	}
	return dp[len(grid[0])-1]
}


func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}


/**
232. 用栈实现队列
https://leetcode.cn/problems/implement-queue-using-stacks/
*/
type MyStack struct {
	Data []int
}

func NewMyStack() *MyStack {
	return &MyStack{}
}

func (s *MyStack) Push(x int)  {
	if len(s.Data) < 1 {
		s.Data = make([]int, 0)
	}
	s.Data = append(s.Data, x)
}

func (s *MyStack) Pop() int {
	if len(s.Data) < 1 {
		return 0
	}
	r := s.Data[len(s.Data)-1]
	s.Data = s.Data[:len(s.Data)-1]
	return r
}

func (s *MyStack) Peek() int {
	if len(s.Data) < 1 {
		return 0
	}
	return s.Data[len(s.Data)-1]
}

func (s *MyStack) Empty() bool {
	return len(s.Data) < 1
}

func (s *MyStack) size() int {
	return len(s.Data)
}

type MyQueue struct {
	popStack *MyStack
	pushStack *MyStack
}

func Constructor1() MyQueue {
	return MyQueue{
		popStack: NewMyStack(),
		pushStack: NewMyStack(),
	}
}

func (q *MyQueue) Push(x int)  {
	if q.pushStack == nil {
		q.pushStack = &MyStack{}
	}
	q.pushStack.Push(x)
}

func (q *MyQueue) Pop() int {
	if len(q.pushStack.Data) < 1 && len(q.popStack.Data) < 1 {
		return 0
	}
	if len(q.popStack.Data) < 1 {
		for !q.pushStack.Empty() {
			q.popStack.Push(q.pushStack.Pop())
		}
	}
	return q.popStack.Pop()
}

func (q *MyQueue) Peek() int {
	if len(q.pushStack.Data) < 1 && len(q.popStack.Data) < 1 {
		return 0
	}
	if len(q.popStack.Data) < 1 {
		for !q.pushStack.Empty() {
			q.popStack.Push(q.pushStack.Pop())
		}
	}
	return q.popStack.Peek()
}

func (q *MyQueue) Empty() bool {
	return len(q.popStack.Data) < 1 && len(q.pushStack.Data) < 1
}


/**
155. 最小栈
https://leetcode.cn/problems/min-stack/
 */
type MinStack struct {
	Data []int
	MinData []int
}

func Constructor() MinStack {
	return MinStack{
		Data: make([]int, 0),
		MinData: make([]int, 0),
	}
}

func (s *MinStack) Push(val int)  {
	s.Data = append(s.Data, val)
	if len(s.MinData) < 1 || s.MinData[len(s.MinData)-1] >= val {
		s.MinData = append(s.MinData, val)
	}
	return
}

func (s *MinStack) Pop()  {
	if len(s.Data) < 1 {
		return
	}
	val := s.Data[len(s.Data)-1]
	s.Data = s.Data[:len(s.Data)-1]
	if s.MinData[len(s.MinData)-1] == val {
		s.MinData = s.MinData[:len(s.MinData)-1]
	}
	return
}

func (s *MinStack) Top() int {
	if len(s.Data) < 1 {
		return 0
	}
	return s.Data[len(s.Data)-1]
}

func (s *MinStack) GetMin() int {
	if len(s.MinData) < 1 {
		return 0
	}
	return s.MinData[len(s.MinData)-1]
}

/**
20. 有效的括号
https://leetcode.cn/problems/valid-parentheses/
 */
func isValid(s string) bool {
	stack := make([]string, 0)
	if len(s) < 1 {
		return true
	}
	for _, v := range s {
		if v == '{' || v == '[' || v == '(' {
			stack = append(stack, string(v))
		} else {
			if len(stack) < 1 {
				return false
			}
			for len(stack) > 0 {
				p := stack[len(stack)-1]
				stack = stack[:len(stack)-1]
				if (v == '}' && p != "{") || (v == ']' && p != "[") || (v == ')' && p != "(") {
					return false
				}
				break
			}
		}
	}
	if len(stack) >0 {
		return false
	}
	return true
}

/**
739. 每日温度
https://leetcode.cn/problems/daily-temperatures/
 */
func dailyTemperatures(temperatures []int) []int {
	// [73,74,75,71,69,72,76,73]
	stack, r := make([]int, 0), make([]int, len(temperatures))
	for k, v := range temperatures {
		for len(stack) > 0 {
			peek := stack[len(stack)-1]
			if temperatures[peek] < v {
				stack = stack[:len(stack)-1]
				r[peek] = k - peek
			} else {
				break
			}
		}
		stack = append(stack, k)
	}
	return r
}

/**
23. 合并 K 个升序链表
https://leetcode.cn/problems/merge-k-sorted-lists/
 */

type ListNode struct {
	Val  int
	Next *ListNode
}

func mergeKLists(lists []*ListNode) *ListNode {
	if len(lists) < 1 {
		return nil
	}
	head := &ListNode {Val: 0}
	tmpHead := head
	for _, node := range lists {
		tmpHead.Next = node
		for tmpHead.Next != nil {
			tmpHead = tmpHead.Next
		}
	}
	r := heapSort(head.Next)
	return r
}

type Heap struct {
	data []*ListNode
}

func NewHeap() *Heap {
	return &Heap{
		data: make([]*ListNode, 0),
	}
}

func (heap *Heap) siftUp(node *ListNode)  {
	if len(heap.data) < 1 {
		heap.data = append(heap.data, []*ListNode{{Val: 0}, node}...)
		return
	}
	heap.data = append(heap.data, node)
	i := len(heap.data)-1
	for i > 1 {
		p := i/2
		if heap.data[i].Val < heap.data[p].Val {
			heap.data[i], heap.data[p] = heap.data[p], heap.data[i]
			i = p
			continue
		}
		break
	}
	return
}

func (heap *Heap) siftDown(node *ListNode)  {
	if len(heap.data) < 1 {
		heap.data = append(heap.data, []*ListNode{{Val: 0}, node}...)
		return
	}
	i := 1
	for i < len(heap.data) {
		p := 2*i
		if p >= len(heap.data) {
			break
		}
		if p+1<len(heap.data) && heap.data[p+1].Val < heap.data[p].Val {
			p ++
		}
		if heap.data[p].Val < node.Val {
			heap.data[p], heap.data[i] = heap.data[i], heap.data[p]
			i = p
			continue
		}
		break
	}
	if i < len(heap.data) {
		heap.data[i] = node
	}
	return
}

func (heap *Heap) extractMin() *ListNode {
	defer func() {
		lastNode := heap.data[len(heap.data)-1]
		heap.data = heap.data[:len(heap.data)-1]
		heap.siftDown(lastNode)
	}()
	return heap.data[1]
}

func heapSort(node *ListNode) (head *ListNode) {
	heap := NewHeap()
	for node != nil {
		tmpNode := &ListNode{
			Val: node.Val,
		}
		heap.siftUp(tmpNode)
		node = node.Next
	}
	heap.printNodeVal()

	head = &ListNode{
		Val: 0,
		Next: nil,
	}
	/**
	1 - 4 - 3 - 2
	 */
	tmp := head
	for len(heap.data) > 1 {
		tmp.Next = heap.extractMin()
		tmp = tmp.Next
	}
	return head.Next
}

func (heap *Heap) printNodeVal() {
	r := []int{}
	for _, node := range heap.data {
		r = append(r, node.Val)
		node = node.Next
	}
	fmt.Println(r)
	return
}

/**
239. 滑动窗口最大值
https://leetcode.cn/problems/sliding-window-maximum/
 */
var a []int
type hp struct {
	sort.IntSlice
}
func (h hp) Less(i, j int) bool {
	return a[h.IntSlice[i]] > a[h.IntSlice[j]]
}

func (h *hp) Push(i interface{}) {
	h.IntSlice = append(h.IntSlice, i.(int))
}

func (h *hp) Pop() interface{} {
	l := h.IntSlice[len(h.IntSlice)-1]
	h.IntSlice = h.IntSlice[:len(h.IntSlice)-1]
	return l
}

func maxSlidingWindow(nums []int, k int) []int {
	a = nums
	hp := &hp{make([]int, k)}
	for i:=0; i<k; i++ {
		hp.IntSlice[i] = i
	}
	heap.Init(hp)
	r := make([]int, 1)
	r[0] = nums[hp.IntSlice[0]]
	for i:=k; i<len(nums); i++ {
	heap.Push(hp, i)
		for hp.IntSlice[0] <= i-k {
			heap.Pop(hp)
		}
		r = append(r, nums[hp.IntSlice[0]])
	}
	return r
}

// 1,3,-1,-3,5,3,6,7
func maxSlidingWindow1(nums []int, k int) []int {
	q := make([]int, 0)

	push := func(i int) {
		for len(q) > 0 && nums[i] > nums[q[len(q)-1]]  {
			q = q[:len(q)-1]
		}
		q = append(q, i)
	}

	for i:=0; i<k; i++ {
		push(i)
	}
	r := make([]int, 0)
	r = append(r, nums[q[0]])
	for i:=k; i<len(nums); i++ {
		push(i)
		for q[0] <= i-k {
			q = q[1:]
		}
		r = append(r, nums[q[0]])
	}
	return r
}

/**
1. 两数之和
https://leetcode.cn/problems/two-sum/description/
 */
func twoSum(nums []int, target int) []int {
	if len(nums) < 1 {
		return []int{}
	}
	mp :=make(map[int]int, len(nums))
	for i:=0; i<len(nums); i++ {
		if k, ok := mp[target-nums[i]]; ok {
			return []int{i, k}
		}
		mp[nums[i]] = i
	}
	return []int{}
}