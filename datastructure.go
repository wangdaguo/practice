package main

import (
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
	r := []int{1,2,3}
	r = r[:len(r)-1]
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

func Constructor() MyQueue {
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