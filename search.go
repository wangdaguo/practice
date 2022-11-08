package main

import (
	"fmt"
	"sort"
)

func main()  {
	//grid := [][]int{
	//	{0,0,1,0,0,0,0,1,0,0,0,0,0},
	//	{0,0,0,0,0,0,0,1,1,1,0,0,0},
	//	{0,1,1,0,1,0,0,0,0,0,0,0,0},
	//	{0,1,0,0,1,1,0,0,1,0,1,0,0},
	//	{0,1,0,0,1,1,0,0,1,1,1,0,0},
	//	{0,0,0,0,0,0,0,0,0,0,1,0,0},
	//	{0,0,0,0,0,0,0,1,1,1,0,0,0},
	//	{0,0,0,0,0,0,0,1,1,0,0,0,0},
	//}
	//r := maxAreaOfIsland1(grid)
	//fmt.Println(r)
	//

	//isConnected := [][]int{
	//	{1,0,0,1},	//0,0|0,3  1
	//	{0,1,1,0},  //1,1|1,2  1
	//	{0,1,1,1},	//2,1|2,2|2,3   1
	//	{1,0,1,1},	//3,0|3,2|3,3
	//}
	//r := findCircleNum123(isConnected)
	//fmt.Println(r)
	//return

	//r := findCircleNumByUnionFind(isConnected)
	//fmt.Println(r)

	//r := findCircleNum(isConnected)
	//fmt.Println(r)

	//nums := []int{0, 0, 2, 1, 3, 2, 4}
	//uf := NewUnionFind(nums)
	//fmt.Println(uf)
	//
	//uf.union(2, 6)
	//fmt.Println(uf)

	//heights := [][]int{
	//	{1,2,3},
	//	{8,9,4},
	//	{7,6,5},
	//}

	//[[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]
	//r := pacificAtlantic(heights)

	//r := permute1([]int{1,2,3})

	//r := combine(3, 2)

	r := permuteUnique([]int{0,1,0,0,9})

	//r := combinationSum([]int{2,3,6,7}, 7)
	//r := combinationSum3([]int{10,1,2,7,6,1,5}, 8)
	//r := subsets([]int{1,2,3})
	//r := subsetsWithDup([]int{2,2,2})
	fmt.Println(r)
	return
}

/**
https://leetcode.cn/problems/subsets-ii/
 */
func subsetsWithDup(nums []int) [][]int {
	if len(nums) < 1 {
		return [][]int{}
	}
	sort.Slice(nums, func(i, j int) bool {
		if nums[i] < nums[j] {
			return true
		}
		return false
	})
	//r, path, check, level := make([][]int, 0), make([]int, 0), make([]bool, len(nums)), 0
	r, path, level := make([][]int, 0), make([]int, 0), 0
	backTrace6(nums, path, level, &r)
	return r
}

func backTrace6(nums, path []int, level int, r *[][]int) {
	tmp := make([]int, 0)
	tmp = append(tmp, path...)
	*r = append(*r, tmp)

	for i := level; i < len(nums); i++ {
		if i > level && nums[i] == nums[i-1]  {
			continue
		}
		path = append(path, nums[i])
		backTrace6(nums, path, i+1, r)
		path = path[:len(path)-1]
	}
}

func subsets(nums []int) [][]int {
	if len(nums) < 1 {
		return [][]int{}
	}
	r, level, path := make([][]int, 0), 0, make([]int, 0)
	backTrace5(nums, path, level, &r)
	r = append(r, []int{})
	return r
}

func backTrace5(nums, path []int, level int, r *[][]int) {
	if len(path) > 0 {
		tmp := make([]int, 0)
		tmp = append(tmp, path...)
		*r = append(*r, tmp)
		if len(path) == len(nums) {
			return
		}
	}
	for i:=level; i<len(nums); i++ {
		path = append(path, nums[i])
		backTrace5(nums, path, i+1, r)
		path = path[:len(path)-1]
	}
}

/**
https://leetcode.cn/problems/combination-sum/
 */
func combinationSum(candidates []int, target int) [][]int {
	if len(candidates) < 1 && target > 0 {
		return [][]int{}
	}
	r, path := make([][]int, 0), make([]int, 0)
	backTrace3(candidates, path, 0, target, &r)
	return r
}

func backTrace3(nums, path []int, level, target int, r *[][]int)  {
	if sumInt(path) > target {
		return
	}
	if sumInt(path) == target {
		tmp := make([]int, 0)
		tmp = append(tmp, path...)
		*r = append(*r, tmp)
		return
	}
	for i:=level; i<len(nums); i++ {
		path = append(path, nums[i])
		backTrace3(nums, path, i, target, r)
		path = path[:len(path)-1]
	}
}

func sumInt(path []int) int {
	var sum int
	for _, val := range path {
		sum += val
	}
	return sum
}

/**
https://leetcode.cn/problems/permutations-ii/
 */
func permuteUnique(nums []int) [][]int {
	if len(nums) < 1 {
		return [][]int{}
	}
	sort.Slice(nums, func(i, j int) bool {
		if nums[i] < nums[j] {
			return true
		}
		return false
	})
	check := make(map[int]bool)
	r, level, path := make([][]int, 0), 0, make([]int, 0)
	backTrace2(nums, path, level, &r, check)
	return r
}

func backTrace2(nums, path []int, level int, r *[][]int, check map[int]bool)  {
	if level == len(nums) {
		tmp := make([]int, 0)
		tmp = append(tmp, path...)
		*r = append(*r, tmp)
		return
	}
	for i:=0; i<len(nums); i++ {
		if check[i] {
			continue
		}
		if i > 0 && check[i-1] == false && nums[i] == nums[i-1] {
			continue
		}
		check[i] = true
		path = append(path, nums[i])
		backTrace2(nums, path, level+1, r, check)
		check[i] = false
		path = path[:len(path)-1]
	}
}

func combine(n int, k int) [][]int {
	if n < k {
		return [][]int{}
	}
	nums, path := make([]int, 0), make([]int, 0)
	for i:=1; i<=n; i++ {
		nums = append(nums, i)
	}
	level := 0
	r := make([][]int, 0)
	backTrack1(nums, path, level, k, &r)
	return r
}

func backTrack1(nums, path []int, level, k int, r *[][]int)  {
	if len(path) == k {
		tmp := make([]int, 0)
		tmp = append(tmp, path...)
		*r = append(*r, tmp)
		return
	}
	for i:=level; i<len(nums); i++ {
		path = append(path, nums[i])
		fmt.Printf("  递归之前 => %v\n", path)
		backTrack1(nums, path, i+1, k, r)
		path = path[0:len(path)-1]
		fmt.Printf("递归之后 => %v\n", path)
	}
}

func permute1(nums []int) [][]int {
	if len(nums) < 1 {
		return [][]int{}
	}
	r, check, path := make([][]int, 0), make([]bool, len(nums)), make([]int, 0)
	backTraceT(nums, path, check, &r)
	return r
}

func backTraceT(nums, path []int, check []bool, r *[][]int)  {
	if len(path) == len(nums) {
		var tmp []int
		tmp = append(tmp, path...)
		*r = append(*r, tmp)
		return
	}
	for i:=0; i<len(nums); i++ {
		if check[i] {
			continue
		}
		check[i] = true
		path = append(path, nums[i])
		backTraceT(nums, path, check, r)
		check[i] = false
		path = path[0:len(path)-1]
	}
}

func pacificAtlantic(heights [][]int) [][]int {
	if len(heights) < 1 {
		return [][]int{}
	}
	pacific := make([][]bool, len(heights))
	atlantic := make([][]bool, len(heights))
	for i := range pacific {
		pacific[i] = make([]bool, len(heights[0]))
		atlantic[i] = make([]bool, len(heights[0]))
	}
	for i:=0; i<len(heights); i++ {
		dfs(heights, i, 0, &pacific)
	}
	for j:=0; j<len(heights[0]); j++ {
		dfs(heights, 0, j, &pacific)
	}
	for i:=0; i<len(heights); i++ {
		dfs(heights, i, len(heights[0])-1, &atlantic)
	}
	for j:=0; j<len(heights[0]); j++ {
		dfs(heights, len(heights)-1, j, &atlantic)
	}

	r := make([][]int, 0)
	for i:=0; i<len(pacific); i++ {
		for j:=0; j<len(pacific[0]); j ++ {
			if pacific[i][j] && atlantic[i][j] {
				r = append(r, []int{i, j})
			}
		}
	}
	return r
}

func dfs(heights [][]int, i, j int, ocean *[][]bool)  {

	//if (*ocean)[i][j] {
	//	return
	//}
	//(*ocean)[i][j] = true
	//for k := 0; k < 4; k++ {
	//	x := i + direction[k]
	//	y := j + direction[k+1]
	//	if x >= 0 && x < len(heights) && y >= 0 && y < len(heights[0]) && heights[x][y] >= heights[i][j] {
	//		dfs(heights, x, y, ocean)
	//	}
	//}

	if (*ocean)[i][j] {
		return
	}
	(*ocean)[i][j] = true
	stack := NewStack()
	stack.push(NewPoint(i, j))
	for !stack.isEmpty() {
		p := stack.pop()
		for k := 0; k < 4; k++ {
			x := p.X + direction[k]
			y := p.Y + direction[k+1]
			if x >= 0 && x < len(heights) && y >= 0 && y < len(heights[0]) && heights[x][y] >= heights[p.X][p.Y] {
				stack.push(NewPoint(x, y))
				(*ocean)[x][y] = true
			}
		}
	}
	return
}

//func dfs1(heights [][]int, i, j int, ocean [][]bool)  {
//	stack := NewStack()
//	for i:=0; i<len(heights); i++ {
//		for j:=0; j<len(heights[0]); j++ {
//			if !ocean[i][j] {
//				ocean[i][j] = true
//				stack.push(NewPoint(i, j))
//			}
//			for !stack.isEmpty() {
//				p := stack.pop()
//				for k:=0; k<4; k++ {
//					x := p.X + direction[k]
//					y := p.Y + direction[k+1]
//					if x >=0 && x < len(heights) && y>=0 && y<len(heights[0]) && heights[x][y] > heights[p.X][p.Y] {
//						dfs(heights, x, y, ocean)
//					}
//				}
//			}
//		}
//	}
//}

func findCircleNumByStack1(isConnected [][]int) int {
	if len(isConnected) < 1 {
		return 0
	}
	vis := make([]bool, len(isConnected))
	cnt := 0
	var dfs func(int)
	dfs = func(i int) {
		vis[i] = true
		for j:=0; j<len(isConnected[0]); j++ {
			if isConnected[i][j] == 1 && !vis[j] {
				dfs(j)
			}
		}
	}
	for i:=0; i<len(vis); i++ {
		if !vis[i] {
			cnt ++
			dfs(i)
		}
	}
	return cnt
}

func findCircleNum123(isConnected [][]int) int {
	if len(isConnected) < 1 {
		return 0
	}
	vis := make([]bool, len(isConnected))
	queue := make([]int, 0)
	cnt := 0
	for i:=0; i<len(vis); i++ {
		if !vis[i] {
			queue = append(queue, i)
			cnt ++
			for len(queue) > 0 {
				j := queue[0]
				queue = queue[1:]
				vis[j] = true
				for t:=0; t<len(isConnected[0]); t++ {
					if isConnected[j][t] == 1 && !vis[t] {
						queue = append(queue, t)
					}
				}
			}
		}
	}
	return cnt
}

func findCircleNumByStack(isConnected [][]int) int {
	if len(isConnected) < 1 {
		return 0
	}
	stack := NewStack()
	cnt := 0
	for i:=0; i<len(isConnected); i++ {
		for j:=0; j<len(isConnected[0]);j ++ {
			if isConnected[i][j] == 1 {
				stack.push(NewPoint(i, j))
				cnt ++
			}
			for !stack.isEmpty() {
				p := stack.pop()
				isConnected[p.X][p.Y] = 0
				for t:=0;t<len(isConnected[0]);t++{
					if isConnected[p.Y][t] == 1 {
						stack.push(NewPoint(p.Y, t))
					}
				}
			}
		}
	}
	return cnt
}

func findCircleNumByUnionFind(isConnected [][]int) int {
	arr := make([]int, len(isConnected))
	for i:=0; i<len(arr); i++ {
		arr[i] = i
	}
	cnt := len(isConnected)
	for i:=0; i<len(isConnected); i++ {
		for j:=i+1; j<len(isConnected[0]); j++ {
			if isConnected[i][j] == 1 && Union(arr, i, j) {
				cnt --
			}
		}
	}
	return cnt
}

func Union(arr []int, i, j int) bool {
	fatherI := findFather(arr, i)
	fatherJ := findFather(arr, j)
	if fatherI != fatherJ {
		arr[fatherJ] = arr[fatherI]
		return true
	}
	return false
}

func findFather(arr []int, i int) int {
	for i != arr[i] {
		i = arr[i]
	}
	return i
}


type UnionFind struct {
	Val []int
	Size map[int]int
}

func NewUnionFind(val []int) *UnionFind {
	uf := &UnionFind{
		Val: val,
	}
	uf.Size = make(map[int]int)
	for i:=0; i<len(val); i++ {
		uf.findRoot(i)
	}
	for i:=0; i<len(val); i++ {
		uf.Size[uf.Val[i]] ++
	}
	return uf
}

func (uf *UnionFind) union(x, y int)  {
	if  uf.connected(x, y) {
		return
	}
	rootX := uf.findRoot(x)
	rootY := uf.findRoot(y)
	if uf.Size[rootX] > uf.Size[rootY] {
		uf.Val[rootY] = rootX
		uf.Size[rootX] += uf.Size[rootY]
		delete(uf.Size, rootY)
		return
	}
	uf.Val[rootX] = rootY
	uf.Size[rootY] += uf.Size[rootX]
	delete(uf.Size, rootX)
	return
}

func (uf *UnionFind) connected(x, y int) bool {
	rootX := uf.findRoot(x)
	rootY := uf.findRoot(y)
	return rootX == rootY
}

func (uf *UnionFind) findRoot(x int) int {
	for x != uf.Val[x] {
		uf.Val[x] = uf.Val[uf.Val[x]]
		x = uf.Val[x]
	}
	return x
}

type Point struct {
	X int
	Y int
}

func NewPoint(x, y int) *Point {
	return &Point{
		X: x,
		Y: y,
	}
}

type Stack struct {
	Data []*Point
}

func NewStack() *Stack {
	data := make([]*Point, 0)
	return &Stack{Data:data}
}

func (stack *Stack) top() *Point {
	return stack.Data[0]
}

func (stack *Stack) pop() *Point {
	r := stack.Data[0]
	stack.Data = stack.Data[1:]
	return r
}

func (stack *Stack) push(p *Point) {
	if stack.isEmpty() {
		stack.Data = append(stack.Data, p)
		return
	}
	stack.Data = append([]*Point{p}, stack.Data...)
	//copy(stack.Data[1:], stack.Data[0:])
	//stack.Data[0] = p
	return
}

func (stack *Stack) isEmpty() bool {
	return len(stack.Data) == 0
}

var direction = []int{-1, 0, 1, 0, -1}

/**
695. 岛屿的最大面积
https://leetcode.cn/problems/max-area-of-island/submissions/
 */
func maxAreaOfIsland(grid [][]int) int {
	stack := NewStack()
	maxArea, tmpArea := 0, 0
	for i:=0; i<len(grid); i++ {
		for j:=0; j<len(grid[0]); j++ {
			if grid[i][j] == 1 {
				stack.push(NewPoint(i, j))
				tmpArea = 1
				grid[i][j] = 0
			}
			for !stack.isEmpty() {
				p := stack.pop()
				for k:=0; k<4; k++ {
					x := p.X + direction[k]
					y := p.Y + direction[k+1]
					if x >=0 && x < len(grid) && y >=0 && y < len(grid[0]) && grid[x][y] == 1 {
						grid[x][y] = 0
						stack.push(NewPoint(x, y))
						tmpArea ++
					}
				}
			}
			maxArea = maxX(maxArea, tmpArea)
		}
	}
	return maxArea
}

func maxX(x, y int) int {
	if x >= y {
		return x
	}
	return y
}

func maxAreaOfIsland1(grid [][]int) int {
	var max int
	for i:=0; i<len(grid); i++ {
		for j:=0; j<len(grid[0]); j++ {
			if grid[i][j] == 1 {
				max = maxX(max, maxAreaOfIslandImpl(grid, i, j))
			}
		}
	}
	return max
}

func maxAreaOfIslandImpl(grid [][]int, i, j int) int {
	if i < 0 || j < 0 || i > len(grid)-1 || j > len(grid[0])-1 || grid[i][j] == 0 {
		return 0
	}
	grid[i][j] = 0
	return 1 + maxAreaOfIslandImpl(grid, i+1, j) + maxAreaOfIslandImpl(grid, i-1, j) + maxAreaOfIslandImpl(grid, i, j-1) +
		maxAreaOfIslandImpl(grid, i, j+1)
}

/**
547. 省份数量
https://leetcode.cn/problems/number-of-provinces/description/
 */
func findCircleNum(isConnected [][]int) int {
	if len(isConnected) < 1 {
		return 0
	}
	cnt := 0
	for i:=0; i<len(isConnected); i++ {
		for j:=0; j<len(isConnected[0]); j++ {
			if isConnected[i][j] == 1 {
				findCircleNumImpl(isConnected, i, j)
				cnt ++
			}
		}
	}
	return cnt
}

func findCircleNumImpl(isConnected [][]int, i, j int) {
	if i<0 || i>len(isConnected)-1 || j < 0 || j > len(isConnected[0])-1 || isConnected[i][j] != 1 {
		return
	}
	isConnected[i][j] = 0
	for t:=0; t<len(isConnected[0]); t++ {
		if isConnected[j][t] == 1 {
			findCircleNumImpl(isConnected, j, t)
		}
	}
}
