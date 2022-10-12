package main

import "fmt"

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
	//	{1,0,0,1},
	//	{0,1,1,0},
	//	{0,1,1,1},
	//	{1,0,1,1},
	//}
	//r := findCircleNum(isConnected)
	//fmt.Println(r)

	nums := []int{0, 0, 2, 1, 3, 2, 4}
	uf := NewUnionFind(nums)
	fmt.Println(uf)

	uf.union(2, 6)
	fmt.Println(uf)
	return
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

func (uf *UnionFind) size() int {
	return len(uf.Val)
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

/**
695. 岛屿的最大面积
https://leetcode.cn/problems/max-area-of-island/submissions/
 */
func maxAreaOfIsland(grid [][]int) int {
	direction := []int{-1, 0, 1, 0, -1}
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
