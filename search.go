package main

import (
	"fmt"
)

func main()  {
	grid := [][]int{
		{0,0,1,0,0,0,0,1,0,0,0,0,0},
		{0,0,0,0,0,0,0,1,1,1,0,0,0},
		{0,1,1,0,1,0,0,0,0,0,0,0,0},
		{0,1,0,0,1,1,0,0,1,0,1,0,0},
		{0,1,0,0,1,1,0,0,1,1,1,0,0},
		{0,0,0,0,0,0,0,0,0,0,1,0,0},
		{0,0,0,0,0,0,0,1,1,1,0,0,0},
		{0,0,0,0,0,0,0,1,1,0,0,0,0},
	}

	//s := NewStack()
	//s.push(NewPoint(1, 1))
	//s.push(NewPoint(2,2))
	//byteList, _ := json.Marshal(s)
	//fmt.Println(string(byteList))
	//a := s.pop()
	//fmt.Println(a)
	//return

	r := maxAreaOfIsland(grid)
	fmt.Println(r)
	return
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