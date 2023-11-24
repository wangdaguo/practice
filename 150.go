package main

import (
	"fmt"
	"math"
)

func main() {
	fmt.Print(r)
}

/**
36. 有效的数独
https://leetcode.cn/problems/valid-sudoku/solutions/1001859/you-xiao-de-shu-du-by-leetcode-solution-50m6/
 */
func isValidSudoku(board [][]byte) bool {
	var rows, cols [9][9]int
	var subboxes [3][3][9] int
	for i, row := range board {
		for j, c := range row {
			if c == '.' {
				continue
			}
			idx := c - '1'
			rows[i][idx] ++
			cols[j][idx] ++
			subboxes[i/3][j/3][idx] ++
			if rows[i][idx] > 1 || cols[j][idx] > 1 || subboxes[i/3][j/3][idx] > 1 {
				return false
			}
		}
	}
	return true
}

/*
*
209. 长度最小的子数组
https://leetcode.cn/problems/minimum-size-subarray-sum/?envType=study-plan-v2&envId=top-interview-150
*/
func minSubArrayLen(s int, nums []int) int {
	if len(nums) < 1 {
		return 0
	}
	left, sum, r := 0, 0, math.MaxInt32
	for k, _ := range nums {
		sum += nums[k]
		for sum >= s {
			r = min(r, k-left+1)
			sum -= nums[left]
			left++
		}
	}
	if r != math.MaxInt32 {
		return r
	}
	return 0
}
func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

/**
54. 螺旋矩阵
https://leetcode.cn/problems/spiral-matrix/solutions/7155/cxiang-xi-ti-jie-by-youlookdeliciousc-3/
 */
func spiralOrder(matrix [][]int) []int {
	u, d, l, r, ans := 0, len(matrix)-1, 0, len(matrix[0])-1, make([]int, 0)
	for {
		for i:=l; i<=r; i++ {  // 向右
			ans = append(ans, matrix[u][i])
		}
		u ++
		if u > d {  // 重新设定上边界
			break
		}
		for i:=u; i<=d; i++ {  // 向下
			ans = append(ans, matrix[i][r])
		}
		r --
		if r < l {   // 重新设定右边界
			break
		}
		for i:=r; i>=l; i-- {  // 向左
			ans = append(ans, matrix[d][i])
		}
		d --
		if d < u {   // 重新设定下边界
			break
		}
		for i:=d; i>=u; i-- {   // 向上
			ans = append(ans, matrix[i][l])
		}
		l ++
		if l > r {
			break
		}
	}
	return ans
}


/*
*
3. 无重复字符的最长子串
https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/?envType=study-plan-v2&envId=top-interview-150
*/
func lengthOfLongestSubstring(s string) int {
	left, mp, r := 0, make(map[byte]int), 0
	for i := range s {
		idx, ok := mp[s[i]]
		if !ok {
			mp[s[i]] = i
			if len(mp) > r {
				r = len(mp)
			}
			continue
		}
		for left <= idx {
			delete(mp, s[left])
			left++
		}
		mp[s[i]] = i
	}
	return r
}

/*
*
73. 矩阵置零
https://leetcode.cn/problems/set-matrix-zeroes/?envType=study-plan-v2&envId=top-interview-150
*/
func setZeroes(matrix [][]int) {
	zeroRows, zeroCols := make([]int, 0), make([]int, 0)
	for i := 0; i < len(matrix); i++ {
		for j := 0; j < len(matrix[0]); j++ {
			if matrix[i][j] == 0 {
				zeroRows = append(zeroRows, i)
				zeroCols = append(zeroCols, j)
			}
		}
	}
	for _, i := range zeroRows {
		matrix[i] = make([]int, len(matrix[i]))
	}
	for _, i := range zeroCols {
		for j := 0; j < len(matrix); j++ {
			matrix[j][i] = 0
		}
	}
	return
}

/*
*
289. 生命游戏
https://leetcode.cn/problems/game-of-life/?envType=study-plan-v2&envId=top-interview-150
*/
func gameOfLife(board [][]int) {
	for i := 0; i < len(board); i++ {
		u, d := true, true
		for j := 0; j < len(board[0]); j++ {
			alive, die, l, r := 0, 0, true, true
			if i > 0 && i<len(board)-1 && j>0 && j<len(board[0])-1 {
				if board[i-1][j] == 0 {
					die++
				} else {
					alive++
				}
			}
			if i == 0 {

			}
			if i == len(board)-1 {
				u, d = false, true
				if board[i-1][j] == 0 {
					die++
				} else {
					alive++
				}
			}
			if j == 0 {

			}
			if j == len(board[0])-1 {

			}

			if board[i][j] == 1 && alive < 2 {
				// 小于2
			} else if board[i][j] == 1 && (alive == 2 || alive == 3) {
				// 2/3个
			} else if board[i][j] == 1 && alive > 3{
				// 超过3个
			} else if board[i][j] == 0 && alive == 3 {

			}
		}
	}
}
