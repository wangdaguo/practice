package main

import (
	"fmt"
	"math"
	"strings"
)

func main() {
	//r := isIsomorphic("paper", "title")
	r := wordPattern("abc", "b c a")
	fmt.Print(r)
}

/*
*
36. 有效的数独
https://leetcode.cn/problems/valid-sudoku/solutions/1001859/you-xiao-de-shu-du-by-leetcode-solution-50m6/
*/
func isValidSudoku(board [][]byte) bool {
	var rows, cols [9][9]int
	var subboxes [3][3][9]int
	for i, row := range board {
		for j, c := range row {
			if c == '.' {
				continue
			}
			idx := c - '1'
			rows[i][idx]++
			cols[j][idx]++
			subboxes[i/3][j/3][idx]++
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

/*
*
54. 螺旋矩阵
https://leetcode.cn/problems/spiral-matrix/solutions/7155/cxiang-xi-ti-jie-by-youlookdeliciousc-3/
*/
func spiralOrder(matrix [][]int) []int {
	u, d, l, r, ans := 0, len(matrix)-1, 0, len(matrix[0])-1, make([]int, 0)
	for {
		for i := l; i <= r; i++ { // 向右
			ans = append(ans, matrix[u][i])
		}
		u++
		if u > d { // 重新设定上边界
			break
		}
		for i := u; i <= d; i++ { // 向下
			ans = append(ans, matrix[i][r])
		}
		r--
		if r < l { // 重新设定右边界
			break
		}
		for i := r; i >= l; i-- { // 向左
			ans = append(ans, matrix[d][i])
		}
		d--
		if d < u { // 重新设定下边界
			break
		}
		for i := d; i >= u; i-- { // 向上
			ans = append(ans, matrix[i][l])
		}
		l++
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
	neighbors := []int{0, 1, -1}
	for row := 0; row < len(board); row++ {
		for col := 0; col < len(board[0]); col++ {

			alive := 0
			for i := 0; i < 3; i++ {
				for j := 0; j < 3; j++ {
					if !(neighbors[i] == 0 && neighbors[j] == 0) {
						r := row + neighbors[i]
						c := col + neighbors[j]
						if r >= 0 && r < len(board) && c >= 0 && c < len(board[0]) && abs(board[r][c]) == 1 {
							alive++
						}
					}
				}
			}

			if board[row][col] == 1 && (alive < 2 || alive > 3) {
				board[row][col] = -1
			}
			if board[row][col] == 0 && alive == 3 {
				board[row][col] = 2
			}
		}
	}
	for row := 0; row < len(board); row++ {
		for col := 0; col < len(board[0]); col++ {
			if board[row][col] > 0 {
				board[row][col] = 1
			} else {
				board[row][col] = 0
			}
		}
	}
	return
}

func abs(value int) int {
	if value < 0 {
		return -value
	}
	return value
}

/**
383. 赎金信
https://leetcode.cn/problems/ransom-note
 */
func canConstruct(ransomNote string, magazine string) bool {
	mp := make(map[byte]int)
	for _, ch := range magazine {
		mp[byte(ch)] ++
	}
	for _, ch := range ransomNote {
		cnt, ok := mp[byte(ch)]
		if !ok || cnt < 1 {
			return false
		}
		mp[byte(ch)] --
	}
	return true
}

/**
205. 同构字符串
https://leetcode.cn/problems/isomorphic-strings/?envType=study-plan-v2&envId=top-interview-150
 */
func isIsomorphic(s string, t string) bool {
	mp, set := make(map[rune]rune), make(map[rune]struct{})
	for i := range s {
		var b2 rune
		b1 := rune(s[i])
		if i < len(t) {
			b2 = rune(t[i])
		} else {
			b2 = rune(0)
		}
		/**
			r := isIsomorphic("paper", "title")  e=>l  r=>e
			r := isIsomorphic("badc", "baba")    b=>a  d=>a
		 */
		b1r, ok1 := mp[b1]
		_, ok2 := set[b2]
		if !ok1 && !ok2{
			mp[b1] = b2
			set[b2] = struct{}{}
		} else if (ok1 && b1r != b2) || (!ok1 && ok2) {   // ok1 || ok2
			return false
		}
	}
	return true
}

/**
290. 单词规律
https://leetcode.cn/problems/word-pattern/?envType=study-plan-v2&envId=top-interview-150
 */
func wordPattern(pattern string, s string) bool {
	wordList := strings.Split(s, " ")
	if len(pattern) != len(wordList) {
		return false
	}
	mp1, mp2 := make(map[string]string), make(map[string]string)
	for i, p := range pattern {
		s1, ok1 := mp1[string(p)]
		s2, ok2 := mp2[wordList[i]]
		if ok1 && ok2 && s1 == mp1[s2] {
			continue
		}
		if !ok1 && !ok2 {
			mp1[string(p)] = wordList[i]
			mp2[wordList[i]] = string(p)
			continue
		}
		return false
	}
	return true
}
