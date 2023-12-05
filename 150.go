package main

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

func main() {
	//r := isIsomorphic("paper", "title")
	//r := wordPattern("abc", "b c a")
	//r := isAnagram("anagram", "nagaram")
	//r := longestConsecutive([]int{100, 4, 200, 1, 3, 2})\
	r := summaryRanges([]int{0, 1, 2, 4, 5, 7})
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

/*
*
383. 赎金信
https://leetcode.cn/problems/ransom-note
*/
func canConstruct(ransomNote string, magazine string) bool {
	mp := make(map[byte]int)
	for _, ch := range magazine {
		mp[byte(ch)]++
	}
	for _, ch := range ransomNote {
		cnt, ok := mp[byte(ch)]
		if !ok || cnt < 1 {
			return false
		}
		mp[byte(ch)]--
	}
	return true
}

/*
*
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
		if !ok1 && !ok2 {
			mp[b1] = b2
			set[b2] = struct{}{}
		} else if (ok1 && b1r != b2) || (!ok1 && ok2) { // ok1 || ok2
			return false
		}
	}
	return true
}

/*
*
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

/*
*
242. 有效的字母异位词
https://leetcode.cn/problems/valid-anagram/?envType=study-plan-v2&envId=top-interview-150
*/
func isAnagram(s string, t string) bool {
	if len(s) != len(t) {
		return false
	}
	mp := make(map[byte]int)
	for i := range s {
		if _, ok := mp[s[i]]; !ok {
			mp[s[i]] = 1
		} else {
			mp[s[i]]++
		}
		if _, ok := mp[t[i]]; !ok {
			mp[t[i]] = -1
		} else {
			mp[t[i]]--
		}
	}
	for _, v := range mp {
		if v != 0 {
			return false
		}
	}
	return true
}

/*
*
49. 字母异位词分组
https://leetcode.cn/problems/group-anagrams/?envType=study-plan-v2&envId=top-interview-150
*/
func groupAnagrams(strs []string) [][]string {
	r, mp := make([][]string, 0), make(map[string][]string)
	for _, s := range strs {
		bl := ByteList(s)
		bl.Sort()
		if _, ok := mp[string(bl)]; !ok {
			mp[string(bl)] = []string{s}
		} else {
			mp[string(bl)] = append(mp[string(bl)], s)
		}
	}
	for _, v := range mp {
		r = append(r, v)
	}
	return r
}

type ByteList []byte

func (bl ByteList) Len() int {
	return len(bl)
}

func (bl ByteList) Less(i, j int) bool {
	return bl[i] < bl[j]
}

func (bl ByteList) Swap(i, j int) {
	bl[i], bl[j] = bl[j], bl[i]
}

func (bl ByteList) Sort() {
	sort.Sort(bl)
}

/*
*
202. 快乐数
https://leetcode.cn/problems/happy-number/?envType=study-plan-v2&envId=top-interview-150
*/
func isHappy(n int) bool {
	sum, i, mp := 0, n, make(map[int]struct{})
	for i > 0 {
		sum += (i % 10) * (i % 10)
		i = i / 10
		if i == 0 {
			if sum == 1 {
				return true
			}
			if _, ok := mp[sum]; ok {
				return false
			}
			mp[sum] = struct{}{}
			i = sum
			sum = 0
		}
	}
	return false
}

/*
219. 存在重复元素 II
*https://leetcode.cn/problems/contains-duplicate-ii/?envType=study-plan-v2&envId=top-interview-150
*/
func containsNearbyDuplicate(nums []int, k int) bool {
	mp := make(map[int]int)
	for i := range nums {
		idx, ok := mp[nums[i]]
		if ok && i-idx <= k {
			return true
		}
		mp[nums[i]] = i
	}
	return false
}

/*
*
128. 最长连续序列
https://leetcode.cn/problems/longest-consecutive-sequence/?envType=study-plan-v2&envId=top-interview-150
*/
func longestConsecutive(nums []int) int {
	r, subLen, mp := 0, 0, make(map[int]bool)
	for i := range nums {
		mp[nums[i]] = true
	}
	for num := range mp {
		if !mp[num-1] {
			cur := num
			subLen = 1
			for mp[cur+1] {
				cur++
				subLen++
			}
			if subLen > r {
				r = subLen
			}
		}
	}
	return r
}

/*
*
228. 汇总区间
https://leetcode.cn/problems/summary-ranges/description/?envType=study-plan-v2&envId=top-interview-150
*/
func summaryRanges(nums []int) []string {
	if len(nums) == 0 {
		return []string{}
	}
	list, start, end := make([][]int, 0), 0, 0
	for i := range nums {
		if i == 0 {
			start, end = i, i
			continue
		}
		if nums[i]-nums[i-1] == 1 {
			end = i
		} else {
			list = append(list, []int{nums[start], nums[end]})
			start, end = i, i
		}
	}
	list = append(list, []int{nums[start], nums[end]})
	r := make([]string, 0)
	for i := range list {
		if list[i][0] == list[i][1] {
			r = append(r, fmt.Sprintf("%s", strconv.Itoa(list[i][1])))
		} else {
			r = append(r, fmt.Sprintf("%s->%s", strconv.Itoa(list[i][0]), strconv.Itoa(list[i][1])))
		}
	}
	return r
}

/*
*
56. 合并区间
https://leetcode.cn/problems/merge-intervals/?envType=study-plan-v2&envId=top-interview-150
*/
func merge(intervals [][]int) [][]int {

}
