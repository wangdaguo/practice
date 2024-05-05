package main

import (
	"container/heap"
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
	//r := summaryRanges([]int{0, 1, 2, 4, 5, 7})
	//r := merge([][]int{
	//	{2, 3},
	//	{4, 5},
	//	{6, 7},
	//	{8, 9},
	//	{1, 10},
	//})
	//r := insert([][]int{{1,2},{3,5},{6,7},{8,10},{12,16}}, []int{4,8})
	//r := insert([][]int{{1, 3}, {6, 9}}, []int{2, 5})
	//r := insert([][]int{{1, 5}}, []int{2, 3})
	//r := findMinArrowShots([][]int{{1, 2}, {3, 4}, {5, 6}, {7, 8}})
	//r := simplifyPath("/a/./b/../../c/")
	//r := evalRPN([]string{"4","13","5","/","+"})
	//node7 := &ListNode{
	//	Val:  7,
	//	Next: nil,
	//}
	//node6 := &ListNode{
	//	Val:  6,
	//	Next: node7,
	//}
	//head1:= &ListNode{
	//	Val:  1,
	//	Next: node6,
	//}

	//node5 := &ListNode{
	//	Val:  5,
	//	Next: nil,
	//}
	//node4 := &ListNode{
	//	Val:  4,
	//	Next: node5,
	//}
	//node3 := &ListNode{
	//	Val:  3,
	//	Next: node4,
	//}
	//node2 := &ListNode{
	//	Val:  2,
	//	Next: node3,
	//}
	//head2 := &ListNode{
	//	Val:  1,
	//	Next: node2,
	//}
	////r := addTwoNumbers(head1, head2)
	//r := reverseBetween(head2, 2, 4)
	//PrintList(r)
	//fmt.Print(r)
	//
	//grid := [][]byte{
	//	//{'1', '1', '0', '0', '0'},
	//	//{'1', '1', '0', '0', '0'},
	//	//{'0', '0', '1', '0', '0'},
	//	//{'0', '0', '0', '1', '1'},
	//	{'1', '1', '1', '1', '0'},
	//	{'1', '1', '0', '1', '0'},
	//	{'1', '1', '0', '0', '0'},
	//	{'0', '0', '0', '0', '0'},
	//}
	//r := numIslands(grid)
	//r := letterCombinations("3")
	//r := combine(4, 2)
	//r := permute([]int{1, 2, 3})
	//r := permuteUnique([]int{1, 1, 2})
	//r := combinationSum3(3, 9)
	//r := subsets([]int{1, 2, 3})
	//r := combinationSum4([]int{1, 2, 3}, 4)
	//r := subsetsWithDup([]int{1, 2, 2})
	//r := exist([][]byte{
	//	{'A', 'B', 'C', 'E'},
	//	{'S', 'F', 'C', 'S'},
	//	{'A', 'D', 'E', 'E'},
	//}, "ABCB")
	//r := generateParenthesis(3)

	//p := Person{
	//	Name:    "Alice",
	//	Age:     30,
	//	Friends: []string{"Bob", "Charlie"},
	//}
	//
	//r1 := []int{30, 40}
	//
	//fmt.Println("Before:")
	//fmt.Printf("person len: %d, cap: %d\n", len(p.Friends), cap(p.Friends))
	//fmt.Printf("r1 len: %d, cap: %d\n", len(r1), cap(r1))
	//
	//p.AddFriend("David")
	//T1(r1)
	//
	//fmt.Println("After:")
	//fmt.Printf("person len: %d, cap: %d\n", len(p.Friends), cap(p.Friends))
	//fmt.Printf("r1 len: %d, cap: %d\n", len(r1), cap(r1))
	//r := sortedArrayToBST([]int{-10, -3, 0, 5, 9})
	//r := maxSubArray([]int{-2, 1})
	//r := minSubArrayLen(7, []int{2, 3, 1, 2, 4, 3})
	//r := searchInsert([]int{1, 3, 5, 6}, 2)
	//r := searchMatrix([][]int{{1, 3}}, 3)
	//r := search([]int{3, 1}, 1)
	//r := findKthLargest2([]int{3, 2, 1, 5, 6, 4}, 2)
	//r := findMaximizedCapital1(2, 0, []int{1, 2, 3}, []int{0, 9, 10})
	obj := ConstructorM()
	obj.AddNum(1)
	obj.AddNum(2)
	fmt.Println(obj.FindMedian())
	obj.AddNum(3)
	fmt.Println(obj.FindMedian())
	//fmt.Println(r)
}

type Person struct {
	Name    string
	Age     int
	Friends []string
}

func (p *Person) AddFriend(friend string) {
	p.Friends = append(p.Friends, friend)
}

func T1(b []int) {
	for i := 0; i < 5; i++ {
		b = append(b, i)
	}
	return
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
	sort.Slice(intervals, func(i, j int) bool {
		if intervals[i][0] < intervals[j][0] || (intervals[i][0] == intervals[j][0] && intervals[i][1] < intervals[j][1]) {
			return true
		}
		return false
	})
	fmt.Println(intervals)
	r, start, end := make([][]int, 0), intervals[0][0], intervals[0][1]
	for i := 1; i < len(intervals); i++ {
		// [1,3] [2,4]
		if end >= intervals[i][0] && intervals[i][1] >= end {
			end = intervals[i][1]
		} else if start <= intervals[i][0] && end >= intervals[i][1] { // [1,4] [1,3]
			continue
		} else if start >= intervals[i][0] && end <= intervals[i][1] { // [1,4] [1, 9]
			end = intervals[i][1]
		} else {
			r = append(r, []int{start, end})
			start, end = intervals[i][0], intervals[i][1]
		}
	}
	r = append(r, []int{start, end})
	return r
}

/*
*
57. 插入区间
https://leetcode.cn/problems/insert-interval/description/?envType=study-plan-v2&envId=top-interview-150
{{1,5}}, []int{2,3}
*/
func insert(intervals [][]int, newInterval []int) [][]int {
	if len(intervals) < 1 {
		return [][]int{newInterval}
	}
	left, right, merged, r := newInterval[0], newInterval[1], false, [][]int{}
	for _, interval := range intervals {
		if interval[0] > newInterval[1] {
			if !merged {
				merged = true
				r = append(r, []int{left, right})
			}
			r = append(r, interval)
		} else if interval[1] < newInterval[0] {
			r = append(r, interval)
		} else {
			left = min(left, interval[0])
			right = max(right, interval[1])
		}
	}
	if !merged {
		r = append(r, []int{left, right})
	}
	return r
}

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

func max(i, j int) int {
	if i > j {
		return i
	}
	return j
}

/*
*
452. 用最少数量的箭引爆气球
https://leetcode.cn/problems/minimum-number-of-arrows-to-burst-balloons/description/?envType=study-plan-v2&envId=top-interview-150
*/
func findMinArrowShots(points [][]int) int {
	if len(points) < 1 {
		return 0
	}
	sort.Slice(points, func(i, j int) bool {
		if points[i][0] < points[j][0] {
			return true
		}
		return false
	})
	cnt, left, right := 1, points[0][0], points[0][1]
	for i := 1; i < len(points); i++ {
		if points[i][0] > right {
			left, right = points[i][0], points[i][1]
			cnt++
		} else {
			left = max(left, points[i][0])
			right = min(right, points[i][1])
		}
	}
	return cnt
}

/*
*
20. 有效的括号
https://leetcode.cn/problems/valid-parentheses/?envType=study-plan-v2&envId=top-interview-150
*/
func isValid(s string) bool {
	if len(s) < 1 {
		return false
	}
	stack := make([]byte, 0)
	for _, c := range s {
		if c == '(' || c == '[' || c == '{' {
			stack = append(stack, byte(c))
		} else {
			if len(stack) < 1 {
				return false
			}
			peek := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if (c == ')' && peek != '(') || (c == ']' && peek != '[') || (c == '}' && peek != '{') {
				return false
			}
		}
	}
	if len(stack) > 0 {
		return false
	}
	return true
}

/*
*
71. 简化路径
https://leetcode.cn/problems/simplify-path/?envType=study-plan-v2&envId=top-interview-150
*/
func simplifyPath(path string) string {
	if len(path) < 1 {
		return ""
	}
	list, stack := strings.Split(path, "/"), make([]string, 0)
	for _, s := range list {
		if s == "" || s == "." {
			continue
		}
		if s == ".." {
			if len(stack) > 0 {
				stack = stack[:len(stack)-1]
			}
			continue
		}
		stack = append(stack, fmt.Sprintf("/%s", s))
	}
	if len(stack) == 0 {
		return "/"
	}
	return strings.Join(stack, "")
}

/*
*
155. 最小栈
https://leetcode.cn/problems/min-stack/?envType=study-plan-v2&envId=top-interview-150
*/
type MinStack struct {
	Data    []int
	MinData []int
}

func Constructor11() MinStack {
	return MinStack{
		Data:    make([]int, 0),
		MinData: make([]int, 0),
	}
}

func (s *MinStack) Push(val int) {
	s.Data = append(s.Data, val)
	if len(s.MinData) == 0 {
		s.MinData = append(s.MinData, val)
	} else {
		peek := s.MinData[len(s.MinData)-1]
		if val <= peek {
			s.MinData = append(s.MinData, val)
		}
	}
	return
}

func (s *MinStack) Pop() {
	if len(s.Data) == 0 {
		return
	}
	peek := s.Data[len(s.Data)-1]
	s.Data = s.Data[:len(s.Data)-1]
	if len(s.MinData) > 0 && peek == s.MinData[len(s.MinData)-1] {
		s.MinData = s.MinData[:len(s.MinData)-1]
	}
	return
}

func (s *MinStack) Top() int {
	if len(s.Data) == 0 {
		return 0
	}
	return s.Data[len(s.Data)-1]
}

func (s *MinStack) GetMin() int {
	if len(s.MinData) == 0 {
		return -1
	}
	return s.MinData[len(s.MinData)-1]
}

/*
*
150. 逆波兰表达式求值
https://leetcode.cn/problems/evaluate-reverse-polish-notation/?envType=study-plan-v2&envId=top-interview-150
*/
func evalRPN(tokens []string) int {
	if len(tokens) < 1 {
		return 0
	}
	stack := make([]int, 0)
	for _, s := range tokens {
		if n, err := strconv.Atoi(s); err == nil {
			stack = append(stack, n)
			continue
		}
		i, j := stack[len(stack)-2], stack[len(stack)-1]
		stack = stack[:len(stack)-2]
		var tmp int
		if s == "+" {
			tmp = i + j
		} else if s == "-" {
			tmp = i - j
		} else if s == "*" {
			tmp = i * j
		} else {
			tmp = int(math.Floor(float64(i) / float64(j)))
		}
		stack = append(stack, tmp)
	}
	return stack[0]
}

/*
*
224. 基本计算器
https://leetcode.cn/problems/basic-calculator/?envType=study-plan-v2&envId=top-interview-150
*/
func calculate(s string) int {
	return 0
}

/*
*
2. 两数相加
https://leetcode.cn/problems/add-two-numbers/?envType=study-plan-v2&envId=top-interview-150
*/
type ListNode struct {
	Val  int
	Next *ListNode
}

func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	if l1 == nil {
		return l2
	}
	if l2 == nil {
		return l1
	}
	rl1, rl2, jw, l := l1, l2, 0, &ListNode{}
	h := l
	for rl1 != nil || rl2 != nil || jw > 0 {
		v1, v2 := 0, 0
		if rl1 != nil {
			v1 = rl1.Val
			rl1 = rl1.Next
		}
		if rl2 != nil {
			v2 = rl2.Val
			rl2 = rl2.Next
		}
		sum := v1 + v2 + jw
		jw = sum / 10
		sum = sum % 10
		node := &ListNode{
			Val: sum,
		}
		l.Next = node
		l = l.Next
	}
	return h.Next
}

func PrintList(head *ListNode) {
	h := head
	var arr []int
	for h != nil {
		arr = append(arr, h.Val)
		h = h.Next
	}
	fmt.Println(arr)
}

func reverseList1(l *ListNode) *ListNode {
	if l == nil {
		return l
	}
	var pre *ListNode
	cur := l
	for cur != nil {
		next := cur.Next
		cur.Next = pre
		pre = cur
		cur = next
	}
	return pre
}

func reverseList2(l *ListNode) *ListNode {
	if l == nil || l.Next == nil {
		return l
	}
	h := reverseList2(l.Next)
	l.Next.Next = l
	l.Next = nil
	return h
}

/*
*
138. 随机链表的复制
https://leetcode.cn/problems/copy-list-with-random-pointer/?envType=study-plan-v2&envId=top-interview-150
*/
type Node struct {
	Val    int
	Next   *Node
	Random *Node
}

func copyRandomList(head *Node) *Node {
	if head == nil {
		return nil
	}
	h, mp := head, make(map[*Node]*Node)
	return deepCopyNode(mp, h)
}

func deepCopyNode(mp map[*Node]*Node, h *Node) *Node {
	if h == nil {
		return nil
	}
	if n, ok := mp[h]; ok {
		return n
	}
	n := &Node{Val: h.Val}
	mp[h] = n
	n.Next = deepCopyNode(mp, h.Next)
	n.Random = deepCopyNode(mp, h.Random)
	return n
}

/*
*
92. 反转链表 II
https://leetcode.cn/problems/reverse-linked-list-ii/?envType=study-plan-v2&envId=top-interview-150
*/
func reverseBetween(head *ListNode, left int, right int) *ListNode {
	if left == right {
		return head
	}
	dummyHead := &ListNode{Next: head}
	pre, end, h, i := dummyHead, dummyHead, dummyHead, 0
	for i < right-left {
		end = end.Next
		i++
	}
	i = 0
	for i < left {
		pre = h
		h = h.Next
		end = end.Next
		i++
	}
	if end != nil {
		end = end.Next
	}
	p := end
	cur := pre.Next
	for cur != end {
		n := cur.Next
		cur.Next = p
		p = cur
		cur = n
	}
	pre.Next = p
	return dummyHead.Next
}

/*
*
82. 删除排序链表中的重复元素 II
https://leetcode.cn/problems/remove-duplicates-from-sorted-list-ii/?envType=study-plan-v2&envId=top-interview-150
[1,2,3,3,4,4,5]
*/
func deleteDuplicates(head *ListNode) *ListNode {
	if head == nil {
		return head
	}
	dummyHead := &ListNode{Next: head, Val: head.Val - 1}
	h := dummyHead
	for h.Next != nil {
		val := h.Next.Val
		if h.Next.Next != nil && h.Next.Next.Val == val {
			for h.Next != nil && h.Next.Val == val {
				h.Next = h.Next.Next
			}
		} else {
			h = h.Next
		}
	}
	return dummyHead.Next
}

/*
*
61. 旋转链表
https://leetcode.cn/problems/rotate-list/?envType=study-plan-v2&envId=top-interview-150
*/
func rotateRight(head *ListNode, k int) *ListNode {
	if head == nil {
		return head
	}
	l, h := 0, head
	for h != nil {
		h = h.Next
		l++
	}
	moveK, i, slow, fast := k%l, 0, head, head
	for i < moveK {
		fast = fast.Next
		i++
	}
	for fast.Next != nil {
		slow = slow.Next
		fast = fast.Next
	}
	newHead := slow.Next
	slow.Next = nil
	fast.Next = head
	return newHead
}

/*
86. 分隔链表
*https://leetcode.cn/problems/partition-list/description/?envType=study-plan-v2&envId=top-interview-150
*/
func partition(head *ListNode, x int) *ListNode {
	if head == nil {
		return head
	}
	large, small, h := &ListNode{}, &ListNode{}, head
	l, s := large, small
	for h != nil {
		if h.Val > x {
			l.Next = h
			l = l.Next
		} else {
			s.Next = h
			s = s.Next
		}
		h = h.Next
	}
	l.Next = nil
	s.Next = large.Next
	return small.Next
}

/*
146. LRU 缓存
https://leetcode.cn/problems/lru-cache/?envType=study-plan-v2&envId=top-interview-150
*/
type DLinkedNode struct {
	key, value int
	prev, next *DLinkedNode
}

type LRUCache struct {
	mp          map[int]*DLinkedNode
	head, tail  *DLinkedNode
	maxLen, len int
}

func Constructor123(capacity int) LRUCache {
	l := LRUCache{
		maxLen: capacity,
		mp:     make(map[int]*DLinkedNode),
		head:   &DLinkedNode{},
		tail:   &DLinkedNode{},
	}
	l.head.next = l.tail
	l.tail.prev = l.head
	return l
}

func (l *LRUCache) Get(key int) int {
	node, ok := l.mp[key]
	if ok {
		l.moveToHead(node)
		return node.value
	}
	return -1
}

func (l *LRUCache) Put(key int, value int) {

	v, ok := l.mp[key]
	if ok {
		v.value = value
		l.moveToHead(v)
		return
	}
	node := &DLinkedNode{
		key:   key,
		value: value,
	}
	l.AddToHead(node)
	l.mp[key] = node
	if l.len < l.maxLen {
		l.len++
	} else {
		removed := l.RemoveTail()
		delete(l.mp, removed.key)
	}
	return
}

func (l *LRUCache) AddToHead(node *DLinkedNode) {
	n := l.head.next
	l.head.next = node
	node.prev = l.head
	node.next = n
	if n != nil {
		n.prev = node
	}
	return
}

func (l *LRUCache) RemoveTail() *DLinkedNode {
	node := l.tail.prev
	l.RemoveNode(node)
	return node
}

func (l *LRUCache) RemoveNode(node *DLinkedNode) {
	node.prev.next = node.next
	node.next.prev = node.prev
	return
}

func (l *LRUCache) moveToHead(node *DLinkedNode) {
	l.RemoveNode(node)
	l.AddToHead(node)
	return
}

/*
*104. 二叉树的最大深度
https://leetcode.cn/problems/maximum-depth-of-binary-tree/description/?envType=study-plan-v2&envId=top-interview-150
*/
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func maxDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	if root.Left == nil && root.Right == nil {
		return 1
	}
	maxDept := 0
	if root.Left != nil {
		maxDept = max(maxDept, maxDepth(root.Left)+1)
	}
	if root.Right != nil {
		maxDept = max(maxDept, maxDepth(root.Right)+1)
	}
	return maxDept
}

/*
*
100. 相同的树
https://leetcode.cn/problems/same-tree/?envType=study-plan-v2&envId=top-interview-150
*/
func isSameTree(p *TreeNode, q *TreeNode) bool {
	if p == nil && q == nil {
		return true
	}
	if (p == nil && q != nil) || (p != nil && q == nil) || (p.Val != q.Val) {
		return false
	}
	return isSameTree(p.Left, q.Left) && isSameTree(p.Right, q.Right)
}

/*
226. 翻转二叉树
https://leetcode.cn/problems/invert-binary-tree/?envType=study-plan-v2&envId=top-interview-150
*/
func invertTree(root *TreeNode) *TreeNode {
	if root == nil || (root.Left == nil && root.Right == nil) {
		return root
	}
	root.Left, root.Right = invertTree(root.Right), invertTree(root.Left)
	return root
}

/*
*
101. 对称二叉树
https://leetcode.cn/problems/symmetric-tree/description/?envType=study-plan-v2&envId=top-interview-150
*/
func isSymmetric(root *TreeNode) bool {
	if root == nil {
		return true
	}
	return isSymmetricImpl(root.Left, root.Right)
}

func isSymmetricImpl(l, r *TreeNode) bool {
	if l == nil && r == nil {
		return true
	} else if l == nil || r == nil {
		return false
	}
	if l.Val != r.Val {
		return false
	}
	return isSymmetricImpl(l.Left, r.Right) && isSymmetricImpl(l.Right, r.Left)
}

/*
102. 二叉树的层序遍历
*https://leetcode.cn/problems/binary-tree-level-order-traversal/?envType=study-plan-v2&envId=top-interview-150
*/
func levelOrder(root *TreeNode) [][]int {
	if root == nil {
		return [][]int{}
	}
	r, queue := make([][]int, 0), make([]*TreeNode, 0)
	queue = append(queue, root)
	for len(queue) > 0 {
		l := len(queue)
		tmp := make([]int, 0)
		for l > 0 {
			node := queue[0]
			queue = queue[1:]
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
			tmp = append(tmp, node.Val)
			l--
		}
		if len(tmp) > 0 {
			r = append(r, tmp)
		}
	}
	return r
}

/*
*
199. 二叉树的右视图
https://leetcode.cn/problems/binary-tree-right-side-view/?envType=study-plan-v2&envId=top-interview-150
*/
func rightSideView(root *TreeNode) []int {
	if root == nil {
		return []int{}
	}
	r, queue := make([]int, 0), make([]*TreeNode, 0)
	queue = append(queue, root)
	for len(queue) > 0 {
		l := len(queue)
		for l > 0 {
			node := queue[0]
			queue = queue[1:]
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
			l--
			if l == 0 {
				r = append(r, node.Val)
			}
		}
	}
	return r
}

/*
*
103. 二叉树的锯齿形层序遍历
https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/description/?envType=study-plan-v2&envId=top-interview-150
*/
func zigzagLevelOrder(root *TreeNode) [][]int {
	if root == nil {
		return [][]int{}
	}
	r, queue, level := make([][]int, 0), make([]*TreeNode, 0), 0
	queue = append(queue, root)
	for len(queue) > 0 {
		tmp := make([]int, 0)
		cnt := len(queue)
		for cnt > 0 {
			node := queue[0]
			queue = queue[1:]
			if level%2 == 0 {
				tmp = append(tmp, node.Val)
			} else {
				tmp = append([]int{node.Val}, tmp...)
			}
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
			cnt--
		}
		level++
		r = append(r, tmp)
	}
	return r
}

/*
*
530. 二叉搜索树的最小绝对差
https://leetcode.cn/problems/minimum-absolute-difference-in-bst/?envType=study-plan-v2&envId=top-interview-150
*/
func getMinimumDifference(root *TreeNode) int {
	if root == nil || (root.Left == nil && root.Right == nil) {
		return root.Val
	}
	r, pre := math.MaxInt64, -1
	var dfs func(*TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		dfs(node.Left)
		if pre != -1 && node.Val-pre < r {
			r = node.Val - pre
		}
		pre = node.Val
		dfs(node.Right)
	}
	dfs(root)
	return r
}

/*
*
230. 二叉搜索树中第K小的元素
https://leetcode.cn/problems/kth-smallest-element-in-a-bst/?envType=study-plan-v2&envId=top-interview-150
*/
func kthSmallest(root *TreeNode, k int) int {
	stack := make([]*TreeNode, 0)
	for {
		for root != nil {
			stack = append(stack, root)
			root = root.Left
		}
		stack, root = stack[:len(stack)-1], stack[len(stack)-1]
		k--
		if k == 0 {
			return root.Val
		}
		root = root.Right
	}
}

/*
*
98. 验证二叉搜索树
https://leetcode.cn/problems/validate-binary-search-tree/?envType=study-plan-v2&envId=top-interview-150
*/
func isValidBST(root *TreeNode) bool {
	pre, r := -1, true
	var dfs func(*TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		dfs(node.Left)
		if pre != -1 && pre < node.Val {
			r = false
			return
		}
		pre = node.Val
		dfs(node.Right)
		if pre > node.Val {
			r = false
			return
		}
	}
	return r
}

/*
200. 岛屿数量
https://leetcode.cn/problems/number-of-islands/?envType=study-plan-v2&envId=top-interview-150
*/
func numIslands(grid [][]byte) int {
	if len(grid) < 1 {
		return 0
	}
	r := 0
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[i]); j++ {
			if grid[i][j] == '1' {
				bfsSearchIslands(grid, i, j)
				r++
			}
		}
	}
	return r
}

// 左、上、右、下
var direction = []int{-1, 0, 1, 0, -1}

func bfsSearchIslands(grid [][]byte, i int, j int) {
	if i < 0 || i >= len(grid) || j < 0 || j >= len(grid[0]) || grid[i][j] == '0' || grid[i][j] == '2' {
		return
	}
	grid[i][j] = '2'
	for k := 0; k < 4; k++ {
		x := i + direction[k]
		y := j + direction[k+1]
		bfsSearchIslands(grid, x, y)
	}
	return
}

/*
*
130. 被围绕的区域
https://leetcode.cn/problems/surrounded-regions/?envType=study-plan-v2&envId=top-interview-150
*/
func solve(board [][]byte) {
	if len(board) < 1 {
		return
	}
	for i := 0; i < len(board); i++ {
		bfsSolve(board, i, 0)
		bfsSolve(board, i, len(board[0])-1)
	}
	for j := 0; j < len(board[0]); j++ {
		bfsSolve(board, 0, j)
		bfsSolve(board, len(board)-1, j)
	}
	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[0]); j++ {
			if board[i][j] == 'O' {
				board[i][j] = 'X'
			} else if board[i][j] == 'Y' {
				board[i][j] = 'O'
			}
		}
	}
	return
}

func bfsSolve(board [][]byte, i int, j int) {
	if i < 0 || i >= len(board) || j < 0 || j >= len(board[0]) || board[i][j] == 'X' || board[i][j] == 'Y' {
		return
	}
	board[i][j] = 'Y'
	for k := 0; k < 4; k++ {
		x := i + direction[k]
		y := j + direction[k+1]
		bfsSolve(board, x, y)
	}
	return
}

/*
133. 克隆图
*https://leetcode.cn/problems/clone-graph/?envType=study-plan-v2&envId=top-interview-150
*/
type NodeG struct {
	Val       int
	Neighbors []*NodeG
}

func cloneGraph(node *NodeG) *NodeG {
	if node == nil {
		return nil
	}
	mp := make(map[*NodeG]*NodeG)
	var cg func(node *NodeG) *NodeG
	cg = func(node *NodeG) *NodeG {
		if node == nil {
			return node
		}
		if _, ok := mp[node]; ok {
			return mp[node]
		}
		cloneNode := &NodeG{
			Val:       node.Val,
			Neighbors: []*NodeG{},
		}
		mp[node] = cloneNode
		for _, n := range node.Neighbors {
			cloneNode.Neighbors = append(cloneNode.Neighbors, cg(n))
		}
		return cloneNode
	}
	return cg(node)
}

/*
*207. 课程表
https://leetcode.cn/problems/course-schedule/description/?envType=study-plan-v2&envId=top-interview-150
*/
func canFinishBFS(numCourses int, prerequisites [][]int) bool {
	edges, indeg, r, q := make([][]int, numCourses), make([]int, numCourses), make([]int, 0), make([]int, 0)
	for _, arr := range prerequisites {
		edges[arr[1]] = append(edges[arr[1]], arr[0])
		indeg[arr[0]]++
	}
	for i := 0; i < numCourses; i++ {
		if indeg[i] == 0 {
			q = append(q, i)
		}
	}
	for len(q) > 0 {
		u := q[0]
		q = q[1:]
		r = append(r, u)
		for _, v := range edges[u] {
			indeg[v]--
			if indeg[v] == 0 {
				q = append(q, v)
			}
		}
	}
	return len(r) == numCourses
}

func canFinishDFS(numCourses int, prerequisites [][]int) bool {
	edges, visited, valid, result := make([][]int, numCourses), make([]int, numCourses), true, make([]int, 0)
	var dfs func(u int)
	dfs = func(u int) {
		visited[u] = 1
		for _, v := range edges[u] {
			if visited[v] == 0 {
				dfs(v)
				if !valid {
					return
				}
			} else if visited[v] == 1 {
				valid = false
				return
			}
		}
		visited[u] = 2
		result = append(result, u)
	}
	for _, arr := range prerequisites {
		edges[arr[1]] = append(edges[arr[1]], arr[0])
	}
	for i := 0; i < numCourses && valid; i++ {
		if visited[i] == 0 {
			dfs(i)
		}
	}
	return len(result) == numCourses && valid
}

/*
210. 课程表 II
https://leetcode.cn/problems/course-schedule-ii/description/
*/
func findOrderBFS(numCourses int, prerequisites [][]int) []int {
	edges, indeg, resut, q := make([][]int, numCourses), make([]int, numCourses), make([]int, 0), make([]int, 0)
	for _, arr := range prerequisites {
		edges[arr[1]] = append(edges[arr[1]], arr[0])
		indeg[arr[0]]++
	}
	for i := 0; i < numCourses; i++ {
		if indeg[i] == 0 {
			q = append(q, i)
		}
	}
	for len(q) > 0 {
		v := q[0]
		q = q[1:]
		resut = append(resut, v)
		for _, t := range edges[v] {
			indeg[t]--
			if indeg[t] == 0 {
				q = append(q, t)
			}
		}
	}
	if len(resut) != numCourses {
		return []int{}
	}
	return resut
}

func findOrderDFS(numCourses int, prerequisites [][]int) []int {
	edges, visited, valid, result := make([][]int, numCourses), make([]int, numCourses), true, make([]int, 0)
	var dfs func(u int)
	dfs = func(u int) {
		visited[u] = 1
		for _, v := range edges[u] {
			if visited[v] == 1 {
				valid = false
				return
			} else if visited[v] == 0 {
				dfs(v)
				if !valid {
					return
				}
			}
		}
		visited[u] = 2
		result = append(result, u)
	}
	for _, arr := range prerequisites {
		edges[arr[1]] = append(edges[arr[1]], arr[0])
	}
	for i := 0; i < numCourses && valid; i++ {
		if visited[i] == 0 {
			dfs(i)
		}
	}
	if !valid {
		return []int{}
	}
	for i := 0; i < len(result)/2; i++ {
		result[i], result[len(result)-i-1] = result[len(result)-i-1], result[i]
	}
	return result
}

/*
*
433. 最小基因变化
https://leetcode.cn/problems/minimum-genetic-mutation/description/?envType=study-plan-v2&envId=top-interview-150
*/
func minMutation(startGene string, endGene string, bank []string) int {
	if startGene == endGene {
		return 0
	}
	mp := make(map[string]struct{})
	for _, s := range bank {
		mp[s] = struct{}{}
	}
	if _, ok := mp[endGene]; !ok {
		return -1
	}
	q, step := []string{startGene}, 0
	for len(q) > 0 {
		step++
		tmp := q
		q = nil
		for _, cur := range tmp {
			for i, x := range cur {
				for _, y := range "ACGT" {
					if x != y {
						s := cur[:i] + string(y) + cur[i+1:]
						if _, ok := mp[s]; !ok {
							continue
						}
						if s == endGene {
							return step
						}
						delete(mp, s)
						q = append(q, s)
					}
				}
			}
		}
	}
	return -1
}

/*
*
208. 实现 Trie (前缀树)
https://leetcode.cn/problems/implement-trie-prefix-tree/?envType=study-plan-v2&envId=top-interview-150
*/
type Trie struct {
	Data  [26]*Trie
	IsEnd bool
}

func Constructor() Trie {
	return Trie{}
}

func (t *Trie) Insert(word string) {
	tmp := t
	for _, b := range word {
		if tmp.Data[b-'a'] == nil {
			tt := Constructor()
			tmp.Data[b-'a'] = &tt
		}
		tmp = tmp.Data[b-'a']
	}
	tmp.IsEnd = true
	return
}

func (t *Trie) SearchPrefix(word string) *Trie {
	tmp := t
	for _, s := range word {
		if tmp.Data[s-'a'] == nil {
			return nil
		}
		tmp = tmp.Data[s-'a']
	}
	return tmp
}

func (t *Trie) Search(word string) bool {
	n := t.SearchPrefix(word)
	return n != nil && n.IsEnd
}

func (t *Trie) StartsWith(prefix string) bool {
	n := t.SearchPrefix(prefix)
	return n != nil
}

/*
*
211. 添加与搜索单词 - 数据结构设计
https://leetcode.cn/problems/design-add-and-search-words-data-structure/description/
*/
type TrieTree struct {
	Data  [26]*TrieTree
	IsEnd bool
}

func NewTrieTree() *TrieTree {
	return &TrieTree{}
}

func (t *TrieTree) Insert(word string) {
	tmp := t
	for _, c := range word {
		if tmp.Data[c-'a'] == nil {
			tmp.Data[c-'a'] = NewTrieTree()
		}
		tmp = tmp.Data[c-'a']
	}
	tmp.IsEnd = true
	return
}

func (t *TrieTree) Search(idx int, word string) bool {
	tmp := t
	for i, c := range word {
		if idx == len(word) {
			return tmp.IsEnd
		}
		if tmp.IsEnd {
			return false
		}
		if c == '.' {
			tmp.Search(i+1, word)
		} else {
			if tmp.Data[c-'a'] == nil {
				return false
			}
		}
	}
	return tmp != nil && tmp.IsEnd
}

type WordDictionary struct {
	trie *TrieTree
}

func NewWordDictionary() WordDictionary {
	return WordDictionary{
		trie: NewTrieTree(),
	}
}

func (w *WordDictionary) AddWord(word string) {
	w.trie.Insert(word)
}

func (w *WordDictionary) Search(word string) bool {
	var dfs func(idx int, node *TrieTree) bool
	dfs = func(idx int, node *TrieTree) bool {
		if idx == len(word) {
			return node.IsEnd
		}
		char := word[idx]
		if char != '.' {
			child := node.Data[char-'a']
			if child != nil && dfs(idx+1, child) {
				return true
			}
		} else {
			for i, _ := range node.Data {
				child := node.Data[i]
				if child != nil && dfs(idx+1, child) {
					return true
				}
			}
		}
		return false
	}
	return dfs(0, w.trie)
}

/*
*
17. 电话号码的字母组合
https://leetcode.cn/problems/letter-combinations-of-a-phone-number
*/
var table = []string{"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"}
var r []string

func letterCombinations(digits string) []string {
	if len(digits) < 1 {
		return []string{}
	}
	r = []string{}
	letterCombinationsImpl(digits, 0, "")
	return r
}

// digits = "23"
func letterCombinationsImpl(digits string, j int, s string) {
	if j == len(digits) {
		r = append(r, s)
		return
	}
	letters := table[digits[j]-'0']
	for i := 0; i < len(letters); i++ {
		letterCombinationsImpl(digits, j+1, fmt.Sprintf("%s%s", s, string(letters[i])))
	}
}

/*
*
77. 组合
https://leetcode.cn/problems/combinations/
*/
func combine(n int, k int) [][]int {
	r, data := make([][]int, 0), make([]int, 0)
	combineImpl(n, k, 1, &r, data)
	return r
}

func combineImpl(n, k, level int, r *[][]int, data []int) {
	if len(data) == k {
		tmp := make([]int, 0)
		tmp = append(tmp, data...)
		*r = append(*r, tmp)
		return
	}
	for i := level; i <= n; i++ {
		data = append(data, i)
		combineImpl(n, k, i+1, r, data)
		data = data[:len(data)-1]
	}
}

/*
46. 全排列
https://leetcode.cn/problems/permutations/description
*/
func permute(nums []int) [][]int {
	if len(nums) < 1 {
		return [][]int{}
	}
	r, data, check := make([][]int, 0), make([]int, 0), make(map[int]bool)
	permuteImpl(nums, data, check, &r)
	return r
}

func permuteImpl(nums []int, data []int, check map[int]bool, r *[][]int) {
	if len(data) == len(nums) {
		tmp := make([]int, 0)
		tmp = append(tmp, data...)
		*r = append(*r, tmp)
		return
	}
	for i := 0; i < len(nums); i++ {
		if b, ok := check[nums[i]]; ok && b {
			continue
		}
		data = append(data, nums[i])
		check[nums[i]] = true
		permuteImpl(nums, data, check, r)
		data = data[:len(data)-1]
		delete(check, nums[i])
	}
}

func combinationSum2(candidates []int, target int) [][]int {
	if len(candidates) < 1 {
		return [][]int{}
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i] <= candidates[j] {
			return true
		}
		return false
	})
	r, data, level := make([][]int, 0), make([]int, 0), 0
	bfsCombinationSum(candidates, data, target, level, &r)
	return r
}

func bfsCombinationSum2(candidates []int, data []int, target, level int, r *[][]int) {
	if sliceSum(data) > target {
		return
	}
	if sliceSum(data) == target {
		tmp := make([]int, 0)
		tmp = append(tmp, data...)
		*r = append(*r, tmp)
		return
	}
	for i := level; i < len(candidates); i++ {
		if i > level && candidates[i] == candidates[i-1] {
			continue
		}
		data = append(data, candidates[i])
		bfsCombinationSum(candidates, data, target, i+1, r)
		data = data[:len(data)-1]
	}
}

func sliceSum(list []int) int {
	var sum int
	for _, n := range list {
		sum += n
	}
	return sum
}

func combinationSum(candidates []int, target int) [][]int {
	if len(candidates) < 1 {
		return [][]int{}
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i] <= candidates[j] {
			return true
		}
		return false
	})
	r, data, level := make([][]int, 0), make([]int, 0), 0
	bfsCombinationSum(candidates, data, target, level, &r)
	return r
}

func bfsCombinationSum(candidates []int, data []int, target, level int, r *[][]int) {
	if sliceSum(data) > target {
		return
	}
	if sliceSum(data) == target {
		tmp := make([]int, 0)
		tmp = append(tmp, data...)
		*r = append(*r, tmp)
		return
	}
	for i := level; i < len(candidates); i++ {
		if target-sliceSum(data)-candidates[i] < 0 {
			break
		}
		data = append(data, candidates[i])
		bfsCombinationSum(candidates, data, target, i, r)
		data = data[:len(data)-1]
	}
}

/*
216. 组合总和 III
https://leetcode.cn/problems/combination-sum-iii/
*/
func combinationSum3(k int, n int) [][]int {
	r, data, level := make([][]int, 0), make([]int, 0), 1
	combinationSum3BT(k, n, level, data, &r)
	return r
}

func combinationSum3BT(k int, n int, level int, data []int, r *[][]int) {
	if len(data) > k || sliceSum(data) > n {
		return
	}
	if sliceSum(data) == n && len(data) == k {
		tmp := make([]int, 0)
		tmp = append(tmp, data...)
		*r = append(*r, tmp)
		return
	}
	for i := level; i < 10; i++ {
		data = append(data, i)
		combinationSum3BT(k, n, i+1, data, r)
		data = data[:len(data)-1]
	}
	return
}

/*
*
377. 组合总和 Ⅳ
https://leetcode.cn/problems/combination-sum-iv/
*/
func combinationSum4(nums []int, target int) int {
	dict, r := make(map[int]int), 0
	sort.Slice(nums, func(i, j int) bool {
		if nums[i] <= nums[j] {
			return true
		}
		return false
	})
	combinationSum4BT(nums, target, dict, &r)
	return r
}

func combinationSum4BT(nums []int, target int, dict map[int]int, r *int) {
	if target == 0 {
		*r++
		return
	}
	for i := 0; i < len(nums); i++ {
		if target < nums[i] {
			break
		}
		target -= nums[i]
		if v, ok := dict[target]; ok {
			*r += v
		} else {
			var tmpR int
			combinationSum4BT(nums, target, dict, &tmpR)
			dict[target] = tmpR
			*r += tmpR
		}
		target += nums[i]
	}
	return
}

/*
*
78. 子集
https://leetcode.cn/problems/subsets/
*/
func subsets(nums []int) [][]int {
	if len(nums) < 1 {
		return [][]int{}
	}
	r, data, level := make([][]int, 0), make([]int, 0), 0
	subsetsBT(nums, data, level, &r)
	return r
}

func subsetsBT(nums []int, data []int, level int, r *[][]int) {
	tmp := make([]int, 0)
	tmp = append(tmp, data...)
	*r = append(*r, tmp)

	for i := level; i < len(nums); i++ {
		data = append(data, nums[i])
		subsetsBT(nums, data, i+1, r)
		data = data[:len(data)-1]
	}
}

/*
*
90. 子集 II
https://leetcode.cn/problems/subsets-ii/
*/
func subsetsWithDup(nums []int) [][]int {
	if len(nums) < 1 {
		return [][]int{}
	}
	sort.Slice(nums, func(i, j int) bool {
		if nums[i] <= nums[j] {
			return true
		}
		return false
	})
	level, r, data := 0, make([][]int, 0), make([]int, 0)
	subsetsWithDupBT(nums, data, level, &r)
	return r
}

func subsetsWithDupBT(nums []int, data []int, level int, r *[][]int) {
	tmp := make([]int, 0)
	tmp = append(tmp, data...)
	*r = append(*r, tmp)
	for i := level; i < len(nums); i++ {
		if level != i && nums[i] == nums[i-1] {
			continue
		}
		data = append(data, nums[i])
		subsetsWithDupBT(nums, data, i+1, r)
		data = data[:len(data)-1]
	}
	return
}

/*
*
93. 复原 IP 地址
https://leetcode.cn/problems/restore-ip-addresses/
*/
func restoreIpAddresses(s string) []string {
	if len(s) < 3 || len(s) > 12 {
		return []string{}
	}
	r := make([]string, 0)
	restoreIpAddressesBT(s, []string{}, r)
	return r
}

func restoreIpAddressesBT(s string, tmp []string, r []string) {
	if len(tmp) == 4 && len(s) == 0 {
		r = append(r, tmp[0]+"."+tmp[1]+"."+tmp[2]+"."+tmp[3])
		return
	}
	for i := 1; i < 4; i++ {
		if len(s) < i {
			return
		}
		str := s[:i]
		if len(str) == 3 && strings.Compare(str, "255") > 0 {
			return
		}
		if len(str) > 1 && str[0] == '0' {
			return
		}
		tmp = append(tmp, str)
		restoreIpAddressesBT(s[i:], tmp, r)
		tmp = tmp[:len(tmp)-1]
	}
	return
}

/*
*
79. 单词搜索
https://leetcode.cn/problems/word-search/
*/
func exist(board [][]byte, word string) bool {
	used, r := make(map[string]struct{}), false
	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[0]); j++ {
			if board[i][j] == word[0] {
				BFSExist(board, word, i, j, 0, used, &r)
				if r {
					return true
				}
			}
		}
	}
	return false
}

func BFSExist(board [][]byte, word string, i, j, n int, used map[string]struct{}, b *bool) {
	_, ok := used[string(i)+string(j)]
	if i < 0 || i >= len(board) || j < 0 || j >= len(board[0]) || n >= len(word) || board[i][j] != word[n] || ok {
		return
	}
	if n == len(word)-1 {
		*b = true
		return
	}
	n++
	used[string(i)+string(j)] = struct{}{}
	for k := 0; k < 4; k++ {
		if *b {
			break
		}
		x := i + direction[k]
		y := j + direction[k+1]
		BFSExist(board, word, x, y, n, used, b)
	}
	delete(used, string(i)+string(j))
	return
}

/*
*
22. 括号生成
https://leetcode.cn/problems/generate-parentheses/description/?envType=study-plan-v2&envId=top-interview-150
*/
func generateParenthesis(n int) []string {
	if n == 0 {
		return []string{}
	}
	l, r, s, res := n, n, "", make([]string, 0)
	generateParenthesisBT(l, r, s, &res)
	return res
}

func generateParenthesisBT(l int, r int, s string, res *[]string) {
	if l == r && l == 0 {
		*res = append(*res, s)
		return
	}
	if l > 0 {
		generateParenthesisBT(l-1, r, s+"(", res)
	}
	if r > 0 && r > l {
		generateParenthesisBT(l, r-1, s+")", res)
	}
}

/*
*
108. 将有序数组转换为二叉搜索树
https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/?envType=study-plan-v2&envId=top-interview-150
*/
func sortedArrayToBST(nums []int) *TreeNode {
	if len(nums) == 0 {
		return nil
	}
	mid := len(nums) / 2
	left := nums[:mid]
	right := nums[mid+1:]
	head := &TreeNode{Val: nums[mid], Left: sortedArrayToBST(left), Right: sortedArrayToBST(right)}
	return head
}

/*
*
面试题 08.06. 汉诺塔问题
https://leetcode.cn/problems/hanota-lcci/
*/
func hanota(A []int, B []int, C []int) []int {
	/**
	A[1:] -> C[1:] = B[1:]
	A[0] -> C[0]
	B[] -> A[] = B
	结束标识：len(A) == len(B) == 0
	*/
	if A == nil {
		return nil
	}
	var move func(n int, a, b, c *[]int)
	move = func(n int, a, b, c *[]int) {
		if n == 0 {
			return
		}
		if n == 1 {
			*c = append(*c, (*a)[len(*a)-1])
			*a = (*a)[:len(*a)-1]
			return
		}
		move(n-1, a, c, b)
		(*c)[0] = (*a)[0]
		move(n-1, b, a, c)
	}
	move(len(A), &A, &B, &C)
	return C
}

/*
148. 排序链表
*https://leetcode.cn/problems/sort-list/?envType=study-plan-v2&envId=top-interview-150
*/
func sortList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	pre, slow, fast := head, head, head
	for fast != nil && fast.Next != nil {
		pre = slow
		slow = slow.Next
		fast = fast.Next.Next
	}
	pre.Next = nil
	return mergeList(sortList(head), sortList(slow))
}

func mergeList(l1, l2 *ListNode) *ListNode {
	h := &ListNode{}
	r := h
	for l1 != nil && l2 != nil {
		if l1.Val > l2.Val {
			h.Next = &ListNode{Val: l2.Val}
			l2 = l2.Next
		} else {
			h.Next = &ListNode{Val: l1.Val}
			l1 = l1.Next
		}
		h = h.Next
	}
	if l1 != nil {
		h.Next = l1
	}
	if l2 != nil {
		h.Next = l2
	}
	return r.Next
}

/*
*
23. 合并 K 个升序链表
https://leetcode.cn/problems/merge-k-sorted-lists/?envType=study-plan-v2&envId=top-interview-150
*/
func mergeKLists(lists []*ListNode) *ListNode {
	if len(lists) < 1 {
		return nil
	}
	if len(lists) == 1 {
		return lists[0]
	}
	return mergeListImpl(0, len(lists)-1, lists)
}

func mergeListImpl(start int, end int, lists []*ListNode) *ListNode {
	if start > end {
		return nil
	}
	if start == end {
		return lists[start]
	}
	//mid := start + (end-start)/2
	mid := (start + end) / 2
	return mergeList(mergeListImpl(start, mid, lists), mergeListImpl(mid+1, end, lists))
}

/*
*
53. 最大子数组和
https://leetcode.cn/problems/maximum-subarray/?envType=study-plan-v2&envId=top-interview-150
*/
func maxSubArray(nums []int) int {
	if len(nums) == 1 {
		return nums[0]
	}
	maxSum, sum := nums[0], nums[0]
	for i := 1; i < len(nums); i++ {
		if sum < 0 && nums[i] > sum {
			sum = nums[i]
		} else {
			sum += nums[i]
		}
		if sum > maxSum {
			maxSum = sum
		}
	}
	return maxSum
}

func maxSubArray11(nums []int) int {
	if len(nums) == 1 {
		return nums[0]
	}
	maxSum, sum := nums[0], nums[0]
	for i := 1; i < len(nums); i++ {
		sum += nums[i]
		if sum < nums[i] {
			sum = nums[i]
		}
		if sum > maxSum {
			maxSum = sum
		}
	}
	return maxSum
}

/*
918. 环形子数组的最大和
*https://leetcode.cn/problems/maximum-sum-circular-subarray/?envType=study-plan-v2&envId=top-interview-150
*/
func maxSubarraySumCircular(nums []int) int {
	n := len(nums)
	leftMax := make([]int, n)
	// 对坐标为 0 处的元素单独处理，避免考虑子数组为空的情况
	leftMax[0] = nums[0]
	leftSum, pre, res := nums[0], nums[0], nums[0]
	for i := 1; i < n; i++ {
		pre = max(pre+nums[i], nums[i])
		res = max(res, pre) // 1：线性数组的最大和
		leftSum += nums[i]
		leftMax[i] = max(leftMax[i-1], leftSum) // 2.1：[0:i] 最大和
	}
	// 从右到左枚举后缀，固定后缀，选择最大前缀
	rightSum := 0
	for i := n - 1; i > 0; i-- {
		rightSum += nums[i]                   // 2.2: [i:n] 最大和
		res = max(res, rightSum+leftMax[i-1]) // 比较  1、2 ，取大值
	}
	return res
}

/*
*
35. 搜索插入位置
https://leetcode.cn/problems/search-insert-position/?envType=study-plan-v2&envId=top-interview-150
*/
func searchInsert(nums []int, target int) int {
	if len(nums) < 1 {
		return 0
	}
	start, end := 0, len(nums)-1
	for start < end {
		mid := start + (end-start)/2
		if nums[mid] == target {
			return mid
		} else if nums[mid] < target {
			start = mid + 1
		} else {
			end = mid - 1
		}
	}
	if nums[start] >= target {
		return start
	}
	return start + 1
}

/*
*
74. 搜索二维矩阵
https://leetcode.cn/problems/search-a-2d-matrix/?envType=study-plan-v2&envId=top-interview-150
*/
func searchMatrix(matrix [][]int, target int) bool {
	if len(matrix) < 0 {
		return false
	}
	start, end, innerLen := 0, len(matrix)-1, len(matrix[0])-1
	for start < end {
		mid := start + (end-start)/2
		if matrix[mid][0] == target {
			return true
		} else if matrix[mid][0] < target {
			start = mid
			if matrix[start][innerLen] >= target {
				break
			} else {
				start++
			}
		} else {
			end = mid
			if end == start+1 {
				break
			}
		}
	}
	arr := matrix[start]
	start, end = 0, len(arr)-1
	for start < end {
		mid := start + (end-start)/2
		if arr[mid] == target {
			return true
		} else if arr[mid] < target {
			start = mid + 1
		} else {
			end = mid - 1
		}
	}
	if start >= len(arr) || arr[start] != target {
		return false
	}
	return true
}

func searchMatrix1(matrix [][]int, target int) bool {
	row := sort.Search(len(matrix), func(i int) bool { return matrix[i][0] > target }) - 1
	if row < 0 {
		return false
	}
	col := sort.SearchInts(matrix[row], target)
	return col < len(matrix[row]) && matrix[row][col] == target
}

/*
*
162. 寻找峰值
https://leetcode.cn/problems/find-peak-element/?envType=study-plan-v2&envId=top-interview-150
*/
func findPeakElement(nums []int) int {
	get := func(index int) int {
		if index < 0 || index > len(nums)-1 {
			return math.MinInt32
		}
		return nums[index]
	}
	start, end := 0, len(nums)-1
	for start < end {
		mid := start + (end-start)/2
		if get(mid-1) < get(mid) && get(mid) > get(mid+1) {
			return mid
		} else if get(mid) < get(mid+1) {
			start = mid + 1
		} else {
			end = mid - 1
		}
	}
	return start
}

/*
33. 搜索旋转排序数组
*https://leetcode.cn/problems/search-in-rotated-sorted-array/?envType=study-plan-v2&envId=top-interview-150
*/
func search(nums []int, target int) int {
	start, end := 0, len(nums)-1
	for start <= end {
		mid := start + (end-start)/2
		if nums[mid] == target {
			return mid
		} else if nums[mid] >= nums[start] {
			if target < nums[mid] && target >= nums[start] {
				end = mid - 1
			} else {
				start = mid + 1
			}
		} else {
			if target > nums[mid] && target <= nums[end] {
				start = mid + 1
			} else {
				end = mid - 1
			}
		}
	}
	return -1
}

/*
*
34. 在排序数组中查找元素的第一个和最后一个位置
https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/?envType=study-plan-v2&envId=top-interview-150
*/
func searchRange(nums []int, target int) []int {
	if len(nums) < 1 {
		return []int{-1, -1}
	}
	return []int{binarySearch12(nums, target, true), binarySearch12(nums, target, false)}
}

func binarySearch12(nums []int, target int, isFindLeft bool) int {
	start, end, r := 0, len(nums)-1, -1
	for start <= end {
		mid := start + (end-start)/2
		if nums[mid] == target {
			r = mid
			if isFindLeft {
				end = mid - 1
			} else {
				start = mid + 1
			}
		} else if nums[mid] > target {
			end = mid - 1
		} else {
			start = mid + 1
		}
	}
	return r
}

/*
153. 寻找旋转排序数组中的最小值
*https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/?envType=study-plan-v2&envId=top-interview-150
*/
func findMin(nums []int) int {
	start, end := 0, len(nums)-1
	for start <= end {
		mid := start + (end-start)/2
		if nums[mid] < nums[end] {
			end = mid
		} else {
			start = mid + 1
		}
	}
	return nums[start]
}

/*
4. 寻找两个正序数组的中位数
*https://leetcode.cn/problems/median-of-two-sorted-arrays/?envType=study-plan-v2&envId=top-interview-150
*/
func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	tLen := len(nums1) + len(nums2)
	if tLen%2 == 0 {
		return float64(getKthElement1(nums1, nums2, tLen/2)+getKthElement1(nums1, nums2, tLen/2+1)) / 2.0
	}
	return float64(getKthElement1(nums1, nums2, tLen/2+1))
}

func findMedianSortedArrays1(nums1 []int, nums2 []int) float64 {
	totalLength := len(nums1) + len(nums2)
	if totalLength%2 == 1 {
		midIndex := totalLength / 2
		return float64(getKthElement1(nums1, nums2, midIndex+1))
	} else {
		midIndex1, midIndex2 := totalLength/2-1, totalLength/2
		return float64(getKthElement1(nums1, nums2, midIndex1+1)+getKthElement1(nums1, nums2, midIndex2+1)) / 2.0
	}
	return 0
}

func getKthElement1(nums1 []int, nums2 []int, k int) int {
	index1, index2 := 0, 0
	for {
		if index1 == len(nums1) {
			return nums2[index2+k-1]
		}
		if index2 == len(nums2) {
			return nums1[index1+k-1]
		}
		if k == 1 {
			return min(nums1[index1], nums2[index2])
		}
		half := k / 2
		newIndex1, newIndex2 := min(index1+half, len(nums1))-1, min(index2+half, len(nums2))-1
		if nums1[newIndex1] <= nums2[newIndex2] {
			k -= (newIndex1 - index1 + 1)
			index1 = newIndex1 + 1
		} else {
			k -= (newIndex2 - index2 + 1)
			index2 = newIndex2 + 1
		}
	}
	return 0
}

/*
*
215. 数组中的第K个最大元素
https://leetcode.cn/problems/kth-largest-element-in-an-array/description/?envType=study-plan-v2&envId=top-interview-150
*/
func findKthLargest(nums []int, target int) int {
	l, r, k := 0, len(nums)-1, len(nums)-target
	for {
		p := GetPartition1(nums, l, r)
		if p == k {
			return nums[p]
		} else if p < k {
			l = p + 1
		} else {
			r = p - 1
		}
	}
	return nums[l]
}

func GetPartition1(nums []int, l int, r int) int {
	val := nums[l]
	for l < r {
		for l < r && val <= nums[r] {
			r--
		}
		nums[l] = nums[r]
		for l < r && val >= nums[l] {
			l++
		}
		nums[r] = nums[l]
	}
	nums[l] = val
	return l
}

type Heap struct {
	Data []int
}

func NewHeap() *Heap {
	return &Heap{
		Data: []int{0},
	}
}

func findKthLargest2(nums []int, target int) int {
	h := NewHeap()
	for _, v := range nums {
		h.siftUp(v)
	}
	if target > len(nums) {
		target = len(nums)
	}
	for target > 1 {
		_ = h.extractMax()
		target--
	}
	return h.Data[1]
}

/*
*
将val放入堆中
*/
func (h *Heap) siftUp(val int) {
	if len(h.Data) < 1 {
		h.Data = append(h.Data, []int{0, val}...)
	}
	h.Data = append(h.Data, val)
	l := len(h.Data) - 1
	for l > 1 {
		p := l / 2
		if h.Data[p] < h.Data[l] {
			h.Data[p], h.Data[l] = h.Data[l], h.Data[p]
			l = p
			continue
		}
		break
	}
	return
}

/*
*
将val从堆末尾加入堆顶，重新组装堆
*/
func (h *Heap) siftDown(val int) {
	if len(h.Data) < 1 {
		h.Data = append(h.Data, []int{0, val}...)
		return
	}
	i, l := 1, len(h.Data)
	for i < l {
		child := 2 * i
		if child >= l {
			break
		}
		if child+1 < l && h.Data[child+1] > h.Data[child] {
			child++
		}
		if h.Data[child] > val {
			h.Data[child], h.Data[i] = h.Data[i], h.Data[child]
			i = child
			continue
		}
		break
	}
	if i < l {
		h.Data[i] = val
	}
	return
}

/*
*
提取堆中的最大值，并对堆中的剩余元素进行重新整理
*/
func (h *Heap) extractMax() int {
	defer func() {
		lastVal := h.Data[len(h.Data)-1]
		h.Data = h.Data[:len(h.Data)-1]
		h.siftDown(lastVal)
	}()
	return h.Data[1]
}

/*
502. IPO
*https://leetcode.cn/problems/ipo/?envType=study-plan-v2&envId=top-interview-150
*/
func findMaximizedCapital(k int, w int, profits []int, capital []int) int {
	h := NewHeapIPO()
	for i := 0; i < len(profits); i++ {
		h.siftUp(profits[i], capital[i])
	}
	i, r, passVal := 0, w, make([]*ValIPO, 0)
	for i < k && len(h.Data) > 1 {
		v := h.extractMax()
		if v.capital > r {
			passVal = append(passVal, v)
			continue
		}
		r += v.profit
		i++
		for _, vv := range passVal {
			h.siftUp(vv.profit, vv.capital)
		}
		passVal = []*ValIPO{}
	}
	return r
}

type ValIPO struct {
	profit  int
	capital int
}

func NewValIPO(profit, capital int) *ValIPO {
	return &ValIPO{
		capital: capital,
		profit:  profit,
	}
}

type HeapIPO struct {
	Data []*ValIPO
}

func NewHeapIPO() *HeapIPO {
	return &HeapIPO{
		Data: []*ValIPO{{capital: 0, profit: 0}},
	}
}

func (h *HeapIPO) siftUp(profit, capital int) {
	if len(h.Data) == 0 {
		h.Data = []*ValIPO{{capital: 0, profit: 0}, {capital: capital, profit: profit}}
		return
	}
	val := NewValIPO(profit, capital)
	h.Data = append(h.Data, val)
	i := len(h.Data) - 1
	for i > 1 {
		p := i / 2
		if h.Data[p].profit < val.profit {
			h.Data[p], h.Data[i] = h.Data[i], h.Data[p]
			i = p
			continue
		}
		break
	}
	return
}

func (h *HeapIPO) siftDown(profit, capital int) {
	if len(h.Data) == 0 {
		h.Data = []*ValIPO{{capital: 0, profit: 0}, {capital: capital, profit: profit}}
		return
	}
	val := NewValIPO(profit, capital)
	i := 1
	for i < len(h.Data) {
		child := 2 * i
		if child >= len(h.Data) {
			break
		}
		if child+1 < len(h.Data) && h.Data[child+1].profit > h.Data[child].profit {
			child++
		}
		if val.profit < h.Data[child].profit {
			h.Data[i], h.Data[child] = h.Data[child], h.Data[i]
			i = child
			continue
		}
		break
	}
	if i < len(h.Data) {
		h.Data[i] = val
	}
	return
}

func (h *HeapIPO) extractMax() *ValIPO {
	defer func() {
		lastVal := h.Data[len(h.Data)-1]
		h.Data = h.Data[:len(h.Data)-1]
		h.siftDown(lastVal.profit, lastVal.capital)
	}()
	return h.Data[1]
}

type pcVal struct {
	profit  int
	capital int
	isUsed  bool
}

func NewPcVal(profit, capital int) *pcVal {
	return &pcVal{
		profit:  profit,
		capital: capital,
		isUsed:  false,
	}
}

type PcValList []*pcVal

func (p PcValList) Len() int {
	return len(p)
}

func (p PcValList) Less(i, j int) bool {
	return p[i].profit >= p[j].profit
}

func (p PcValList) Swap(i, j int) {
	p[i], p[j] = p[j], p[i]
}

func (p PcValList) Sort() {
	sort.Sort(p)
}

func findMaximizedCapital1(k int, w int, profits []int, capital []int) int {
	l := make([]*pcVal, 0)
	list := PcValList(l)
	for i := 0; i < len(profits); i++ {
		v := NewPcVal(profits[i], capital[i])
		list = append(list, v)
	}
	list.Sort()
	if k > len(profits) {
		k = len(profits)
	}
	i, start := 0, 0
	for i < k {
		flag := false
		for j := start; j < len(list); j++ {
			if w < list[j].capital || list[j].isUsed {
				continue
			}
			w += list[j].profit
			list[j].isUsed = true
			flag = true
			if j == start {
				start++
			}
			i++
			break
		}
		if !flag {
			break
		}
	}
	return w
}

/*
373. 查找和最小的 K 对数字
*https://leetcode.cn/problems/find-k-pairs-with-smallest-sums/?envType=study-plan-v2&envId=top-interview-150
*/
func kSmallestPairs(nums1 []int, nums2 []int, k int) [][]int {
	hp := &hp{
		num1: nums1,
		num2: nums2,
		data: make([]pair, 0),
	}
	for i := 0; i < len(nums1) && i < k; i++ {
		hp.data = append(hp.data, pair{i: i, j: 0})
	}
	j, r := 0, make([][]int, 0)
	for hp.Len() > 0 && j < k {
		v := heap.Pop(hp).(pair)
		r = append(r, []int{nums1[v.i], nums2[v.j]})
		if v.j+1 < len(nums2) {
			heap.Push(hp, pair{v.i, v.j + 1})
		}
		j++
	}
	return r
}

type pair struct {
	i, j int
}

type hp struct {
	data       []pair
	num1, num2 []int
}

func (h *hp) Push(x interface{}) {
	h.data = append(h.data, x.(pair))
}

func (h *hp) Pop() interface{} {
	v := h.data[len(h.data)-1]
	h.data = h.data[:len(h.data)-1]
	return v
}

func (h hp) Len() int {
	return len(h.data)
}

func (h hp) Less(i, j int) bool {
	a, b := h.data[i], h.data[j]
	return h.num1[a.i]+h.num2[a.j] < h.num1[b.i]+h.num2[b.j]
}

func (h hp) Swap(i, j int) {
	h.data[i], h.data[j] = h.data[j], h.data[i]
}

/*
295. 数据流的中位数
*https://leetcode.cn/problems/find-median-from-data-stream/?envType=study-plan-v2&envId=top-interview-150
*/
type MedianFinder struct {
	queMin, queMax hp1
}

type hp1 struct {
	sort.IntSlice
}

func (h *hp1) Push(x interface{}) {
	h.IntSlice = append(h.IntSlice, x.(int))
}

func (h *hp1) Pop() interface{} {
	v := h.IntSlice[len(h.IntSlice)-1]
	h.IntSlice = h.IntSlice[:len(h.IntSlice)-1]
	return v
}

func ConstructorM() MedianFinder {
	return MedianFinder{}
}

func (m *MedianFinder) AddNum(num int) {
	if m.queMin.Len() == 0 || num <= -m.queMin.IntSlice[0] {
		heap.Push(&m.queMin, -num)
		if m.queMax.Len()+1 < m.queMin.Len() {
			heap.Push(&m.queMax, -heap.Pop(&m.queMin).(int))
		}
	} else {
		heap.Push(&m.queMax, num)
		if m.queMax.Len() > m.queMin.Len() {
			heap.Push(&m.queMin, -heap.Pop(&m.queMax).(int))
		}
	}
}

func (m *MedianFinder) FindMedian() float64 {
	if m.queMin.Len() > m.queMax.Len() {
		return float64(-m.queMin.IntSlice[0])
	}
	return float64(m.queMax.IntSlice[0]-m.queMin.IntSlice[0]) / 2
}
