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

	node5 := &ListNode{
		Val:  5,
		Next: nil,
	}
	node4 := &ListNode{
		Val:  4,
		Next: node5,
	}
	node3 := &ListNode{
		Val:  3,
		Next: node4,
	}
	node2 := &ListNode{
		Val:  2,
		Next: node3,
	}
	head2 := &ListNode{
		Val:  1,
		Next: node2,
	}
	//r := addTwoNumbers(head1, head2)
	r := reverseBetween(head2, 2, 4)
	PrintList(r)
	//fmt.Print(r)
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

func Constructor() MinStack {
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
