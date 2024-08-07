package main

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"strings"
)

func main() {
	//r := isIsomorphic("paper", "title")
	//r := wordPattern("abc", "b c a")
	//r := isAnagram("anagram", "nagaram")
	//r := longestConsecutive1([]int{100, 4, 200, 1, 3, 2})
	//r := longestConsecutive1([]int{0})

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
	//obj := ConstructorM()
	//obj.AddNum(1)
	//obj.AddNum(2)
	//fmt.Println(obj.FindMedian())
	//obj.AddNum(3)
	//fmt.Println(obj.FindMedian())
	//r := addBinary("11", "1")
	//r := mySqrt(1)
	//r := rob([]int{1, 2, 3, 1})
	//r := coinChange([]int{2}, 3)
	//r := lengthOfLIS([]int{1, 3, 6, 7, 9, 4, 10, 5, 6})
	//r := minPathSum([][]int{{9, 1, 4, 8}})
	//r := uniquePathsWithObstacles1([][]int{{0, 0, 0}, {0, 1, 0}, {0, 0, 0}})
	//r := longestPalindrome("babad")
	//r := maximalSquare([][]byte{
	//	{'1', '0', '1', '0', '0'},
	//	{'1', '0', '1', '1', '1'},
	//	{'1', '1', '1', '1', '1'},
	//	{'1', '0', '0', '1', '0'}})
	//r := removeDuplicates12([]int{1, 2, 2})
	//r := romanToInt("MCMXCIV")
	//r := convert("AB", 1)
	//r := isSubsequence("abc", "ahbgdc")
	//r := Add[int](100, 200)
	//fmt.Println(r)
}

type Wow[T int | string] int

type NewType[T interface{ *int }] []T

func Add[T int | string | float64](a T, b T) T {
	return a + b
}

/*
*
42. 接雨水
https://leetcode.cn/problems/trapping-rain-water/
*/
func trap(height []int) int {
	res := 0
	for i := 1; i < len(height)-1; i++ {
		l, maxL, r, maxR := i-1, height[i-1], i+1, height[i+1]
		for l >= 0 {
			if height[l] > height[i] {
				maxL = max(height[l], maxL)
			}
			l--
		}
		for r < len(height) {
			if height[r] > height[i] {
				maxR = max(height[r], maxR)
			}
			r++
		}
		res += max((min(maxL, maxR) - height[i]), 0)
	}
	return res
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
88. 合并两个有序数组
*https://leetcode.cn/problems/merge-sorted-array/?envType=study-plan-v2&envId=top-interview-150
*/
func merge123(nums1 []int, m int, nums2 []int, n int) {
	if n == 0 {
		return
	}
	for i := m + n - 1; i >= 0; i-- {
		if m == 0 {
			nums1[i] = nums2[n-1]
			n--
			continue
		}
		if n == 0 {
			return
		}
		if nums1[m-1] >= nums2[n-1] {
			nums1[i] = nums1[m-1]
			m--
		} else {
			nums1[i] = nums2[n-1]
			n--
		}
	}
	return
}

/*
*
27. 移除元素
https://leetcode.cn/problems/remove-element/?envType=study-plan-v2&envId=top-interview-150
*/
func removeElement(nums []int, val int) int {
	if len(nums) < 1 {
		return 0
	}
	left, right, r := 0, 0, len(nums)
	for right < len(nums) {
		if nums[right] != val {
			if left != right {
				nums[left] = nums[right]
			}
			left++
			right++
			continue
		}
		right++
		r--
	}
	return r
}

/*
*
26. 删除有序数组中的重复项
https://leetcode.cn/problems/remove-duplicates-from-sorted-array/?envType=study-plan-v2&envId=top-interview-150
*/
func removeDuplicates(nums []int) int {
	if len(nums) <= 1 {
		return len(nums)
	}
	left, right, r := 0, 1, len(nums)
	for right < len(nums) {
		if nums[right] != nums[left] {
			nums[left+1] = nums[right]
			left++
			right++
			continue
		}
		right++
		r--
	}
	return r
}

/*
*
80. 删除有序数组中的重复项 II
https://leetcode.cn/problems/remove-duplicates-from-sorted-array-ii/?envType=study-plan-v2&envId=top-interview-150
*/
func removeDuplicates12(nums []int) int {
	if len(nums) <= 2 {
		return len(nums)
	}
	/**
	1 2 2
	*/
	left, right, val, r := 0, 1, nums[0], len(nums)
	for right < len(nums) {
		if nums[right] == val {
			if left > 0 && nums[left-1] == nums[left] && nums[left] == val {
				right++
				if right > left+1 {
					r--
				}
			} else {
				nums[left+1] = nums[right]
				left++
				right++
			}
			continue
		}
		nums[left+1] = nums[right]
		val = nums[right]
		left++
		right++
	}
	fmt.Println(nums)
	return r
}

func removeDuplicates123(nums []int) int {
	if len(nums) <= 2 {
		return len(nums)
	}
	left, right := 2, 2
	for right < len(nums) {
		if nums[right] != nums[left-2] {
			nums[left] = nums[right]
			left++
		}
		right++
	}
	return left
}

/*
买卖股票的最佳时机 II
https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/?envType=study-plan-v2&envId=top-interview-150
*/
func maxProfit(prices []int) int {
	if len(prices) < 1 {
		return 0
	}
	dp := make([][2]int, len(prices))
	dp[0][1] = -prices[0]
	for i := 1; i < len(prices); i++ {
		dp[i][0] = max(dp[i-1][0], dp[i-1][1]+prices[i])
		dp[i][1] = max(dp[i-1][1], dp[i-1][0]-prices[i])
	}
	return dp[len(prices)-1][0]
}

func maxProfit1(prices []int) int {
	r := 0
	for i := 1; i < len(prices); i++ {
		r += max(0, prices[i]-prices[i-1])
	}
	return r
}

/*
55. 跳跃游戏
*https://leetcode.cn/problems/jump-game/?envType=study-plan-v2&envId=top-interview-150
*/
func canJump(nums []int) bool {
	if len(nums) < 1 {
		return true
	}
	i, dis, maxDis := 0, nums[0], nums[0]
	for i <= dis {
		for ; i <= dis; i++ {
			if nums[i]+i > maxDis {
				maxDis = nums[i] + i
			}
			if maxDis >= len(nums)-1 {
				return true
			}
		}
		dis = maxDis
	}
	return false
}

func jump(nums []int) int {
	if len(nums) < 1 {
		return 0
	}
	start, end, cnt := 0, 1, 0
	for end < len(nums) {
		maxPos := 0
		for i := start; i < end; i++ {
			maxPos = max(maxPos, i+nums[i])
		}
		start = end
		end = maxPos + 1
		cnt++
	}
	return cnt
}

func hIndex(citations []int) int {
	sort.Ints(citations)
	cnt := 0
	for i := len(citations) - 1; i >= 0; i-- {
		if citations[i] > cnt {
			cnt++
		}
	}
	return cnt
}

/*
*
380. O(1) 时间插入、删除和获取随机元素
https://leetcode.cn/problems/insert-delete-getrandom-o1/?envType=study-plan-v2&envId=top-interview-150
*/
type RandomizedSet struct {
	mp   map[int]int
	nums []int
}

func ConstructorRs() RandomizedSet {
	return RandomizedSet{
		nums: make([]int, 0),
		mp:   make(map[int]int),
	}
}

func (r *RandomizedSet) Insert(val int) bool {
	if _, ok := r.mp[val]; ok {
		return false
	}
	r.nums = append(r.nums, val)
	r.mp[val] = len(r.nums) - 1
	return true
}

func (r *RandomizedSet) Remove(val int) bool {
	idx, ok := r.mp[val]
	if !ok {
		return false
	}
	r.nums[idx] = r.nums[len(r.nums)-1]
	r.mp[r.nums[idx]] = idx
	delete(r.mp, val)
	r.nums = r.nums[:len(r.nums)-1]
	return true
}

func (r *RandomizedSet) GetRandom() int {
	return r.nums[rand.Intn(len(r.nums))]
}

/*
238. 除自身以外数组的乘积
*https://leetcode.cn/problems/product-of-array-except-self/?envType=study-plan-v2&envId=top-interview-150
*/
func productExceptSelf(nums []int) []int {
	prefixAns, answer := make([]int, len(nums)), make([]int, len(nums))
	for k, _ := range nums {
		if k == 0 {
			prefixAns[k] = 1
		} else {
			prefixAns[k] = prefixAns[k-1] * nums[k-1]
		}
	}
	suffix := 1
	for i := len(nums) - 1; i >= 0; i-- {
		if i == len(nums)-1 {
			answer[i] = prefixAns[i]
		} else {
			suffix *= nums[i+1]
			answer[i] = prefixAns[i] * suffix
		}
	}
	return answer
}

/*
*
13. 罗马数字转整数
https://leetcode.cn/problems/roman-to-integer/?envType=study-plan-v2&envId=top-interview-150
*/
func romanToInt(s string) int {
	if len(s) < 1 {
		return 0
	}
	mp := map[byte]int{
		'I': 1,
		'V': 5,
		'X': 10,
		'L': 50,
		'C': 100,
		'D': 500,
		'M': 1000,
	}
	var r int
	for k, _ := range s {
		t := mp[s[k]]
		if k == len(s)-1 || mp[s[k+1]] <= t {
			r += t
		} else {
			r -= t
		}
	}
	return r
}

/*
*
14. 最长公共前缀
https://leetcode.cn/problems/longest-common-prefix/?envType=study-plan-v2&envId=top-interview-150
*/
func longestCommonPrefix(strs []string) string {
	if len(strs) < 1 {
		return ""
	}
	for i := 0; i < len(strs[0]); i++ {
		c := strs[0][i : i+1]
		for j := 1; j < len(strs); j++ {
			if i == len(strs[j]) || c != strs[j][i:i+1] {
				return strs[0][0:i]
			}
		}
	}
	return strs[0]
}

/*
151. 反转字符串中的单词
*https://leetcode.cn/problems/reverse-words-in-a-string/?envType=study-plan-v2&envId=top-interview-150
*/
func reverseWords(s string) string {
	if len(s) <= 1 {
		return s
	}
	s1 := reverseStr(s)
	list, r := strings.Split(s1, " "), ""
	for i := 0; i < len(list); i++ {
		if list[i] == "" {
			continue
		}
		subStr := reverseStr(list[i])
		if i == 0 {
			r = fmt.Sprintf("%s%s", r, subStr)
		} else {
			r = fmt.Sprintf("%s %s", r, subStr)
		}
	}
	return r
}

func reverseStr(s string) string {
	s = strings.Trim(s, " ")
	if len(s) < 1 {
		return s
	}
	r, i, j := []byte(s), 0, len(s)-1
	for i < j {
		r[i], r[j] = r[j], r[i]
		i++
		j--
	}
	return string(r)
}

/*
*
https://leetcode.cn/problems/is-subsequence/?envType=study-plan-v2&envId=top-interview-150
*/
func isSubsequence(s string, t string) bool {
	if len(s) == 0 {
		return true
	}
	i, j := 0, 0
	for i < len(s) && j < len(t) {
		if s[i] == t[j] {
			i++
			j++
		} else {
			j++
		}
	}
	if i != len(s) {
		return false
	}
	return true
}

/*
15. 三数之和
https://leetcode.cn/problems/3sum/?envType=study-plan-v2&envId=top-interview-150
*/
func threeSum1(nums []int) [][]int {
	r := make([][]int, 0)
	if len(nums) < 3 {
		return r
	}
	sort.Ints(nums)
	for i := 0; i < len(nums); i++ {
		if nums[i] > 0 {
			return r
		}
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}
		start, end := i+1, len(nums)-1
		for start < end {
			if nums[i]+nums[start]+nums[end] > 0 {
				end--
			} else if nums[i]+nums[start]+nums[end] < 0 {
				start++
			} else {
				r = append(r, []int{nums[i], nums[start], nums[end]})
				for start < end && nums[start] == nums[start+1] {
					start++
				}
				for start < end && nums[end] == nums[end-1] {
					end--
				}
				start++
				end--
			}
		}
	}
	return r
}

/*
*
6. Z 字形变换
https://leetcode.cn/problems/zigzag-conversion/?envType=study-plan-v2&envId=top-interview-150
*/
func convert1(s string, numRows int) string {
	if numRows == 1 {
		return s
	}
	res, flag, cnt, i := make([]string, numRows), 1, 0, 0
	for cnt < len(s) {
		if i == numRows-1 || (cnt != 0 && i == 0) {
			flag = -flag
		}
		res[i] += string(s[cnt])
		i += flag
		cnt++
	}
	var r string
	for _, v := range res {
		r += v
	}
	return r
}

/*
125. 验证回文串
*https://leetcode.cn/problems/valid-palindrome/?envType=study-plan-v2&envId=top-interview-150
*/
func isPalindromeStr(s string) bool {
	if len(s) <= 1 {
		return true
	}
	s = strings.ToLower(s)
	i, j := 0, len(s)-1
	for i < j {
		if (s[i] >= '0' && s[i] <= '9' || s[i] >= 'a' && s[i] <= 'z') &&
			(s[j] >= '0' && s[j] <= '9' || s[j] >= 'a' && s[j] <= 'z') {
			if s[i] != s[j] {
				return false
			}
			i++
			j--
		} else if s[i] >= '0' && s[i] <= '9' || s[i] >= 'a' && s[i] <= 'z' {
			j--
		} else {
			i++
		}
	}
	return true
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
	// u:上边界、d:下边界、l:左边界、r:右边界
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

func rotate(matrix [][]int) {
	n := len(matrix)
	for i := 0; i < n/2; i++ {
		for j := 0; j < (n+1)/2; j++ {
			/**
			i = 1, j = 0, n = 4
			1,0 => 0,2 | 0,2 => 2,3  | 2,3 => 3,1  | 3,1 => 1,0

			公式：
				matrix[j][n-i-1] = a[i][j]
			通过推导，matrix[j][n-i-1] 、matrix[n-i-1][n-j-1]、matrix[n-j-1][i] 带入 公式，得到下面的等式
				matrix[n-i-1][n-j-1] = matrix[j][n-i-1]
				matrix[n-j-1][i] = matrix[n-i-1][n-j-1]
				matrix[i][j] = matrix[n-j-1][i]

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
func isIsomorphic1(s string, t string) bool {
	if len(s) != len(t) {
		return false
	}
	m1, m2 := make(map[byte]byte), make(map[byte]byte)
	for i := 0; i < len(s); i++ {
		b1, ok1 := m1[s[i]]
		b2, ok2 := m2[t[i]]
		if ok1 && ok2 { // true true
			if b1 != t[i] || b2 != s[i] {
				return false
			}
		} else if !ok1 && !ok2 {
			m1[s[i]] = t[i]
			m2[t[i]] = s[i]
		} else {
			return false
		}
	}
	return true
}

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
// 数量多时候超时，用下一个
func longestConsecutive1(nums []int) int {
	if len(nums) < 1 {
		return 0
	}
	mp, maxLen, r := make(map[int]int), 0, 1
	for _, n := range nums {
		mp[n] = 1
	}
	/**
	100, 4, 200, 1, 3, 2
	*/
	for _, n := range nums {
		if preV, ok := mp[n-1]; ok {
			mp[n] = preV + 1
			maxLen = preV + 1
		}
		if _, ok := mp[n+1]; ok {
			cnt := n + 1
			for {
				if _, ok := mp[cnt]; !ok {
					break
				}
				mp[cnt] = mp[cnt-1] + 1
				cnt++
			}
			maxLen = mp[cnt-1]
		}
		if maxLen > r {
			r = maxLen
		}
	}
	return r
}

/*
*
找到每一个连续队列的头节点开始遍历，效率高
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

func reverseBetween123(head *ListNode, left int, right int) *ListNode {
	/**
	1 c2 n3 4 5  left=2, right = 4
	1 3 c2 4 5 => 1 4 3 2 5
	*/
	if head == nil {
		return nil
	}
	dummyHead := &ListNode{Next: head}
	pre := dummyHead
	for i := 0; i < left-1; i++ {
		pre = pre.Next
	}
	cur := pre.Next
	for i := 0; i < right-left; i++ {
		next := cur.Next
		cur.Next = next.Next
		next.Next = pre.Next
		pre.Next = next
	}
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
*
首尾相连，然后找到要断开的位置进行切断
*/
func rotateRight12(head *ListNode, k int) *ListNode {
	if head == nil {
		return nil
	}
	h, n := head, 1
	for h.Next != nil {
		h = h.Next
		n++
	}
	add := n - k%n
	if add == n {
		return head
	}
	h.Next = head
	for add > 0 {
		h = h.Next
		add--
	}
	r := h.Next
	h.Next = nil
	return r
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
func maxDepth1(root *TreeNode) int {
	if root == nil {
		return 0
	}
	if root.Left == nil && root.Right == nil {
		return 1
	}
	return max(maxDepth(root.Left)+1, maxDepth(root.Right)+1)
}

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
*
105. 从前序与中序遍历序列构造二叉树
https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/?envType=study-plan-v2&envId=top-interview-150
*/
func buildTree(preorder []int, inorder []int) *TreeNode {
	if len(preorder) == 0 || len(inorder) == 0 {
		return nil
	}
	var build func(preorder, inorder []int) *TreeNode
	build = func(preorder, inorder []int) *TreeNode {
		if len(preorder) == 0 || len(inorder) == 0 {
			return nil
		}
		rootIndex := 0
		for idx, val := range inorder {
			if val == preorder[0] {
				rootIndex = idx
				break
			}
		}
		preLeft, preRight := preorder[1:rootIndex+1], preorder[rootIndex+1:]
		inLeft, inRight := inorder[:rootIndex], inorder[rootIndex+1:]
		root := &TreeNode{
			Val:   preorder[0],
			Left:  build(preLeft, inLeft),
			Right: build(preRight, inRight),
		}
		return root
	}
	return build(preorder, inorder)
}

/*
106. 从中序与后序遍历序列构造二叉树
*https://leetcode.cn/problems/construct-binary-tree-from-inorder-and-postorder-traversal/?envType=study-plan-v2&envId=top-interview-150
*/
func buildTree1(inorder []int, postorder []int) *TreeNode {
	if len(postorder) == 0 || len(inorder) == 0 {
		return nil
	}
	var build func(inorder, postorder []int) *TreeNode
	build = func(inorder, postorder []int) *TreeNode {
		if len(postorder) == 0 || len(inorder) == 0 {
			return nil
		}
		rootIndex := 0
		for idx, val := range inorder {
			if val == postorder[len(postorder)-1] {
				rootIndex = idx
				break
			}
		}
		postLeft, postRight := postorder[:rootIndex], postorder[rootIndex:len(postorder)-1]
		inLeft, inRight := inorder[:rootIndex], inorder[rootIndex+1:]
		root := &TreeNode{
			Val:   postorder[len(postorder)-1],
			Left:  build(inLeft, postLeft),
			Right: build(inRight, postRight),
		}
		return root
	}
	return build(inorder, postorder)
}

/*
*
117. 填充每个节点的下一个右侧节点指针 II
https://leetcode.cn/problems/populating-next-right-pointers-in-each-node-ii/?envType=study-plan-v2&envId=top-interview-150
*/

type NodeN struct {
	Val   int
	Left  *NodeN
	Right *NodeN
	Next  *NodeN
}

func connect(root *NodeN) *NodeN {
	if root == nil {
		return root
	}
	queue := make([]*NodeN, 0)
	queue = append(queue, root)
	for len(queue) > 0 {
		cnt := len(queue)
		for i := 0; i < cnt; i++ {
			var nextNode *NodeN
			if i+1 < cnt {
				nextNode = queue[1]
			}
			curNode := queue[0]
			queue = queue[1:]
			curNode.Next = nextNode
			if curNode.Left != nil {
				queue = append(queue, curNode.Left)
			}
			if curNode.Right != nil {
				queue = append(queue, curNode.Right)
			}
		}
	}
	return root
}

/*
114. 二叉树展开为链表
https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/description/?envType=study-plan-v2&envId=top-interview-150
*/
func flatten(root *TreeNode) {
	if root == nil {
		return
	}
	for root != nil {
		if root.Left == nil {
			root = root.Right
		} else {
			l := root.Left
			for l.Right != nil {
				l = l.Right
			}
			l.Right = root.Right
			root.Right = root.Left
			root.Left = nil
			root = root.Right
		}
	}
	return
}

func flatten1(root *TreeNode) {
	if root == nil {
		return
	}
	stack := make([]*TreeNode, 0)
	stack = append(stack, root)
	var pre *TreeNode
	for len(stack) > 0 {
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if pre != nil {
			pre.Right = node
			pre.Left = nil
		}
		if node.Right != nil {
			stack = append(stack, node.Right)
		}
		if node.Left != nil {
			stack = append(stack, node.Left)
		}
		pre = node
	}
}

/*
*
112. 路径总和
https://leetcode.cn/problems/path-sum/description/?envType=study-plan-v2&envId=top-interview-150
*/
func hasPathSum(root *TreeNode, targetSum int) bool {
	if root == nil {
		return false
	}
	stack, valStack := make([]*TreeNode, 0), make([]int, 0)
	stack = append(stack, root)
	valStack = append(valStack, root.Val)
	for len(stack) > 0 {
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		sum := valStack[len(valStack)-1]
		valStack = valStack[:len(valStack)-1]
		if node.Left == nil && node.Right == nil && sum == targetSum {
			return true
		}
		if node.Right != nil {
			stack = append(stack, node.Right)
			valStack = append(valStack, sum+node.Right.Val)
		}
		if node.Left != nil {
			stack = append(stack, node.Left)
			valStack = append(valStack, sum+node.Left.Val)
		}
	}
	return false
}

func hasPathSum1(root *TreeNode, targetSum int) bool {
	if root == nil {
		return false
	}
	if root.Left == nil && root.Right == nil && targetSum == root.Val {
		return true
	}
	return hasPathSum1(root.Left, targetSum-root.Val) || hasPathSum1(root.Right, targetSum-root.Val)
}

func hasPathSum2(root *TreeNode, targetSum int) bool {
	if root == nil {
		return false
	}
	path, list := make([]int, 0), make([][]int, 0)
	hasPathSum2Impl(root, &path, &list)
	for _, l := range list {
		if sumInt(l) == targetSum {
			return true
		}
	}
	return false
}

func sumInt(path []int) int {
	var sum int
	for _, val := range path {
		sum += val
	}
	return sum
}

func hasPathSum2Impl(root *TreeNode, path *[]int, list *[][]int) {
	if root == nil {
		return
	}
	*path = append(*path, root.Val)
	if root.Left == nil && root.Right == nil {
		tmp := make([]int, 0)
		tmp = append(tmp, *path...)
		*list = append(*list, tmp)
	}
	if root.Left != nil {
		hasPathSum2Impl(root.Left, path, list)
	}
	if root.Right != nil {
		hasPathSum2Impl(root.Right, path, list)
	}
	*path = (*path)[:len(*path)-1]
	return
}

/*
129. 求根节点到叶节点数字之和
https://leetcode.cn/problems/sum-root-to-leaf-numbers/description/?envType=study-plan-v2&envId=top-interview-150
*/
func sumNumbers1(root *TreeNode) int {
	if root == nil {
		return 0
	}
	return sumNumbersDfs(root, 0)
}

func sumNumbersDfs(root *TreeNode, preSum int) int {
	if root == nil {
		return 0
	}
	sum := preSum*10 + root.Val
	if root.Left == nil && root.Right == nil {
		return sum
	}
	return sumNumbersDfs(root.Left, sum) + sumNumbersDfs(root.Right, sum)
}

func sumNumbers(root *TreeNode) int {
	if root == nil {
		return 0
	}
	var pathSlice []int
	treePath(root, 0, &pathSlice)
	return sumPath(pathSlice)
}
func treePath(root *TreeNode, pathVal int, pathSlice *[]int) {
	if root.Left == nil && root.Right == nil {
		pathVal = pathVal*10 + root.Val
		*pathSlice = append(*pathSlice, pathVal)
		return
	}
	pathVal = pathVal*10 + root.Val
	if root.Left != nil {
		treePath(root.Left, pathVal, pathSlice)
	}
	if root.Right != nil {
		treePath(root.Right, pathVal, pathSlice)
	}
}

func sumPath(path []int) int {
	if len(path) < 1 {
		return 0
	}
	var r int
	for _, v := range path {
		r += v
	}
	return r
}

type pv struct {
	node *TreeNode
	num  int
}

func sumNumbers2(root *TreeNode) int {
	if root == nil {
		return 0
	}
	queue, sum := make([]pv, 0), 0
	queue = append(queue, pv{root, root.Val})
	for len(queue) > 0 {
		val := queue[0]
		queue = queue[1:]
		left, right, num := val.node.Left, val.node.Right, val.num
		if left == nil && right == nil {
			sum += num
		} else {
			if left != nil {
				queue = append(queue, pv{left, num*10 + left.Val})
			}
			if right != nil {
				queue = append(queue, pv{right, num*10 + right.Val})
			}
		}
	}
	return sum
}

/*
*
124. 二叉树中的最大路径和
https://leetcode.cn/problems/binary-tree-maximum-path-sum/?envType=study-plan-v2&envId=top-interview-150
*/
func maxPathSum(root *TreeNode) int {
	maxSum := math.MinInt32
	var maxGain func(root *TreeNode) int
	maxGain = func(root *TreeNode) int {
		if root == nil {
			return 0
		}
		leftGain, rightGain := max(maxGain(root.Left), 0), max(maxGain(root.Right), 0)
		sum := root.Val + leftGain + rightGain
		maxSum = max(maxSum, sum)
		return root.Val + max(leftGain, rightGain)
	}
	maxGain(root)
	return maxSum
}

func maxPathSum1(root *TreeNode) int {
	if root == nil {
		return 0
	}
	r, nodeList := math.MinInt32, make([]*TreeNode, 0)
	preTravel(root, &nodeList)
	for _, n := range nodeList {
		s := getMaxSumByNode(n)
		if r < s {
			r = s
		}
	}
	return r
}

func preTravel(root *TreeNode, nodeList *[]*TreeNode) {
	if root == nil {
		return
	}
	*nodeList = append(*nodeList, root)
	preTravel(root.Left, nodeList)
	preTravel(root.Right, nodeList)
}

func getMaxSumByNode(root *TreeNode) int {
	path, leftList, rightList := make([]int, 0), make([][]int, 0), make([][]int, 0)
	maxPathSumImpl(root.Left, path, &leftList)
	maxPathSumImpl(root.Right, path, &rightList)
	var maxLeft, maxRight int
	for _, l := range leftList {
		s := maxSum(l)
		if maxLeft < s {
			maxLeft = s
		}
	}
	for _, l := range rightList {
		s := maxSum(l)
		if maxRight < s {
			maxRight = s
		}
	}
	return maxLeft + maxRight + root.Val
}

func maxSum(list []int) int {
	sum, maxSum := 0, 0
	for i := 0; i < len(list); i++ {
		sum += list[i]
		if sum < 0 {
			sum = 0
		}
		if sum > maxSum {
			maxSum = sum
		}
	}
	return maxSum
}

func maxPathSumImpl(root *TreeNode, path []int, list *[][]int) {
	if root == nil {
		return
	}
	path = append(path, root.Val)
	if root.Left == nil && root.Right == nil {
		tmp := make([]int, 0)
		tmp = append(tmp, path...)
		*list = append(*list, tmp)
		return
	}
	if root.Left != nil {
		maxPathSumImpl(root.Left, path, list)
	}
	if root.Right != nil {
		maxPathSumImpl(root.Right, path, list)
	}
}

/*
*
173. 二叉搜索树迭代器
https://leetcode.cn/problems/binary-search-tree-iterator/?envType=study-plan-v2&envId=top-interview-150
*/
type BSTIterator struct {
	root     *TreeNode
	nodeList []*TreeNode
	cur      int
}

func ConstructorBSTIterator(root *TreeNode) BSTIterator {
	nodeList := make([]*TreeNode, 0)
	inOrderT(root, &nodeList)
	bst := BSTIterator{
		root:     root,
		nodeList: nodeList,
		cur:      -1,
	}
	return bst
}

func inOrderT(root *TreeNode, nodeList *[]*TreeNode) {
	if root == nil {
		return
	}
	inOrderT(root.Left, nodeList)
	*nodeList = append(*nodeList, root)
	inOrderT(root.Right, nodeList)
}

func (bst *BSTIterator) Next() int {
	if !bst.HasNext() {
		return -1
	}
	bst.cur++
	return bst.nodeList[bst.cur].Val
}

func (bst *BSTIterator) HasNext() bool {
	return bst.cur+1 >= len(bst.nodeList)
}

/*
*
236. 二叉树的最近公共祖先
https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/description/?envType=study-plan-v2&envId=top-interview-150
*/
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	parent, visited := make(map[int]*TreeNode), make(map[int]bool)
	var dfs func(root *TreeNode)
	dfs = func(root *TreeNode) {
		if root == nil {
			return
		}
		if root.Left != nil {
			parent[root.Left.Val] = root
			dfs(root.Left)
		}
		if root.Right != nil {
			parent[root.Right.Val] = root
			dfs(root.Right)
		}
	}
	dfs(root)
	for p != nil {
		visited[p.Val] = true
		p = parent[p.Val]
	}
	for q != nil {
		if _, ok := visited[q.Val]; ok {
			return q
		}
		q = parent[q.Val]
	}
	return nil
}

func lowestCommonAncestor1(root, p, q *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	pPath, qPath, path := make([]*TreeNode, 0), make([]*TreeNode, 0), make([]*TreeNode, 0)
	findPath(root, p, q, path, &pPath, &qPath)
	fmt.Println(pPath, qPath)
	return findCommonRoot(pPath, qPath)
}

func findCommonRoot(path1 []*TreeNode, path2 []*TreeNode) *TreeNode {
	mp := make(map[*TreeNode]struct{})
	var r *TreeNode
	for _, n := range path1 {
		mp[n] = struct{}{}
	}
	for _, n := range path2 {
		if _, ok := mp[n]; ok {
			r = n
		}
	}
	return r
}

func findPath(root, p, q *TreeNode, path []*TreeNode, path1, path2 *[]*TreeNode) {
	if root == nil {
		return
	}
	path = append(path, root)
	if root == p {
		*path1 = append(*path1, path...)
	}
	if root == q {
		*path2 = append(*path2, path...)
	}
	if root.Left != nil {
		findPath(root.Left, p, q, path, path1, path2)
	}
	if root.Right != nil {
		findPath(root.Right, p, q, path, path1, path2)
	}
}

/*
*
144. 二叉树的前序遍历
https://leetcode.cn/problems/binary-tree-preorder-traversal/description/
*/
func preorderTraversal(root *TreeNode) []int {
	r, stack := make([]int, 0), make([]*TreeNode, 0)
	stack = append(stack, root)
	for len(stack) > 0 {
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		r = append(r, node.Val)
		if node.Right != nil {
			stack = append(stack, node.Right)
		}
		if node.Left != nil {
			stack = append(stack, node.Left)
		}
	}
	return r
}

/*
94. 二叉树的中序遍历
https://leetcode.cn/problems/binary-tree-inorder-traversal/
*/
func inorderTraversal(root *TreeNode) []int {
	r, stack := make([]int, 0), make([]*TreeNode, 0)
	stack, node := append(stack, root), root
	for len(stack) > 0 || node != nil {
		for node != nil {
			stack = append(stack, node)
			node = node.Left
		}
		node = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		r = append(r, node.Val)
		node = node.Right
	}
	return r
}

/*
145. 二叉树的后序遍历
https://leetcode.cn/problems/binary-tree-postorder-traversal/description/
*/
func postorderTraversal(root *TreeNode) []int {
	r, stack := make([]int, 0), make([]*TreeNode, 0)
	stack, node := append(stack, root), root
	var pre *TreeNode
	for len(stack) > 0 || node != nil {
		for node != nil {
			stack = append(stack, node)
			node = node.Left
		}
		peekNode := stack[len(stack)-1]
		if peekNode.Right == nil || peekNode.Right == pre {
			r = append(r, peekNode.Val)
			pre = peekNode
			stack = stack[:len(stack)-1]
		} else {
			node = peekNode.Right
		}
	}
	return r
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

func reverseKGroup(head *ListNode, k int) *ListNode {
	dummyHead := &ListNode{}
	dummyHead.Next = head
	pre, tail, next := dummyHead, dummyHead, head
	for next != nil {
		tail = head
		for i := 1; i < k && tail != nil; i++ {
			tail = tail.Next
		}
		if tail == nil {
			break
		}
		next = tail.Next
		tail.Next = nil

		pre.Next = reverseListk1(head)

		pre = head
		head.Next = next
		head = head.Next
	}
	return dummyHead.Next
}

func reverseListk1(head *ListNode) *ListNode {
	if head == nil {
		return head
	}
	var pre *ListNode
	cur := head
	for cur != nil {
		next := cur.Next
		cur.Next = pre
		pre = cur
		cur = next
	}
	return pre
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
637. 二叉树的层平均值
https://leetcode.cn/problems/average-of-levels-in-binary-tree/description/?envType=study-plan-v2&envId=top-interview-150
*/
func averageOfLevels(root *TreeNode) []float64 {
	r := make([]float64, 0)
	if root == nil {
		return r
	}
	queue := make([]*TreeNode, 0)
	queue = append(queue, root)
	for len(queue) > 0 {
		i, cnt, sum := len(queue), len(queue), 0
		for cnt > 0 {
			node := queue[0]
			sum += node.Val
			queue = queue[1:]
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
			cnt--
			if cnt == 0 {
				average := float64(sum) / float64(i)
				r = append(r, average)
			}
		}
	}
	return r
}

type data struct {
	sum, count int
}

func averageOfLevels1(root *TreeNode) []float64 {
	r := make([]float64, 0)
	if root == nil {
		return r
	}
	levelData, dfs := make([]data, 0), func(root *TreeNode, level int) {}
	dfs = func(root *TreeNode, level int) {
		if root == nil {
			return
		}
		if level < len(levelData) {
			levelData[level].sum += root.Val
			levelData[level].count++
		} else {
			levelData = append(levelData, data{sum: root.Val, count: 1})
		}
		if root.Left != nil {
			dfs(root.Left, level+1)
		}
		if root.Right != nil {
			dfs(root.Right, level+1)
		}
	}
	dfs(root, 0)
	for _, val := range levelData {
		r = append(r, float64(val.sum)/float64(val.count))
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

func kthSmallest1(root *TreeNode, k int) int {
	if root == nil {
		return -1
	}
	var dfs func(node *TreeNode)
	var r, i int
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		dfs(node.Left)
		i++
		if i == k {
			r = node.Val
			return
		}
		dfs(node.Right)
	}
	dfs(root)
	return r
}

/*
*
98. 验证二叉搜索树
https://leetcode.cn/problems/validate-binary-search-tree/?envType=study-plan-v2&envId=top-interview-150
*/
func isValidBST(root *TreeNode) bool {
	var preNode *TreeNode
	r := true
	var dfs func(*TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		dfs(node.Left)
		if preNode != nil && preNode.Val >= node.Val {
			r = false
			return
		}
		preNode = node
		dfs(node.Right)
	}
	dfs(root)
	return r
}

func isValidBST123(root *TreeNode) bool {
	if root == nil {
		return true
	}
	var pre *TreeNode
	var isValidBSTImpl func(node *TreeNode) bool
	isValidBSTImpl = func(node *TreeNode) bool {
		if node == nil {
			return true
		}
		if !isValidBSTImpl(node.Left) || (pre != nil && pre.Val >= node.Val) {
			return false
		}
		pre = node
		return isValidBSTImpl(node.Right)
	}
	return isValidBSTImpl(root)
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
	return valid
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

func letterCombinations32(digits string) []string {
	r := make([]string, 0)
	if len(digits) < 1 {
		return r
	}
	letterCombinations32Impl(digits, 0, []string{}, &r)
	return r
}

func letterCombinations32Impl(digits string, idx int, path []string, r *[]string) {
	if len(path) == len(digits) {
		s := strings.Join(path, "")
		*r = append(*r, s)
		return
	}
	str := table[digits[idx]-'0']
	for i := 0; i < len(str); i++ {
		path = append(path, string(str[i]))
		letterCombinations32Impl(digits, idx+1, path, r)
		path = path[:len(path)-1]
	}
	return
}

func combinationSum123(candidates []int, target int) [][]int {
	r, data := make([][]int, 0), make([]int, 0)
	combinationSum123Impl(candidates, target, 0, data, &r)
	return r
}

func sumList(list []int) int {
	var r int
	for _, v := range list {
		r += v
	}
	return r
}

func combinationSum123Impl(candidates []int, target, idx int, data []int, r *[][]int) {
	if sumList(data) == target {
		tmp := make([]int, 0)
		tmp = append(tmp, data...)
		*r = append(*r, tmp)
		return
	} else if sumList(data) > target {
		return
	}
	for i := idx; i < len(candidates); i++ {
		data = append(data, candidates[i])
		combinationSum123Impl(candidates, target, i, data, r)
		data = data[:len(data)-1]
	}
	return
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
	for start <= end {
		mid := start + (end-start)/2
		if nums[mid] == target {
			return mid
		} else if nums[mid] < target {
			start = mid + 1
		} else {
			end = mid - 1
		}
	}
	return start
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
	for start <= end {
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
394. 字符串解码
https://leetcode.cn/problems/decode-string/description/
*/
func decodeString(s string) string {
	stack, idx := make([]string, 0), 0
	for idx < len(s) {
		if s[idx] >= '0' && s[idx] <= '9' {
			num := getNum(s, &idx)
			stack = append(stack, num)
		} else if s[idx] >= 'a' && s[idx] <= 'z' || s[idx] >= 'A' && s[idx] <= 'Z' || s[idx] == '[' {
			stack = append(stack, string(s[idx]))
			idx++
		} else {
			idx++
			strList := []string{}
			for stack[len(stack)-1] != "[" {
				strList = append(strList, stack[len(stack)-1])
				stack = stack[:len(stack)-1]
			}
			for i := 0; i < len(strList)/2; i++ {
				strList[i], strList[len(strList)-i-1] = strList[len(strList)-i-1], strList[i]
			}
			stack = stack[:len(stack)-1]
			repeat, _ := strconv.Atoi(stack[len(stack)-1])
			stack = stack[:len(stack)-1]
			t := strings.Repeat(getString(strList), repeat)
			stack = append(stack, t)
		}
	}
	return getString(stack)
}

func getNum(s string, ptr *int) string {
	ret := ""
	for ; s[*ptr] >= '0' && s[*ptr] <= '9'; *ptr++ {
		ret += string(s[*ptr])
	}
	return ret
}

func getString(v []string) string {
	ret := ""
	for _, s := range v {
		ret += s
	}
	return ret
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

/*
67. 二进制求和
https://leetcode.cn/problems/add-binary/?envType=study-plan-v2&envId=top-interview-150
*/
func addBinary(a string, b string) string {
	if len(a) < 1 {
		return b
	}
	if len(b) < 1 {
		return a
	}
	maxLen, m, n, sum, jw, r := max(len(a), len(b)), 0, 0, 0, 0, ""
	for i := 0; i < maxLen; i++ {
		if i >= len(a) {
			m = 0
		} else {
			m = int(a[len(a)-i-1] - '0')
		}

		if i >= len(b) {
			n = 0
		} else {
			n = int(b[len(b)-i-1] - '0')
		}
		sum += (m + n + jw) % 2
		r = fmt.Sprintf("%d%s", sum, r)
		jw = (m + n + jw) / 2
		sum = 0
	}
	if jw > 0 {
		r = fmt.Sprintf("%d%s", jw, r)
	}
	return r
}

/*
190. 颠倒二进制位
*https://leetcode.cn/problems/reverse-bits/?envType=study-plan-v2&envId=top-interview-150
*/
func reverseBits(num uint32) uint32 {
	var r uint32
	for i := 0; i < 32; i++ {
		r <<= 1
		r += num & 1
		num >>= 1
	}
	return r
}

/*
*
191. 位1的个数
https://leetcode.cn/problems/number-of-1-bits/?envType=study-plan-v2&envId=top-interview-150
*/
func hammingWeight(n int) int {
	cnt := 0
	for n > 0 {
		i := n & 0x1
		if i == 1 {
			cnt++
		}
		n >>= 1
	}
	return cnt
}

/*
*
136. 只出现一次的数字
https://leetcode.cn/problems/single-number/?envType=study-plan-v2&envId=top-interview-150
*/
func singleNumber(nums []int) int {
	if len(nums) < 1 {
		return 0
	}
	var r int
	for _, v := range nums {
		r ^= v
	}
	return r
}

/*
*
137. 只出现一次的数字 II
https://leetcode.cn/problems/single-number-ii/?envType=study-plan-v2&envId=top-interview-150
*/
func singleNumber1(nums []int) int {
	ans := int32(0)
	for i := 0; i < 32; i++ {
		total := int32(0)
		for _, num := range nums {
			total += int32(num) >> i & 1
		}
		if total%3 != 0 {
			ans |= 1 << i
		}
	}
	return int(ans)
}

/*
201. 数字范围按位与
https://leetcode.cn/problems/bitwise-and-of-numbers-range/description/?envType=study-plan-v2&envId=top-interview-150
*/
func rangeBitwiseAnd(left int, right int) int {
	shift := 0
	for left < right {
		left, right = left>>1, right>>1
		shift++
	}
	return left << shift
}

func rangeBitwiseAnd1(left int, right int) int {
	for left < right {
		right &= right - 1
	}
	return right
}

/*
9. 回文数
https://leetcode.cn/problems/palindrome-number/?envType=study-plan-v2&envId=top-interview-150
*/
func isPalindrome(x int) bool {
	if x < 0 {
		return false
	}
	t, p := x, 0
	for t > 0 {
		p *= 10
		p += t % 10
		t /= 10
	}
	return p == x
}

/*
66. 加一
https://leetcode.cn/problems/plus-one/?envType=study-plan-v2&envId=top-interview-150
*/
func plusOne(digits []int) []int {
	if len(digits) < 1 {
		return []int{1}
	}
	jw, r := 0, make([]int, 0)
	for i := len(digits) - 1; i >= 0; i-- {
		var t int
		if i == len(digits)-1 {
			t = digits[i] + 1
			jw = t / 10
			r = append(r, t%10)
			continue
		}
		t = digits[i] + jw
		jw = t / 10
		r = append([]int{t % 10}, r...)
	}
	if jw > 0 {
		r = append([]int{jw}, r...)
	}
	return r
}

/*
*
172. 阶乘后的零
https://leetcode.cn/problems/factorial-trailing-zeroes/?envType=study-plan-v2&envId=top-interview-150
*/
func trailingZeroes(n int) int {
	r := 0
	for i := 5; i <= n; i += 5 {
		for x := i; x%5 == 0; x /= 5 {
			r++
		}
	}
	return r
}

/*
69. x 的平方根
https://leetcode.cn/problems/sqrtx/?envType=study-plan-v2&envId=top-interview-150
*/
func mySqrt(x int) int {
	if x == 0 {
		return 0
	}
	l, n := 1, x
	for l <= n {
		m := (l + n) / 2
		if m*m < x {
			l = m + 1
		} else if m*m > x {
			n = m - 1
		} else {
			return m
		}
	}
	return n
}

/*
*
50. Pow(x, n)
https://leetcode.cn/problems/powx-n/description/?envType=study-plan-v2&envId=top-interview-150
*/
func myPow(x float64, n int) float64 {
	if n == 0 {
		return 1
	}
	if x == 0 {
		return 0
	}
	if n == 1 {
		return x
	}
	if n == -1 {
		return 1 / x
	}
	half := myPow(x, n/2)
	res := myPow(x, n%2)
	return half * half * res
}

/*
*
70. 爬楼梯
https://leetcode.cn/problems/climbing-stairs/?envType=study-plan-v2&envId=top-interview-150
*/
func climbStairs(n int) int {
	if n <= 2 {
		return n
	}
	dp := make(map[int]int)
	dp[1] = 1
	dp[2] = 2
	for i := 3; i <= n; i++ {
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[n]
}

/*
198. 打家劫舍
https://leetcode.cn/problems/house-robber/?envType=study-plan-v2&envId=top-interview-150
*/
func rob(nums []int) int {
	if len(nums) < 1 {
		return 0
	}
	dp := make([]int, len(nums)+1)
	/*
		dp[i] = max(dp[i-1], nums[i] + dp[i-2])
	*/
	dp[0] = 0
	dp[1] = nums[0]
	for i := 1; i < len(nums); i++ {
		dp[i+1] = max(dp[i], nums[i]+dp[i-1])
	}
	return dp[len(dp)-1]
}

/*
139. 单词拆分
https://leetcode.cn/problems/word-break/description/?envType=study-plan-v2&envId=top-interview-150
*/
func wordBreak(s string, wordDict []string) bool {
	mp := make(map[string]bool)
	for _, w := range wordDict {
		mp[w] = true
	}
	dp := make([]bool, len(s)+1)
	dp[0] = true
	for i := 0; i <= len(s); i++ {
		for j := 0; j < i; j++ {
			if dp[j] && mp[s[j:i]] {
				dp[i] = true
				break
			}
		}
	}
	return dp[len(s)]
}

/*
322. 零钱兑换
*https://leetcode.cn/problems/coin-change/?envType=study-plan-v2&envId=top-interview-150
*/
var res = math.MaxInt32

// 简单递归 leetcode 超时
func coinChange(coins []int, amount int) int {
	if len(coins) == 0 {
		return -1
	}
	findWay(coins, amount, 0)
	if res == math.MaxInt32 {
		return -1
	}
	return res
}

func findWay(coins []int, amount, cnt int) {
	if amount < 0 {
		return
	}
	if amount == 0 {
		res = min(res, cnt)
	}
	for i := 0; i < len(coins); i++ {
		findWay(coins, amount-coins[i], cnt+1)
	}
}

func coinChange1(coins []int, amount int) int {
	if len(coins) == 0 {
		return -1
	}
	dp := make([]int, amount+1)
	for k, _ := range dp {
		dp[k] = amount + 1
	}
	dp[0] = 0
	for i := 1; i <= amount; i++ {
		for j := 0; j < len(coins); j++ {
			if i >= coins[j] {
				dp[i] = min(dp[i], dp[i-coins[j]]+1)
			}
		}
	}
	if dp[amount] > amount {
		return -1
	}
	return dp[amount]
}

/*
300. 最长递增子序列
*https://leetcode.cn/problems/longest-increasing-subsequence/?envType=study-plan-v2&envId=top-interview-150
*/
func lengthOfLIS(nums []int) int {
	if len(nums) < 1 {
		return 0
	}
	dp := make([]int, len(nums))
	for i := 0; i < len(dp); i++ {
		dp[i] = 1
	}
	/**
	10, 9, 2, 5, 3, 7, 101, 18
	dp[i] = (x in 0...i-1 && nums[i] > nums[x]) max(dp[x]+1, dp[i])
	*/
	for i := 1; i < len(nums); i++ {
		for j := 0; j < i; j++ {
			if nums[i] > nums[j] {
				dp[i] = max(dp[i], dp[j]+1)
			}
		}
	}
	fmt.Println(dp)
	r := 0
	for _, v := range dp {
		r = max(r, v)
	}
	return r
}

/*
120. 三角形最小路径和
*https://leetcode.cn/problems/triangle/?envType=study-plan-v2&envId=top-interview-150
*/
func minimumTotal(triangle [][]int) int {
	dp := make([][]int, len(triangle))
	for i := 0; i < len(triangle); i++ {
		dp[i] = make([]int, len(triangle))
	}
	dp[0][0] = triangle[0][0]
	for i := 1; i < len(triangle); i++ {
		dp[i][0] = dp[i-1][0] + triangle[i][0]
		for j := 1; j < i; j++ {
			dp[i][j] = min(dp[i-1][j], dp[i-1][j-1]) + triangle[i][j]
		}
		dp[i][i] = dp[i-1][i-1] + triangle[i][i]
	}
	ans := math.MaxInt32
	for i := 0; i < len(triangle); i++ {
		ans = min(ans, dp[len(triangle)-1][i])
	}
	return ans
}

/*
*64. 最小路径和
https://leetcode.cn/problems/minimum-path-sum/description/?envType=study-plan-v2&envId=top-interview-150
*/
func minPathSum(grid [][]int) int {
	if len(grid) < 1 {
		return 0
	}
	dp := make([][]int, len(grid))
	for i := 0; i < len(grid); i++ {
		dp[i] = make([]int, len(grid[i]))
	}
	dp[0][0] = grid[0][0]
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[0]); j++ {
			if i == 0 && j == 0 {
				dp[i][j] = grid[0][0]
			} else if i == 0 {
				dp[i][j] = dp[i][j-1] + grid[i][j]
			} else if j == 0 {
				dp[i][j] = dp[i-1][j] + grid[i][j]
			} else {
				dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
			}
		}
	}
	fmt.Println(dp)
	return dp[len(dp)-1][len(dp[0])-1]
}

/*
63. 不同路径 II
*https://leetcode.cn/problems/unique-paths-ii/?envType=study-plan-v2&envId=top-interview-150
记忆化递归
*/
func uniquePathsWithObstacles(obstacleGrid [][]int) int {
	if len(obstacleGrid) < 1 {
		return 0
	}
	mp := make(map[string]int)
	return uniquePathsWithObstaclesBt(obstacleGrid, 0, 0, mp)
}

func uniquePathsWithObstaclesBt(grid [][]int, i int, j int, mp map[string]int) int {
	if v, ok := mp[fmt.Sprintf("%d%d", i, j)]; ok {
		return v
	}
	if i >= len(grid) || j >= len(grid[0]) || grid[i][j] == 1 || grid[i][j] == 2 {
		return 0
	}
	if i == len(grid)-1 && j == len(grid[0])-1 {
		return 1
	}
	r := uniquePathsWithObstaclesBt(grid, i+1, j, mp) + uniquePathsWithObstaclesBt(grid, i, j+1, mp)
	mp[fmt.Sprintf("%d%d", i, j)] = r
	return r
}

/*
*
动态规划
dp[i][j] = dp[i-1][j] + dp[i][j-1]

[

	[0 1 2]
	[1 0 2]
	[2 2 4]

]
*/
func uniquePathsWithObstacles1(obstacleGrid [][]int) int {
	dp := make([][]int, len(obstacleGrid))
	for i := 0; i < len(obstacleGrid); i++ {
		dp[i] = make([]int, len(obstacleGrid[i]))
	}
	for i := 0; i < len(obstacleGrid); i++ {
		for j := 0; j < len(obstacleGrid[i]); j++ {
			if obstacleGrid[i][j] == 1 {
				dp[i][j] = 0
				continue
			}
			if i == 0 && j == 0 {
				dp[i][j] = 1
			} else if i == 0 {
				if obstacleGrid[i][j-1] == 1 {
					obstacleGrid[i][j] = 1
					dp[i][j] = 0
				} else {
					dp[i][j] = dp[i][j-1]
				}
			} else if j == 0 {
				if obstacleGrid[i-1][j] == 1 {
					obstacleGrid[i][j] = 1
					dp[i][j] = 0
				} else {
					dp[i][j] = dp[i-1][j]
				}
			} else {
				dp[i][j] = dp[i][j-1] + dp[i-1][j]
			}
		}
	}
	fmt.Println(dp)
	return dp[len(dp)-1][len(dp[0])-1]
}

/*
5. 最长回文子串
*https://leetcode.cn/problems/longest-palindromic-substring/?envType=study-plan-v2&envId=top-interview-150
*/
func longestPalindrome(s string) string {
	if len(s) == 0 {
		return ""
	}
	if len(s) == 1 {
		return s
	}
	/**
	dp[i] = max(dp[i-1], s[x:i] + s[i])
	*/
	dp, maxStr := make([]string, len(s)+1), ""
	dp[0] = ""
	for i := 0; i <= len(s); i++ {
		for j := 0; j < i; j++ {
			if isPalindrome11(s[j:i]) {
				if len(dp[i]) < len(s[j:i]) {
					dp[i] = s[j:i]
				} else {
					dp[i] = dp[i]
				}
			}
		}
		if len(dp[i]) > len(maxStr) {
			maxStr = dp[i]
		}
	}
	fmt.Println(dp)
	return maxStr
}

func isPalindrome11(s string) bool {
	fmt.Println(s)
	if len(s) == 0 || len(s) == 1 {
		return true
	}
	i, j := 0, len(s)-1
	for i < j {
		if s[i] != s[j] {
			return false
		}
		i++
		j--
	}
	return true
}

func longestPalindrome11(s string) string {
	start, end := 0, 0
	for i := 0; i < len(s); i++ {
		s1, e1 := expandAroundCenter(s, i, i)
		s2, e2 := expandAroundCenter(s, i, i+1)
		if e1-s1 > end-start {
			start, end = s1, e1
		}
		if e2-s2 > end-start {
			start, end = s2, e2
		}
	}
	return s[start : end+1]
}

func expandAroundCenter(s string, i int, j int) (l, r int) {
	for i >= 0 && j < len(s) && s[i] == s[j] {
		i, j = i-1, j+1
	}
	return i + 1, j - 1
}

/*
97. 交错字符串
*https://leetcode.cn/problems/interleaving-string/?envType=study-plan-v2&envId=top-interview-150
*/
func isInterleave(s1 string, s2 string, s3 string) bool {
	/**
	f[i][j] = (f[i-1][j] && s1[i-1] == s3[i+j-1]) || (f[i][j-1] && s2[j-1] == s3[i+j-1])
	*/
	f, n, m := make([][]bool, len(s1)+1), len(s1), len(s2)
	if (n + m) != len(s3) {
		return false
	}
	for i := 0; i < len(f); i++ {
		f[i] = make([]bool, m+1)
	}
	f[0][0] = true
	for i := 0; i <= n; i++ {
		for j := 0; j <= m; j++ {
			p := i + j - 1
			if i > 0 {
				f[i][j] = f[i][j] || (f[i-1][j] && s1[i-1] == s3[p])
			}
			if j > 0 {
				f[i][j] = f[i][j] || (f[i][j-1] && s2[j-1] == s3[p])
			}
		}
	}
	return f[n][m]
}

/*
72. 编辑距离
*https://leetcode.cn/problems/edit-distance/?envType=study-plan-v2&envId=top-interview-150
*/
func minDistance(word1 string, word2 string) int {
	dp := make([][]int, len(word1)+1)
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, len(word2)+1)
		dp[i][0] = i
	}
	for j := 0; j < len(dp[0]); j++ {
		dp[0][j] = j
	}

	for i := 1; i <= len(word1); i++ {
		for j := 1; j <= len(word2); j++ {
			if word1[i-1] == word2[j-1] {
				dp[i][j] = dp[i-1][j-1]
			} else {
				dp[i][j] = minM(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1
			}
		}
	}
	return dp[len(dp)-1][len(dp[0])-1]
}

func minM(x ...int) int {
	min := x[0]
	for i := 1; i < len(x); i++ {
		if x[i] < min {
			min = x[i]
		}
	}
	return min
}

/*
221. 最大正方形
https://leetcode.cn/problems/maximal-square/?envType=study-plan-v2&envId=top-interview-150
*/
func maximalSquare(matrix [][]byte) int {
	/*
		dp[i][j] = minVal(dp[i-1][j], dp[i][j-1], dp[i][j]) + 1
	*/
	dp, maxSide := make([][]int, len(matrix)), 0
	for i := 0; i < len(matrix); i++ {
		dp[i] = make([]int, len(matrix[i]))
		for j := 0; j < len(matrix[i]); j++ {
			dp[i][j] = int(matrix[i][j] - '0')
			if dp[i][j] == 1 {
				maxSide = 1
			}
		}
	}
	for i := 1; i < len(matrix); i++ {
		for j := 1; j < len(matrix[i]); j++ {
			if dp[i][j] == 1 {
				dp[i][j] = minVal(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
				if dp[i][j] > maxSide {
					maxSide = dp[i][j]
				}
			}
		}
	}
	fmt.Println(dp)
	return maxSide * maxSide
}

func minVal(list ...int) int {
	if len(list) == 0 {
		return 0
	}
	minV := list[0]
	for i := 1; i < len(list); i++ {
		if list[i] < minV {
			minV = list[i]
		}
	}
	return minV
}
