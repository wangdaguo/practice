package main

import (
	"container/heap"
	"fmt"
	"hash/crc32"
	"math"
	"sort"
	"strconv"
	"strings"
)

func main() {
	//nums := []int{0, 1, 0, 3, 12}
	//moveZeroes(nums)
	//r := maxArea([]int{1, 8, 6, 2, 5, 4, 8, 3, 7})
	//r := findAnagrams("cbaebabacd", "abc")
	//r := subarraySum([]int{-1, -1, 1}, 0)
	//r := coinChange([]int{1, 2, 5}, 11)
	//nums := [][]int{{0, 0, 0, 5}, {4, 3, 1, 4}, {0, 1, 1, 4}, {1, 2, 1, 3}, {0, 0, 1, 1}}
	//setZeroes(nums)
	//fmt.Println(nums)
	//nums := [][]int{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
	//r := spiralOrder(nums)
	//grid := [][]int{{2, 1, 1}, {1, 1, 0}, {0, 1, 1}}
	//r := orangesRotting(grid)
	//r := permute([]int{1, 2, 3})
	//r := letterCombinations("23")
	//r := exist([][]byte{
	//	{'A', 'B', 'C', 'E'},
	//	{'S', 'F', 'C', 'S'},
	//	{'A', 'D', 'E', 'E'},
	//}, "ABCB")
	//r := substring1("100", "199")
	//r := findMin1([]int{4, 5, 6, 7, 0, 1, 2})
	//r := findPeakElement([]int{4, 5, 6, 7, 0, 1, 2})
	//r := decodeString("3[a]2[bc]")
	//r := topKFrequent([]int{1, 1, 1, 2, 2, 3}, 2)
	//r := maxProfit([]int{7, 1, 5, 3, 6, 4})
	r := Crc32(369436432338804)
	fmt.Println(r)
}

func Crc32(num int64) int64 {
	numStr := strconv.FormatInt(num, 10)
	data := []byte(numStr)
	return int64(crc32.ChecksumIEEE(data))
}

/*
*
旋转数组找最大值
*/
func findMax1(nums []int) int {
	start, end := 0, len(nums)-1
	for start < end {
		mid := start + (end-start)/2
		if nums[mid] > nums[start] {
			start = mid
		} else {
			end = mid - 1
		}
	}
	return nums[start]
}

/*
*
旋转数组找最小值
*/
func findMin1(nums []int) int {
	start, end := 0, len(nums)-1
	for start < end {
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
*
旋转数组随意找一个峰值
*/
func findPeakElement(nums []int) int {
	get := func(idx int) int {
		if idx < 0 || idx >= len(nums) {
			return math.MinInt32
		}
		return nums[idx]
	}
	start, end := 0, len(nums)-1
	for start <= end {
		mid := start + (end-start)/2
		if get(mid) > get(mid-1) && get(mid) > get(mid+1) {
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
	if len(nums) < 1 {
		return -1
	}
	start, end := 0, len(nums)-1
	for start <= end {
		mid := start + (end-start)/2
		if nums[mid] == target {
			return mid
		} else if nums[mid] < nums[end] {
			if nums[mid] < target && target <= mid {
				start = mid + 1
			} else {
				end = mid - 1
			}
		} else { // nums[mid] >= nums[start]
			if nums[mid] > target && nums[start] <= target {
				end = mid - 1
			} else {
				start = mid + 1
			}
		}
	}
	return -1
}

func t2w(data []int) {
	for i := 0; i < 10; i++ {
		data = append(data, i)
		fmt.Printf("data in : %p; &data in %p\n", data, &data) // data 代表切片指向数组地址；&data代表切片本身地址
	}
	return
}

/*
*
1. 两数之和
https://leetcode.cn/problems/two-sum/?envType=study-plan-v2&envId=top-100-liked
*/
func twoSum(nums []int, target int) []int {
	mp := make(map[int]int)
	for k, v := range nums {
		if idx, ok := mp[target-v]; ok {
			return []int{idx, k}
		}
		mp[v] = k
	}
	return []int{}
}

/*
*
49. 字母异位词分组
https://leetcode.cn/problems/group-anagrams/description/?envType=study-plan-v2&envId=top-100-liked
*/
func groupAnagrams(strs []string) [][]string {
	mp := make(map[string][]string)
	for _, str := range strs {
		s := []byte(str)
		sort.Slice(str, func(i, j int) bool {
			return s[i] < s[j]
		})
		mp[string(s)] = append(mp[string(s)], str)
	}
	r := make([][]string, 0)
	for _, v := range mp {
		r = append(r, v)
	}
	return r
}

/*
*
128. 最长连续序列
https://leetcode.cn/problems/longest-consecutive-sequence/?envType=study-plan-v2&envId=top-100-liked
*/
func longestConsecutive(nums []int) int {
	r, mp := 0, make(map[int]bool)
	for _, v := range nums {
		mp[v] = true
	}
	for num := range mp {
		if _, ok := mp[num-1]; !ok {
			cur, len := num, 1
			for mp[cur+1] {
				cur++
				len++
			}
			if len > r {
				r = len
			}
		}
	}
	return r
}

/*
*
283. 移动零
https://leetcode.cn/problems/move-zeroes/?envType=study-plan-v2&envId=top-100-liked
*/
func moveZeroes(nums []int) {
	idx := 0
	for i := 0; i < len(nums); i++ {
		if nums[i] != 0 {
			nums[idx] = nums[i]
			idx++
		}
	}
	for idx < len(nums) {
		nums[idx] = 0
		idx++
	}
	return
}

/*
*
11. 盛最多水的容器
https://leetcode.cn/problems/container-with-most-water/?envType=study-plan-v2&envId=top-100-liked
*/
func maxArea(height []int) int {
	area, maxArea, i, j := 0, 0, 0, len(height)-1
	for i < j {
		area = min(height[i], height[j]) * int(math.Abs(float64(i-j)))
		if area > maxArea {
			maxArea = area
		}
		if height[i] > height[j] {
			j--
		} else {
			i++
		}
	}
	return maxArea
}
func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

/*
*
3. 无重复字符的最长子串
https://leetcode.cn/problems/longest-substring-without-repeating-characters/?envType=study-plan-v2&envId=top-100-liked
*/
func lengthOfLongestSubstring(s string) int {
	if len(s) < 1 {
		return 0
	}
	mp, left, cnt := make(map[byte]int), 0, 0
	for i := 0; i < len(s); i++ {
		idx, ok := mp[s[i]]
		if !ok {
			mp[s[i]] = i
			if len(mp) > cnt {
				cnt = len(mp)
			}
			continue
		}
		for left <= idx {
			delete(mp, s[left])
			left++
		}
		mp[s[i]] = i
	}
	return cnt
}

/*
*
438. 找到字符串中所有字母异位词
https://leetcode.cn/problems/find-all-anagrams-in-a-string/?envType=study-plan-v2&envId=top-100-liked
*/
func findAnagrams(s string, p string) []int {
	var sCount, pCount [26]int
	r := make([]int, 0)
	for k, v := range p {
		pCount[v-'a']++
		sCount[s[k]-'a']++
	}
	if pCount == sCount {
		r = append(r, 0)
	}
	/*
		s = "abcabc" 6
		p = "abc"  3
		s[:3] = abc 0/1/2
	*/
	for k, v := range s[:len(s)-len(p)] {
		sCount[v-'a']--
		sCount[s[k+len(p)]-'a']++
		if pCount == sCount {
			r = append(r, k)
		}
	}
	return r
}

/*
*
560. 和为 K 的子数组
https://leetcode.cn/problems/subarray-sum-equals-k/?envType=study-plan-v2&envId=top-100-liked
*/
func subarraySum(nums []int, k int) int {
	pre, mp, r := 0, make(map[int]int), 0
	for i := 0; i < len(nums); i++ {
		pre += nums[i]
		if v, ok := mp[pre-k]; ok {
			r += v
		}
		mp[pre]++
	}
	return r
}

var list []int

type heap12 struct {
	sort.IntSlice
}

func (h heap12) Less(i, j int) bool {
	return list[h.IntSlice[i]] > list[h.IntSlice[j]]
}

func (h heap12) Push(x any) {
	h.IntSlice = append(h.IntSlice, x.(int))
}

func (h heap12) Pop() any {
	val := h.IntSlice[len(h.IntSlice)-1]
	h.IntSlice = h.IntSlice[:len(h.IntSlice)-1]
	return val
}

func maxSlidingWindow1(nums []int, k int) []int {
	list = nums
	hp := &heap12{make(sort.IntSlice, k)}
	for i := 0; i < k; i++ {
		hp.IntSlice[i] = i
	}
	heap.Init(hp)
	r := make([]int, 1)
	r[0] = nums[hp.IntSlice[0]]
	for i := k; i < len(nums); i++ {
		heap.Push(hp, i)
		for i-k >= hp.IntSlice[0] {
			heap.Pop(hp)
		}
		r = append(r, nums[hp.IntSlice[0]])
	}
	return r
}

/*
*
76. 最小覆盖子串
https://leetcode.cn/problems/minimum-window-substring/description/?envType=study-plan-v2&envId=top-100-liked
*/
func minWindow(s string, t string) string {
	cnt, left, mp, char, minStart, minSize := 0, 0, make(map[byte]bool), make(map[byte]int), 0, len(s)+1
	for i := 0; i < len(t); i++ {
		char[t[i]]++
		mp[t[i]] = true
	}
	for i := 0; i < len(s); i++ {
		if _, ok := mp[s[i]]; ok {
			char[s[i]]--
			if char[s[i]] >= 0 {
				cnt++
			}

		}
		for cnt == len(t) {
			if minSize > i-left+1 {
				minSize = i - left + 1
				minStart = left
			}
			char[s[left]]++
			if _, ok := mp[s[left]]; ok && char[s[left]] > 0 {
				cnt--
			}
			left++
		}
	}
	if minSize > len(s) {
		return ""
	}
	return s[minStart : minStart+minSize]
}

/*
53. 最大子数组和
https://leetcode.cn/problems/maximum-subarray/?envType=study-plan-v2&envId=top-100-liked
*/
func maxSubArray(nums []int) int {
	sum, maxSum := nums[0], nums[0]
	for i := 1; i < len(nums); i++ {
		if sum < 0 {
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

/*
*
56. 合并区间
https://leetcode.cn/problems/merge-intervals/?envType=study-plan-v2&envId=top-100-liked
*/
func merge(intervals [][]int) [][]int {
	if len(intervals) <= 1 {
		return intervals
	}
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0] || intervals[i][0] == intervals[j][0] && intervals[i][1] < intervals[j][1]
	})
	start, end, r := intervals[0][0], intervals[0][1], make([][]int, 0)
	for i := 1; i < len(intervals); i++ {
		if end < intervals[i][0] { // [1,2] [3,4]
			r = append(r, []int{start, end})
			start, end = intervals[i][0], intervals[i][1]
		} else if end >= intervals[i][0] && end <= intervals[i][1] { // [1,3] [2,4]
			end = intervals[i][1]
		} else if start <= intervals[i][0] && end >= intervals[i][1] { // [1,4] [2,3]
			continue
		} else {
			end = intervals[i][1]
		}
	}
	r = append(r, []int{start, end})
	return r
}

/*
189. 轮转数组
https://leetcode.cn/problems/rotate-array/?envType=study-plan-v2&envId=top-100-liked
*/
func rotate1(nums []int, k int) {
	k = k % (len(nums))
	rev(nums, 0, len(nums)-1)
	rev(nums, 0, k-1)
	rev(nums, k, len(nums)-1)
	return
}

func rev(nums []int, i, j int) {
	for i < j {
		nums[i], nums[j] = nums[j], nums[i]
		i++
		j--
	}
	return
}

/*
*238. 除自身以外数组的乘积
https://leetcode.cn/problems/product-of-array-except-self/?envType=study-plan-v2&envId=top-100-liked
*/
func productExceptSelf(nums []int) []int {
	pre := make([]int, len(nums)-1)
	pre[0] = 1
	for i := 1; i < len(nums)-1; i++ {
		pre[i] = pre[i-1] * nums[i-1]
	}
	suffix, r := 1, make([]int, len(nums))
	for i := len(nums) - 1; i >= 0; i-- {
		r[i] = suffix * pre[i]
		suffix *= nums[i]
	}
	return r
}

/*
*73. 矩阵置零
https://leetcode.cn/problems/set-matrix-zeroes/?envType=study-plan-v2&envId=top-100-liked
*/
func setZeroes(matrix [][]int) {
	col, row := make([]int, 0), make([]int, 0)
	for i := 0; i < len(matrix); i++ {
		for j := 0; j < len(matrix[i]); j++ {
			if matrix[i][j] == 0 {
				col = append(col, j)
				row = append(row, i)
			}
		}
	}
	for i := 0; i < len(row); i++ {
		for j := 0; j < len(matrix[row[i]]); j++ {
			matrix[row[i]][j] = 0
		}
	}
	for i := 0; i < len(col); i++ {
		for j := 0; j < len(matrix); j++ {
			matrix[j][col[i]] = 0
		}
	}
}

/*
*41. 缺失的第一个正数
https://leetcode.cn/problems/first-missing-positive/?envType=study-plan-v2&envId=top-100-liked
*/
func firstMissingPositive(nums []int) int {
	/**
	nums = [2, 1]
	*/
	for i := 0; i < len(nums); i++ {
		for nums[i] > 0 && nums[i] <= len(nums) && nums[nums[i]-1] != nums[i] {
			nums[nums[i]-1], nums[i] = nums[i], nums[nums[i]-1]
		}
	}
	for i := 0; i < len(nums); i++ {
		if nums[i] != i+1 {
			return i + 1
		}
	}
	return len(nums) + 1
}

/*
54. 螺旋矩阵
*https://leetcode.cn/problems/spiral-matrix/?envType=study-plan-v2&envId=top-100-liked
*/
func spiralOrder(matrix [][]int) []int {
	up, down, left, right, r := 0, len(matrix)-1, 0, len(matrix[0])-1, make([]int, 0)
	for {
		for i := left; i <= right; i++ {
			r = append(r, matrix[up][i])
		}
		up++
		if up > down {
			break
		}
		for i := up; i <= down; i++ {
			r = append(r, matrix[i][right])
		}
		right--
		if left > right {
			break
		}
		for i := right; i >= left; i-- {
			r = append(r, matrix[down][i])
		}
		down--
		if up > down {
			break
		}
		for i := down; i >= up; i-- {
			r = append(r, matrix[i][left])
		}
		left++
		if left > right {
			break
		}
	}
	return r
}

/*
48. 旋转图像
https://leetcode.cn/problems/rotate-image/?envType=study-plan-v2&envId=top-100-liked
*/
func rotate(matrix [][]int) {
	for i := 0; i < len(matrix)/2; i++ {
		for j := 0; j < (len(matrix)+1)/2; j++ {
			/**
			base : i, j => j, len(matrix)-i-1
			matrix[j][len(matrix)-i-1] = matrix[i][j]
			matrix[len(matrix)-i-1][len(matrix)-j-1] = matrix[j][len(matrix)-i-1]
			matrix[len(matrix)-j-1][i] = matrix[len(matrix)-i-1][len(matrix)-j-1]
			matrix[i][j] = matrix[len(matrix)-j-1][i]
			*/
			matrix[i][j], matrix[j][len(matrix)-i-1], matrix[len(matrix)-i-1][len(matrix)-j-1], matrix[len(matrix)-j-1][i] =
				matrix[len(matrix)-j-1][i], matrix[i][j], matrix[j][len(matrix)-i-1], matrix[len(matrix)-i-1][len(matrix)-j-1]
		}
	}
	return
}

/*
*240. 搜索二维矩阵 II
https://leetcode.cn/problems/search-a-2d-matrix-ii/?envType=study-plan-v2&envId=top-100-liked
*/
func searchMatrix(matrix [][]int, target int) bool {
	firstRow := sort.Search(len(matrix), func(i int) bool { return matrix[i][len(matrix[i])-1] >= target })
	if firstRow == len(matrix) {
		return false
	}
	lastRow := sort.Search(len(matrix), func(i int) bool { return matrix[i][0] > target }) - 1
	if lastRow < 0 {
		return false
	}
	for row := firstRow; row <= lastRow; row++ {
		col := sort.SearchInts(matrix[row], target)
		if col < len(matrix[row]) && matrix[row][col] == target {
			return true
		}
	}
	return false
}

/*
*160. 相交链表
https://leetcode.cn/problems/intersection-of-two-linked-lists/description/?envType=study-plan-v2&envId=top-100-liked
*/
type ListNode struct {
	Val  int
	Next *ListNode
}

func getIntersectionNode(headA, headB *ListNode) *ListNode {
	pa, pb := headA, headB
	for pa != pb {
		if pa == nil {
			pa = headB
		} else {
			pa = pa.Next
		}

		if pb == nil {
			pb = headA
		} else {
			pb = pb.Next
		}
	}
	return pa
}

/*
206. 反转链表
*https://leetcode.cn/problems/reverse-linked-list/?envType=study-plan-v2&envId=top-100-liked
*/
func reverseList1(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
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

func reverseList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	newHead := reverseList(head.Next)
	head.Next.Next = head
	head.Next = nil
	return newHead
}

/*
*234. 回文链表
https://leetcode.cn/problems/palindrome-linked-list/?envType=study-plan-v2&envId=top-100-liked
*/
func isPalindrome(head *ListNode) bool {
	if head == nil || head.Next == nil {
		return true
	}
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		fast = fast.Next.Next
		slow = slow.Next
	}
	if fast != nil {
		slow = slow.Next
	}
	nh, hh := reverseList(slow), head
	for nh != nil {
		if nh.Val != hh.Val {
			return false
		}
		nh = nh.Next
		hh = hh.Next
	}
	return true
}

/*
*141. 环形链表
https://leetcode.cn/problems/linked-list-cycle/?envType=study-plan-v2&envId=top-100-liked
*/
func hasCycle(head *ListNode) bool {
	if head == nil || head.Next == nil {
		return false
	}
	slow, fast := head, head.Next
	for slow != fast {
		if fast == nil || fast.Next == nil {
			return false
		}
		fast = fast.Next.Next
		slow = slow.Next
	}
	return true
}

/*
*142. 环形链表 II
https://leetcode.cn/problems/linked-list-cycle-ii/submissions/?envType=study-plan-v2&envId=top-100-liked
*/
func detectCycle(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return nil
	}
	slow, fast := head, head
	for fast != nil {
		slow = slow.Next
		if fast.Next == nil {
			return nil
		}
		fast = fast.Next.Next
		if slow == fast {
			p := head
			for p != slow {
				p = p.Next
				slow = slow.Next
			}
			return p
		}
	}
	return nil
}

/*
24. 两两交换链表中的节点
https://leetcode.cn/problems/swap-nodes-in-pairs/?envType=study-plan-v2&envId=top-100-liked
*/
func swapPairs(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	dummyHead := &ListNode{Next: head}
	tmp := dummyHead
	for tmp.Next != nil && tmp.Next.Next != nil {
		n1 := tmp.Next
		n2 := tmp.Next.Next
		tmp.Next = n2
		n1.Next = n2.Next
		n2.Next = n1
		tmp = n1
	}
	return dummyHead.Next
}

/*
25. K 个一组翻转链表
https://leetcode.cn/problems/reverse-nodes-in-k-group/description/?envType=study-plan-v2&envId=top-100-liked
*/
func reverseKGroup(head *ListNode, k int) *ListNode {
	dummyHead := &ListNode{Next: head}
	pre, cur, next, cnt := dummyHead, dummyHead.Next, dummyHead.Next, 1
	for cur != nil {
		for cnt < k && next != nil {
			next = next.Next
			cnt++
		}
		if next == nil {
			break
		}
		cnt = 1
		n := next.Next
		next.Next = nil
		tmpH := reverseList(cur)
		pre.Next = tmpH
		cur.Next = n
		if n == nil {
			break
		}
		pre = cur
		cur = n
		next = cur
	}
	return dummyHead.Next
}

/*
*
138. 随机链表的复制
https://leetcode.cn/problems/copy-list-with-random-pointer?envType=study-plan-v2&envId=top-100-liked
*/
type Node struct {
	Val    int
	Next   *Node
	Random *Node
}

func copyRandomList(head *Node) *Node {
	if head == nil {
		return head
	}
	mp := make(map[*Node]*Node)
	var dcp func(head *Node) *Node
	dcp = func(head *Node) *Node {
		if head == nil {
			return head
		}
		if node, ok := mp[head]; ok {
			return node
		}
		node := &Node{
			Val: head.Val,
		}
		mp[head] = node
		node.Next = dcp(head.Next)
		node.Random = dcp(head.Random)
		return node
	}
	return dcp(head)
}

/*
*
148. 排序链表
https://leetcode.cn/problems/sort-list/description/?envType=study-plan-v2&envId=top-100-liked
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
	return mergeList3(sortList(head), sortList(slow))
}

func mergeList3(node1 *ListNode, node2 *ListNode) *ListNode {
	head := &ListNode{}
	h := head
	for node1 != nil && node2 != nil {
		if node1.Val < node2.Val {
			n := &ListNode{Val: node1.Val}
			h.Next = n
			node1 = node1.Next
		} else {
			n := &ListNode{Val: node2.Val}
			h.Next = n
			node2 = node2.Next
		}
		h = h.Next
	}
	if node1 != nil {
		h.Next = node1
	}
	if node2 != nil {
		h.Next = node2
	}
	return head.Next
}

/*
*23. 合并 K 个升序链表
https://leetcode.cn/problems/merge-k-sorted-lists/?envType=study-plan-v2&envId=top-100-liked
*/
func mergeKLists(lists []*ListNode) *ListNode {
	if len(lists) < 1 {
		return nil
	}
	if len(lists) == 1 {
		return lists[0]
	}
	start, end := 0, len(lists)
	mid := start + (end-start)/2
	return mergeList3(mergeKLists(lists[start:mid]), mergeKLists(lists[mid:end]))
}

/*
146. LRU 缓存
https://leetcode.cn/problems/lru-cache/description/?envType=study-plan-v2&envId=top-100-liked
*/
type LRUCache struct {
	capacity, size int
	mp             map[int]*DLinkedNode
	head, tail     *DLinkedNode
}

type DLinkedNode struct {
	key, value int
	prev, next *DLinkedNode
}

func Constructor(capacity int) LRUCache {
	l := LRUCache{
		capacity: capacity,
		mp:       make(map[int]*DLinkedNode),
		head:     &DLinkedNode{},
		tail:     &DLinkedNode{},
	}
	l.head.next = l.tail
	l.tail.prev = l.head
	return l
}

func (l *LRUCache) Get(key int) int {
	if _, ok := l.mp[key]; !ok {
		return -1
	}
	node := l.mp[key]
	l.moveToHead(node)
	return node.value
}

func (l *LRUCache) Put(key int, value int) {
	if node, ok := l.mp[key]; ok {
		node.value = value
		l.moveToHead(node)
		return
	}
	node := &DLinkedNode{
		key:   key,
		value: value,
	}
	l.mp[key] = node
	l.size++
	l.addToHead(node)
	if l.size > l.capacity {
		tail := l.removeTail()
		delete(l.mp, tail.key)
		l.size--
	}
}

func (l *LRUCache) remove(node *DLinkedNode) {
	node.prev.next = node.next
	node.next.prev = node.prev
}

func (l *LRUCache) addToHead(node *DLinkedNode) {
	next := l.head.next
	l.head.next = node
	node.prev = l.head
	node.next = next
	next.prev = node
}

func (l *LRUCache) moveToHead(node *DLinkedNode) {
	l.remove(node)
	l.addToHead(node)
}

func (l *LRUCache) removeTail() *DLinkedNode {
	node := l.tail.prev
	l.remove(node)
	return node
}

/*
*94. 二叉树的中序遍历
https://leetcode.cn/problems/binary-tree-inorder-traversal/?envType=study-plan-v2&envId=top-100-liked
*/
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func inorderTraversal(root *TreeNode) []int {
	if root == nil {
		return []int{}
	}
	r, stack := make([]int, 0), make([]*TreeNode, 0)
	for root != nil || len(stack) > 0 {
		for root != nil {
			stack = append(stack, root)
			root = root.Left
		}
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		r = append(r, node.Val)
		root = node.Right
	}
	return r
}

/*
*104. 二叉树的最大深度
https://leetcode.cn/problems/maximum-depth-of-binary-tree/description/?envType=study-plan-v2&envId=top-100-liked
*/
func maxDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	if root.Left == nil && root.Right == nil {
		return 1
	}
	return max(maxDepth(root.Left), maxDepth(root.Right)) + 1
}
func max(i, j int) int {
	if i > j {
		return i
	}
	return j
}

/*
*
226. 翻转二叉树
https://leetcode.cn/problems/invert-binary-tree/?envType=study-plan-v2&envId=top-100-liked
*/
func invertTree(root *TreeNode) *TreeNode {
	if root == nil || root.Left == nil && root.Right == nil {
		return root
	}
	root.Left, root.Right = invertTree(root.Right), invertTree(root.Left)
	return root
}

/*
*
101. 对称二叉树
https://leetcode.cn/problems/symmetric-tree/?envType=study-plan-v2&envId=top-100-liked
*/
func isSymmetric(root *TreeNode) bool {
	var isSymmetricImpl func(left, right *TreeNode) bool
	isSymmetricImpl = func(left, right *TreeNode) bool {
		if left == nil && right == nil {
			return true
		}
		if left == nil && right != nil || (left != nil && right == nil) || (left.Val != right.Val) {
			return false
		}
		return isSymmetricImpl(left.Left, right.Right) && isSymmetricImpl(left.Right, right.Left)
	}
	if root == nil {
		return true
	}
	return isSymmetricImpl(root.Left, root.Right)
}

/*
543. 二叉树的直径
https://leetcode.cn/problems/diameter-of-binary-tree/?envType=study-plan-v2&envId=top-100-liked
*/
func diameterOfBinaryTree(root *TreeNode) int {
	var deep func(root *TreeNode) int
	r, mp := 1, make(map[*TreeNode]int)
	deep = func(root *TreeNode) int {
		if root == nil {
			return 0
		}
		if d, ok := mp[root]; ok {
			return d
		}
		lDeep, rDeep := deep(root.Left), deep(root.Right)
		r = max(r, lDeep+rDeep+1)
		mp[root] = max(deep(root.Left), deep(root.Right)) + 1
		return mp[root]
	}
	deep(root)
	return r - 1
}

/*
*
102. 二叉树的层序遍历
https://leetcode.cn/problems/binary-tree-level-order-traversal/?envType=study-plan-v2&envId=top-100-liked
*/
func levelOrder(root *TreeNode) [][]int {
	if root == nil {
		return [][]int{}
	}
	queue, r := make([]*TreeNode, 0), make([][]int, 0)
	queue = append(queue, root)
	for len(queue) > 0 {
		cnt, tmp := len(queue), make([]int, 0)
		for cnt > 0 {
			cnt--
			node := queue[0]
			queue = queue[1:]
			tmp = append(tmp, node.Val)
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
		if len(tmp) > 0 {
			r = append(r, tmp)
		}
	}
	return r
}

/*
108. 将有序数组转换为二叉搜索树
*https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/?envType=study-plan-v2&envId=top-100-liked
*/
func sortedArrayToBST(nums []int) *TreeNode {
	if len(nums) < 1 {
		return nil
	}
	mid := len(nums) / 2
	left := nums[:mid]
	right := nums[mid+1:]
	head := &TreeNode{Val: nums[mid], Left: sortedArrayToBST(left), Right: sortedArrayToBST(right)}
	return head
}

func isValidBST(root *TreeNode) bool {
	if root == nil || root.Left == nil && root.Right == nil {
		return true
	}
	var preNode *TreeNode
	var r bool
	var isValidBSTImpl func(root *TreeNode)
	isValidBSTImpl = func(root *TreeNode) {
		if root == nil {
			return
		}
		isValidBSTImpl(root.Left)
		if preNode != nil && preNode.Val >= root.Val {
			r = false
			return
		}
		preNode = root
		isValidBSTImpl(root.Right)
	}
	isValidBSTImpl(root)
	return r
}

/*
230. 二叉搜索树中第 K 小的元素
https://leetcode.cn/problems/kth-smallest-element-in-a-bst/?envType=study-plan-v2&envId=top-100-liked
*/
func kthSmallest(root *TreeNode, k int) int {
	r, i := 0, 0
	var dfs func(root *TreeNode)
	dfs = func(root *TreeNode) {
		if root == nil {
			return
		}
		dfs(root.Left)
		i++
		if k == i {
			r = root.Val
			return
		}
		dfs(root.Right)
		return
	}
	dfs(root)
	return r
}

/*
199. 二叉树的右视图
https://leetcode.cn/problems/binary-tree-right-side-view/?envType=study-plan-v2&envId=top-100-liked
*/
func rightSideView(root *TreeNode) []int {
	if root == nil {
		return []int{}
	}
	rightView, maxDepth, nodeStack, depthStack := make(map[int]int), 0, make([]*TreeNode, 0), make([]int, 0)
	nodeStack = append(nodeStack, root)
	depthStack = append(depthStack, 0)
	for len(nodeStack) > 0 {
		node := nodeStack[len(nodeStack)-1]
		nodeStack = nodeStack[:len(nodeStack)-1]

		depth := depthStack[len(depthStack)-1]
		depthStack = depthStack[:len(depthStack)-1]

		if node == nil {
			continue
		}

		maxDepth = max(maxDepth, depth)
		if _, ok := rightView[depth]; !ok {
			rightView[depth] = node.Val
		}
		nodeStack = append(nodeStack, node.Left)
		nodeStack = append(nodeStack, node.Right)

		depthStack = append(depthStack, depth+1)
		depthStack = append(depthStack, depth+1)
	}
	r := []int{}
	for i := 0; i <= maxDepth; i++ {
		r = append(r, rightView[i])
	}
	return r
}

/*
114. 二叉树展开为链表
*https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/?envType=study-plan-v2&envId=top-100-liked
*/
func flatten(root *TreeNode) {
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

func flatten12(root *TreeNode) {
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
}

/*
105. 从前序与中序遍历序列构造二叉树
https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/?envType=study-plan-v2&envId=top-100-liked
*/
func buildTree(preorder []int, inorder []int) *TreeNode {
	if len(preorder) == 0 || len(inorder) == 0 {
		return nil
	}
	var build func(preorder []int, inorder []int) *TreeNode
	build = func(preorder []int, inorder []int) *TreeNode {
		if len(preorder) == 0 || len(inorder) == 0 {
			return nil
		}
		rootIdx := 0
		for k, v := range inorder {
			if v == preorder[rootIdx] {
				rootIdx = k
				break
			}
		}
		preLeft, preRight := preorder[1:rootIdx+1], preorder[rootIdx+1:]
		inLeft, inRight := inorder[:rootIdx], inorder[rootIdx+1:]
		root := &TreeNode{
			Val:   inorder[rootIdx],
			Left:  build(preLeft, inLeft),
			Right: build(preRight, inRight),
		}
		return root
	}
	return build(preorder, inorder)
}

/*
https://leetcode.cn/problems/path-sum-iii/?envType=study-plan-v2&envId=top-100-liked
*/
func pathSum(root *TreeNode, targetSum int) int {
	if root == nil {
		return 0
	}
	res := rootSum(root, targetSum)
	res += pathSum(root.Left, targetSum)
	res += pathSum(root.Right, targetSum)
	return res
}

func rootSum(root *TreeNode, sum int) int {
	var r int
	if root == nil || sum < 0 {
		return r
	}
	if root.Val == sum {
		r++
	}
	r += rootSum(root.Left, sum-root.Val)
	r += rootSum(root.Right, sum-root.Val)
	return r
}

func pathSum1(root *TreeNode, targetSum int) (ans int) {
	preSum := make(map[int]int)
	var dfs func(root *TreeNode, curr int)
	dfs = func(root *TreeNode, curr int) {
		if root == nil {
			return
		}
		curr += root.Val
		ans += preSum[curr-targetSum]
		preSum[curr]++
		dfs(root.Left, curr)
		dfs(root.Right, curr)
		preSum[curr]--
	}
	dfs(root, 0)
	return
}

func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	mp, queue := make(map[*TreeNode]*TreeNode), make([]*TreeNode, 0)
	queue = append(queue, root)
	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]
		if node.Left != nil {
			queue = append(queue, node.Left)
			mp[node.Left] = node
		}
		if node.Right != nil {
			queue = append(queue, node.Right)
			mp[node.Right] = node
		}
	}
	path := make(map[*TreeNode]struct{})
	for p != nil {
		path[p] = struct{}{}
		parent, ok := mp[p]
		if !ok {
			break
		}
		p = parent
	}
	for q != nil {
		if _, ok := path[q]; ok {
			return q
		}
		parent, ok := mp[q]
		if !ok {
			break
		}
		q = parent
	}
	return root
}

/*
200. 岛屿数量
https://leetcode.cn/problems/number-of-islands/?envType=study-plan-v2&envId=top-100-liked
*/
func numIslands(grid [][]byte) int {
	if len(grid) < 1 {
		return 0
	}
	cnt := 0
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[i]); j++ {
			if grid[i][j] == '1' {
				numIslandsBfs(grid, i, j)
				cnt++
			}
		}
	}
	return cnt
}

var direction = []int{-1, 0, 1, 0, -1}

func numIslandsBfs(grid [][]byte, i int, j int) {
	if i < 0 || i >= len(grid) || j < 0 || j >= len(grid[i]) || grid[i][j] == '0' || grid[i][j] == '2' {
		return
	}
	grid[i][j] = '2'
	for k := 0; k < 4; k++ {
		numIslandsBfs(grid, i+direction[k], j+direction[k+1])
	}
	return
}

/*
994. 腐烂的橘子
https://leetcode.cn/problems/rotting-oranges/?envType=study-plan-v2&envId=top-100-liked
*/
type pair struct {
	i, j int
}

func orangesRotting(grid [][]int) int {
	if len(grid) < 1 {
		return 0
	}
	queue, goodOrangeCnt, depth := make([]pair, 0), 0, 0
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[i]); j++ {
			if grid[i][j] == 1 {
				goodOrangeCnt++
			} else if grid[i][j] == 2 {
				queue = append(queue, pair{i, j})
			}
		}
	}
	if goodOrangeCnt == 0 {
		return depth
	}
	if len(queue) < 1 {
		return -1
	}
	for len(queue) > 0 {
		n := len(queue)
		for n > 0 {
			n--
			i, j := queue[0].i, queue[0].j
			queue = queue[1:]
			for k := 0; k < 4; k++ {
				x := i + direction[k]
				y := j + direction[k+1]
				if x < 0 || x >= len(grid) || y < 0 || y >= len(grid[x]) || grid[x][y] == 0 || grid[x][y] == 2 {
					continue
				}
				grid[x][y] = 2
				goodOrangeCnt--
				queue = append(queue, pair{x, y})
			}
		}
		if len(queue) > 0 {
			depth++
		}
	}
	if goodOrangeCnt > 0 {
		return -1
	}
	return depth
}

/*
207. 课程表
https://leetcode.cn/problems/course-schedule/?envType=study-plan-v2&envId=top-100-liked
*/
func canFinish(numCourses int, prerequisites [][]int) bool {
	var (
		edges   = make([][]int, numCourses)
		visited = make([]int, numCourses)
		result  []int
		valid   = true
		dfs     func(u int)
	)
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
	for _, info := range prerequisites {
		edges[info[1]] = append(edges[info[1]], info[0])
	}
	for i := 0; i < numCourses && valid; i++ {
		if visited[i] == 0 {
			dfs(i)
		}
	}
	return valid
}

func canFinishBfs(numCourses int, prerequisites [][]int) bool {
	var (
		edges  = make([][]int, numCourses)
		indeg  = make([]int, numCourses)
		result []int
	)
	for _, info := range prerequisites {
		edges[info[1]] = append(edges[info[1]], info[0])
		indeg[info[0]]++
	}
	queue := make([]int, 0)
	for k, _ := range edges {
		if indeg[k] == 0 {
			queue = append(queue, k)
		}
	}
	for len(queue) > 0 {
		u := queue[0]
		queue = queue[1:]
		result = append(result, u)
		for _, v := range edges[u] {
			indeg[v]--
			if indeg[v] == 0 {
				queue = append(queue, v)
			}
		}
	}
	return len(result) == numCourses
}

/*
208. 实现 Trie (前缀树)
https://leetcode.cn/problems/implement-trie-prefix-tree/description/?envType=study-plan-v2&envId=top-100-liked
*/
type Trie struct {
	data  [26]*Trie
	isEnd bool
}

func Constructor1() Trie {
	return Trie{}
}

func (t *Trie) Insert(word string) {
	tmp := t
	for _, b := range word {
		if tmp.data[b-'a'] == nil {
			tt := Constructor1()
			tmp.data[b-'a'] = &tt
		}
		tmp = tmp.data[b-'a']
	}
	tmp.isEnd = true
	return
}

func (t *Trie) SearchPrefix(word string) *Trie {
	tmp := t
	for _, b := range word {
		if tmp.data[b-'a'] == nil {
			return nil
		}
		tmp = tmp.data[b-'a']
	}
	return tmp
}

func (t *Trie) Search(word string) bool {
	trie := t.SearchPrefix(word)
	return trie != nil && trie.isEnd
}

func (t *Trie) StartsWith(prefix string) bool {
	return t.SearchPrefix(prefix) != nil
}

/*
124. 二叉树中的最大路径和
https://leetcode.cn/problems/binary-tree-maximum-path-sum/?envType=study-plan-v2&envId=top-100-liked
*/
func maxPathSum(root *TreeNode) int {
	maxSum := math.MinInt32
	var maxGain func(root *TreeNode) int
	maxGain = func(root *TreeNode) int {
		if root == nil {
			return 0
		}
		leftSum, rightSum := max(0, maxGain(root.Left)), max(0, maxGain(root.Right))
		maxSum = max(root.Val+leftSum+rightSum, maxSum)
		return root.Val + max(leftSum, rightSum)
	}
	maxGain(root)
	return maxSum
}

/*
*
221. 最大正方形
https://leetcode.cn/problems/maximal-square/?envType=study-plan-v2&envId=top-interview-150
*/
func maximalSquare(matrix [][]byte) int {
	dp, maxSlide := make([][]int, len(matrix)), 0
	for i := 0; i < len(matrix); i++ {
		dp[i] = make([]int, len(matrix[i]))
		for j := 0; j < len(matrix[i]); j++ {
			if matrix[i][j] == '1' {
				dp[i][j] = 1
				maxSlide = 1
			}
		}
	}
	for i := 1; i < len(dp); i++ {
		for j := 1; j < len(dp[i]); j++ {
			if dp[i][j] == 1 {
				dp[i][j] = minVal(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
				if dp[i][j] > maxSlide {
					maxSlide = dp[i][j]
				}
			}
		}
	}
	return maxSlide * maxSlide
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

/*
46. 全排列
https://leetcode.cn/problems/permutations/?envType=study-plan-v2&envId=top-100-liked
*/
func permute(nums []int) [][]int {
	if len(nums) < 1 {
		return [][]int{}
	}
	r, mp, data := make([][]int, 0), make(map[int]struct{}), make([]int, 0)
	permuteImpl1(nums, &data, mp, &r)
	return r
}

func permuteImpl1(nums []int, data *[]int, mp map[int]struct{}, r *[][]int) {
	if len(*data) == len(nums) {
		tmp := make([]int, 0)
		tmp = append(tmp, *data...)
		*r = append(*r, tmp)
		return
	}
	for i := 0; i < len(nums); i++ {
		if _, ok := mp[nums[i]]; ok {
			continue
		}
		mp[nums[i]] = struct{}{}
		*data = append(*data, nums[i])
		permuteImpl1(nums, data, mp, r)
		delete(mp, nums[i])
		*data = (*data)[:len(*data)-1]
	}
	return
}

/*
78. 子集
https://leetcode.cn/problems/subsets/description/?envType=study-plan-v2&envId=top-100-liked
*/
func subsets(nums []int) [][]int {
	if len(nums) < 1 {
		return [][]int{}
	}
	r, data := make([][]int, 0), make([]int, 0)
	subsetsImpl(nums, &data, 0, &r)
	return r
}

func subsetsImpl(nums []int, data *[]int, level int, r *[][]int) {
	tmp := make([]int, 0)
	tmp = append(tmp, *data...)
	*r = append(*r, tmp)

	for i := level; i < len(nums); i++ {
		*data = append(*data, nums[i])
		subsetsImpl(nums, data, level+1, r)
		*data = (*data)[:len(*data)-1]
	}
	return
}

/*
39. 组合总和
https://leetcode.cn/problems/combination-sum/description/
题解：https://leetcode.cn/problems/subsets/solutions/7812/hui-su-python-dai-ma-by-liweiwei1419/
*/
func combinationSum(candidates []int, target int) [][]int {
	if len(candidates) < 1 {
		return [][]int{}
	}
	r, data, level := make([][]int, 0), make([]int, 0), 0
	combinationSumBT(candidates, &data, target, level, &r)
	return r
}

func combinationSumBT(candidates []int, data *[]int, target, level int, r *[][]int) {
	if sumList(*data) == target {
		tmp := make([]int, 0)
		tmp = append(tmp, *data...)
		*r = append(*r, tmp)
		return
	} else if sumList(*data) > target {
		return
	}
	for i := level; i < len(candidates); i++ {
		*data = append(*data, candidates[i])
		combinationSumBT(candidates, data, target, i, r)
		*data = (*data)[:len(*data)-1]
	}
	return
}

func sumList(list []int) int {
	var r int
	for _, v := range list {
		r += v
	}
	return r
}

/*
*40. 组合总和 IIg
https://leetcode.cn/problems/combination-sum-ii/description/
题解：https://leetcode.cn/problems/combination-sum-ii/solutions/14753/hui-su-suan-fa-jian-zhi-python-dai-ma-java-dai-m-3/
*/
func combinationSum2(candidates []int, target int) [][]int {
	if len(candidates) < 1 {
		return [][]int{}
	}
	sort.Ints(candidates)
	r, data, level := make([][]int, 0), make([]int, 0), 0
	combinationSum2BT(candidates, data, target, level, &r)
	return r
}

func combinationSum2BT(candidates []int, data []int, target int, level int, r *[][]int) {
	if sumList(data) == target {
		tmp := make([]int, 0)
		tmp = append(tmp, data...)
		*r = append(*r, tmp)
		return
	} else if sumList(data) > target {
		return
	}
	/*
		这个方法最重要的作用是，可以让同一层级，不出现相同的元素。即
		                  1
		                 / \
		                2   2  这种情况不会发生 但是却允许了不同层级之间的重复即：
		               /     \
		              5       5
		                例2
		                  1
		                 /
		                2      这种情况确是允许的
		               /
		              2

		为何会有这种神奇的效果呢？
		首先 cur-1 == cur 是用于判定当前元素是否和之前元素相同的语句。这个语句就能砍掉例1。
		可是问题来了，如果把所有当前与之前一个元素相同的都砍掉，那么例二的情况也会消失。
		因为当第二个2出现的时候，他就和前一个2相同了。

		那么如何保留例2呢？
		那么就用cur > begin 来避免这种情况，你发现例1中的两个2是处在同一个层级上的，
		例2的两个2是处在不同层级上的。
		***在一个for循环中，所有被遍历到的数都是属于一个层级的***。我们要让一个层级中，
		必须出现且只出现一个2，那么就放过第一个出现重复的2，但不放过后面出现的2。
		第一个出现的2的特点就是 cur == begin. 第二个出现的2 特点是cur > begin.
	*/
	for i := level; i < len(candidates); i++ {
		if i > level && candidates[i-1] == candidates[i] {
			continue
		}
		data = append(data, candidates[i])
		combinationSum2BT(candidates, data, target, i+1, r)
		data = data[:len(data)-1]
	}
	/*
		所谓的同一个for循环是指  << for i := level; i < len(candidates); i++ >> 这个循环里，这个循环会不断地append、然后再pop，总是在尝试同一层的数据
		例如：[1,2,2,2,5] 中，i=0 代表我们第一层的选择（例如选择1）；此时 i++ 代表我们第一层尝试其他选择（例如选择2）
	*/
	return
}

/*
90. 子集 II
https://leetcode.cn/problems/subsets-ii/
*/
func subsetsWithDup(nums []int) [][]int {
	if len(nums) < 1 {
		return [][]int{}
	}
	sort.Ints(nums)
	r, data, level := make([][]int, 0), make([]int, 0), 0
	subsetsWithDupBT1(nums, data, level, &r)
	return r
}

func subsetsWithDupBT1(nums []int, data []int, level int, r *[][]int) {
	tmp := make([]int, 0)
	tmp = append(tmp, data...)
	*r = append(*r, tmp)

	for i := level; i < len(nums); i++ {
		if i > level && nums[i] == nums[i-1] {
			continue
		}
		data = append(data, nums[i])
		subsetsWithDupBT1(nums, data, i+1, r)
		data = data[:len(data)-1]
	}
	return
}

/*
47. 全排列 II
https://leetcode.cn/problems/permutations-ii/description/
*/
func permuteUnique(nums []int) [][]int {
	if len(nums) < 1 {
		return [][]int{}
	}
	sort.Ints(nums)
	r, check, data := make([][]int, 0), make(map[int]bool), make([]int, 0)
	permuteUniqueBT(nums, data, check, &r)
	return r
}

func permuteUniqueBT(nums []int, data []int, check map[int]bool, r *[][]int) {
	if len(data) == len(nums) {
		tmp := make([]int, 0)
		tmp = append(tmp, data...)
		*r = append(*r, tmp)
		return
	}
	for i := 0; i < len(nums); i++ {
		if check[i] {
			continue
		}
		if i > 0 && nums[i] == nums[i-1] && !check[i-1] {
			continue
		}
		check[i] = true
		data = append(data, nums[i])
		permuteUniqueBT(nums, data, check, r)
		data = data[:len(data)-1]
		delete(check, i)
	}
	return
}

/*
17. 电话号码的字母组合
https://leetcode.cn/problems/letter-combinations-of-a-phone-number
*/
var table = []string{"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"}

func letterCombinations(digits string) []string {
	if len(digits) < 1 {
		return []string{}
	}
	r, idx, str := make([]string, 0), 0, ""
	letterCombinationsBT(digits, idx, str, &r)
	return r
}

func letterCombinationsBT(l string, idx int, data string, r *[]string) {
	if len(data) == len(l) {
		*r = append(*r, data)
		return
	}
	str := table[l[idx]-'0']
	for i := 0; i < len(str); i++ {
		data = fmt.Sprintf("%s%s", data, string(str[i]))
		letterCombinationsBT(l, idx+1, data, r)
		data = data[:len(data)-1]
	}
	return
}

/*
79. 单词搜索
https://leetcode.cn/problems/word-search/description/?envType=study-plan-v2&envId=top-100-liked
*/
func exist(board [][]byte, word string) bool {
	if len(board) < 1 {
		return false
	}

	r, selected := false, make(map[string]bool)
	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[i]); j++ {
			if board[i][j] == word[0] {
				existBFS(board, word, i, j, 0, selected, &r)
				if r {
					return r
				}
			}
		}
	}
	return r
}

func existBFS(board [][]byte, word string, i, j, idx int, selected map[string]bool, r *bool) {

	key := string(i) + string(j)
	if i < 0 || i >= len(board) || j < 0 || j >= len(board[i]) || selected[key] || board[i][j] != word[idx] {
		return
	}
	if idx == len(word)-1 {
		*r = true
		return
	}
	selected[key] = true
	direction := []int{-1, 0, 1, 0, -1}
	for k := 0; k < 4; k++ {
		x := i + direction[k]
		y := j + direction[k+1]
		existBFS(board, word, x, y, idx+1, selected, r)
	}
	delete(selected, key)
	return
}

/*
22. 括号生成
https://leetcode.cn/problems/generate-parentheses/?envType=study-plan-v2&envId=top-100-liked
*/
func generateParenthesis(n int) []string {
	if n < 1 {
		return []string{}
	}
	left, right, data, r := n, n, "", make([]string, 0)
	generateParenthesisImpl(n, left, right, data, &r)
	return r
}

func generateParenthesisImpl(n int, left int, right int, s string, r *[]string) {
	if left == right && left == 0 {
		*r = append(*r, s)
		return
	}
	if left > 0 {
		generateParenthesisImpl(n, left-1, right, fmt.Sprintf("%s%s", s, "("), r)
	}
	if right > left {
		generateParenthesisImpl(n, left, right-1, fmt.Sprintf("%s%s", s, ")"), r)
	}
}

/*
131. 分割回文串
题解：https://leetcode.cn/problems/palindrome-partitioning/solutions/2059414/hui-su-bu-hui-xie-tao-lu-zai-ci-pythonja-fues/?envType=study-plan-v2&envId=top-100-liked
https://leetcode.cn/problems/palindrome-partitioning/?envType=study-plan-v2&envId=top-100-liked
*/
func partition(s string) [][]string {
	if len(s) < 1 {
		return [][]string{}
	}
	r, path := make([][]string, 0), []string{}
	var dfs func(i int)
	dfs = func(i int) {
		if i == len(s) {
			r = append(r, append([]string{}, path...)) // 复制 path
			return
		}
		for j := i; j < len(s); j++ { // 枚举子串的结束位置
			if isPalindrome12(s, i, j) {
				path = append(path, s[i:j+1])
				dfs(j + 1)
				path = path[:len(path)-1] // 恢复现场
			}
		}
	}
	dfs(0)
	return r
}

func isPalindrome12(s string, i int, j int) bool {
	for i < j {
		if s[i] != s[j] {
			return false
		}
		i++
		j--
	}
	return true
}

func canFinish1(numCourses int, prerequisites [][]int) bool {
	var (
		edges, indeg, r, q = make([][]int, numCourses), make([]int, numCourses), make([]int, 0), make([]int, 0)
	)
	for _, list := range prerequisites {
		edges[list[1]] = append(edges[list[1]], list[0])
		indeg[list[0]]++
	}
	for k, v := range indeg {
		if v == 0 {
			q = append(q, k)
		}
	}
	for len(q) > 0 {
		n := q[0]
		q = q[1:]
		r = append(r, n)
		for _, v := range edges[n] {
			indeg[v]--
			if indeg[v] == 0 {
				q = append(q, v)
			}
		}
	}
	return len(r) == numCourses
}

func canFinishDFS(numCourses int, prerequisites [][]int) bool {
	var (
		edges, visited, r, valid = make([][]int, numCourses), make([]int, numCourses), make([]int, 0), true
	)
	var dfs func(i int)
	dfs = func(i int) {
		visited[i] = 1
		for _, v := range edges[i] {
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
		visited[i] = 2
		r = append(r, i)
	}
	for _, list := range prerequisites {
		edges[list[1]] = append(edges[list[1]], list[0])
	}
	for i := 0; i < numCourses && valid; i++ {
		if visited[i] == 0 {
			dfs(i)
		}
	}
	return valid
}

/*
35. 搜索插入位置
https://leetcode.cn/problems/search-insert-position/?envType=study-plan-v2&envId=top-100-liked
题解：https://leetcode.cn/problems/search-insert-position/solutions/8017/hua-jie-suan-fa-35-sou-suo-cha-ru-wei-zhi-by-guanp/?envType=study-plan-v2&envId=top-100-liked
*/
func searchInsert(nums []int, target int) int {
	if len(nums) < 1 {
		return 0
	}
	start, end := 0, len(nums)-1
	for start < end {
		mid := (end-start)/2 + start
		if nums[mid] == target {
			return mid
		} else if nums[mid] < target {
			start = mid + 1
		} else {
			end = mid - 1
		}
	}
	return start + 1
}

func searchMatrix1(matrix [][]int, target int) bool {
	if len(matrix) < 1 {
		return false
	}
	row := sort.Search(len(matrix), func(i int) bool { // 根据表达式求值，如果找不到返回数据len
		return matrix[i][0] > target
	}) - 1
	if row < 0 {
		return false
	}
	col := sort.SearchInts(matrix[row], target) // 寻找int值所在的位置，如果找不到返回应该插入的位置
	if col < len(matrix[row]) && matrix[row][col] == target {
		return true
	}
	return false
}

/*
215. 数组中的第K个最大元素
https://leetcode.cn/problems/kth-largest-element-in-an-array/description/
*/
func findKthLargest(nums []int, k int) int {
	if len(nums) < 1 {
		return -1
	}
	n, start, end := len(nums)-k, 0, len(nums)-1
	for {
		p := getPartitionPos(nums, start, end)
		if p == n {
			return nums[p]
		} else if p < n {
			start = p + 1
		} else {
			end = p - 1
		}
	}
	return nums[start]
}

func getPartitionPos(nums []int, start, end int) int {
	val := nums[start]
	for start < end {
		for start < end && val <= nums[end] {
			end--
		}
		nums[start] = nums[end]
		for start < end && val >= nums[start] {
			start++
		}
		nums[end] = nums[start]
	}
	nums[start] = val
	return start
}

/*
236. 二叉树的最近公共祖先
https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/description/
*/
func lowestCommonAncestor3(root, p, q *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	mp := make(map[*TreeNode]*TreeNode)
	var dfs func(parentNode, node *TreeNode)
	dfs = func(parentNode, node *TreeNode) {
		mp[node] = parentNode
		if node.Left != nil {
			dfs(node, node.Left)
		}
		if node.Right != nil {
			dfs(node, node.Right)
		}
	}
	var parentNode *TreeNode
	dfs(parentNode, root)
	pPath := make(map[*TreeNode]struct{})
	for p != nil {
		pPath[p] = struct{}{}
		p = mp[p]
	}
	for q != nil {
		if _, ok := pPath[q]; ok {
			return q
		}
		q = mp[q]
	}
	return root
}

func substring1(num1 string, num2 string) string {
	if isLess(num1, num2) {
		r := sub(num2, num1)
		return "-" + r
	}
	return sub(num1, num2)
}

func sub(num1 string, num2 string) string {
	r, borrow, i, j := "", 0, len(num1)-1, len(num2)-1
	for i >= 0 || j >= 0 {
		x, y := 0, 0
		if i >= 0 {
			x = int(num1[i] - '0')
		}
		if j >= 0 {
			y = int(num2[j] - '0')
		}
		diff := 0
		if x-y-borrow < 0 {
			diff = x - y - borrow + 10
			borrow = 1
		} else {
			diff = x - y - borrow
			borrow = 0
		}
		r = fmt.Sprintf("%d%s", diff, r)
		i--
		j--
	}
	idx := 0
	for ; idx < len(r)-1; idx++ {
		if r[idx] != '0' {
			break
		}
	}
	return r[idx:]
}

func isLess(num1 string, num2 string) bool {
	if len(num1) == len(num2) {
		return num1 < num2
	}
	return len(num1) < len(num2)
}

/*
20. 有效的括号
https://leetcode.cn/problems/valid-parentheses/?envType=study-plan-v2&envId=top-100-liked
*/
func isValid(s string) bool {
	if len(s) < 1 {
		return true
	}
	stack := make([]byte, 0)
	for i := 0; i < len(s); i++ {
		if s[i] == '(' || s[i] == '[' || s[i] == '{' {
			stack = append(stack, s[i])
		}
		if s[i] == ')' || s[i] == ']' || s[i] == '}' {
			if len(stack) == 0 {
				return false
			}
			peek := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if s[i] == ')' && peek == '(' || (s[i] == ']' && peek == '[') || (s[i] == '}' && peek == '{') {
				continue
			} else {
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
155. 最小栈
https://leetcode.cn/problems/min-stack/?envType=study-plan-v2&envId=top-100-liked
*/
type MinStack struct {
	data    []int
	minData []int
}

func ConstructorMinStack() MinStack {
	return MinStack{
		data:    make([]int, 0),
		minData: make([]int, 0),
	}
}

func (m *MinStack) Push(val int) {
	m.data = append(m.data, val)
	if len(m.minData) == 0 {
		m.minData = append(m.minData, val)
	} else {
		if m.minData[len(m.minData)-1] > val {
			m.minData = append(m.minData, val)
		} else {
			m.minData = append(m.minData, m.minData[len(m.minData)-1])
		}
	}
}

func (m *MinStack) Pop() {
	if len(m.data) == 0 {
		return
	}
	m.data = m.data[:len(m.data)-1]
	m.minData = m.minData[:len(m.minData)-1]
}

func (m *MinStack) Top() int {
	if len(m.data) == 0 {
		return 0
	}
	return m.data[len(m.data)-1]
}

func (m *MinStack) GetMin() int {
	if len(m.minData) == 0 {
		return 0
	}
	return m.minData[len(m.minData)-1]
}

/*
394. 字符串解码
https://leetcode.cn/problems/decode-string/?envType=study-plan-v2&envId=top-100-liked
*/
func decodeString(s string) string {
	stack, idx := make([]string, 0), 0
	for idx < len(s) {
		if s[idx] >= '0' && s[idx] <= '9' {
			numStr := ""
			for ; s[idx] >= '0' && s[idx] <= '9'; idx++ {
				numStr += string(s[idx])
			}
			stack = append(stack, numStr)
		} else if s[idx] >= 'a' && s[idx] <= 'z' || (s[idx] >= 'A' && s[idx] <= 'Z') || s[idx] == '[' {
			stack = append(stack, string(s[idx]))
			idx++
		} else {
			idx++
			strList := make([]string, 0)
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

func getString(v []string) string {
	ret := ""
	for _, s := range v {
		ret += s
	}
	return ret
}

/*
739. 每日温度
https://leetcode.cn/problems/daily-temperatures/?envType=study-plan-v2&envId=top-100-liked
*/
func dailyTemperatures(temperatures []int) []int {
	ans, stack := make([]int, len(temperatures)), make([]int, 0)
	for i := 0; i < len(temperatures); i++ {
		if len(stack) == 0 {
			stack = append(stack, i)
			continue
		}
		for len(stack) > 0 && temperatures[stack[len(stack)-1]] < temperatures[i] {
			ans[stack[len(stack)-1]] = i - stack[len(stack)-1]
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, i)
	}
	return ans
}

/**
 * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
 * @param num1 string字符串
 *GoInstallBinaries @param num2 string字符串
 * @return string字符串
 */
func substring(num1 string, num2 string) string {
	var (
		flag = true
	)
	/*
	   判断结果符号
	*/
	if len(num1) < len(num2) {
		flag = false
	} else if len(num1) == len(num2) {
		for i := 0; i < len(num1); i++ {
			if num1[i] > num2[i] {
				break
			} else if num1[i] == num2[i] {
				continue
			} else {
				flag = false
			}
		}
	}
	if !flag {
		num1, num2 = num2, num1
	}
	r, jw, jwVal := "", math.MaxInt32, 0
	for i := 0; i < len(num1); i++ {
		n1, n2 := 0, 0
		if jw < len(num1)-i-1 {
			jwVal = 9
		} else if jw == len(num1)-i-1 {
			jwVal = -1
		}
		if len(num1)-i-1 >= 0 {
			n1 = int(num1[len(num1)-i-1]-'0') + jwVal
		}
		if len(num2)-i-1 >= 0 {
			n2 = int(num2[len(num2)-i-1] - '0')
		}
		c := 0
		if n1 < n2 {
			for j := i + 1; j < len(num1); j++ {
				if num1[len(num1)-j-1] != '0' {
					jw = len(num1) - j - 1
					break
				}
			}
			c = n1 + 10 - n2
		} else {
			c = n1 - n2
		}
		r = fmt.Sprintf("%d%s", c, r)
	}
	r = strings.TrimLeft(r, "0")
	if !flag {
		r = fmt.Sprintf("%s%s", "-", r)
	}
	return r
}

/*
34. 在排序数组中查找元素的第一个和最后一个位置
https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/?envType=study-plan-v2&envId=top-100-liked
*/
func searchRange(nums []int, target int) []int {
	left := sort.SearchInts(nums, target)
	if left == len(nums) || nums[left] != target {
		return []int{-1, -1}
	}
	right := sort.SearchInts(nums, target+1)
	return []int{left, right - 1}
}

/*
33. 搜索旋转排序数组
https://leetcode.cn/problems/search-in-rotated-sorted-array/description/?envType=study-plan-v2&envId=top-100-liked
*/
func search1(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := left + (right-left)/2
		if nums[mid] == target {
			return mid
		} else if nums[mid] > nums[left] {
			if target >= nums[left] && target < nums[mid] {
				right = mid - 1
			} else {
				left = mid + 1
			}
		} else {
			if target > nums[mid] && target <= nums[right] {
				left = mid + 1
			} else {
				right = mid - 1
			}
		}
	}
	return -1
}

/*
4. 寻找两个正序数组的中位数
https://leetcode.cn/problems/median-of-two-sorted-arrays/?envType=study-plan-v2&envId=top-100-liked
*/
func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	len1, len2 := len(nums1), len(nums2)

	if (len1+len2)%2 == 0 {
		return float64((getKLargestVal(nums1, nums2, (len1+len2)/2) + getKLargestVal(nums1, nums2, (len1+len2)/2+1))) / 2.0
	}
	return float64(getKLargestVal(nums1, nums2, (len1+len2)/2+1))
}

func getKLargestVal(nums1 []int, nums2 []int, k int) int {
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

func findKthLargest123(nums []int, k int) int {
	if len(nums) < 1 {
		return -1
	}
	n, start, end := len(nums)-k, 0, len(nums)-1
	for {
		idx := getPartitionPos123(nums, start, end)
		if idx == n {
			return nums[idx]
		} else if idx < n {
			start = idx + 1
		} else {
			end = idx - 1
		}
	}
	return nums[start]
}

func getPartitionPos123(nums []int, start int, end int) int {
	val := nums[start]
	for start < end {
		for start < end && nums[end] >= val {
			end--
		}
		nums[start] = nums[end]
		for start < end && nums[end] >= val {
			start++
		}
		nums[end] = nums[start]
	}
	nums[start] = val
	return start
}

func topKFrequent(nums []int, k int) []int {
	hp, mp := NewHp(), make(map[int]int)
	for i := 0; i < len(nums); i++ {
		mp[nums[i]] += 1
	}
	for kk, vv := range mp {
		heap.Push(hp, kv{key: kk, val: vv})
		if hp.Len() > k {
			heap.Pop(hp)
		}
		//hp.Push(nums[i])
	}
	r := make([]int, k)
	for i := 0; i < k; i++ {
		r[k-i-1] = heap.Pop(hp).(int)
	}
	return r
}

type kv struct {
	key, val int
}

type hp struct {
	data []kv
}

func NewHp() *hp {
	return &hp{
		data: make([]kv, 0),
	}
}

func (h *hp) Less(i, j int) bool {
	return h.data[i].val < h.data[j].val
}

func (h *hp) Swap(i, j int) {
	h.data[i], h.data[j] = h.data[j], h.data[i]
}

func (h *hp) Len() int {
	return len(h.data)
}

func (h *hp) Push(x any) {
	h.data = append(h.data, x.(kv))
}

func (h *hp) Pop() any {
	d := h.data[len(h.data)-1]
	defer func() {
		h.data = h.data[:len(h.data)-1]
	}()
	return d.key
}

/*
121. 买卖股票的最佳时机
https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/?envType=study-plan-v2&envId=top-100-liked
*/
func maxProfit(prices []int) int {
	if len(prices) < 1 {
		return 0
	}
	buyIndex, r := 0, math.MinInt32
	for i := 1; i < len(prices); i++ {
		if prices[i] < prices[buyIndex] {
			buyIndex = i
			continue
		}
		if prices[i]-prices[buyIndex] > r {
			r = prices[i] - prices[buyIndex]
		}
	}
	return r
}

/*
55. 跳跃游戏
https://leetcode.cn/problems/jump-game/?envType=study-plan-v2&envId=top-100-liked
*/
func canJump(nums []int) bool {
	i, dis, maxDis := 0, nums[0], nums[0]
	for i <= dis {
		for ; i <= dis; i++ {
			if i+nums[i] > maxDis {
				maxDis = i + nums[i]
			}
			if maxDis >= len(nums)-1 {
				return true
			}
		}
		dis = maxDis
	}
	return false
}

/*
45. 跳跃游戏 II
https://leetcode.cn/problems/jump-game-ii/?envType=study-plan-v2&envId=top-100-liked
*/
func jump(nums []int) int {
	if len(nums) <= 1 {
		return 0
	}
	r, i, dis, maxDis := 1, 0, nums[0], nums[0]
	for dis < len(nums)-1 {
		r++
		for ; i <= dis; i++ {
			if i+nums[i] > maxDis {
				maxDis = i + nums[i]
			}
		}
		dis = maxDis
	}
	return r
}

/*
763. 划分字母区间
https://leetcode.cn/problems/partition-labels/?envType=study-plan-v2&envId=top-100-liked
*/
func partitionLabels(s string) []int {
	if len(s) < 1 {
		return []int{}
	}
	pos, start, end, r := [26]int{}, 0, 0, make([]int, 0)
	for i := 0; i < len(s); i++ {
		pos[s[i]-'a'] = i
	}
	for i := 0; i < len(s); i++ {
		if pos[s[i]-'a'] > end {
			end = pos[s[i]-'a']
		}
		if i == end {
			r = append(r, end-start+1)
			start, end = i+1, i+1
		}
	}
	return r
}

/*
70. 爬楼梯
https://leetcode.cn/problems/climbing-stairs/?envType=study-plan-v2&envId=top-100-liked
*/
func climbStairs(n int) int {
	dp := make([]int, n+1)
	dp[0] = 0
	dp[1] = 1
	dp[2] = 2
	for i := 3; i <= n; i++ {
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[n]
}

/*
118. 杨辉三角
https://leetcode.cn/problems/pascals-triangle/description/?envType=study-plan-v2&envId=top-100-liked
*/
func generate(numRows int) [][]int {
	if numRows == 0 {
		return [][]int{}
	}
	dp := make([][]int, numRows)
	for i := 0; i < numRows; i++ {
		dp[i] = make([]int, i+1)
		dp[i][0], dp[i][i] = 1, 1
		for j := 1; j < i; j++ {
			dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
		}
	}
	return dp
}

/*
198. 打家劫舍
https://leetcode.cn/problems/house-robber/?envType=study-plan-v2&envId=top-100-liked
*/
func rob(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	if len(nums) == 1 {
		return nums[0]
	}
	dp := make([]int, len(nums))
	dp[0], dp[1] = nums[0], max(nums[0], nums[1])
	for i := 2; i < len(nums); i++ {
		dp[i] = max(dp[i-1], dp[i-2]+nums[i])
	}
	return dp[len(nums)-1]
}

/*
279. 完全平方数
https://leetcode.cn/problems/perfect-squares/?envType=study-plan-v2&envId=top-100-liked
*/
func numSquares(n int) int {
	/*
		总结动态规划方程得出的思路
		找到最小子问题，涉及到当前数和上一个数的跨度，以及上一个数的结果如何变成当前数的结果这两个点。
		1，当前数n和上一个数的跨度：
		假设n=12， 上一个数可以是11，11 + 1 = 12，OK；
		上一个数可以是8， 因为8 + 4 = 12；
		上一个数可以是3， 因为3 + 9 = 12；
		为什么11、8、3可以？因为题目要求是完全平方数相加。只有11加上1（11）， 8+4（22），3 + 9（3*3）才满足要求。

		  反过来说， 上一个数=当前数 - 某个数的平方，即：
							上一个数可以是12 - 1*1 = 11；
							上一个数可以是12 - 2*2 = 8；
							上一个数可以是12 - 3*3 = 3；

		 总结来说，如果把“某个数的平方”用j*j代表，那么相邻两个数的跨度就是n、n - j * j
		2，上一个数n - j * j 所对应的完全平方数最少数量 f[n - j * j]， 与当前数n对应的最少数量f[n]就是+1的关系：f[n] = f[n - j * j] + 1。即上一个数最少要再加一次完全平方数，才能等于n；
		另外， 11 + 1 = 12、8 + 4 = 12、3 + 9 = 12有三种方式都可以得到n=12， 要选哪个呢？选最少的那个。f[n] = Math.min(f[n - j * j] + 1, f[n]);
	*/
	dp := make([]int, n+1)
	for i := 0; i < len(dp); i++ {
		dp[i] = math.MaxInt16 / 2
	}
	dp[0] = 0
	for i := 1; i <= n; i++ {
		for j := 1; j*j <= i; j++ {
			dp[i] = minVal(dp[i], dp[i-j*j]+1)
		}
	}
	return dp[n]
}

/*
322. 零钱兑换
https://leetcode.cn/problems/coin-change/?envType=study-plan-v2&envId=top-100-liked
*/
func coinChange(coins []int, amount int) int {
	/*
		dp[i] i 元钱需要几个硬币
		dp[i] = min(dp[i-coins[0...i]] + 1)
	*/
	dp := make([]int, amount+1)
	for i := 0; i < len(dp); i++ {
		dp[i] = amount + 1
	}
	dp[0] = 0
	for i := 1; i <= amount; i++ {
		for j := 0; j < len(coins); j++ {
			if coins[j] > i {
				continue
			}
			dp[i] = min(dp[i], dp[i-coins[j]]+1)
		}
	}
	if dp[amount] > amount {
		return -1
	}
	return dp[amount]
}

/*
139. 单词拆分
https://leetcode.cn/problems/word-break/description/?envType=study-plan-v2&envId=top-100-liked
*/
func wordBreak(s string, wordDict []string) bool {
	mp := make(map[string]bool)
	for _, word := range wordDict {
		mp[word] = true
	}
	/*
		dp[i] 代表前i个字符是否可以用wordDict里面的表示
		dp[i] = dp[j] && mp[s[j:i]]  // j < i
	*/
	dp := make([]bool, len(s)+1)
	dp[0] = true
	for i := 0; i <= len(s); i++ {
		for j := 0; j < i; j++ {
			if dp[j] && mp[s[j:i]] {
				// dp[i] = dp[0:0~j] && mp[s[j+1:i+1]]
				dp[i] = true
				break
			}
		}
	}
	return dp[len(s)]
}

/*
300. 最长递增子序列
https://leetcode.cn/problems/longest-increasing-subsequence/?envType=study-plan-v2&envId=top-100-liked
*/
func lengthOfLIS(nums []int) int {
	/*
		dp[i] 代表以第i个数结尾的最大递增子序列
		dp[i] = max(dp[j...i] + 1)
	*/
	dp := make([]int, len(nums)+1)
	for i := 0; i < len(dp); i++ {
		dp[i] = 1
	}
	for i := 0; i < len(nums); i++ {
		for j := 0; j < i; j++ {
			if nums[i] > nums[j] {
				dp[i] = max(dp[j]+1, dp[i])
			}
		}
	}
	r := 0
	for _, v := range dp {
		r = max(r, v)
	}
	return r
}

/*
718. 最长重复子数组
https://leetcode.cn/problems/maximum-length-of-repeated-subarray/description/
*/
func findLength(nums1 []int, nums2 []int) int {
	/*
		dp[i][j] 代表 nums1[i:] 与 nums2[j:] 的最长公共前缀
		if nums1[i] == nums2[j] {
			dp[i][j] = dp[i+1][j+1] + 1
		} else {
			dp[i][j] = 0
		}
	*/
	n, m := len(nums1), len(nums2)
	dp := make([][]int, n+1)
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, m+1)
	}
	ans := 0
	for i := n - 1; i >= 0; i-- {
		for j := m - 1; j >= 0; j-- {
			if nums1[i] == nums2[j] {
				dp[i][j] = dp[i+1][j+1] + 1
			} else {
				dp[i][j] = 0
			}
			if ans < dp[i][j] {
				ans = dp[i][j]
			}
		}
	}
	return ans
}

/*
152. 乘积最大子数组
https://leetcode.cn/problems/maximum-product-subarray/?envType=study-plan-v2&envId=top-100-liked
*/
func maxProduct(nums []int) int {
	/*
		dp[i] 代表以i为结尾的最大非空连续乘积；这里面需要考虑正负号的情况，因为如果当前数字为正数，则前面的乘机越大越好，如果当前数字为负数，则前面的乘积越小越好
		dpMax[i] = max(dpMax[i-1]*num[i], max(num[i], dpMin[i-1]*num[i]))
		dpMin[i] = min(dpMin[i-1]*num[i], min(num[i], dpMax[i-1]*num[i]))
	*/
	dpMax, dpMin := make([]int, len(nums)), make([]int, len(nums))
	for i := 0; i < len(nums); i++ {
		dpMax[i], dpMin[i] = nums[i], nums[i]
	}
	for i := 1; i < len(nums); i++ {
		dpMax[i] = max(dpMax[i-1]*nums[i], max(dpMin[i-1]*nums[i], nums[i]))
		dpMin[i] = min(dpMin[i-1]*nums[i], min(nums[i], dpMax[i-1]*nums[i]))
		if dpMin[i] < math.MinInt32 {
			dpMin[i] = nums[i]
		}
	}
	r := dpMax[0]
	for i := 0; i < len(nums); i++ {
		r = max(r, dpMax[i])
	}
	return r
}

/*
416. 分割等和子集
https://leetcode.cn/problems/partition-equal-subset-sum/?envType=study-plan-v2&envId=top-100-liked
*/
func canPartition(nums []int) bool {
	sum, maxVal := 0, math.MinInt32
	if len(nums) < 2 {
		return false
	}
	for i := 0; i < len(nums); i++ {
		sum += nums[i]
		if nums[i] > maxVal {
			maxVal = nums[i]
		}
	}
	if sum%2 != 0 || maxVal > sum/2 {
		return false
	}
	/*
		dp[i][j] 代表前i个数字和是否等于j
		if j == 0; dp[i][j] 都不选，则一定为true；if i == 0; dp[i][nums[0]] 为true；
	*/
	dp, target := make([][]bool, len(nums)), sum/2
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]bool, target+1)
	}
	for i := 0; i < len(nums); i++ {
		dp[i][0] = true
	}
	dp[0][nums[0]] = true
	for i := 1; i < len(nums); i++ {
		for j := 1; j <= target; j++ {
			if j >= nums[i] {
				dp[i][j] = dp[i-1][j] || dp[i-1][j-nums[i]]
			} else {
				dp[i][j] = dp[i-1][j]
			}
		}
	}
	return dp[len(nums)-1][target]
}

/*
======================================================================  动态规划专题  ============================================================
https://leetcode.cn/circle/discuss/tXLS3i/
*/

/*
377. 组合总和 Ⅳ
https://leetcode.cn/problems/combination-sum-iv/
*/
func combinationSum4(nums []int, target int) int {
	/*
		dp[x] 表示选取的元素之和等于 x 的方案数
	*/
	dp := make([]int, target+1)
	dp[0] = 1
	for i := 1; i < len(dp); i++ {
		for _, num := range nums {
			if num <= i {
				dp[i] += dp[i-num]
			}
		}
	}
	return dp[len(dp)-1]
}

/*
62. 不同路径
https://leetcode.cn/problems/unique-paths/?envType=study-plan-v2&envId=top-100-liked
*/
func uniquePaths(m int, n int) int {
	/*
		dp[i][j] = dp[i-1][j] + dp[i][j-1]
	*/
	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)
		for j := 0; j < n; j++ {
			if i == 0 || j == 0 {
				dp[i][j] = 1
			} else {
				dp[i][j] = dp[i-1][j] + dp[i][j-1]
			}
		}
	}
	return dp[m-1][n-1]
}

/*
64. 最小路径和
https://leetcode.cn/problems/minimum-path-sum/?envType=study-plan-v2&envId=top-100-liked
*/
func minPathSum(grid [][]int) int {
	/*
		dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
	*/
	dp := make([][]int, len(grid))
	for i := 0; i < len(grid); i++ {
		dp[i] = make([]int, len(grid[i]))
		for j := 0; j < len(grid[i]); j++ {
			if i == 0 && j == 0 {
				dp[i][j] = grid[i][j]
			} else if i == 0 {
				dp[i][j] = dp[i][j-1] + grid[i][j]
			} else if j == 0 {
				dp[i][j] = dp[i-1][j] + grid[i][j]
			} else {
				dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
			}
		}
	}
	return dp[len(dp)-1][len(dp[0])-1]
}

/*
5. 最长回文子串
https://leetcode.cn/problems/longest-palindromic-substring/?envType=study-plan-v2&envId=top-100-liked
*/
func longestPalindrome(s string) string {
	dp := make([][]bool, len(s))
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]bool, len(s))
		dp[i][i] = true
	}
	maxLen, begin := 1, 0
	for length := 2; length <= len(s); length++ {
		for left := 0; left < len(s); left++ {
			right := left + length - 1
			if right >= len(s) {
				break
			}
			if s[left] != s[right] {
				dp[left][right] = false
			} else {
				if right-left < 3 {
					dp[left][right] = true
				} else {
					dp[left][right] = dp[left+1][right-1]
				}
			}
			if dp[left][right] && right-left+1 > maxLen {
				begin = left
				maxLen = right - left + 1
			}
		}
	}
	return s[begin : begin+maxLen]
}

func longestPalindrome21(s string) string {
	if len(s) <= 1 {
		return s
	}
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

func expandAroundCenter(s string, i, j int) (l, r int) {
	for i >= 0 && j <= len(s)-1 && s[i] == s[j] {
		i--
		j++
	}
	return i + 1, j - 1
}

/*
1143. 最长公共子序列
https://leetcode.cn/problems/longest-common-subsequence/?envType=study-plan-v2&envId=top-100-liked
*/
func longestCommonSubsequence(text1 string, text2 string) int {
	/*
		dp[i][j] 表示 text1[0:i] 和 text2[0:j] 的最长公共子序列的长度。
		if text1[i] == text2[j] {
			dp[i][j] = dp[i-1][j-1] + 1
		} else {
			dp[i][j] = max(dp[i-1][j], dp[i][j-1])
		}
	*/
	dp := make([][]int, len(text1)+1)
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, len(text2)+1)
	}
	for i := 0; i < len(text1); i++ {
		for j := 0; j < len(text2); j++ {
			if text1[i] == text2[j] {
				dp[i+1][j+1] = dp[i][j] + 1
			} else {
				dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
			}
		}
	}
	return dp[len(dp)-1][len(dp[0])-1]
}

/*
72. 编辑距离
https://leetcode.cn/problems/edit-distance/?envType=study-plan-v2&envId=top-100-liked
*/
func minDistance(word1 string, word2 string) int {
	/*
		dp[i][j] 代表 word1 前i个 与 word2 前j个 的编辑距离
		if word1[i] == word2[j] {
			dp[i][j] = dp[i-1][j-1]
		} else {
			dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
		}
	*/
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
				dp[i][j] = minM(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
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
