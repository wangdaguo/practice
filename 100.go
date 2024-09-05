package main

import (
	"container/heap"
	"fmt"
	"math"
	"sort"
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
	nums := [][]int{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
	r := spiralOrder(nums)
	fmt.Println(r)
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
func rotate(nums []int, k int) {
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
	isSymmetricImpl := func(left, right *TreeNode) bool {
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
322. 零钱兑换
https://leetcode.cn/problems/coin-change/?envType=study-plan-v2&envId=top-100-liked
*/
func coinChange(coins []int, amount int) int {
	dp := make([]int, amount+1)
	dp[0] = 0
	for i := 1; i <= amount; i++ {
		dp[i] = amount + 1
	}
	/**
	i元钱需要的coin数量   [1, 2, 5]  11
	dp[i] = min(dp[i], dp[i-coins[i]]+1)
	*/
	for i := 1; i <= amount; i++ {
		for j := 0; j < len(coins); j++ {
			if i < coins[j] {
				continue
			}
			dp[i] = min(dp[i], dp[i-coins[j]]+1)
		}
	}
	if dp[len(dp)-1] > amount {
		return -1
	}
	return dp[len(dp)-1]
}
