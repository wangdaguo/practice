package main

import (
	"fmt"
	"math"
	"time"
)

func main() {
	//l := Constructor(2)
	//l.Put(1, 1)    // 0->1
	//l.Put(2, 2)    // 0->2->1
	//g1 := l.Get(1) // 0->1->2
	//l.Put(3, 3)    // 0->3->1
	//g2 := l.Get(2)
	////fmt.Println(g1, g2)
	////return
	//l.Put(4, 4) // 0->4->3
	//g3 := l.Get(1)
	//g4 := l.Get(3)
	//g5 := l.Get(4)
	//fmt.Println(g1, g2, g3, g4, g5)

	//node5 := &ListNode{
	//	Val:  2,
	//	Next: nil,
	//}
	//node4 := &ListNode{
	//	Val:  1,
	//	Next: node5,
	//}
	//node3 := &ListNode{
	//	Val:  -3,
	//	Next: node4,
	//}
	//node2 := &ListNode{
	//	Val:  2,
	//	Next: node3,
	//}
	//list1 := &ListNode{
	//	Val:  2,
	//	Next: node2,
	//}

	//node102 := &ListNode{
	//	Val:  6,
	//	Next: nil,
	//}
	//node101 := &ListNode{
	//	Val:  5,
	//	Next: node102,
	//}
	//list2 := &ListNode{
	//	Val:  4,
	//	Next: node101,
	//}

	//r := addTwoNumbers1(list1, list2)
	//h := removeZeroSumSublists(list1)
	//b, _ := json.Marshal(h)
	//fmt.Println(string(b))

	//b := Constructor1("leetcode.com")
	//b.Visit("google.com")
	//b.Visit("facebook.com")
	//b.Visit("youtube.com")
	//s1 := b.Back(1)
	//s2 := b.Back(1)
	//fmt.Println(s1, s2)
	//b.Forward(1)

	//q := Constructor2()
	//q.PushMiddle(1) // [1]
	//q.PushMiddle(2) // [1]
	//q.PushMiddle(3) // [1]
	//PrintList(q.head.Next)
	//r1 := q.PopMiddle()
	//r2 := q.PopMiddle()
	//r3 := q.PopMiddle()
	//r5 := q.PopBack() // 返回 2 -> []
	//r6 := q.PopBack() // 返回 2 -> []
	//r7 := q.PopBack() // 返回 2 -> []
	//r8 := q.PopBack() // 返回 2 -> []
	//fmt.Println(r1, r2, r3)

	//q.PushBack(2)   // [1, 2]
	//q.PushMiddle(3) // [1, 3, 2]
	//q.PushMiddle(4) // [1, 4, 3, 2]
	//
	//r1 := q.PopFront()  // 返回 1 -> [4, 3, 2]
	//r2 := q.PopMiddle() // 返回 3 -> [4, 2]
	//r3 := q.PopMiddle() // 返回 4 -> [2]
	//PrintList(q.head.Next)
	//return
	//r4 := q.PopBack() // 返回 2 -> []
	//
	//r5 := q.PopFront() // 返回 -1 -> [] （队列为空）

	//fmt.Println(r1, r2, r3, r4, r5)
	//PrintList(q.head.Next)

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
	//list1 := &ListNode{
	//	Val:  1,
	//	Next: node2,
	//}
	//reorderList1(list1)
	//PrintList(list1)

	//r := maxProfit([]int{2, 1, 2, 1, 0, 1, 2})
	//fmt.Println(r)

	//twitter := ConstructorTwitter()
	//twitter.PostTweet(1, 5)      // 用户 1 发送了一条新推文 (用户 id = 1, 推文 id = 5)
	//f1 := twitter.GetNewsFeed(1) // 用户 1 的获取推文应当返回一个列表，其中包含一个 id 为 5 的推文
	//twitter.Follow(1, 2)         // 用户 1 关注了用户 2
	//twitter.PostTweet(2, 6)      // 用户 2 发送了一个新推文 (推文 id = 6)
	//f2 := twitter.GetNewsFeed(1) // 用户 1 的获取推文应当返回一个列表，其中包含两个推文，id 分别为 -> [6, 5] 。推文 id 6 应当在推文 id 5 之前，因为它是在 5 之后发送的
	//twitter.Unfollow(1, 2)       // 用户 1 取消关注了用户 2
	//f3 := twitter.GetNewsFeed(1) // 用户 1 获取推文应当返回一个列表，其中包含一个 id 为 5 的推文。因为用户 1 已经不再关注用户 2

	/**
	["Twitter","postTweet","postTweet","postTweet","postTweet","postTweet","postTweet","postTweet","postTweet","postTweet","postTweet","postTweet","postTweet","postTweet","postTweet","follow","follow","follow","follow","follow","follow","follow","follow","follow","follow","follow","follow","getNewsFeed","getNewsFeed","getNewsFeed","getNewsFeed","getNewsFeed"]

	[[],[1,6765],[5,671],[3,2868],[4,8148],[4,386],[3,6673],[3,7946],[3,1445],[4,4822],[1,3781],[4,9038],[1,9643],[3,5917],[2,8847],[1,3],[1,4],[4,2],[4,1],[3,2],[3,5],[3,1],[2,3],[2,1],[2,5],[5,1],[5,2],[1],[2],[3],[4],[5]]
	 */
	twitter := ConstructorTwitter()
	twitter.PostTweet(1,6765)   // userTweetList[1] => {val:6765, time:1}
	twitter.PostTweet(5,671) 	 // userTweetList[5] => {val:671, time:2}
	twitter.PostTweet(3,2868) 	 // userTweetList[3] => {val:2868, time:3}
	twitter.PostTweet(4,8148) 	 // userTweetList[4] => {val:8148, time:4}
	twitter.PostTweet(4,386)	 // userTweetList[4] => {val:8148, time:4} -> {val:386, time:5}
	twitter.PostTweet(3,6673) 	 // userTweetList[3] => {val:2868, time:3} -> {val:6673, time:6}
	twitter.PostTweet(3,7946)	 // userTweetList[3] => {val:2868, time:3} -> {val:6673, time:6} -> {val:7946, time:7}
	twitter.PostTweet(3,1445)	 // userTweetList[3] => {val:2868, time:3} -> {val:6673, time:6} -> {val:7946, time:7} -> {val:1445, time:8}
	twitter.PostTweet(4,4822)  // userTweetList[4] => {val:8148, time:4} -> {val:386, time:5} -> {val:4822, time:9}
	twitter.PostTweet(1,3781)  // userTweetList[1] => {val:6765, time:1} -> {val:3781, time:10}
	twitter.PostTweet(4,9038)  // userTweetList[4] => {val:8148, time:4} -> {val:386, time:5} -> {val:4822, time:9} -> {val:9038, time:11}
	twitter.PostTweet(1,9643)  // userTweetList[1] => {val:6765, time:1} -> {val:3781, time:10} -> {val:9643, time:12}
	twitter.PostTweet(3,5917)  // userTweetList[3] => {val:2868, time:3} -> {val:6673, time:6} -> {val:7946, time:7} -> {val:1445, time:8}-> {val:5917, time:13}
	twitter.PostTweet(2,8847)   // userTweetList[2] => {val:8847, time:14}


	/**
	userTweetList[1] => {val:6765, time:1} -> {val:3781, time:10} -> {val:9643, time:12}
	userTweetList[2] => {val:8847, time:14}
	userTweetList[3] => {val:2868, time:3} -> {val:6673, time:6} -> {val:7946, time:7} -> {val:1445, time:8} -> {val:5917, time:13}
	userTweetList[4] => {val:8148, time:4} -> {val:386, time:5} -> {val:4822, time:9} -> {val:9038, time:11}
	userTweetList[5] => {val:671, time:2}
	 */

	twitter.Follow(1,3)
	twitter.Follow(1,4)
	twitter.Follow(4,2)
	twitter.Follow(4,1)
	twitter.Follow(3,2)
	twitter.Follow(3,5)
	twitter.Follow(3,1)
	twitter.Follow(2,3)
	twitter.Follow(2,1)
	twitter.Follow(2,5)
	twitter.Follow(5,1)
	twitter.Follow(5,2)

	//f1 := twitter.GetNewsFeed(1)
	f2 := twitter.GetNewsFeed(2)  // {3,1,5}
	f3 := twitter.GetNewsFeed(3)
	f4 := twitter.GetNewsFeed(4)
	f5 := twitter.GetNewsFeed(5)

	fmt.Println(f2, f3, f4, f5)
	//fmt.Println(f1, f2, f3, f4, f5)
	return
}


/**
355. 设计推特
https://leetcode.cn/problems/design-twitter/submissions/
 */
type TwitterNode struct {
	Val  int
	Time int64
	Next *TwitterNode
}

type Twitter struct {
	followeeList  map[int]map[int]int  // 关注用户列表
	userTweetList map[int]*TwitterNode // 每个人发的消息列表
	maxLen        int
}

func ConstructorTwitter() Twitter {
	t := Twitter{
		followeeList:  make(map[int]map[int]int),
		userTweetList: make(map[int]*TwitterNode),
		maxLen:        10,
	}
	return t
}

func (t *Twitter) PostTweet(userId int, tweetId int) {
	newNode := &TwitterNode{
		Val:  tweetId,
		Time: time.Now().UnixNano(),
	}
	n, ok := t.userTweetList[userId]
	if ok {
		newNode.Next = n
	}
	t.userTweetList[userId] = newNode
	return
}

func (t *Twitter) GetNewsFeed(userId int) []int {

	var nodeList []*TwitterNode
	// 自己发送的推文
	if n, ok := t.userTweetList[userId]; ok {
		nodeList = append(nodeList, n)
	}
	mp, ok := t.followeeList[userId]
	if !ok {
		mp = map[int]int{}
	}
	// 关注者发送的推文
	for uID, _ := range mp {
		nodeList = append(nodeList, t.userTweetList[uID])
	}
	// 排序
	head := mergeMoreList(nodeList, 0, len(nodeList)-1)
	r, h, cnt := make([]int, 0), head, 0
	for h != nil && cnt < 10 {
		r = append(r, h.Val)
		cnt++
		h = h.Next
	}
	return r
}

func mergeMoreList(list []*TwitterNode, start, end int) *TwitterNode {
	if len(list) < 1 || start > end {
		return nil
	}
	if start == end {
		return list[start]
	}
	middle := start + (end-start)/2
	return mergeSortList(mergeMoreList(list, start, middle), mergeMoreList(list, middle+1, end))
}

func mergeSortList(l1, l2 *TwitterNode) *TwitterNode {
	head := &TwitterNode{}
	h := head
	for l1 != nil && l2 != nil {
		if l1.Time < l2.Time {
			newNode := &TwitterNode{
				Val: l2.Val,
				Time: l2.Time,
			}
			h.Next = newNode
			l2 = l2.Next
		} else {
			newNode := &TwitterNode{
				Val: l1.Val,
				Time: l1.Time,
			}
			h.Next = newNode
			l1 = l1.Next
		}
		h = h.Next
	}
	for l1 != nil {
		newNode := &TwitterNode{
			Val: l1.Val,
			Time: l1.Time,
		}
		h.Next = newNode
		l1 = l1.Next
		h = h.Next
	}
	for l2 != nil {
		newNode := &TwitterNode{
			Val: l2.Val,
			Time: l2.Time,
		}
		h.Next = newNode
		l2 = l2.Next
		h = h.Next
	}
	return head.Next
}

func (t *Twitter) Follow(followerId int, followeeId int) {
	if _, ok := t.followeeList[followerId]; !ok {
		t.followeeList[followerId] = map[int]int{}
	}
	t.followeeList[followerId][followeeId] = followeeId
	return
}

func (t *Twitter) Unfollow(followerId int, followeeId int) {
	mp, ok := t.followeeList[followerId]
	if !ok {
		return
	}
	if _, ok := mp[followeeId]; !ok {
		return
	}
	delete(mp, followeeId)
	return
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

type ListNode struct {
	Val  int
	Next *ListNode
}

type Node struct {
	Val    int
	Next   *Node
	Random *Node
}

type Node2 struct {
	Val  string
	Pre  *Node2
	Next *Node2
}

type Node3 struct {
	Val   int
	Left  *Node3
	Right *Node3
	Next  *Node3
}

func maxProfit(prices []int) int {
	if len(prices) < 1 {
		return 0
	}
	buyIndex, r := 0, 0
	for i := 1; i < len(prices); i++ {
		if prices[i] < prices[buyIndex] {
			buyIndex = i
		}
		if prices[i] > prices[buyIndex] && prices[i]-prices[buyIndex] > r {
			r = prices[i] - prices[buyIndex]
		}
	}
	return r
}

/*
*
116. 填充每个节点的下一个右侧节点指针
https://leetcode.cn/problems/populating-next-right-pointers-in-each-node/
*/
func connect(root *Node3) *Node3 {
	if root == nil {
		return root
	}
	queue := make([]*Node3, 0)
	queue = append(queue, root)
	for len(queue) > 0 {
		h := &Node3{}
		cnt := len(queue)
		for cnt > 0 {
			next := queue[0]
			queue = queue[1:]
			h.Next = next
			h = h.Next
			if next.Left != nil {
				queue = append(queue, next.Left)
			}
			if next.Right != nil {
				queue = append(queue, next.Right)
			}
			cnt--
		}
	}
	return root
}

/*
*
LCR 057. 存在重复元素 III
https://leetcode.cn/problems/7WqeDu/description/
*/
func containsNearbyAlmostDuplicate(nums []int, k int, t int) bool {
	mp := map[int]int{}
	for i, x := range nums {
		ID := getID(x, t+1)
		if _, ok := mp[ID]; ok {
			return true
		}
		if v, ok := mp[ID+1]; ok && abs(v-x) <= t {
			return true
		}
		if v, ok := mp[ID-1]; ok && abs(v-x) <= t {
			return true
		}
		mp[ID] = x
		if i >= k {
			delete(mp, getID(nums[i-k], t+1))
		}
	}
	return false
}

func getID(x, t int) int {
	if x >= 0 {
		return x / t
	}
	return (x+1)/t - 1
}

func abs(x int) int {
	if x >= 0 {
		return x
	}
	return -x
}

/*
*
链表中环的入口节点
https://leetcode.cn/problems/c32eOV/solutions/1037744/lian-biao-zhong-huan-de-ru-kou-jie-dian-vvofe/
*/
func detectCycle(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
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
*
LCR 022. 环形链表 II
https://leetcode.cn/problems/partition-list-lcci/description/
*/
func partition(head *ListNode, x int) *ListNode {
	if head == nil {
		return head
	}
	smallHead, bigHead, h := &ListNode{}, &ListNode{}, head
	sh, bh := smallHead, bigHead
	for h != nil {
		if h.Val < x {
			sh.Next = h
			sh = sh.Next
		} else {
			bh.Next = h
			bh = bh.Next
		}
		h = h.Next
	}
	sh.Next = bigHead.Next
	bh.Next = nil
	return smallHead.Next
}

func reorderList1(head *ListNode) {
	middle := findMiddleNode(head)
	l1 := head
	l2 := middle.Next
	middle.Next = nil
	l2 = reverseL(l2)
	mergeL(l1, l2)
}

func findMiddleNode(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	slow, fast := head, head
	for fast.Next != nil && fast.Next.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	return slow
}

func reverseL(head *ListNode) *ListNode {
	if head.Next == nil {
		return head
	}
	h := reverseL(head.Next)
	head.Next.Next = head
	head.Next = nil
	return h
}

func mergeL(h1, h2 *ListNode) {
	for h1 != nil && h2 != nil {
		tmp1 := h1.Next
		tmp2 := h2.Next
		h1.Next = h2
		h1 = tmp1

		h2.Next = h1
		h2 = tmp2
	}
}

func removeNodes(head *ListNode) *ListNode {
	if head.Next == nil {
		return head
	}
	node := removeNodes(head.Next)
	if node.Val > head.Val {
		return node
	}
	head.Next = node
	return head
}

/*
*
2095. 删除链表的中间节点  [1,2,3,4,5]  =>   [1,2,4]
https://leetcode.cn/problems/delete-the-middle-node-of-a-linked-list/
*/
func deleteMiddle(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return nil
	}
	dummyHead := &ListNode{Next: head}
	pre, slow, fast := head, head, head
	for fast != nil && fast.Next != nil {
		pre = slow
		slow = slow.Next
		fast = fast.Next.Next
	}
	pre.Next = slow.Next
	return dummyHead.Next
}

/*
*725. 分隔链表
https://leetcode.cn/problems/split-linked-list-in-parts/description/
*/
func splitListToParts(head *ListNode, k int) []*ListNode {
	if head == nil {
		tmp, r := 0, make([]*ListNode, 0)
		for tmp <= k {
			r = append(r, nil)
			tmp++
		}
		return r
	}
	r, h, l := make([]*ListNode, 0), head, 0
	for h != nil {
		l++
		h = h.Next
	}
	var emptyListCnt, avgCnt, prefixAddCnt = k - l, 1, 0
	if l >= k {
		emptyListCnt, avgCnt, prefixAddCnt = 0, l/k, l%k
	}
	i, cur, newHead := 0, head, head
	var pre *ListNode
	// 1, 2 , 3
	for cur != nil || i == avgCnt {
		if i < avgCnt || (prefixAddCnt > 0 && len(r) <= prefixAddCnt && i < (avgCnt+1)) {
			pre = cur
			cur = cur.Next
			i++
			continue
		}
		pre.Next = nil
		r = append(r, newHead)
		// 重置指针
		newHead = cur
		pre = cur
		i = 0
	}
	for emptyListCnt > 0 {
		r = append(r, nil)
		emptyListCnt--
	}
	return r
}

/*
1721. 交换链表中的节点
*https://leetcode.cn/problems/swapping-nodes-in-a-linked-list/
*/
func swapNodes(head *ListNode, k int) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	n, dn, fast, i := head, head, head, 1
	for i < k {
		i++
		fast = fast.Next
	}
	for fast.Next != nil {
		dn = dn.Next
		fast = fast.Next
	}
	i = 1
	for i < k {
		n = n.Next
		i++
	}
	n.Val, dn.Val = dn.Val, n.Val
	return head
}

/*
1670. 设计前中后队列
*https://leetcode.cn/problems/design-front-middle-back-queue/
*/
type FrontMiddleBackQueue struct {
	head *ListNode
}

func Constructor2() FrontMiddleBackQueue {
	return FrontMiddleBackQueue{
		head: &ListNode{},
	}
}

func (f *FrontMiddleBackQueue) PushFront(val int) {
	node := &ListNode{
		Val: val,
	}
	next := f.head.Next
	f.head.Next = node
	node.Next = next
	return
}

func (f *FrontMiddleBackQueue) PushMiddle(val int) {
	pre := f.head
	slow, fast := f.head.Next, f.head.Next
	node := &ListNode{
		Val: val,
	}
	for fast != nil && fast.Next != nil {
		pre = slow
		slow = slow.Next
		fast = fast.Next.Next
	}
	preNext := pre.Next
	pre.Next = node
	node.Next = preNext
	return
}

func (f *FrontMiddleBackQueue) PushBack(val int) {
	node := &ListNode{
		Val: val,
	}
	h := f.head
	for h.Next != nil {
		h = h.Next
	}
	h.Next = node
}

func (f *FrontMiddleBackQueue) PopFront() int {
	if f.head.Next == nil {
		return -1
	}
	cur := f.head.Next
	f.head.Next = f.head.Next.Next
	return cur.Val
}

// 4, 2
func (f *FrontMiddleBackQueue) PopMiddle() int {
	var pre *ListNode
	slow, fast := f.head, f.head
	for fast != nil && fast.Next != nil {
		pre = slow
		slow = slow.Next
		fast = fast.Next.Next
	}
	if pre == nil {
		return -1
	}
	cur := pre.Next
	pre.Next = pre.Next.Next
	return cur.Val
}

func (f *FrontMiddleBackQueue) PopBack() int {
	pre, h := f.head, f.head.Next
	for h != nil && h.Next != nil {
		pre = h
		h = h.Next
	}
	if pre == f.head && pre.Next == nil {
		return -1
	}
	pre.Next = nil
	return h.Val
}

/*
*
1472. 设计浏览器历史记录
https://leetcode.cn/problems/design-browser-history/
*/
type BrowserHistory struct {
	head    *Node2
	curNode *Node2
}

func Constructor1(homepage string) BrowserHistory {
	b := BrowserHistory{
		head: &Node2{
			Val: homepage,
		},
	}
	b.curNode = b.head
	return b
}

func (b *BrowserHistory) Visit(url string) {
	n := &Node2{
		Val: url,
	}
	n.Pre = b.curNode
	b.curNode.Next = n
	b.curNode = n
	return
}

func (b *BrowserHistory) Back(steps int) string {
	i := 0
	for i < steps && b.curNode.Pre != nil {
		i++
		b.curNode = b.curNode.Pre
	}
	return b.curNode.Val
}

func (b *BrowserHistory) Forward(steps int) string {
	i := 0
	for i < steps && b.curNode.Next != nil {
		i++
		b.curNode = b.curNode.Next
	}
	return b.curNode.Val
}

/*
*
1171. 从链表中删去总和值为零的连续节点
https://leetcode.cn/problems/remove-zero-sum-consecutive-nodes-from-linked-list/

[2, 10, -3, 1, 2]

2， 10， 9， 10， 12
*/
func removeZeroSumSublists(head *ListNode) *ListNode {
	dummy := &ListNode{Val: 0}
	dummy.Next = head
	seen := map[int]*ListNode{}
	prefix := 0
	for node := dummy; node != nil; node = node.Next {
		prefix += node.Val
		seen[prefix] = node
	}
	prefix = 0
	for node := dummy; node != nil; node = node.Next {
		prefix += node.Val
		node.Next = seen[prefix].Next
	}
	return dummy.Next
}

/*
*
1019. 链表中的下一个更大节点
https://leetcode.cn/problems/next-greater-node-in-linked-list/description/
*/
func nextLargerNodes(head *ListNode) []int {
	var ans []int
	var stack [][]int
	cur := head
	idx := -1
	for cur != nil {
		idx++
		ans = append(ans, 0)
		for len(stack) > 0 && stack[len(stack)-1][0] < cur.Val {
			top := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			ans[top[1]] = cur.Val
		}
		stack = append(stack, []int{cur.Val, idx})
		cur = cur.Next
	}
	return ans
}

/*
*
445. 两数相加 II
https://leetcode.cn/problems/add-two-numbers-ii/
*/
func addTwoNumbers1(l1 *ListNode, l2 *ListNode) *ListNode {
	if l1 == nil {
		return l2
	}
	if l2 == nil {
		return l1
	}
	r1, r2, dummyHead, jw := reverseList(l1), reverseList(l2), &ListNode{}, 0
	h := dummyHead
	for r1 != nil || r2 != nil || jw != 0 {
		v1, v2 := 0, 0
		if r1 != nil {
			v1 = r1.Val
			r1 = r1.Next
		}
		if r2 != nil {
			v2 = r2.Val
			r2 = r2.Next
		}
		val := (v1 + v2 + jw) % 10
		jw = (v1 + v2 + jw) / 10
		tmpNode := &ListNode{Val: val}
		h.Next = tmpNode
		h = h.Next
	}
	return reverseList(dummyHead.Next)
}

func reverseList(list *ListNode) *ListNode {
	if list == nil || list.Next == nil {
		return list
	}
	var pre, cur *ListNode
	cur = list
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
146. LRU 缓存
https://leetcode.cn/problems/lru-cache/
*/
type LRUCache struct {
	head   *Node1
	mp     map[int]*Node1
	maxLen int
}

type Node1 struct {
	Key  int
	Val  int
	Pre  *Node1
	Next *Node1
}

func Constructor(capacity int) LRUCache {
	l := LRUCache{
		head:   &Node1{},
		mp:     make(map[int]*Node1, capacity),
		maxLen: capacity,
	}
	return l
}

func (l *LRUCache) Get(key int) int {
	h, ok := l.mp[key]
	if !ok {
		return -1
	}

	l.removeNode(h)
	l.insertHead(h)
	return h.Val
}

func (l *LRUCache) insertHead(h *Node1) {

	next := l.head.Next

	l.head.Next = h
	h.Pre = l.head

	if next != nil {
		h.Next = next
		next.Pre = h
	}

	return
}

func (l *LRUCache) removeTail() *Node1 {

	h := l.head
	for h.Next != nil {
		h = h.Next
	}
	delete(l.mp, h.Key)
	l.removeNode(h)
	return h
}

func (l *LRUCache) removeNode(h *Node1) {
	if h.Next == nil {
		if h.Pre != nil {
			h.Pre.Next = nil
		}
		h.Pre = nil
	} else {
		h.Pre.Next = h.Next
		if h.Next != nil {
			h.Next.Pre = h.Pre
		}
	}
	return
}

/*
*
["LRUCache","put","put","get","put","put","get"]
[[2],[2,1],[2,2],[2],[1,1],[4,1],[2]]
*/
func (l *LRUCache) Put(key int, value int) {
	n, ok := l.mp[key]
	if ok {
		n.Val = value
		l.removeNode(n)
		l.insertHead(n)
		return
	}

	newNode := &Node1{Key: key, Val: value}
	if len(l.mp) < l.maxLen {
		l.mp[key] = newNode
		l.insertHead(newNode)
		return
	}

	l.mp[key] = newNode

	// 去掉最后一个
	tail := l.removeTail()

	// 去掉最后一个在map中的值
	delete(l.mp, tail.Key)

	// 加入新的
	l.insertHead(newNode)

	return
}

/*
138. 随机链表的复制
*https://leetcode.cn/problems/copy-list-with-random-pointer/
*/
func copyRandomList(head *Node) *Node {
	if head == nil {
		return nil
	}
	mp := make(map[*Node]*Node)
	return deepCopy(head, mp)
}

func deepCopy(head *Node, mp map[*Node]*Node) *Node {
	if head == nil {
		return nil
	}
	if n, ok := mp[head]; ok {
		return n
	}
	newNode := &Node{Val: head.Val}
	mp[head] = newNode
	newNode.Next = deepCopy(head.Next, mp)
	newNode.Random = deepCopy(head.Random, mp)
	return newNode
}

/*
*
206. 反转链表
https://leetcode-cn.com/problems/reverse-linked-list/
*/
func reverseList1(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}
	cur := head
	var next, pre *ListNode
	for cur != nil {
		next = cur.Next
		cur.Next = pre
		pre = cur
		cur = next
	}
	return pre
}

func reverseList2(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}
	cur := head
	pre := reverseList2(head.Next)
	cur.Next.Next = cur
	return pre
}

func hasCycle(head *ListNode) bool {
	if head == nil || head.Next == nil {
		return false
	}
	slow, fast := head, head.Next
	for slow != fast {
		if fast == nil && fast.Next == nil {
			return false
		}
		slow = slow.Next
		fast = fast.Next.Next
	}
	return true
}

/*
*
24. 两两交换链表中的节点
https://leetcode-cn.com/problems/swap-nodes-in-pairs/
*/
func swapPairs(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	newHead := head.Next
	pre := swapPairs(head.Next.Next)
	head.Next.Next = head
	head.Next = pre
	return newHead
}

func swapPairs1(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	dummy := &ListNode{0, head}
	n1, n2 := dummy, head
	for n2 != nil && n2.Next != nil {
		n1.Next = n2.Next
		n2.Next = n2.Next.Next
		n1.Next.Next = n2

		newN2 := n2.Next
		n1 = n2
		n2 = newN2
	}
	return dummy.Next
}

func swapPairs2(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	newHead := head.Next
	tmp := swapPairs2(head.Next.Next)
	head.Next = tmp
	newHead.Next = head
	return newHead
}

/*
*
328. 奇偶链表
https://leetcode-cn.com/problems/odd-even-linked-list/
输入: 1->2->3->4->5->NULL
输出: 1->3->5->2->4->NULL
*/
func oddEvenList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	oddHead := head.Next
	m1, m2 := head, oddHead
	for {
		m1.Next = m2.Next
		m1 = m1.Next
		if m1 == nil {
			break
		}
		m2.Next = m1.Next
		m2 = m2.Next
		if m2 == nil {
			break
		}
	}
	//fmt.Printf("m1=%+v||m2=%+v", m1, oddHead)
	//fmt.Println()
	//return oddHead
	if m1 == nil {
		m1 = head
		for m1.Next != nil {
			m1 = m1.Next
		}
	}
	m1.Next = oddHead
	return head
}

/*
*
92. 反转链表 II
https://leetcode-cn.com/problems/reverse-linked-list-ii/
输入: 1->2->3->4->5->NULL, m = 2, n = 4
输出: 1->4->3->2->5->NULL
*/
func reverseBetween(head *ListNode, m int, n int) *ListNode {
	if head == nil || m == n {
		return head
	}
	newHead := &ListNode{Val: -1, Next: head}
	cur, start := head, newHead
	var i int
	for i = 1; i < m; i++ {
		start = cur
		cur = cur.Next
	}
	if cur == nil {
		return head
	}
	end := cur
	var pre *ListNode
	for cur != nil && i <= n {
		next := cur.Next
		cur.Next = pre
		pre = cur
		cur = next
		i++
	}
	start.Next = pre
	end.Next = cur
	return newHead.Next
}

/*
*
237. 删除链表中的节点
https://leetcode-cn.com/problems/delete-node-in-a-linked-list/
输入: head = [4,5,1,9], node = 5
输出: [4,1,9]
解释: 给定你链表中值为 5 的第二个节点，那么在调用了你的函数之后，该链表应变为 4 -> 1 -> 9.
*/
func deleteNode(node *ListNode) {
	node.Val = node.Next.Val
	node.Next = node.Next.Next
}

/*
*
19. 删除链表的倒数第N个节点
https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/
给定一个链表: 1->2->3->4->5, 和 n = 2.
当删除了倒数第二个节点后，链表变为 1->2->3->5.
*/
func removeNthFromEnd(head *ListNode, n int) *ListNode {
	if head == nil || n < 1 {
		return head
	}
	newHead := &ListNode{Val: -1, Next: head}
	var i int
	start, end := newHead, newHead
	for i < n {
		end = end.Next
		i++
	}
	//fmt.Println(end.Val)
	for end.Next != nil {
		start = start.Next
		end = end.Next
	}
	//fmt.Println(start.Val, end.Val)
	//return nil
	start.Next = start.Next.Next
	return newHead.Next
}

/*
*
83. 删除排序链表中的重复元素
https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/
给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次。
输入: 1->1->2->3->3
输出: 1->2->3
*/
func deleteDuplicates(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	pre, cur := head, head.Next
	for cur != nil {
		for pre.Val == cur.Val {
			pre.Next = cur.Next
			cur = cur.Next
			if cur == nil {
				return head
			}
		}
		pre = cur
		cur = cur.Next
	}
	return head
}

/*
*
203. 移除链表元素
https://leetcode-cn.com/problems/remove-linked-list-elements/
删除链表中等于给定值 val 的所有节点。
输入: 1->2->6->3->4->5->6, val = 6
输出: 1->2->3->4->5
*/
func removeElements(head *ListNode, val int) *ListNode {
	if head == nil {
		return head
	}
	newHead := &ListNode{Val: -1, Next: head}
	tmp := newHead
	for tmp != nil && tmp.Next != nil {
		for tmp.Next != nil && tmp.Next.Val == val {
			tmp.Next = tmp.Next.Next
		}
		tmp = tmp.Next
	}
	return newHead.Next
}

/*
*
82. 删除排序链表中的重复元素 II
https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/
给定一个排序链表，删除所有含有重复数字的节点，只保留原始链表中 没有重复出现 的数字。
输入: 1->2->3->3->4->4->5
输出: 1->2->5
*/
func deleteDuplicates2(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	newHead := &ListNode{Val: -1, Next: head}
	pre, cur := newHead, head
	for cur != nil {
		next := cur.Next
		if next != nil && cur.Val == next.Val {
			for next != nil && cur.Val == next.Val {
				next = next.Next
			}
			cur = next
			pre.Next = cur
		} else {
			pre.Next = cur
			pre = cur
			cur = next
		}
	}
	return newHead.Next
}

/*
*
2. 两数相加
https://leetcode-cn.com/problems/add-two-numbers/
输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
输出：7 -> 0 -> 8
原因：342 + 465 = 807
*/
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	if l1 == nil {
		return l2
	}
	if l2 == nil {
		return l1
	}
	r := &ListNode{Val: -1, Next: nil}
	pre := r
	t := 0
	for l1 != nil || l2 != nil || t != 0 {
		if l1 != nil {
			t += l1.Val
			l1 = l1.Next
		}
		if l2 != nil {
			t += l2.Val
			l2 = l2.Next
		}
		node := &ListNode{Val: t % 10, Next: nil}
		pre.Next = node
		pre = pre.Next
		t /= 10
	}
	return r.Next
}

/*
*
160. 相交链表
https://leetcode-cn.com/problems/intersection-of-two-linked-lists/
编写一个程序，找到两个单链表相交的起始节点。
intersectVal = 0,
listA = [2,6,4],
listB = [1,5],
skipA = 3, skipB = 2
*/
func getIntersectionNode(headA, headB *ListNode) *ListNode {
	if headA == nil || headB == nil {
		return nil
	}
	pA, pB := headA, headB
	for pA != pB {
		if pA == nil {
			pA = headB
		} else {
			pA = pA.Next
		}

		if pB == nil {
			pB = headA
		} else {
			pB = pB.Next
		}
	}
	return pA
}

/*
*
21. 合并两个有序链表
https://leetcode-cn.com/problems/merge-two-sorted-lists/
输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4
*/
func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
	r := &ListNode{Val: -1, Next: nil}
	tmp := r
	for l1 != nil || l2 != nil {
		if l1 == nil {
			tmp.Next = l2
			break
		}
		if l2 == nil {
			tmp.Next = l1
			break
		}
		if l1.Val >= l2.Val {
			tmp.Next = &ListNode{Val: l2.Val, Next: nil}
			l2 = l2.Next
		} else {
			tmp.Next = &ListNode{Val: l1.Val, Next: nil}
			l1 = l1.Next
		}
		tmp = tmp.Next
	}
	return r.Next
}

/*
*
234. 回文链表
https://leetcode-cn.com/problems/palindrome-linked-list/
输入: 1->2
输出: false
输入: 1->2->2->1
输出: true
*/
func isPalindrome1(head *ListNode) bool {
	if head == nil || head.Next == nil {
		return true
	}
	tmp := head
	var l int
	for tmp != nil {
		tmp = tmp.Next
		l++
	}
	middle := int(math.Floor(float64(l / 2)))
	var pre, next *ListNode
	cur := head
	var i int
	for i < middle {
		next = cur.Next
		cur.Next = pre
		pre = cur
		cur = next
		i++
	}
	//fmt.Println(middle, pre.Val, cur.Val, next.Val)
	//return false
	sHead := cur
	if l%2 != 0 {
		sHead = sHead.Next
	}
	for sHead != nil && pre != nil {
		if sHead.Val != pre.Val {
			return false
		}
		sHead = sHead.Next
		pre = pre.Next
	}
	return true
}

func isPalindrome1Test(head *ListNode) bool {
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	if fast != nil {
		slow = slow.Next
	}
	tail := rev(slow)
	cur1, cur2 := head, tail
	for cur2 != nil {
		if cur1.Val != cur2.Val {
			return false
		}
		cur1 = cur1.Next
		cur2 = cur2.Next
	}
	return true
}

func rev(head *ListNode) *ListNode {
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
143. 重排链表
https://leetcode-cn.com/problems/reorder-list/
给定一个单链表 L：L0→L1→…→Ln-1→Ln ，
将其重新排列后变为： L0→Ln→L1→Ln-1→L2→Ln-2→…

你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

给定链表 1->2->3->4, 重新排列为 1->4->2->3.
*/
func reorderList(head *ListNode) {
	if head == nil || head.Next == nil {
		return
	}
	slow, fast, pre := head, head, head
	for fast != nil && fast.Next != nil {
		pre = slow
		slow = slow.Next
		fast = fast.Next.Next
	}
	if fast != nil {
		pre = slow
		slow = slow.Next
	}
	pre.Next = nil
	tail := rev(slow)
	m1, m2 := head, tail
	var next1, next2 *ListNode
	for m2 != nil {
		next1 = m1.Next
		next2 = m2.Next
		m1.Next = m2
		m2.Next = next1
		m1 = next1
		m2 = next2
	}
	return
}

/*
*
148. 排序链表
https://leetcode-cn.com/problems/sort-list/
在 O(n log n) 时间复杂度和常数级空间复杂度下，对链表进行排序。

输入: 4->2->1->3
输出: 1->2->3->4
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
	return merge1(sortList(head), sortList(slow))
}

func merge1(l1 *ListNode, l2 *ListNode) *ListNode {
	h := &ListNode{Val: -1, Next: nil}
	r := h
	for l1 != nil && l2 != nil {
		if l1.Val >= l2.Val {
			r.Next = &ListNode{Val: l2.Val, Next: nil}
			l2 = l2.Next
		} else {
			r.Next = &ListNode{Val: l1.Val, Next: nil}
			l1 = l1.Next
		}
		r = r.Next
	}
	if l1 != nil {
		r.Next = l1
	}
	if l2 != nil {
		r.Next = l2
	}
	return h.Next
}

/*
*
25. K 个一组翻转链表
https://leetcode-cn.com/problems/reverse-nodes-in-k-group/
给你这个链表：1->2->3->4->5
当 k = 2 时，应当返回: 2->1->4->3->5
当 k = 3 时，应当返回: 3->2->1->4->5
*/
func reverseKGroup(head *ListNode, k int) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	var i int
	tmp := head
	for tmp != nil {
		i++
		tmp = tmp.Next
	}
	totalCnt := int(math.Floor(float64(i / k)))
	var cnt, j int
	newHead := &ListNode{Val: -1, Next: nil}
	preStart, start, end := newHead, head, head
	for cnt < totalCnt {
		cur := start
		var pre, next *ListNode
		end = cur
		for j < k {
			next = cur.Next
			cur.Next = pre
			pre = cur
			cur = next
			j++
		}
		preStart.Next = pre
		preStart = end
		start = cur
		cnt++
		j = 0
	}
	if start != nil {
		preStart.Next = start
	}
	return newHead.Next
}

/*
*
61. 旋转链表
https://leetcode-cn.com/problems/rotate-list/
给定一个链表，旋转链表，将链表每个节点向右移动 k 个位置，其中 k 是非负数。
输入: 1->2->3->4->5->NULL, k = 2
输出: 4->5->1->2->3->NULL
解释:
向右旋转 1 步: 5->1->2->3->4->NULL
向右旋转 2 步: 4->5->1->2->3->NULL
*/
func rotateRight1(head *ListNode, k int) *ListNode {
	if head == nil {
		return head
	}
	var l int
	tmp := head
	for tmp != nil {
		tmp = tmp.Next
		l++
	}
	moveLen := k % l
	if moveLen == 0 {
		return head
	}
	var cnt int
	pre, cur := head, head
	for cnt < l-moveLen {
		cnt++
		pre = cur
		cur = cur.Next
	}
	pre.Next = nil
	t := cur
	for t != nil && t.Next != nil {
		t = t.Next
	}
	t.Next = head
	return cur
}

func rotateRight(head *ListNode, k int) *ListNode {
	if head == nil || head.Next == nil || k == 0 {
		return head
	}
	len := 1
	tmp := head
	for tmp.Next != nil {
		tmp = tmp.Next
		len++
	}
	k = k % len
	if k == 0 {
		return head
	}
	tmp.Next = head
	for cnt := len - k; cnt > 0 && tmp != nil; cnt-- {
		tmp = tmp.Next
	}
	newHead := tmp.Next
	tmp.Next = nil
	return newHead
}

/*
*
86. 分隔链表
https://leetcode-cn.com/problems/partition-list/
输入: head = 1->4->3->2->5->2, x = 3
输出: 1->2->2->4->3->5
*/
func partition1(head *ListNode, x int) *ListNode {
	if head == nil {
		return head
	}
	less := &ListNode{}
	great := &ListNode{}
	curLess := less
	curGreat := great
	for head != nil {
		if head.Val < x {
			curLess.Next = head
			curLess = curLess.Next
		} else {
			curGreat.Next = head
			curGreat = curGreat.Next
		}
		head = head.Next
	}
	curGreat.Next = nil
	curLess.Next = great.Next
	return less.Next
}

/*
*
23. 合并K个排序链表
https://leetcode-cn.com/problems/merge-k-sorted-lists/
输入:
[

	1->4->5,
	1->3->4,
	2->6

]
输出: 1->1->2->3->4->4->5->6
*/
func mergeKLists(lists []*ListNode) *ListNode {
	if len(lists) < 1 {
		return nil
	}
	return merge2(lists, 0, len(lists)-1)
}

func merge2(lists []*ListNode, start, end int) *ListNode {
	if start == end {
		return lists[start]
	}
	if start > end {
		return nil
	}
	mid := (start + end) / 2
	return merge2List(merge2(lists, start, mid), merge2(lists, mid+1, end))
}

func merge2List(h1, h2 *ListNode) *ListNode {
	head := &ListNode{Val: -1}
	tmp := head
	for h1 != nil && h2 != nil {
		if h1.Val > h2.Val {
			tmp.Next = h2
			h2 = h2.Next
		} else {
			tmp.Next = h1
			h1 = h1.Next
		}
		tmp = tmp.Next
	}
	if h1 != nil {
		tmp.Next = h1
	} else if h2 != nil {
		tmp.Next = h2
	}
	return head.Next
}

/*
*
147. 对链表进行插入排序
https://leetcode-cn.com/problems/insertion-sort-list/
147. 对链表进行插入排序
输入: -1-> 4->2->1->3
输出: 1->2->3->4
*/
func insertionSortList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	newHead := &ListNode{Val: -1, Next: head}
	tmp := head
	pre := newHead
	for tmp != nil {
		next := tmp.Next
		pre = tmp
		h := head
		preH := newHead
		for h != tmp {
			if h.Val > tmp.Val {
				pre.Next = tmp.Next
				tmp.Next = preH.Next
				preH.Next = tmp

				break
			}
			preH = h
			h = h.Next
		}
		pre = tmp
		tmp = next
	}
	return newHead.Next
}

func insertionSortList1(head *ListNode) *ListNode {
	if head == nil {
		return head
	}
	h := &ListNode{Next: head}
	p := head.Next
	head.Next = nil
	for p != nil {
		pre := h
		next := p.Next
		for pre.Next != nil && pre.Next.Val <= p.Val {
			pre = pre.Next
		}
		p.Next = pre.Next
		pre.Next = p
		p = next
	}
	return h.Next
}

func mainLinkedList() {
	//node5 := &ListNode{
	//	Val:3,
	//	Next:nil,
	//}
	//node4 := &ListNode{
	//	Val:3,
	//	Next:node5,
	//}
	//node3 := &ListNode{
	//	Val:4,
	//	Next:nil,
	//}
	//node2 := &ListNode{
	//	Val:6,
	//	Next:node3,
	//}
	//node1 := &ListNode{
	//	Val:2,
	//	Next:node2,
	//}

	//l5 := &ListNode{
	//	Val:5,
	//	Next:nil,
	//}
	l4 := &ListNode{
		Val:  3,
		Next: nil,
	}
	l3 := &ListNode{
		Val:  1,
		Next: l4,
	}
	l2 := &ListNode{
		Val:  2,
		Next: l3,
	}
	l1 := &ListNode{
		Val:  4,
		Next: l2,
	}
	//r := swapPairs2(node1)
	//r := oddEvenList(node1)
	//r := reverseBetween(node1, 1, 2)
	//r := removeNthFromEnd(node1, 2)
	//r := deleteDuplicates(node1)
	//r := deleteDuplicates2(node1)
	//r := addTwoNumbers(node1, l1)
	//r := getIntersectionNode(node1, l1)
	//r := isPalindrome1(l1)
	//reorderList(r)
	//fmt.Println(r)
	//r := reverseKGroup(l1, 2)
	//r := rotateRight(l1, 1)
	r := insertionSortList(l1)
	var i int64
	for r != nil {
		i++
		fmt.Println(r.Val)
		r = r.Next
		if i > 10 {
			break
		}
	}
}
