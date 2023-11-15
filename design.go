package main

import (
	"time"
)

func main() {
	//matrix := [][]int{
	//	{3,0,1,4,2},
	//	{5,6,3,2,1},
	//	{1,2,0,1,5},
	//	{4,1,0,1,7},
	//	{1,0,3,0,5},
	//}
	//numMatrix := ConstructorNumMatrix(matrix)
	////fmt.Print(numMatrix)
	////return
	//s1 := numMatrix.SumRegion(2, 1, 4, 3) // return 8 (红色矩形框的元素总和)
	//s2 := numMatrix.SumRegion(1, 1, 2, 2) // return 11 (绿色矩形框的元素总和)
	//s3 := numMatrix.SumRegion(1, 2, 2, 4) // return 12 (蓝色矩形框的元素总和)
	//fmt.Print(s1, s2, s3)

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
	//twitter := ConstructorTwitter()
	//twitter.PostTweet(1,6765)   // userTweetList[1] => {val:6765, time:1}
	//twitter.PostTweet(5,671) 	 // userTweetList[5] => {val:671, time:2}
	//twitter.PostTweet(3,2868) 	 // userTweetList[3] => {val:2868, time:3}
	//twitter.PostTweet(4,8148) 	 // userTweetList[4] => {val:8148, time:4}
	//twitter.PostTweet(4,386)	 // userTweetList[4] => {val:8148, time:4} -> {val:386, time:5}
	//twitter.PostTweet(3,6673) 	 // userTweetList[3] => {val:2868, time:3} -> {val:6673, time:6}
	//twitter.PostTweet(3,7946)	 // userTweetList[3] => {val:2868, time:3} -> {val:6673, time:6} -> {val:7946, time:7}
	//twitter.PostTweet(3,1445)	 // userTweetList[3] => {val:2868, time:3} -> {val:6673, time:6} -> {val:7946, time:7} -> {val:1445, time:8}
	//twitter.PostTweet(4,4822)  // userTweetList[4] => {val:8148, time:4} -> {val:386, time:5} -> {val:4822, time:9}
	//twitter.PostTweet(1,3781)  // userTweetList[1] => {val:6765, time:1} -> {val:3781, time:10}
	//twitter.PostTweet(4,9038)  // userTweetList[4] => {val:8148, time:4} -> {val:386, time:5} -> {val:4822, time:9} -> {val:9038, time:11}
	//twitter.PostTweet(1,9643)  // userTweetList[1] => {val:6765, time:1} -> {val:3781, time:10} -> {val:9643, time:12}
	//twitter.PostTweet(3,5917)  // userTweetList[3] => {val:2868, time:3} -> {val:6673, time:6} -> {val:7946, time:7} -> {val:1445, time:8}-> {val:5917, time:13}
	//twitter.PostTweet(2,8847)   // userTweetList[2] => {val:8847, time:14}

	/**
	userTweetList[1] => {val:6765, time:1} -> {val:3781, time:10} -> {val:9643, time:12}
	userTweetList[2] => {val:8847, time:14}
	userTweetList[3] => {val:2868, time:3} -> {val:6673, time:6} -> {val:7946, time:7} -> {val:1445, time:8} -> {val:5917, time:13}
	userTweetList[4] => {val:8148, time:4} -> {val:386, time:5} -> {val:4822, time:9} -> {val:9038, time:11}
	userTweetList[5] => {val:671, time:2}
	*/

	//twitter.Follow(1,3)
	//twitter.Follow(1,4)
	//twitter.Follow(4,2)
	//twitter.Follow(4,1)
	//twitter.Follow(3,2)
	//twitter.Follow(3,5)
	//twitter.Follow(3,1)
	//twitter.Follow(2,3)
	//twitter.Follow(2,1)
	//twitter.Follow(2,5)
	//twitter.Follow(5,1)
	//twitter.Follow(5,2)
	//
	////f1 := twitter.GetNewsFeed(1)
	//f2 := twitter.GetNewsFeed(2)  // {3,1,5}
	//f3 := twitter.GetNewsFeed(3)
	//f4 := twitter.GetNewsFeed(4)
	//f5 := twitter.GetNewsFeed(5)
	//
	//fmt.Println(f2, f3, f4, f5)
	//fmt.Println(f1, f2, f3, f4, f5)
}

/**
208. 实现 Trie (前缀树)
https://leetcode.cn/problems/implement-trie-prefix-tree/description/
*/

type Trie struct {
	children [26]*Trie
	isEnd    bool
}

func ConstructorTrie() Trie {
	return Trie{}
}

func (t *Trie) Insert(word string) {
	node := t
	for _, char := range word {
		idx := char - 'a'
		if node.children[idx] == nil {
			node.children[idx] = &Trie{}
		}
		node = node.children[idx]
	}
	node.isEnd = true
	return
}

func (t *Trie) SearchPrefix(prefix string) *Trie {
	node := t
	for _, char := range prefix {
		idx := char - 'a'
		if node.children[idx] == nil {
			return nil
		}
		node = node.children[idx]
	}
	return node
}

func (t *Trie) Search(word string) bool {
	node := t.SearchPrefix(word)
	return node != nil && node.isEnd
}

func (t *Trie) StartsWith(prefix string) bool {
	node := t.SearchPrefix(prefix)
	return node != nil
}

type TrieNode struct {
	children [26]*TrieNode
	isEnd    bool
}

func (t *TrieNode) insert(word string) {
	node := t
	for _, char := range word {
		idx := char - 'a'
		if node.children[idx] == nil {
			node.children[idx] = &TrieNode{}
		}
		node = node.children[idx]
	}
	node.isEnd = true
	return
}

func (t *TrieNode) search(word string) bool {
	node := t
	for _, char := range word {
		idx := char - 'a'
		if node.children[idx] == nil {
			return false
		}
		node = node.children[idx]
	}
	return node != nil && node.isEnd
}

type WordDictionary struct {
	trieRoot *TrieNode
}

func ConstructorWordDictionary() WordDictionary {
	return WordDictionary{
		trieRoot: &TrieNode{},
	}
}

func (w *WordDictionary) AddWord(word string) {
	w.trieRoot.insert(word)
}

func (w *WordDictionary) Search(word string) bool {
	var dfs func(idx int, node *TrieNode) bool
	dfs = func(idx int, node *TrieNode) bool {
		if idx == len(word) {
			return node.isEnd
		}
		char := word[idx]
		if char != '.' {
			child := node.children[char-'a']
			if child != nil && dfs(idx+1, child) {
				return true
			}
		} else {
			for i, _ := range node.children {
				child := node.children[i]
				if child != nil && dfs(idx+1, child) {
					return true
				}
			}
		}
		return false
	}
	return dfs(0, w.trieRoot)
}

/*
*
284. 窥视迭代器
https://leetcode.cn/problems/peeking-iterator/
*/

type Iterator struct {
}

func (i *Iterator) hasNext() bool {
	// Returns true if the iteration has more elements.
	return false
}

func (i *Iterator) next() int {
	// Returns the next element in the iteration.
	return -1
}

type PeekingIterator struct {
	iter     *Iterator
	_hasNext bool
	_next    int
}

func ConstructorPeekingIterator(iter *Iterator) *PeekingIterator {
	return &PeekingIterator{
		iter:     iter,
		_hasNext: iter.hasNext(),
		_next:    iter.next(),
	}
}

func (p *PeekingIterator) hasNext() bool {
	return p._hasNext
}

func (p *PeekingIterator) next() int {
	ret := p._next
	p._hasNext = p.iter.hasNext()
	if p.iter.hasNext() {
		p._next = p.iter.next()
	}
	return ret
}

func (p *PeekingIterator) peek() int {
	return p._next
}

/**
304. 二维区域和检索 - 矩阵不可变
https://leetcode.cn/problems/range-sum-query-2d-immutable/
 */
type NumMatrix struct {
	matrixSum [][]int
}

func ConstructorNumMatrix(matrix [][]int) NumMatrix {
	nm := NumMatrix{}
	matrixSum := make([][]int, len(matrix))
	for i:=0; i<len(matrix); i++ {
		matrixSum[i] = make([]int, len(matrix[0]))
	}
	for i:=0; i<len(matrix); i++ {
		for j:=0; j<len(matrix[i]); j++ {
			if i == 0 && j == 0 {
				matrixSum[i][j] = matrix[i][j]
			} else if i == 0 {
				matrixSum[i][j] = matrixSum[i][j-1] + matrix[i][j]
			} else if j == 0 {
				matrixSum[i][j] = matrixSum[i-1][j] + matrix[i][j]
			} else {
				matrixSum[i][j] = matrixSum[i-1][j] + matrixSum[i][j-1] - matrixSum[i-1][j-1] + matrix[i][j]
			}
		}
	}
	nm.matrixSum = matrixSum
	return nm
}

func (n *NumMatrix) SumRegion(row1 int, col1 int, row2 int, col2 int) int {
	/**
	sum(row2, col2) - sum(row2, col1-1) - sum(row1-1, col2) + sum(row1-1, col1-1)
	 */
	sum1, sum2, sum3, sum4 := n.matrixSum[row2][col2], 0, 0, 0

	if col1 >= 1 && row1 >= 1 {
		sum2 = n.matrixSum[row2][col1-1]
		sum3 = n.matrixSum[row1-1][col2]
		sum4 = n.matrixSum[row1-1][col1-1]
	} else if col1 >= 1 {
		sum2 = n.matrixSum[row2][col1-1]
	} else if row1 >= 1 {
		sum3 = n.matrixSum[row1-1][col2]
	}
	return sum1 - sum2 -sum3 + sum4
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
