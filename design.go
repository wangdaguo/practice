package main

func main() {

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
