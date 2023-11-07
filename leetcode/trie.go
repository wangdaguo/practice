package main

/**
字典树节点
 */
type TrieNode struct {
	Nodes map[string]*TrieNode
	isEnd bool
}

func NewTrieNode() *TrieNode {
	tree := new(TrieNode)
	tree.Nodes = make(map[string]*TrieNode, 26)
	return tree
}

func (n *TrieNode) containsKey(s string) bool {
	_, ok := n.Nodes[s]
	return ok
}

func (n *TrieNode) get(s string) *TrieNode {
	return n.Nodes[s]
}

func (n *TrieNode) put(s string, node *TrieNode) {
	n.Nodes[s] = node
}

func (n *TrieNode) setEnd()  {
	n.isEnd = true
}

/**
字典树
 */
type Tree struct {
	root *TrieNode
}

func NewTree() *Tree {
	tree := new(Tree)
	node := NewTrieNode()
	tree.root = node
	return tree
}

func (tree *Tree) insert(word string)  {
	node := tree.root
	for i:=0; i<len(word); i++ {
		if !node.containsKey(word[i:i+1]) {
			node.put(word[i:i+1], NewTrieNode())
		}
		node = node.get(word[i:i+1])
	}
	node.setEnd()
}

func (tree *Tree) searchPrefix(word string) *TrieNode {
	node := tree.root
	for i := 0; i < len(word); i++ {
		curChar := word[i:i+1]
		if node.containsKey(curChar) {
			node = node.get(curChar)
		} else {
			return node
		}
	}
	return node
}

func (tree *Tree) search(word string) bool {
	node := tree.searchPrefix(word)
	return node != nil && node.isEnd
}


