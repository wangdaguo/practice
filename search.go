package main

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

func main()  {
	//grid := [][]int{
	//	{0,0,1,0,0,0,0,1,0,0,0,0,0},
	//	{0,0,0,0,0,0,0,1,1,1,0,0,0},
	//	{0,1,1,0,1,0,0,0,0,0,0,0,0},
	//	{0,1,0,0,1,1,0,0,1,0,1,0,0},
	//	{0,1,0,0,1,1,0,0,1,1,1,0,0},
	//	{0,0,0,0,0,0,0,0,0,0,1,0,0},
	//	{0,0,0,0,0,0,0,1,1,1,0,0,0},
	//	{0,0,0,0,0,0,0,1,1,0,0,0,0},
	//}
	//r := maxAreaOfIsland1(grid)
	//fmt.Println(r)
	//

	//isConnected := [][]int{
	//	{1,0,0,1},	//0,0|0,3  1
	//	{0,1,1,0},  //1,1|1,2  1
	//	{0,1,1,1},	//2,1|2,2|2,3   1
	//	{1,0,1,1},	//3,0|3,2|3,3
	//}
	//r := findCircleNum123(isConnected)
	//fmt.Println(r)
	//return

	//r := findCircleNumByUnionFind(isConnected)
	//fmt.Println(r)

	//r := findCircleNum(isConnected)
	//fmt.Println(r)

	//nums := []int{0, 0, 2, 1, 3, 2, 4}
	//uf := NewUnionFind(nums)
	//fmt.Println(uf)
	//
	//uf.union(2, 6)
	//fmt.Println(uf)

	//heights := [][]int{
	//	{1,2,3},
	//	{8,9,4},
	//	{7,6,5},
	//}

	//[[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]
	//r := pacificAtlantic(heights)

	//r := permute1([]int{1,2,3})

	//r := combine(3, 2)

	//r := permuteUnique([]int{0,1,0,0,9})

	//r := combinationSum([]int{2,3,6,7}, 7)
	//r := combinationSum3([]int{10,1,2,7,6,1,5}, 8)
	//r := subsets([]int{1,2,3})
	//r := subsetsWithDup([]int{2,2,2})

	//board := [][]byte{
	//	{'A', 'B', 'C', 'E'},
	//	{'S', 'F', 'C', 'S'},
	//	{'A', 'D', 'E', 'E'},
	//}
	//r := exist(board, "ABCCED")
	//r := solveNQueens(4)

	//aList := make([]int, 0)
	//fmt.Printf("main alist pointer is %p\n", aList)
	//tL(aList)
	//fmt.Println(aList)
	//return

	//grid := [][]int{
	//	{1,1,1,1,1},
	//	{1,0,0,0,1},
	//	{1,0,1,0,1},
	//	{1,0,0,0,1},
	//	{1,1,1,1,1},
	//}
	//r := shortestBridge(grid)
	//var beginWord, endWord, wordList = "hit", "cog", []string{"hot","dot","dog","lot","log","cog"}
	//var beginWord, endWord, wordList = "aaaaa", "uuuuu", []string{"aaaaa","waaaa","wbaaa","xaaaa","xbaaa","bbaaa","bbwaa","bbwba","bbxaa","bbxba","bbbba","wbbba","wbbbb","xbbba","xbbbb","cbbbb","cwbbb","cwcbb","cxbbb","cxcbb","cccbb","cccwb","cccwc","cccxb","cccxc","ccccc","wcccc","wdccc","xcccc","xdccc","ddccc","ddwcc","ddwdc","ddxcc","ddxdc","ddddc","wdddc","wdddd","xdddc","xdddd","edddd","ewddd","ewedd","exddd","exedd","eeedd","eeewd","eeewe","eeexd","eeexe","eeeee","weeee","wfeee","xeeee","xfeee","ffeee","ffwee","ffwfe","ffxee","ffxfe","ffffe","wfffe","wffff","xfffe","xffff","gffff","gwfff","gwgff","gxfff","gxgff","gggff","gggwf","gggwg","gggxf","gggxg","ggggg","wgggg","whggg","xgggg","xhggg","hhggg","hhwgg","hhwhg","hhxgg","hhxhg","hhhhg","whhhg","whhhh","xhhhg","xhhhh","ihhhh","iwhhh","iwihh","ixhhh","ixihh","iiihh","iiiwh","iiiwi","iiixh","iiixi","iiiii","wiiii","wjiii","xiiii","xjiii","jjiii","jjwii","jjwji","jjxii","jjxji","jjjji","wjjji","wjjjj","xjjji","xjjjj","kjjjj","kwjjj","kwkjj","kxjjj","kxkjj","kkkjj","kkkwj","kkkwk","kkkxj","kkkxk","kkkkk","wkkkk","wlkkk","xkkkk","xlkkk","llkkk","llwkk","llwlk","llxkk","llxlk","llllk","wlllk","wllll","xlllk","xllll","mllll","mwlll","mwmll","mxlll","mxmll","mmmll","mmmwl","mmmwm","mmmxl","mmmxm","mmmmm","wmmmm","wnmmm","xmmmm","xnmmm","nnmmm","nnwmm","nnwnm","nnxmm","nnxnm","nnnnm","wnnnm","wnnnn","xnnnm","xnnnn","onnnn","ownnn","owonn","oxnnn","oxonn","ooonn","ooown","ooowo","oooxn","oooxo","ooooo","woooo","wpooo","xoooo","xpooo","ppooo","ppwoo","ppwpo","ppxoo","ppxpo","ppppo","wpppo","wpppp","xpppo","xpppp","qpppp","qwppp","qwqpp","qxppp","qxqpp","qqqpp","qqqwp","qqqwq","qqqxp","qqqxq","qqqqq","wqqqq","wrqqq","xqqqq","xrqqq","rrqqq","rrwqq","rrwrq","rrxqq","rrxrq","rrrrq","wrrrq","wrrrr","xrrrq","xrrrr","srrrr","swrrr","swsrr","sxrrr","sxsrr","sssrr","ssswr","sssws","sssxr","sssxs","sssss","wssss","wtsss","xssss","xtsss","ttsss","ttwss","ttwts","ttxss","ttxts","tttts","wttts","wtttt","xttts","xtttt","utttt","uwttt","uwutt","uxttt","uxutt","uuutt","uuuwt","uuuwu","uuuxt","uuuxu","uuuuu","zzzzz","zzzzy","zzzyy","zzyyy","zzyyx","zzyxx","zzxxx","zzxxw","zzxww","zzwww","zzwwv","zzwvv","zzvvv","zzvvu","zzvuu","zzuuu","zzuut","zzutt","zzttt","zztts","zztss","zzsss","zzssr","zzsrr","zzrrr","zzrrq","zzrqq","zzqqq","zzqqp","zzqpp","zzppp","zzppo","zzpoo","zzooo","zzoon","zzonn","zznnn","zznnm","zznmm","zzmmm","zzmml","zzmll","zzlll","zzllk","zzlkk","zzkkk","zzkkj","zzkjj","zzjjj","zzjji","zzjii","zziii","zziih","zzihh","zzhhh","zzhhg","zzhgg","zzggg","zzggf","zzgff","zzfff","zzffe","zzfee","zzeee","zzeed","zzedd","zzddd","zzddc","zzdcc","zzccc","zzccz","azccz","aaccz","aaacz","aaaaz","uuuzu","uuzzu","uzzzu","zzzzu"}
	//r := findLadders(beginWord, endWord, wordList)
	//fmt.Println(r)
	//board := [][]byte{
	//	{'X', 'X', 'X', 'X'},
	//	{'X', 'O', 'O', 'X'},
	//	{'X', 'X', 'O', 'X'},
	//	{'X', 'O', 'X', 'X'},
	//}
	//solve(board)
	//fmt.Println(board)
	//node5 := &TreeNode{
	//	Val:   5,
	//	Left:  nil,
	//	Right: nil,
	//}
	//node2 := &TreeNode{
	//	Val:   2,
	//	Left:  nil,
	//	Right: node5,
	//}
	//node3 := &TreeNode{
	//	Val:   3,
	//	Left:  nil,
	//	Right: nil,
	//}
	//root := &TreeNode{
	//	Val:   1,
	//	Left:  node2,
	//	Right: node3,
	//}
	//r := binaryTreePaths(root)
	//r := combinationSum2([]int{10,1,2,7,6,1,5}, 8)

	//n := 4
	//edges := [][]int{
	//	{1,0},
	//	{1,2},
	//	{1,3},
	//}
	///**
	// */
	//r := findMinHeightTrees(n, edges)
	//fmt.Println(r)
	return
}

/**
310. 最小高度树
https://leetcode.cn/problems/minimum-height-trees/
 */
func findMinHeightTrees(n int, edges [][]int) []int {
	if n == 0 || len(edges) == 0 {
		return []int{0}
	}
	//nodeList := make(map[int][]int)
	//for i:=0; i<n; i++ {
	//	nodeList[i] = make([]int, 0)
	//}
	graph := make([][]int, n)
	for i:=0; i<n; i++ {
		graph[i] = make([]int, n)
	}
	for i:=0; i<len(edges); i++ {
		for j:=0; j<len(edges[0]); j+=2 {
			//nodeList[edges[i][j]] = append(nodeList[edges[i][j]], edges[i][j+1])
			//nodeList[edges[i][j+1]] = append(nodeList[edges[i][j+1]], edges[i][j])
			graph[edges[i][j]][edges[i][j+1]] = 1
			graph[edges[i][j+1]][edges[i][j]] = 1
		}
	}
	rMp := make(map[int]int)
	for i:=0; i<n; i++ {
		tmp := make([]int, 0)
		var max int
		for j:=0; j<n; j++ {
			if graph[i][j] == 0 {
				continue
			}
			Dfs3(edges, graph,0, i, i, j, &tmp)
			max = findMax(tmp)
		}
		rMp[i] = max
	}
	//fmt.Println(rMp)
	//return []int{}

	mp := make(map[int][]int)
	for node, l := range rMp {
		mp[l] = append(mp[l], node)
	}
	minLen := n
	for key, _ := range mp {
		if key < minLen {
			minLen = key
		}
	}
	return mp[minLen]
}

func findMax(list []int) int {
	if len(list) < 1 {
		return 0
	}
	r := list[0]
	for i:=1; i<len(list); i++ {
		if r < list[i] {
			r = list[i]
		}
	}
	return r
}

/**
0 1 0 0
1 0 1 1
0 1 0 0
0 1 0 0
*/
func Dfs3(edges [][]int, graph [][]int, cnt, root, x, y int, t *[]int) {
	if graph[x][y] == 0 {
		if y == len(graph[x])-1 {
			*t = append(*t, cnt)
			//(*t)[0] = cnt
		}
		return
	}
	graph[x][y] = 0
	graph[y][x] = 0
	for i:=0; i<len(graph[y]); i++ {
		Dfs3(edges, graph, cnt+1, root, y, i, t)
	}
	graph[x][y] = 1
	graph[y][x] = 1
}

func combinationSum2(candidates []int, target int) [][]int {
	var r [][]int
	if len(candidates) < 1 {
		return r
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i] < candidates[j] {
			return true
		}
		return false
	})
	path, level := make([]int, 0), 0
	btcs(candidates, path, level, target, &r)
	return r
}

func btcs(nums []int, path []int, level, target int, r *[][]int) {
	if sumList(path) > target {
		return
	}
	if sumList(path) == target {
		var tmp []int
		tmp = append(tmp, path...)
		*r = append(*r, tmp)
		return
	}
	for i:=level; i<len(nums); i++ {
		if i > level && nums[i] == nums[i-1] {
			continue
		}
		path = append(path, nums[i])
		btcs(nums, path, i+1, target, r)
		path = path[:len(path)-1]
	}
}

func sumList(list []int) int {
	var r int
	for _, v := range list {
		r += v
	}
	return r
}

/**
47. 全排列 II
https://leetcode.cn/problems/permutations-ii/
*/
func permuteUnique(nums []int) [][]int {
	r := make([][]int, 0)
	if len(nums)  < 1 {
		return r
	}
	sort.Slice(nums, func(i, j int) bool {
		if nums[i] < nums[j] {
			return true
		}
		return false
	})
	check, path := make(map[int]bool), make([]int, 0)
	ptpu2(nums, check, path, &r)
	return r
}

func ptpu2(nums []int, check map[int]bool, path []int, r *[][]int) {
	if len(path) == len(nums) {
		var tmp []int
		tmp = append(tmp, path...)
		*r = append(*r, tmp)
		return
	}
	for i:=0; i<len(nums); i ++ {
		if check[i] {
			continue
		}
		if i>0 && nums[i-1] == nums[i] && check[i-1] == false {
			continue
		}
		check[i] = true
		path = append(path, nums[i])
		ptpu2(nums, check, path, r)
		check[i] = false
		path = path[:len(path)-1]
	}
}

/**
257. 二叉树的所有路径
https://leetcode.cn/problems/binary-tree-paths/
 */
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func binaryTreePaths(root *TreeNode) []string {
	if root == nil {
		return []string{}
	}
	r := make([]string, 0)
	Dfs2(root, []string{}, &r)
	return r
}

func Dfs2(node *TreeNode, t []string, r *[]string) {
	if node == nil {
		return
	}
	if node.Left == nil && node.Right == nil {
		t = append(t, strconv.Itoa(node.Val))
		*r = append(*r, strings.Join(t, "->"))
		return
	}
	t = append(t, strconv.Itoa(node.Val))
	Dfs2(node.Left, t, r)
	Dfs2(node.Right, t, r)
	return
}

/**
130. 被围绕的区域
https://leetcode.cn/problems/surrounded-regions/
 */
func solve(board [][]byte)  {
	if len(board) < 1 {
		return
	}
	// 左侧
	for i:=0; i<len(board); i++ {
		Dfs11(board, i, 0)
	}
	// 右侧
	for i:=0; i<len(board); i++ {
		Dfs11(board, i, len(board[0])-1)
	}
	// 上侧
	for i:=0; i<len(board[0]); i++ {
		Dfs11(board, 0, i)
	}
	// 下侧
	for i:=0; i<len(board[0]); i++ {
		Dfs11(board, len(board)-1, i)
	}
	for i:=0; i<len(board); i++ {
		for j:=0; j<len(board[0]); j++ {
			if board[i][j] == 'O' {
				board[i][j] = 'X'
			}  else if  board[i][j] == 'Y' {
				board[i][j] = 'O'
			}
		}
	}
	return
}

func Dfs11(board [][]byte, x, y int)  {
	if x < 0 || x >= len(board) || y < 0 || y >= len(board[0]) || board[x][y] == 'X' || board[x][y] == 'Y' {
		return
	}
	board[x][y] = 'Y'
	for i:=0; i<4; i ++ {
		Dfs11(board, x+d[i], y+d[i+1])
	}
}

/**
126. 单词接龙 II
https://leetcode.cn/problems/word-ladder-ii/
 */
func findLadders(beginWord string, endWord string, wordList []string) [][]string {
	if len(beginWord) != len(endWord) {
		return [][]string{}
	}
	wordMap, next, ans := make(map[string]struct{}), make(map[string][]string), make([][]string, 0)
	for _, v := range wordList {
		wordMap[v] = struct{}{}
	}
	if _, ok := wordMap[endWord]; !ok {
		return [][]string{}
	}
	q1 := make(map[string]struct{})
	q1[beginWord] = struct{}{}
	q2 := make(map[string]struct{})
	q2[endWord] = struct{}{}

	var found, reversed  bool
	delete(wordMap, beginWord)
	delete(wordMap, endWord)

	for len(q1) > 0 {
		q := make(map[string]struct{})
		for k, _ := range q1 {
			str := k
			for i:=0; i<len(str); i++ {
				char := str[i]
				for j:=0; j<26; j++ {
					str = replaceChar(str, byte('a'+j), i)
					if _, ok := q2[str]; ok {
						found = true
						if reversed {
							next[str] = append(next[str], k)
						} else {
							next[k] = append(next[k], str)
						}
					}
					if _, ok := wordMap[str]; ok {
						if reversed {
							next[str] = append(next[str], k)
						} else {
							next[k] = append(next[k], str)
						}
						q[str] = struct{}{}
					}
				}
				str = replaceChar(str, char, i)
			}
		}
		if found {
			break
		}
		for k := range q {
			delete(wordMap, k)
		}
		if len(q) <= len(q2) {
			q1 = q
		} else {
			q1 = q2
			q2 = q
			reversed = !reversed
		}
	}
	if found {
		path := []string{beginWord}
		bt11(beginWord, endWord, next, path, &ans)
	}
	return ans
}

func bt11(beginWord, endWord string, next map[string][]string, path []string, ans *[][]string) {
	if beginWord == endWord {
		tmp := make([]string, 0)
		tmp = append(tmp, path...)
		*ans = append(*ans, tmp)
		return
	}
	for _, v := range next[beginWord] {
		path = append(path, v)
		bt11(v, endWord, next, path, ans)
		path = path[:len(path)-1]
	}
}

func replaceChar(str string, ch byte, n int) string {
	byteList := []byte(str)
	if n >= len(byteList) {
		return str
	}
	byteList[n] = ch
	return string(byteList)
}

/**
934. 最短的桥
DFS找到一个岛所有点，然后BFS一层层外扩找到另一个岛
https://leetcode.cn/problems/shortest-bridge/
 */
func shortestBridge(grid [][]int) int {
	if len(grid) < 1 {
		return 0
	}
	tag, queue := false, NewQueue()
	for i:=0; i<len(grid); i++ {
		if tag {
			break
		}
		for j:=0; j<len(grid[0]); j++ {
			if grid[i][j] == 1 {
				tag = true
				fillOneIsland(grid, queue, i, j)
				break
			}
		}
	}
	fmt.Println(grid)

	var r int
	for !queue.isEmpty() {
		size := queue.Size()
		for size > 0 {
			size --
			point := queue.pop()
			for c:=0; c<4; c++ {
				if point.X + d[c] < 0 || point.X + d[c] >= len(grid) || point.Y + d[c+1] < 0 || point.Y + d[c+1] >= len(grid[0]) ||
					grid[point.X + d[c]][point.Y+d[c+1]] == 2 {
					continue
				}
				if grid[point.X + d[c]][point.Y+d[c+1]] == 1 {
					return r
				}
				if grid[point.X + d[c]][point.Y+d[c+1]] == 0 {
					grid[point.X + d[c]][point.Y+d[c+1]] = 2
					queue.push(NewPoint(point.X + d[c], point.Y+d[c+1]))
				}
			}
		}
		r ++
	}
	return 0
}

var d = []int{-1, 0, 1, 0, -1}

func fillOneIsland(grid [][]int, queue *Queue, i, j int) {
	if i < 0 || i >= len(grid) || j < 0 || j >= len(grid[0]) || grid[i][j] == 0 || grid[i][j] == 2 {
		return
	}
	grid[i][j] = 2
	queue.push(NewPoint(i, j))
	for c:=0; c<4; c++ {
		fillOneIsland(grid, queue, i+d[c], j+d[c+1])
	}
}

/**
51. N 皇后
https://leetcode.cn/problems/n-queens/
 */
func solveNQueens(n int) [][]string {
	if n <=0 {
		return [][]string{}
	}
	ans, i, board, col, ldiag, rdiag := make([][]string, 0), 0, make([][]string, n), make([]bool, n), make([]bool, 2*n), make([]bool, 2*n)
	for c:=0; c<n; c++ {
		var t []string
		for ci:=0; ci<n; ci++ {
			t = append(t, ".")
		}
		board[c] = t
	}
	bt(&ans, n, i, board, col, ldiag, rdiag)
	return ans
}

func bt(ans *[][]string , n, cnt int, board [][]string, col, ldiag, rdiag []bool)  {
	if cnt == n {
		var tmp []string
		for _, v := range board {
			tmp = append(tmp,  strings.Join(v, ""))
		}
		*ans = append(*ans, tmp)
		return
	}
	for i:=0; i<n; i++ {
		if col[i] || ldiag[n-1-cnt+i] || rdiag[i+cnt] {
			continue
		}
		col[i], ldiag[n-1-cnt+i], rdiag[i+cnt] = true, true, true
		board[cnt][i] = "Q"
		bt(ans, n, cnt+1, board, col, ldiag, rdiag)
		board[cnt][i] = "."
		col[i], ldiag[n-1-cnt+i], rdiag[i+cnt] = false, false, false
	}
	return
}

/**
79. 单词搜索
https://leetcode.cn/problems/word-search/
 */
func exist(board [][]byte, word string) bool {
	if len(board) < 1 {
		return false
	}
	if len(word) < 1 {
		return true
	}
	r , byteList, check, iStart, jStart, cnt := false, make([]byte, 0), make([][]int, 0), 0, 0, 0
	for i:=0; i<len(board); i++ {
		check = append(check, make([]int, len(board[0])))
	}
	for i:=iStart; i<len(board); i++ {
		for j := jStart; j < len(board[0]); j++ {
			backTrace8(board, byteList, check, i, j, cnt, word, &r)
		}
	}
	return r
}

func backTrace8(board [][]byte, byteList []byte, check [][]int, iStart, jStart, cnt int, word string, r *bool) {
	if iStart < 0 || iStart >= len(board) || jStart < 0 || jStart >= len(board[0]) {
		return
	}
	if check[iStart][jStart] == 1 || *r == true || board[iStart][jStart] != word[cnt]{
		return
	}
	if cnt == len(word)-1 {
		*r = true
		return
	}
	check[iStart][jStart] = 1
	for k := 0; k < 4; k++ {
		backTrace8(board, byteList, check, iStart+direction[k], jStart+direction[k+1], cnt+1, word, r)
	}
	check[iStart][jStart] = 0
	return
}

/**
https://leetcode.cn/problems/subsets-ii/
 */
func subsetsWithDup(nums []int) [][]int {
	if len(nums) < 1 {
		return [][]int{}
	}
	sort.Slice(nums, func(i, j int) bool {
		if nums[i] < nums[j] {
			return true
		}
		return false
	})
	//r, path, check, level := make([][]int, 0), make([]int, 0), make([]bool, len(nums)), 0
	r, path, level := make([][]int, 0), make([]int, 0), 0
	backTrace6(nums, path, level, &r)
	return r
}

func backTrace6(nums, path []int, level int, r *[][]int) {
	tmp := make([]int, 0)
	tmp = append(tmp, path...)
	*r = append(*r, tmp)

	for i := level; i < len(nums); i++ {
		if i > level && nums[i] == nums[i-1]  {
			continue
		}
		path = append(path, nums[i])
		backTrace6(nums, path, i+1, r)
		path = path[:len(path)-1]
	}
}

func subsets(nums []int) [][]int {
	if len(nums) < 1 {
		return [][]int{}
	}
	r, level, path := make([][]int, 0), 0, make([]int, 0)
	backTrace5(nums, path, level, &r)
	r = append(r, []int{})
	return r
}

func backTrace5(nums, path []int, level int, r *[][]int) {
	if len(path) > 0 {
		tmp := make([]int, 0)
		tmp = append(tmp, path...)
		*r = append(*r, tmp)
		if len(path) == len(nums) {
			return
		}
	}
	for i:=level; i<len(nums); i++ {
		path = append(path, nums[i])
		backTrace5(nums, path, i+1, r)
		path = path[:len(path)-1]
	}
}

/**
https://leetcode.cn/problems/combination-sum/
 */
func combinationSum(candidates []int, target int) [][]int {
	if len(candidates) < 1 && target > 0 {
		return [][]int{}
	}
	r, path := make([][]int, 0), make([]int, 0)
	backTrace3(candidates, path, 0, target, &r)
	return r
}

func backTrace3(nums, path []int, level, target int, r *[][]int)  {
	if sumInt(path) > target {
		return
	}
	if sumInt(path) == target {
		tmp := make([]int, 0)
		tmp = append(tmp, path...)
		*r = append(*r, tmp)
		return
	}
	for i:=level; i<len(nums); i++ {
		path = append(path, nums[i])
		backTrace3(nums, path, i, target, r)
		path = path[:len(path)-1]
	}
}

func sumInt(path []int) int {
	var sum int
	for _, val := range path {
		sum += val
	}
	return sum
}

func combine(n int, k int) [][]int {
	if n < k {
		return [][]int{}
	}
	nums, path := make([]int, 0), make([]int, 0)
	for i:=1; i<=n; i++ {
		nums = append(nums, i)
	}
	level := 0
	r := make([][]int, 0)
	backTrack1(nums, path, level, k, &r)
	return r
}

func backTrack1(nums, path []int, level, k int, r *[][]int)  {
	if len(path) == k {
		tmp := make([]int, 0)
		tmp = append(tmp, path...)
		*r = append(*r, tmp)
		return
	}
	for i:=level; i<len(nums); i++ {
		path = append(path, nums[i])
		fmt.Printf("  递归之前 => %v\n", path)
		backTrack1(nums, path, i+1, k, r)
		path = path[0:len(path)-1]
		fmt.Printf("递归之后 => %v\n", path)
	}
}

func permute1(nums []int) [][]int {
	if len(nums) < 1 {
		return [][]int{}
	}
	r, check, path := make([][]int, 0), make([]bool, len(nums)), make([]int, 0)
	backTraceT(nums, path, check, &r)
	return r
}

func backTraceT(nums, path []int, check []bool, r *[][]int)  {
	if len(path) == len(nums) {
		var tmp []int
		tmp = append(tmp, path...)
		*r = append(*r, tmp)
		return
	}
	for i:=0; i<len(nums); i++ {
		if check[i] {
			continue
		}
		check[i] = true
		path = append(path, nums[i])
		backTraceT(nums, path, check, r)
		check[i] = false
		path = path[0:len(path)-1]
	}
}

func pacificAtlantic(heights [][]int) [][]int {
	if len(heights) < 1 {
		return [][]int{}
	}
	pacific := make([][]bool, len(heights))
	atlantic := make([][]bool, len(heights))
	for i := range pacific {
		pacific[i] = make([]bool, len(heights[0]))
		atlantic[i] = make([]bool, len(heights[0]))
	}
	for i:=0; i<len(heights); i++ {
		dfs(heights, i, 0, &pacific)
	}
	for j:=0; j<len(heights[0]); j++ {
		dfs(heights, 0, j, &pacific)
	}
	for i:=0; i<len(heights); i++ {
		dfs(heights, i, len(heights[0])-1, &atlantic)
	}
	for j:=0; j<len(heights[0]); j++ {
		dfs(heights, len(heights)-1, j, &atlantic)
	}

	r := make([][]int, 0)
	for i:=0; i<len(pacific); i++ {
		for j:=0; j<len(pacific[0]); j ++ {
			if pacific[i][j] && atlantic[i][j] {
				r = append(r, []int{i, j})
			}
		}
	}
	return r
}

func dfs(heights [][]int, i, j int, ocean *[][]bool)  {

	//if (*ocean)[i][j] {
	//	return
	//}
	//(*ocean)[i][j] = true
	//for k := 0; k < 4; k++ {
	//	x := i + direction[k]
	//	y := j + direction[k+1]
	//	if x >= 0 && x < len(heights) && y >= 0 && y < len(heights[0]) && heights[x][y] >= heights[i][j] {
	//		dfs(heights, x, y, ocean)
	//	}
	//}

	if (*ocean)[i][j] {
		return
	}
	(*ocean)[i][j] = true
	stack := NewStack()
	stack.push(NewPoint(i, j))
	for !stack.isEmpty() {
		p := stack.pop()
		for k := 0; k < 4; k++ {
			x := p.X + direction[k]
			y := p.Y + direction[k+1]
			if x >= 0 && x < len(heights) && y >= 0 && y < len(heights[0]) && heights[x][y] >= heights[p.X][p.Y] {
				stack.push(NewPoint(x, y))
				(*ocean)[x][y] = true
			}
		}
	}
	return
}

//func dfs1(heights [][]int, i, j int, ocean [][]bool)  {
//	stack := NewStack()
//	for i:=0; i<len(heights); i++ {
//		for j:=0; j<len(heights[0]); j++ {
//			if !ocean[i][j] {
//				ocean[i][j] = true
//				stack.push(NewPoint(i, j))
//			}
//			for !stack.isEmpty() {
//				p := stack.pop()
//				for k:=0; k<4; k++ {
//					x := p.X + direction[k]
//					y := p.Y + direction[k+1]
//					if x >=0 && x < len(heights) && y>=0 && y<len(heights[0]) && heights[x][y] > heights[p.X][p.Y] {
//						dfs(heights, x, y, ocean)
//					}
//				}
//			}
//		}
//	}
//}

func findCircleNumByStack1(isConnected [][]int) int {
	if len(isConnected) < 1 {
		return 0
	}
	vis := make([]bool, len(isConnected))
	cnt := 0
	var dfs func(int)
	dfs = func(i int) {
		vis[i] = true
		for j:=0; j<len(isConnected[0]); j++ {
			if isConnected[i][j] == 1 && !vis[j] {
				dfs(j)
			}
		}
	}
	for i:=0; i<len(vis); i++ {
		if !vis[i] {
			cnt ++
			dfs(i)
		}
	}
	return cnt
}

func findCircleNum123(isConnected [][]int) int {
	if len(isConnected) < 1 {
		return 0
	}
	vis := make([]bool, len(isConnected))
	queue := make([]int, 0)
	cnt := 0
	for i:=0; i<len(vis); i++ {
		if !vis[i] {
			queue = append(queue, i)
			cnt ++
			for len(queue) > 0 {
				j := queue[0]
				queue = queue[1:]
				vis[j] = true
				for t:=0; t<len(isConnected[0]); t++ {
					if isConnected[j][t] == 1 && !vis[t] {
						queue = append(queue, t)
					}
				}
			}
		}
	}
	return cnt
}

func findCircleNumByStack(isConnected [][]int) int {
	if len(isConnected) < 1 {
		return 0
	}
	stack := NewStack()
	cnt := 0
	for i:=0; i<len(isConnected); i++ {
		for j:=0; j<len(isConnected[0]);j ++ {
			if isConnected[i][j] == 1 {
				stack.push(NewPoint(i, j))
				cnt ++
			}
			for !stack.isEmpty() {
				p := stack.pop()
				isConnected[p.X][p.Y] = 0
				for t:=0;t<len(isConnected[0]);t++{
					if isConnected[p.Y][t] == 1 {
						stack.push(NewPoint(p.Y, t))
					}
				}
			}
		}
	}
	return cnt
}

func findCircleNumByUnionFind(isConnected [][]int) int {
	arr := make([]int, len(isConnected))
	for i:=0; i<len(arr); i++ {
		arr[i] = i
	}
	cnt := len(isConnected)
	for i:=0; i<len(isConnected); i++ {
		for j:=i+1; j<len(isConnected[0]); j++ {
			if isConnected[i][j] == 1 && Union(arr, i, j) {
				cnt --
			}
		}
	}
	return cnt
}

func Union(arr []int, i, j int) bool {
	fatherI := findFather(arr, i)
	fatherJ := findFather(arr, j)
	if fatherI != fatherJ {
		arr[fatherJ] = arr[fatherI]
		return true
	}
	return false
}

func findFather(arr []int, i int) int {
	for i != arr[i] {
		i = arr[i]
	}
	return i
}


type UnionFind struct {
	Val []int
	Size map[int]int
}

func NewUnionFind(val []int) *UnionFind {
	uf := &UnionFind{
		Val: val,
	}
	uf.Size = make(map[int]int)
	for i:=0; i<len(val); i++ {
		uf.findRoot(i)
	}
	for i:=0; i<len(val); i++ {
		uf.Size[uf.Val[i]] ++
	}
	return uf
}

func (uf *UnionFind) union(x, y int)  {
	if  uf.connected(x, y) {
		return
	}
	rootX := uf.findRoot(x)
	rootY := uf.findRoot(y)
	if uf.Size[rootX] > uf.Size[rootY] {
		uf.Val[rootY] = rootX
		uf.Size[rootX] += uf.Size[rootY]
		delete(uf.Size, rootY)
		return
	}
	uf.Val[rootX] = rootY
	uf.Size[rootY] += uf.Size[rootX]
	delete(uf.Size, rootX)
	return
}

func (uf *UnionFind) connected(x, y int) bool {
	rootX := uf.findRoot(x)
	rootY := uf.findRoot(y)
	return rootX == rootY
}

func (uf *UnionFind) findRoot(x int) int {
	for x != uf.Val[x] {
		uf.Val[x] = uf.Val[uf.Val[x]]
		x = uf.Val[x]
	}
	return x
}

type Point struct {
	X int
	Y int
}

func NewPoint(x, y int) *Point {
	return &Point{
		X: x,
		Y: y,
	}
}

type Queue struct {
	Data []*Point
}

func NewQueue() *Queue {
	data := make([]*Point, 0)
	return &Queue{Data:data}
}

func (queue *Queue) pop() *Point {
	r := queue.Data[0]
	queue.Data = queue.Data[1:]
	return r
}

func (queue *Queue) push(p *Point) {
	if queue.isEmpty() {
		queue.Data = append(queue.Data, p)
		return
	}
	queue.Data = append(queue.Data, p)
	return
}

func (queue *Queue) isEmpty() bool {
	return len(queue.Data) == 0
}

func (queue *Queue) Size() int {
	return len(queue.Data)
}

type Stack struct {
	Data []*Point
}

func NewStack() *Stack {
	data := make([]*Point, 0)
	return &Stack{Data:data}
}

func (stack *Stack) top() *Point {
	return stack.Data[0]
}

func (stack *Stack) pop() *Point {
	r := stack.Data[0]
	stack.Data = stack.Data[1:]
	return r
}

func (stack *Stack) push(p *Point) {
	if stack.isEmpty() {
		stack.Data = append(stack.Data, p)
		return
	}
	stack.Data = append([]*Point{p}, stack.Data...)
	//copy(stack.Data[1:], stack.Data[0:])
	//stack.Data[0] = p
	return
}

func (stack *Stack) isEmpty() bool {
	return len(stack.Data) == 0
}

func (stack *Stack) Size() int {
	return len(stack.Data)
}

var direction = []int{-1, 0, 1, 0, -1}

/**
695. 岛屿的最大面积
https://leetcode.cn/problems/max-area-of-island/description/
*/
func maxAreaOfIsland(grid [][]int) int {
	stack := NewStack()
	maxArea, tmpArea := 0, 0
	for i:=0; i<len(grid); i++ {
		for j:=0; j<len(grid[0]); j++ {
			if grid[i][j] == 1 {
				stack.push(NewPoint(i, j))
				tmpArea = 1
				grid[i][j] = 0
			}
			for !stack.isEmpty() {
				p := stack.pop()
				for k:=0; k<4; k++ {
					x := p.X + direction[k]
					y := p.Y + direction[k+1]
					if x >=0 && x < len(grid) && y >=0 && y < len(grid[0]) && grid[x][y] == 1 {
						grid[x][y] = 0
						stack.push(NewPoint(x, y))
						tmpArea ++
					}
				}
			}
			maxArea = maxX(maxArea, tmpArea)
		}
	}
	return maxArea
}

func maxX(x, y int) int {
	if x >= y {
		return x
	}
	return y
}

func maxAreaOfIsland1(grid [][]int) int {
	var max int
	for i:=0; i<len(grid); i++ {
		for j:=0; j<len(grid[0]); j++ {
			if grid[i][j] == 1 {
				max = maxX(max, maxAreaOfIslandImpl(grid, i, j))
			}
		}
	}
	return max
}

func maxAreaOfIslandImpl(grid [][]int, i, j int) int {
	if i < 0 || j < 0 || i > len(grid)-1 || j > len(grid[0])-1 || grid[i][j] == 0 {
		return 0
	}
	grid[i][j] = 0
	return 1 + maxAreaOfIslandImpl(grid, i+1, j) + maxAreaOfIslandImpl(grid, i-1, j) + maxAreaOfIslandImpl(grid, i, j-1) +
		maxAreaOfIslandImpl(grid, i, j+1)
}

/**
547. 省份数量
https://leetcode.cn/problems/number-of-provinces/description/
 */
func findCircleNum(isConnected [][]int) int {
	if len(isConnected) < 1 {
		return 0
	}
	isVisited := make([]bool, len(isConnected))
	var r int
	for i := 0; i < len(isConnected); i++ {
		if isVisited[i] == false {
			isVisited[i] = true
			findDFS(isConnected, i, isVisited)
			r ++
		}
	}
	return r
}

func findDFS(isConnected [][]int, i int, isVisited []bool) {
	for j:=0; j<len(isConnected[0]); j++ {
		if isConnected[i][j] == 1 && isVisited[j] == false {
			isVisited[j] = true
			findDFS(isConnected, j, isVisited)
		}
	}
}

func pacificAtlantic1(heights [][]int) [][]int {
	if len(heights) < 1 {
		return [][]int{}
	}
	pacific, atlantic := make([][]bool, len(heights)), make([][]bool, len(heights))
	for i:=0; i<len(heights); i++ {
		pacific[i] = make([]bool, len(heights[0]))
		atlantic[i] = make([]bool, len(heights[0]))
	}

	for i:=0; i<len(heights); i++ {
		flowDFS(heights, i, 0, pacific)
		flowDFS(heights, i, len(heights[0])-1, atlantic)
	}
	for j:=0; j<len(heights[0]); j++ {
		flowDFS(heights, 0, j, pacific)
		flowDFS(heights, len(heights)-1, j, atlantic)
	}
	var r [][]int
	for i:=0; i<len(pacific); i++ {
		for j:=0; j<len(pacific[0]); j++ {
			if pacific[i][j] && atlantic[i][j] {
				r = append(r, []int{i, j})
			}
		}
	}
	return r
}

func flowDFS(heights [][]int, i int, j int, check [][]bool) {
	if check[i][j] {
		return
	}
	check[i][j] = true
	for k:=0; k<4; k++ {
		x := i+direction[k]
		y := j+direction[k+1]
		if x<0 || x>=len(heights) || y<0 || y>=len(heights[0]) || heights[i][j] > heights[x][y] {
			continue
		}
		flowDFS(heights, x, y, check)
	}
}

