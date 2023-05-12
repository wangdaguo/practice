package main

import (
	"fmt"
	"math"
)

func main() {
	//r := climbStairs(47)
	//r := rob([]int{1,2,3,1})
	//mat := [][]int{{0,0,0}, {0,1,0}, {0,0,0}}
	//r := updateMatrix1(mat)
	//r := numSquares(12)
	//r := lengthOfLIS([]int{1,3,6,7,9,4,10,5,6})
	//weight, value, count, bagCap := []int{1,2,3}, []int{10,20,30}, 3, 4
	//r := bag01Optimize(weight, value, count, bagCap)
	weight, value, count, bagCap := []int{1,2,3}, []int{12,20,100}, 3, 4
	r := bagCompleteOptimize(weight, value, count, bagCap)
	fmt.Println(r)
}

/**
完全背包问题
 */
func bagComplete(weight []int, value []int, count, bagCap int) int {
	dp := make([][]int, count+1)
	for i:=0; i<=count; i++ {
		dp[i] = make([]int, bagCap+1)
	}
	dp[0][0] = 0
	for i:=1; i<=count; i++ {
		w, v := weight[i-1], value[i-1]
		for j:=1; j<=bagCap; j++ {
			if j >= w {
				dp[i][j] = max(dp[i-1][j], dp[i][j-w]+v)
			} else {
				dp[i][j] = dp[i-1][j]
			}
		}
	}
	fmt.Println(dp)
	return dp[count][bagCap]
}

/**
完全背包问题空间压缩
*/
func bagCompleteOptimize(weight []int, value []int, count, bagCap int) int {
	dp := make([]int, bagCap+1)
	dp[0] = 0
	for i := 1; i <= count; i++ {
		w, v := weight[i-1], value[i-1]
		for j := w; j <= bagCap; j++ {
			dp[j] = max(dp[j], dp[j-w]+v)
		}
	}
	fmt.Println(dp)
	return dp[bagCap]
}

/**
01背包问题
 */
func bag01(weight []int, value []int, count, bagCap int) int {
	dp := make([][]int, count)
	for i:=0; i<count; i++ {
		dp[i] = make([]int, bagCap+1)
	}
	dp[0][0] = 0
	for i:=1; i<count; i++ {
		w, v := weight[i-1], value[i-1]
		for j:=1; j<bagCap+1; j++ {
			if j >= w {
				dp[i][j] = max(dp[i-1][j], dp[i-1][j-w]+v)
			} else {
				dp[i][j] = dp[i-1][j]
			}
		}
	}
	fmt.Println(dp)
	return dp[count-1][bagCap]
}

/**
01背包空间压缩
*/
func bag01Optimize(weight []int, value []int, count, bagCap int) int {
	dp := make([]int, bagCap+1)
	dp[0] = 0
	for i := 1; i < count; i++ {
		w, v := weight[i-1], value[i-1]
		for j := bagCap; j >= w; j-- {
			dp[j] = max(dp[j], dp[j-w]+v)
		}
	}
	fmt.Println(dp)
	return dp[bagCap]
}

/**
70. 爬楼梯
https://leetcode.cn/problems/climbing-stairs/
输入：n = 2
输出：2
解释：有两种方法可以爬到楼顶。
1. 1 阶 + 1 阶
2. 2 阶
 */
func climbStairs(n int) int {
	if n <= 2 {
		return n
	}
	dp := make([]int, n+1)
	dp[1] = 1
	dp[2] = 2
	for i:=3; i<=n; i++ {
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[n]
}

/**
198. 打家劫舍
https://leetcode.cn/problems/house-robber/description/
输入：[1,2,3,1]
输出：4
解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
偷窃到的最高金额 = 1 + 3 = 4 。
 */
func rob(nums []int) int {
	if len(nums) < 1 {
		return 0
	}
	if len(nums) < 2 {
		return nums[0]
	}
	dp := make([]int, len(nums))
	max := func(i, j int) int {
		if i >= j {
			return i
		}
		return j
	}
	dp[0] = nums[0]
	dp[1] = max(nums[0], nums[1])
	for i:=2; i<len(nums); i++ {
		dp[i] = max(dp[i-1], dp[i-2]+nums[i])
	}
	return dp[len(nums)-1]
}

/**
413. 等差数列划分
https://leetcode.cn/problems/arithmetic-slices/description/
 */
func numberOfArithmeticSlices(nums []int) int {
	if len(nums) < 3 {
		return 0
	}
	diff, t, r := nums[1] - nums[0], 0, 0
	for i:=2; i<len(nums); i++ {
		if nums[i] - nums[i-1] == diff {
			t ++
		} else {
			diff = nums[i] - nums[i-1]
			t = 0
		}
		r += t
	}
	return r
}

/**
64. 最小路径和
https://leetcode.cn/problems/minimum-path-sum/?utm_source=LCUS&utm_medium=ip_redirect&utm_campaign=transfer2china
 */
func minPathSum(grid [][]int) int {
	dp := make([][]int, len(grid))
	for i:=0; i<len(grid); i++ {
		dp[i] = make([]int, len(grid[i]))
	}
	for i:=0; i<len(grid); i++ {
		for j:=0; j<len(grid[i]); j++ {
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
	return dp[len(grid)-1][len(grid[0])-1]
}

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

func minPathSum1(grid [][]int) int {
	dp := make([]int, len(grid[0]))
	for i:=0; i<len(grid); i++ {
		for j := 0; j < len(grid[i]); j++ {
			if i == 0 && j == 0 {
				dp[j] = grid[i][j]
			} else if i == 0 {
				dp[j] = dp[j-1] + grid[i][j]
			} else if j == 0 {
				dp[j] = dp[j] + grid[i][j]
			} else {
				dp[j] = min(dp[j], dp[j-1]) + grid[i][j]
			}
		}
	}
	return dp[len(grid[0])-1]
}

/**
542. 01 矩阵
https://leetcode.cn/problems/01-matrix/?utm_source=LCUS&utm_medium=ip_redirect&utm_campaign=transfer2china
 */
func updateMatrix(mat [][]int) [][]int {
	seen, dist := make([][]int, len(mat)), make([][]int, len(mat))
	for i := 0; i < len(mat); i++ {
		seen[i] = make([]int, len(mat[0]))
		dist[i] = make([]int, len(mat[0]))
	}
	if len(mat) < 1 {
		return dist
	}
	q := make([]*P, 0)
	for i := 0; i < len(mat); i++ {
		for j := 0; j < len(mat[i]); j++ {
			if mat[i][j] == 0 {
				q = append(q, NewP(i, j))
				seen[i][j] = 1
			}
		}
	}
	var v *P
	for len(q) > 0 {
		v, q = q[0], q[1:]
		for i := 0; i < 4; i++ {
			x := v.X + direction[i]
			y := v.Y + direction[i+1]
			if x < 0 || x >= len(mat) || y < 0 || y >= len(mat[0]) || seen[x][y] == 1 {
				continue
			}
			seen[x][y] = 1
			dist[x][y] = dist[v.X][v.Y] + 1
			q = append(q, NewP(x, y))
		}
	}
	return dist
}

var direction = []int{-1, 0, 1, 0, -1}

type P struct {
	X int
	Y int
}

func NewP(x, y int) *P {
	return &P{
		X:x,
		Y:y,
	}
}

func updateMatrix1(mat [][]int) [][]int {
	dist := make([][]int, len(mat))
	for i := 0; i < len(mat); i++ {
		tmp := make([]int, len(mat[0]))
		for j:=0; j<len(tmp); j++ {
			// 初始化动态规划的数组，所有的距离值都设置为一个很大的数
			tmp[j] = math.MaxInt16 / 2
		}
		dist[i] = append(dist[i], tmp...)
	}
	for i := 0; i < len(mat); i++ {
		for j := 0; j < len(mat[i]); j++ {
			if mat[i][j] == 0 {
				dist[i][j] = 0
			}
		}
	}
	// 只有 水平向左移动 和 竖直向上移动，注意动态规划的计算顺序
	for i := 0; i < len(mat); i++ {
		for j := 0; j < len(mat[i]); j++ {
			if i-1 >= 0 {
				dist[i][j] = min(dist[i][j], dist[i-1][j]+1)
			}
			if j-1 >= 0 {
				dist[i][j] = min(dist[i][j], dist[i][j-1]+1)
			}
		}
	}
	// 只有 水平向右移动 和 竖直向下移动，注意动态规划的计算顺序
	for i:=len(mat)-1; i>=0; i-- {
		for j:=len(mat[0])-1; j>=0; j-- {
			if i+1 < len(mat) {
				dist[i][j] = min(dist[i][j], dist[i+1][j]+1)
			}
			if j+1 < len(mat[0]) {
				dist[i][j] = min(dist[i][j], dist[i][j+1]+1)
			}
		}
	}
	return dist
}

/**
221. 最大正方形
https://leetcode.cn/problems/maximal-square/?utm_source=LCUS&utm_medium=ip_redirect&utm_campaign=transfer2china
 */
func maximalSquare(matrix [][]byte) int {
	if len(matrix) < 1 {
		return 0
	}
	dp := make([][]int, len(matrix))
	var maxSide int
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
				dp[i][j] = minVal(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])+1
				if dp[i][j] > maxSide {
					maxSide = dp[i][j]
				}
			}
		}
	}
	return maxSide * maxSide
}

func minVal(list... int) int {
	if len(list) == 0 {
		return 0
	}
	minV := list[0]
	for i:=1; i<len(list); i++ {
		if list[i] < minV {
			minV = list[i]
		}
	}
	return minV
}

/**
279. 完全平方数
https://leetcode.cn/problems/perfect-squares/?utm_source=LCUS&utm_medium=ip_redirect&utm_campaign=transfer2china
 */
func numSquares(n int) int {
	if n <= 0 {
		return 0
	}
	dp := make([]int, n+1)
	for i:=0; i<len(dp); i++ {
		dp[i] = math.MaxInt16 / 2
	}
	dp[0] = 0
	for i:=1; i<=n; i++ {
		for j:=1; j*j<=i; j++ {
			dp[i] = minVal(dp[i], dp[i-j*j] + 1)
		}
	}
	return dp[n]
}

/**
91. 解码方法
https://leetcode.cn/problems/decode-ways/?utm_source=LCUS&utm_medium=ip_redirect&utm_campaign=transfer2china
 */
func numDecodings(s string) int {
	if len(s) < 1 {
		return 0
	}
	dp := make([]int, len(s)+1)
	dp[0] = 1
	for i:=1; i<=len(s); i++ {
		if s[i-1] != '0' {
			dp[i] += dp[i-1]
		}
		if i-2>=0 && s[i-2] != '0' && ((s[i-2]-'0')*10+(s[i-1]-'0') <= 26) {
			dp[i] += dp[i-2]
		}
	}
	return dp[len(s)]
}

func numDecodings1(s string) int {
	if len(s) < 1 {
		return 0
	}
	a, b, c := 0, 1, 0 // a=s[i-2], b=s[i-1], c=s[i]
	for i:=1; i<=len(s); i++ {
		c = 0
		if s[i-1] != '0' {
			c += b
		}
		if i-2>=0 && s[i-2] != '0' && ((s[i-2]-'0')*10+(s[i-1]-'0') <= 26) {
			c += a
		}
		a, b = b, c
	}
	return c
}

/**
139. 单词拆分
https://leetcode.cn/problems/word-break/
 */
func wordBreak(s string, wordDict []string) bool {
	set := make(map[string]bool)
	for _, v := range wordDict {
		set[v] = true
	}
	dp := make([]bool, len(s)+1)
	dp[0] = true
	for i:=1; i<=len(s); i++ {
		for j:=0; j<i; j++ {
			if dp[j] && set[s[j:i]] {
				dp[i] = true
				break
			}
		}
	}
	return dp[len(s)]
}

/**
300. 最长递增子序列
https://leetcode.cn/problems/longest-increasing-subsequence/?utm_source=LCUS&utm_medium=ip_redirect&utm_campaign=transfer2china
 */
func lengthOfLIS(nums []int) int {
	dp := make([]int, len(nums))
	dp[0] = 1
	maxVal := 1
	for i:=1; i<len(nums); i++ {
		dp[i] = 1
		for j:=0; j<i; j++ {
			if nums[j] < nums[i] {
				dp[i] = max(dp[j]+1, dp[i])
			}
		}
		if dp[i] > maxVal {
			maxVal = dp[i]
		}
	}
	fmt.Println(dp)
	return maxVal
}

func max(i, j int) int {
	if i > j {
		return i
	}
	return j
}

/**
1143. 最长公共子序列
https://leetcode.cn/problems/longest-common-subsequence/
 */
func longestCommonSubsequence(text1 string, text2 string) int {
	len1, len2 := len(text1), len(text2)
	dp := make([][]int, len1+1)
	for i:=0; i<len1+1; i++ {
		dp[i] = make([]int, len2+1)
	}
	for i:=1; i<=len1; i++ {
		for j:=1; j<=len2; j++ {
			if text1[i-1] == text2[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = max(dp[i-1][j], dp[i][j-1])
			}
		}
	}
	return dp[len1][len2]
}


