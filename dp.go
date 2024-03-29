package main

import (
	"fmt"
	"math"
	"strings"
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
	//weight, value, count, bagCap := []int{1,2,3}, []int{12,20,100}, 3, 4
	//r := bagComplete(weight, value, count, bagCap)
	//r := canPartition1([]int{2,2,1,1})
	//r := coinChange([]int{1, 2, 5}, 11)
	//r := minDistance("horse", "ros")
	//r := minSteps(3)
	//r := maxProfit([]int{7,1,5,3,6,4})
	//r := maxProfit3([]int{1,2,3,0,2})
	//r := rob1([]int{1,2,3})
	//r := maxSubArray2([]int{-1,-7,-3})
	//r := integerBreak(10)
	r := minDistance1("sea", "eat")
	fmt.Println(r)
}

/**
完全背包问题
*/
func bagComplete(weight []int, value []int, count, bagCap int) int {
	dp := make([][]int, count+1)
	for i := 0; i <= count; i++ {
		dp[i] = make([]int, bagCap+1)
	}
	dp[0][0] = 0
	for i := 1; i <= count; i++ {
		w, v := weight[i-1], value[i-1]
		for j := 1; j <= bagCap; j++ {
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
	for i := 0; i < count; i++ {
		dp[i] = make([]int, bagCap+1)
	}
	dp[0][0] = 0
	for i := 1; i < count; i++ {
		w, v := weight[i-1], value[i-1]
		for j := 1; j < bagCap+1; j++ {
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
	for i := 3; i <= n; i++ {
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
	for i := 2; i < len(nums); i++ {
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
	diff, t, r := nums[1]-nums[0], 0, 0
	for i := 2; i < len(nums); i++ {
		if nums[i]-nums[i-1] == diff {
			t++
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
	for i := 0; i < len(grid); i++ {
		dp[i] = make([]int, len(grid[i]))
	}
	for i := 0; i < len(grid); i++ {
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
	for i := 0; i < len(grid); i++ {
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
		X: x,
		Y: y,
	}
}

func updateMatrix1(mat [][]int) [][]int {
	dist := make([][]int, len(mat))
	for i := 0; i < len(mat); i++ {
		tmp := make([]int, len(mat[0]))
		for j := 0; j < len(tmp); j++ {
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
	for i := len(mat) - 1; i >= 0; i-- {
		for j := len(mat[0]) - 1; j >= 0; j-- {
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
				dp[i][j] = minVal(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
				if dp[i][j] > maxSide {
					maxSide = dp[i][j]
				}
			}
		}
	}
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

/**
279. 完全平方数
https://leetcode.cn/problems/perfect-squares/?utm_source=LCUS&utm_medium=ip_redirect&utm_campaign=transfer2china
*/
func numSquares(n int) int {
	if n <= 0 {
		return 0
	}
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
	for i := 1; i <= len(s); i++ {
		if s[i-1] != '0' {
			dp[i] += dp[i-1]
		}
		if i-2 >= 0 && s[i-2] != '0' && ((s[i-2]-'0')*10+(s[i-1]-'0') <= 26) {
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
	for i := 1; i <= len(s); i++ {
		c = 0
		if s[i-1] != '0' {
			c += b
		}
		if i-2 >= 0 && s[i-2] != '0' && ((s[i-2]-'0')*10+(s[i-1]-'0') <= 26) {
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
	for i := 1; i <= len(s); i++ {
		for j := 0; j < i; j++ {
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
	for i := 1; i < len(nums); i++ {
		dp[i] = 1
		for j := 0; j < i; j++ {
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
	for i := 0; i < len1+1; i++ {
		dp[i] = make([]int, len2+1)
	}
	for i := 1; i <= len1; i++ {
		for j := 1; j <= len2; j++ {
			if text1[i-1] == text2[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = max(dp[i-1][j], dp[i][j-1])
			}
		}
	}
	return dp[len1][len2]
}

/**
416. 分割等和子集
https://leetcode.cn/problems/partition-equal-subset-sum/?utm_source=LCUS&utm_medium=ip_redirect&utm_campaign=transfer2china
*/
func canPartition(nums []int) bool {
	if len(nums) < 1 {
		return true
	}
	sum := sumInt(nums)
	if sum%2 != 0 {
		return false
	}
	halfSum := sum / 2 // val, weight := halfSum, len(nums)
	dp := make([]bool, halfSum+1)
	dp[0] = true
	for i := 1; i <= len(nums); i++ {
		for j := halfSum; j >= nums[i-1]; j-- {
			dp[j] = dp[j] || dp[j-nums[i-1]]
		}
	}
	fmt.Println(dp, halfSum)
	return dp[halfSum]
}

func sumInt(path []int) int {
	var sum int
	for _, val := range path {
		sum += val
	}
	return sum
}

func canPartition1(nums []int) bool {
	if len(nums) < 1 {
		return true
	}
	sum := sumInt(nums)
	if sum%2 != 0 {
		return false
	}
	halfSum := sum / 2 // val, weight := halfSum, len(nums)
	dp := make([][]bool, len(nums)+1)
	for k, _ := range dp {
		dp[k] = make([]bool, halfSum+1)
		dp[k][0] = true
	}

	for i := 1; i <= len(nums); i++ {
		for j := nums[i-1]; j <= halfSum; j++ {
			dp[i][j] = dp[i-1][j] || dp[i-1][j-nums[i-1]]
		}
	}
	return dp[len(nums)][halfSum]
}

/**
474. 一和零
https://leetcode.cn/problems/ones-and-zeroes/?utm_source=LCUS&utm_medium=ip_redirect&utm_campaign=transfer2china
*/
func findMaxForm(strs []string, m int, n int) int {
	dp := make([][]int, m+1)
	for k, _ := range dp {
		dp[k] = make([]int, n+1)
	}
	for _, str := range strs {
		zeroCnt := strings.Count(str, "0")
		oneCnt := len(str) - zeroCnt
		for i := m; i >= zeroCnt; i-- {
			for j := n; j >= oneCnt; j-- {
				dp[i][j] = max(dp[i][j], dp[i-zeroCnt][j-oneCnt]+1)
			}
		}
	}
	return dp[m][n]
}

/**
322. 零钱兑换
https://leetcode.cn/problems/coin-change/?utm_source=LCUS&utm_medium=ip_redirect&utm_campaign=transfer2china
*/
func coinChange(coins []int, amount int) int {
	dp := make([]int, amount+1)
	for i := 1; i < len(dp); i++ {
		dp[i] = amount + 1
	}
	dp[0] = 0
	for i := 1; i <= amount; i++ {
		for _, coin := range coins {
			if i >= coin {
				dp[i] = min(dp[i], dp[i-coin]+1)
			}
		}
	}
	fmt.Println(dp)
	if dp[amount] > amount {
		return -1
	}
	return dp[amount]
}

/**
72. 编辑距离
https://leetcode.cn/problems/edit-distance/?utm_source=LCUS&utm_medium=ip_redirect&utm_campaign=transfer2china
*/
func minDistance(word1 string, word2 string) int {
	dp := make([][]int, len(word1)+1)
	for i, _ := range dp {
		dp[i] = make([]int, len(word2)+1)
	}
	for i := 0; i <= len(word1); i++ {
		dp[i][0] = i
	}
	for j := 0; j <= len(word2); j++ {
		dp[0][j] = j
	}
	for i := 1; i <= len(word1); i++ {
		for j := 1; j <= len(word2); j++ {
			if word1[i-1] == word2[j-1] {
				dp[i][j] = dp[i-1][j-1]
			} else {
				dp[i][j] = mostMin(dp[i-1][j-1], dp[i][j-1], dp[i-1][j]) + 1
			}
		}
	}
	fmt.Println(dp)
	return dp[len(word1)][len(word2)]
}

func mostMin(list ...int) int {
	if len(list) < 1 {
		return 0
	}
	min := list[0]
	for i := 1; i < len(list); i++ {
		if min > list[i] {
			min = list[i]
		}
	}
	return min
}

/**
650. 只有两个键的键盘
https://leetcode.cn/problems/2-keys-keyboard/?utm_source=LCUS&utm_medium=ip_redirect&utm_campaign=transfer2china
*/
func minSteps(n int) int {
	dp := make([]int, n+1)
	dp[0] = 0
	h := int(math.Sqrt(float64(n)))
	for i := 2; i <= n; i++ {
		dp[i] = i
		for j := 2; j <= h; j++ {
			if i%j == 0 {
				dp[i] = dp[j] + dp[i/j]
			}
		}
	}
	fmt.Println(dp)
	return dp[n]
}

/**
10. 正则表达式匹配
https://leetcode.cn/problems/regular-expression-matching/?utm_source=LCUS&utm_medium=ip_redirect&utm_campaign=transfer2china
*/
func isMatch(s string, p string) bool {
	dp := make([][]bool, len(s)+1)
	for i := range dp {
		dp[i] = make([]bool, len(p)+1)
	}
	dp[0][0] = true
	match := func(i, j int) bool {
		if i == 0 {
			return false
		}
		if p[j-1] == '.' {
			return true
		}
		return s[i-1] == p[j-1]
	}
	for i := 0; i <= len(s); i++ {
		for j := 1; j <= len(p); j++ {
			if p[j-1] == '*' {
				dp[i][j] = dp[i][j] || dp[i][j-2]
				if match(i, j-1) {
					dp[i][j] = dp[i][j] || dp[i-1][j]
				}
			} else if match(i, j) {
				dp[i][j] = dp[i][j] || dp[i-1][j-1]
			}
		}
	}
	return dp[len(s)][len(p)]
}

/**
121. 买卖股票的最佳时机
https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/?utm_source=LCUS&utm_medium=ip_redirect&utm_campaign=transfer2china
*/
func maxProfit(prices []int) int {
	if len(prices) < 1 {
		return 0
	}
	var sell int
	buy := math.MaxInt16
	for i := 0; i < len(prices); i++ {
		buy = min(buy, prices[i])
		sell = max(sell, prices[i]-buy)
	}
	return sell
}

/**
1049. 最后一块石头的重量 II
https://leetcode.cn/problems/last-stone-weight-ii/
*/
func lastStoneWeightII(stones []int) int {
	/**
	dp[i] = dp[j] - dp[i-j]
	*/
	return 0
}

/**
188. 买卖股票的最佳时机 IV
https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iv/
*/
func maxProfit2(k int, prices []int) int {
	if len(prices) < 2 {
		return 0
	}
	if k > len(prices) {
		return maxProfitUnlimited(prices)
	}
	buy, sell := make([]int, len(prices)+1), make([]int, len(prices)+1)
	for i := range buy {
		buy[i] = math.MinInt32
	}
	for i := 0; i < len(prices); i++ {
		for j := 1; j <= k; j++ {
			buy[j] = max(buy[j], sell[j-1]-prices[i])
			sell[j] = max(sell[j], buy[j]+prices[i])
		}
	}
	return sell[k]
}

func maxProfitUnlimited(prices []int) int {
	var maxProfit int
	for i := 1; i < len(prices); i++ {
		if prices[i] > prices[i-1] {
			maxProfit += prices[i] - prices[i-1]
		}
	}
	return maxProfit
}

/**
309. 最佳买卖股票时机含冷冻期
https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-cooldown/
*/
func maxProfit3(prices []int) int {
	// 分别是  买入、不动、卖出、冻结
	buy, s1, sell, s2 := make([]int, len(prices)+1), make([]int, len(prices)+1), make([]int, len(prices)+1), make([]int, len(prices)+1)
	buy[0], s1[0] = -prices[0], -prices[0]
	sell[0], s2[0] = 0, 0
	for i := 1; i < len(prices); i++ {
		buy[i] = s2[i-1] - prices[i]
		s1[i] = max(s1[i-1], buy[i-1])
		sell[i] = max(buy[i-1], s1[i-1]) + prices[i]
		s2[i] = max(s2[i-1], sell[i-1])
	}
	return max(s2[len(prices)-1], sell[len(prices)-1])
}

/**
213. 打家劫舍 II
https://leetcode.cn/problems/house-robber-ii/
*/
func rob1(nums []int) int {
	if len(nums) < 1 {
		return 0
	}
	have1Dp, notHave1Dp := make([]int, len(nums)+1), make([]int, len(nums)+1)
	have1Dp[0], have1Dp[1] = 0, nums[0]
	notHave1Dp[0], notHave1Dp[1] = 0, 0

	for i := 2; i <= len(nums); i++ {
		if i == len(nums) {
			have1Dp[i] = have1Dp[i-1]
			notHave1Dp[i] = max(notHave1Dp[i-1], notHave1Dp[i-2]+nums[i-1])
		} else {
			have1Dp[i] = max(have1Dp[i-1], have1Dp[i-2]+nums[i-1])
			notHave1Dp[i] = max(notHave1Dp[i-1], notHave1Dp[i-2]+nums[i-1])
		}
	}
	fmt.Println(have1Dp, notHave1Dp)
	return max(have1Dp[len(nums)], notHave1Dp[len(nums)])
}

/**
53. 最大子数组和
https://leetcode.cn/problems/maximum-subarray/
*/
func maxSubArray2(nums []int) int {
	if len(nums) < 1 {
		return 0
	}
	tmp, max := nums[0], nums[0]
	for i, v := range nums {
		if i == 0 {
			continue
		}
		tmp += v
		if tmp < v {
			tmp = v
		}
		if tmp > max {
			max = tmp
		}
	}
	return max
}

/**
343. 整数拆分
https://leetcode.cn/problems/integer-break/
*/
func integerBreak(n int) int {
	dp := make([]int, n+1)
	dp[0], dp[1] = 0, 1
	for i := 2; i <= n; i++ {
		for j := 1; j <= i; j++ {
			dp[i] = max(dp[i], dp[i-j]*j)
			dp[i] = max(dp[i], j*(i-j))
		}
	}
	fmt.Println(dp)
	return dp[n]
}

/**
583. 两个字符串的删除操作
https://leetcode.cn/problems/delete-operation-for-two-strings/
*/
func minDistance1(word1 string, word2 string) int {
	dp := make([][]int, len(word1)+1)
	for i := range dp {
		dp[i] = make([]int, len(word2)+1)
		dp[i][0] = i
	}
	for j := range dp[0] {
		dp[0][j] = j
	}

	//dp[i][j] 表示word1[:i] 与 word2[:j] 保持相同所需要删除的最小步数
	for i := 1; i <= len(word1); i++ {
		for j := 1; j <= len(word2); j++ {
			if word1[i-1] == word2[j-1] {
				dp[i][j] = dp[i-1][j-1]
			} else {
				dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + 1
			}
		}
	}
	fmt.Println(dp)
	return dp[len(word1)][len(word2)]
}
