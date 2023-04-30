package main

import (
	"fmt"
)

func main() {
	//r := climbStairs(47)
	r := rob([]int{1,2,3,1})
	fmt.Println(r)
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

}