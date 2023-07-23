package main

import (
	"fmt"
)

func main() {
	//r := hammingDistance1(3, 1)
	//r := reverseBits(0b00000010100101000001111010011100)
	//r := singleNumber([]int{2,2,1})
	r := countBits(5)
	fmt.Println(r)
}

/**
461. 汉明距离
https://leetcode.cn/problems/hamming-distance/?utm_source=LCUS&utm_medium=ip_redirect&utm_campaign=transfer2china
 */
func hammingDistance(x int, y int) int {
	i, r := x ^ y, 0
	for i > 0 {
		i = i & (i-1)
		r ++
	}
	return r
}

func hammingDistance1(x int, y int) int {
	i, r := x ^ y, 0
	for i > 0 {
		r += i & 1
		i >>= 1
	}
	return r
}

/**
190. 颠倒二进制位
https://leetcode.cn/problems/reverse-bits/
 */
func reverseBits(num uint32) uint32 {
	r := uint32(0)
	for i:=0; i<32; i++ {
		r <<= 1
		r += num & 1
		num >>= 1
	}
	return r
}

/**
136. 只出现一次的数字
https://leetcode.cn/problems/single-number/
 */
func singleNumber(nums []int) int {
	if len(nums) < 1 {
		return 0
	}
	var r int
	for _, v := range nums {
		r ^= v
	}
	return r
}

/**
342. 4的幂
https://leetcode.cn/problems/power-of-four/
 */
func isPowerOfFour(n int) bool {
	if n < 0 {
		return false
	}
	if n & (n-1) != 0 {
		return false
	}
	if n & 0xaaaaaaaa != 0 {
		return false
	}
	return true
}

/**
318. 最大单词长度乘积
https://leetcode.cn/problems/maximum-product-of-word-lengths/
 */
func maxProduct(words []string) int {
	mp, ans := make(map[int]int), 0
	for _, word := range words {
		mask := 0
		for i:=0; i<len(word); i++ {
			mask |= 1 << (word[i]-'a')
		}
		mp[mask] = max(mp[mask], len(word))
		for _m, _len := range mp {
			if _m & mask == 0 {
				ans = max(ans, len(word) * _len)
			}
		}
	}
	return ans
}

func max(i, j int) int {
	if i > j {
		return i
	}
	return j
}

/**
338. 比特位计数
https://leetcode.cn/problems/counting-bits/
 */
func countBits(n int) []int {
	dp := make([]int, n+1)
	dp[0] = 0
	for i:=1; i<=n; i++ {
		if i & 1 == 1 {
			dp[i] = dp[i-1] + 1
		} else {
			dp[i] = dp[i>>1]
		}
	}
	return dp
}