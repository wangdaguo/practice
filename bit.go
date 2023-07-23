package main

import (
	"fmt"
)

func main() {
	//r := hammingDistance1(3, 1)
	//r := reverseBits(0b00000010100101000001111010011100)
	//r := singleNumber([]int{2,2,1})
	//r := countBits(5)
	//r := missingNumber([]int{19,58,31,51,54,47,68,25,85,9,83,70,24,75,30,78,62,38,41,21,56,60,94,1,45,15,72,52,28,93,14,96,35,17,95,89,74,46,13,82,57,76,55,20,36,63,44,61,6,92,65,50,91,42,98,34,8,33,40,12,7,48,11,80,10,71,97,39,73,26,99,43,90,5,3,2,23,29,0,79,53,64,4,27,37,84,69,81,22,86,67,32,66,18,16,77,59,49,87})
	//r := hasAlternatingBits(17)
	r := findComplement(1)
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

/**
268. 丢失的数字
https://leetcode.cn/problems/missing-number/
 */
func missingNumber(nums []int) int {
	var r int
	for i, v := range nums {
		r ^= i ^ v
	}
	return r ^ len(nums)
}

func missingNumber1(nums []int) int {
	mask, end, cnt := 0, len(nums), 0
	for i:=0; i<len(nums); i++ {
		mask |= 1 << nums[i]
	}
	for cnt <= end {
		if mask & 1 == 0 {
			return cnt
		}
		mask >>= 1
		cnt ++
	}
	return -1
}

/**
693. 交替位二进制数
https://leetcode.cn/problems/binary-number-with-alternating-bits/
 */
func hasAlternatingBits(n int) bool {
	a := n ^ (n >> 1)
	return a & (a+1) == 0
}

/**
476. 数字的补数
https://leetcode.cn/problems/number-complement/
 */
func findComplement(num int) int {
	r, cnt, tmp := 0, 0, 0
	for num > 0 {
		tmp = (num & 1) ^ 1
		tmp <<= cnt
		r += tmp
		num >>= 1
		cnt ++
	}
	return r
}

/**
260. 只出现一次的数字 III
https://leetcode.cn/problems/single-number-iii/
 */
func singleNumber1(nums []int) []int {
	xorSum, type1, type2 := 0, 0, 0
	for _, v := range nums {
		xorSum ^= v
	}
	l1 := xorSum & -xorSum
	for _, v := range nums {
		if v & l1 > 0 {
			type1 ^= v
		} else {
			type2 ^= v
		}
	}
	return []int{type1, type2}
}