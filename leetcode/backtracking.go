package main

import (
	"fmt"
	"sort"
	"strings"
)

/*
78. 子集
https://leetcode-cn.com/problems/subsets/
 */
func subsets(nums []int) [][]int {
	if len(nums) < 1 {
		return [][]int{}
	}
	var r [][]int
	r = append(r, []int{})
	subsetsImpl1(nums, 0, &[]int{}, &r)
	return r
}

func subsetsImpl1(nums []int, start int, tmp *[]int, r *[][]int) {
	if start >= len(nums) {
		return
	}
	for i := start; i < len(nums); i++ {
		if i > start && nums[i] == nums[i-1] {
			continue
		}
		*tmp = append(*tmp, nums[i])
		subsetsImpl1(nums, i+1, tmp, r)
		t := make([]int, len(*tmp))
		copy(t, *tmp)
		*r = append(*r, t)
		*tmp = (*tmp)[:len(*tmp)-1]
	}
}

/*
39. 组合总和 元素可重复使用
https://leetcode-cn.com/problems/combination-sum/
 */
func combinationSum(candidates []int, target int) [][]int {
	if len(candidates) < 1 {
		return [][]int{}
	}
	var r [][]int
	combinationSumImpl(candidates, target, &r, []int{}, 0)
	return r
}

func combinationSumImpl(candidates []int, target int, r *[][]int, tmp []int, start int) {
	if sumSlice(tmp) == target {
		t := make([]int, len(tmp))
		copy(t, tmp)
		*r = append(*r, t)
		return
	}
	if sumSlice(tmp) > target {
		return
	}
	for i:=start; i<len(candidates); i++ {
		tmp = append(tmp, candidates[i])
		combinationSumImpl(candidates, target, r, tmp, i)
		tmp = tmp[:len(tmp)-1]
	}
}

/*
40. 组合总和 II  元素不可重复使用
https://leetcode-cn.com/problems/combination-sum-ii/
 */
func combinationSum2(candidates []int, target int) [][]int {
	if len(candidates) < 1 {
		return [][]int{}
	}
	var r [][]int
	c := intSlice(candidates)
	c.Sort()
	combinationSum2Impl(c, target, []int{}, &r, 0)
	return r
}
func combinationSum2Impl(candidates []int, target int, tmp []int, r *[][]int, start int)  {
	if sumSlice(tmp) == target {
		t := make([]int, len(tmp))
		copy(t, tmp)
		*r = append(*r, t)
		return
	}
	if sumSlice(tmp) > target {
		return
	}
	for i := start; i < len(candidates); i++ {
		if i > start && candidates[i] == candidates[i-1] {
			continue
		}
		tmp = append(tmp, candidates[i])
		combinationSum2Impl(candidates, target, tmp, r, i+1)
		tmp = tmp[:len(tmp)-1]

	}
}
type intSlice []int
func (s intSlice) Len() int {
	return len(s)
}
func (s intSlice) Swap(a, b int) {
	s[a] , s[b] = s[b], s[a]
}
func (s intSlice) Less(a, b int) bool {
	if s[a] > s[b] {
		return true
	}
	return false
}
func(s intSlice) Sort() {
	sort.Sort(s)
}

/*
46. 全排列
https://leetcode-cn.com/problems/permutations/
 */
func permute(nums []int) [][]int {
	if len(nums) < 1 {
		return [][]int{}
	}
	var r [][]int
	permuteImpl(nums, &r, 0)
	return r
}
func permuteImpl(nums []int, r *[][]int, start int) {
	if start == len(nums) {
		t := make([]int, len(nums))
		copy(t, nums)
		*r = append(*r, t)
		return
	}
	for i := start; i < len(nums); i++ {
		nums[start], nums[i] = nums[i], nums[start]
		permuteImpl(nums, r, start+1)
		nums[start], nums[i] = nums[i], nums[start]
	}
}

/*
47. 全排列 II
https://leetcode-cn.com/problems/permutations-ii/
 */
func permuteUnique(nums []int) [][]int {
	if len(nums) < 1 {
		return [][]int{}
	}
	//c := intSlice(nums)
	//c.Sort()
	sort.Ints(nums)
	var r [][]int
	permuteUniqueImpl(nums, &r, 0)
	return r
}
func permuteUniqueImpl(nums []int, r *[][]int, start int)  {
	if start == len(nums) {
		t := make([]int, len(nums))
		copy(t, nums)
		*r = append(*r, t)
		return
	}
	mp := make(map[int]bool)
	for i:=start; i<len(nums); i++ {
		if i > start && nums[i] == nums[i-1] || mp[nums[i]] {
			continue
		}
		mp[nums[i]] = true
		nums[i], nums[start] = nums[start], nums[i]
		permuteUniqueImpl(nums, r, start+1)
		nums[i], nums[start] = nums[start], nums[i]
	}
}

/*
面试题 08.01. 三步问题
https://leetcode-cn.com/problems/three-steps-problem-lcci/
 */
func waysToStep(n int) int {
	if n == 1 {
		return 1
	}
	if n == 2 {
		return 2
	}
	if n == 3 {
		return 4
	}
	n1, n2, n3 := 1, 2, 4
	for i:=4; i<=n; i++ {
		tmp := ((n1 % 1000000007 + n2 % 1000000007) % 1000000007 + n3 %1000000007) % 1000000007
		n1 = n2 % 1000000007
		n2 = n3 % 1000000007
		n3 = tmp
	}
	return n3
}

/**
90. 子集 II
https://leetcode-cn.com/problems/subsets-ii/
 */
func subsetsWithDup(nums []int) [][]int {
	if len(nums) < 1 {
		return [][]int{}
	}
	var r [][]int
	r = append(r, []int{})
	sort.Ints(nums)
	subsetsWithDupImpl(nums, &r, &[]int{}, 0)
	return r
}
func subsetsWithDupImpl(nums []int, r *[][]int, tmp *[]int, start int)  {
	if start == len(nums) {
		return
	}
	for i:=start; i<len(nums); i++ {
		if i > start && nums[i] == nums[i-1] {
			continue
		}
		*tmp = append(*tmp, nums[i])
		subsetsWithDupImpl(nums, r, tmp, i+1)
		t := make([]int, len(*tmp))
		copy(t, *tmp)
		*r = append(*r, t)
		*tmp = (*tmp)[:len(*tmp)-1]
	}
}

/**
77. 组合
https://leetcode-cn.com/problems/combinations/
 */
func combine(n int, k int) [][]int {
	if n == 0 {
		return [][]int{}
	}
	if n == 1 {
		return [][]int{{n}}
	}
	var nums []int
	for i:=1; i<=n; i++ {
		nums = append(nums, i)
	}
	var r [][]int
	combineImpl(nums, &r, &[]int{}, k, 0)
	return r
}
func combineImpl(nums []int, r *[][]int, tmp *[]int, k, start int)  {
	if len(*tmp) == k {
		t := make([]int, len(*tmp))
		copy(t, *tmp)
		*r = append(*r, t)
		return
	}
	for i:=start; i<len(nums); i++ {
		*tmp = append(*tmp, nums[i])
		combineImpl(nums, r, tmp, k, i+1)
		*tmp = (*tmp)[:len(*tmp)-1]
	}
}

/*
216. 组合总和 III
https://leetcode-cn.com/problems/combination-sum-iii/
 */
func combinationSum3(k int, n int) [][]int {
	if k == 0 || n == 0 {
		return  [][]int{}
	}
	var nums []int
	for i:=1; i<10; i++ {
		nums = append(nums, i)
	}
	var r [][]int
	combinationSum3Impl(nums, &r, &[]int{}, n, k, 0)
	return r
}

func combinationSum3Impl(nums []int, r *[][]int, tmp *[]int, n, k, start int) {
	if len(*tmp) == k && sumSlice(*tmp) == n {
		t := make([]int, len(*tmp))
		copy(t, *tmp)
		*r = append(*r, t)
		return
	}
	if len(*tmp) > k || sumSlice(*tmp) > n {
		return
	}
	for i:=start; i<len(nums); i++ {
		*tmp = append(*tmp, nums[i])
		combinationSum3Impl(nums, r, tmp, n, k, i+1)
		*tmp = (*tmp)[:len(*tmp)-1]
	}
}

/*
377. 组合总和 Ⅳ
https://leetcode-cn.com/problems/combination-sum-iv/
 */
func combinationSum4(nums []int, target int) int {
	if len(nums) < 1 {
		return 0
	}
	sort.Ints(nums)
	var r, tmp int
	dict := make(map[int]int)
	combinationSum4Impl(nums, &r, &tmp, target, 0, dict)
	return r
}

func combinationSum4Impl(nums []int, r *int, tmp *int, target, start int, dict map[int]int) {
	if *tmp == target {
		*r ++
		return
	}
	if target < nums[0] {
		return
	}
	if *tmp > target {
		return
	}
	for i:=start; i<len(nums); i++ {
		*tmp += nums[i]
		if v, ok := dict[target-*tmp]; ok {
			*r += v
		} else {
			var rr, tt int
			combinationSum4Impl(nums, &rr, &tt, target-*tmp, start, dict)
			dict[target-*tmp] = rr
			*r += rr
		}
		*tmp -= nums[i]
	}
}

func combinationSum41(nums []int, target int) int {
	if len(nums) < 1 {
		return 0
	}
	dp := make([]int, target+1)
	dp[0] = 1
	for i:=1; i<=target; i++ {
		for _, num := range nums {
			if i >= num {
				dp[i] += dp[i-num]
			}
		}
	}
	return dp[target]
}

/**
31. 下一个排列
https://leetcode-cn.com/problems/next-permutation/
 */
func nextPermutation(nums []int)  {
	if len(nums) < 1 {
		return
	}
	i:=len(nums)-2
	for i >= 0 && nums[i] >= nums[i+1] {
		i --
	}
	if i >= 0 {
		j := len(nums)-1
		for j >= 0 && nums[j] <= nums[i] {
			j --
		}
		nums[j], nums[i] = nums[i], nums[j]
	}
	reverseSort(nums, i+1)
	return
}

func reverseSort(nums []int, s int)  {
	i, j := s, len(nums)-1
	for i <= j {
		nums[i], nums[j] = nums[j], nums[i]
		i ++
		j --
	}
	return
}

/**
17. 电话号码的字母组合
https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/
 */
var table = []string {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"}
func letterCombinations(digits string) []string {
	if len(digits) < 1 {
		return []string{}
	}
	res := []string{""}
	for i:=0; i<len(digits); i++ {
		t := table[digits[i]-'0']
		tmp := []string{}
		for j:=0; j<len(t); j++ {
			for z:=0; z<len(res); z++ {
				tmp = append(tmp, res[z] + string([]byte{t[j]}))
			}
		}
		res = tmp
	}
	return res
}

/**
93. 复原IP地址
https://leetcode-cn.com/problems/restore-ip-addresses/
 */
func restoreIpAddresses(s string) []string {
	if len(s) < 3 || len(s) > 12 {
		return []string{}
	}
	var res []string
	restoreIpAddressesImpl(s, &res, []string{})
	return res
}

func restoreIpAddressesImpl(s string, res *[]string, tmp []string)  {
	if len(tmp) == 4 {
		if len(s) == 0 {
			*res = append(*res, tmp[0]+"."+tmp[1]+"."+tmp[2]+"."+tmp[3])
		}
		return
	}
	for i:=1; i<=3; i++ {
		if len(s) < i {
			return
		}
		str := s[:i]
		if len(str) == 3 && strings.Compare(str, "255") > 0 {
			return
		}
		if len(str) > 1 && s[0] == '0' {
			return
		}
		tmp = append(tmp, str)
		restoreIpAddressesImpl(s[i:], res, tmp)
		tmp = tmp[:len(tmp)-1]
	}
}


func mainxx()  {
	//nums := []int{1,2}
	//r := subsets(nums)
	//r := waysToStep(5)
	//r := permuteUnique([]int{0,0,9,1})
	//r := subsetsWithDup([]int{1,2,2})
	//r := combine(4, 2)
	//r := combinationSum3(3, 9)
	//r := combinationSum41([]int{1,2,3}, 2)
	r := []int{1,3,2}
	nextPermutation(r)
	fmt.Println(r)
}