package main

import (
	"container/heap"
	"fmt"
	"math"
	"sort"
)

func maing() {
	//nums := []int{0, 1, 0, 3, 12}
	//moveZeroes(nums)
	//r := maxArea([]int{1, 8, 6, 2, 5, 4, 8, 3, 7})
	//r := findAnagrams("cbaebabacd", "abc")
	r := subarraySum([]int{-1, -1, 1}, 0)
	fmt.Println(r)
	//list := []int{1, 2, 3, 4, 5, 6}
	//fmt.Println(list[0:2])
}

/*
*
1. 两数之和
https://leetcode.cn/problems/two-sum/?envType=study-plan-v2&envId=top-100-liked
*/
func twoSum(nums []int, target int) []int {
	mp := make(map[int]int)
	for k, v := range nums {
		if idx, ok := mp[target-v]; ok {
			return []int{idx, k}
		}
		mp[v] = k
	}
	return []int{}
}

/*
*
49. 字母异位词分组
https://leetcode.cn/problems/group-anagrams/description/?envType=study-plan-v2&envId=top-100-liked
*/
func groupAnagrams(strs []string) [][]string {
	mp := make(map[string][]string)
	for _, str := range strs {
		s := []byte(str)
		sort.Slice(str, func(i, j int) bool {
			return s[i] < s[j]
		})
		mp[string(s)] = append(mp[string(s)], str)
	}
	r := make([][]string, 0)
	for _, v := range mp {
		r = append(r, v)
	}
	return r
}

/*
*
128. 最长连续序列
https://leetcode.cn/problems/longest-consecutive-sequence/?envType=study-plan-v2&envId=top-100-liked
*/
func longestConsecutive(nums []int) int {
	r, mp := 0, make(map[int]bool)
	for _, v := range nums {
		mp[v] = true
	}
	for num := range mp {
		if _, ok := mp[num-1]; !ok {
			cur, len := num, 1
			for mp[cur+1] {
				cur++
				len++
			}
			if len > r {
				r = len
			}
		}
	}
	return r
}

/*
*
283. 移动零
https://leetcode.cn/problems/move-zeroes/?envType=study-plan-v2&envId=top-100-liked
*/
func moveZeroes(nums []int) {
	idx := 0
	for i := 0; i < len(nums); i++ {
		if nums[i] != 0 {
			nums[idx] = nums[i]
			idx++
		}
	}
	for idx < len(nums) {
		nums[idx] = 0
		idx++
	}
	return
}

/*
*
11. 盛最多水的容器
https://leetcode.cn/problems/container-with-most-water/?envType=study-plan-v2&envId=top-100-liked
*/
func maxArea(height []int) int {
	area, maxArea, i, j := 0, 0, 0, len(height)-1
	for i < j {
		area = min(height[i], height[j]) * int(math.Abs(float64(i-j)))
		if area > maxArea {
			maxArea = area
		}
		if height[i] > height[j] {
			j--
		} else {
			i++
		}
	}
	return maxArea
}
func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

/*
*
3. 无重复字符的最长子串
https://leetcode.cn/problems/longest-substring-without-repeating-characters/?envType=study-plan-v2&envId=top-100-liked
*/
func lengthOfLongestSubstring(s string) int {
	if len(s) < 1 {
		return 0
	}
	mp, left, cnt := make(map[byte]int), 0, 0
	for i := 0; i < len(s); i++ {
		idx, ok := mp[s[i]]
		if !ok {
			mp[s[i]] = i
			if len(mp) > cnt {
				cnt = len(mp)
			}
			continue
		}
		for left <= idx {
			delete(mp, s[left])
			left++
		}
		mp[s[i]] = i
	}
	return cnt
}

/*
*
438. 找到字符串中所有字母异位词
https://leetcode.cn/problems/find-all-anagrams-in-a-string/?envType=study-plan-v2&envId=top-100-liked
*/
func findAnagrams(s string, p string) []int {
	var sCount, pCount [26]int
	r := make([]int, 0)
	for k, v := range p {
		pCount[v-'a']++
		sCount[s[k]-'a']++
	}
	if pCount == sCount {
		r = append(r, 0)
	}
	for k, v := range s[:len(s)-len(p)] {
		sCount[v-'a']--
		sCount[s[k+len(p)]-'a']++
		if pCount == sCount {
			r = append(r, k)
		}
	}
	return r
}

/*
*
560. 和为 K 的子数组
https://leetcode.cn/problems/subarray-sum-equals-k/?envType=study-plan-v2&envId=top-100-liked
*/
func subarraySum(nums []int, k int) int {
	pre, mp, r := 0, make(map[int]int), 0
	for i := 0; i < len(nums); i++ {
		pre += nums[i]
		if v, ok := mp[pre-k]; ok {
			r += v
		}
		mp[pre]++
	}
	return r
}

var list []int

type heap12 struct {
	sort.IntSlice
}

func (h heap12) Less(i, j int) bool {
	return list[h.IntSlice[i]] > list[h.IntSlice[j]]
}

func (h heap12) Push(x any) {
	h.IntSlice = append(h.IntSlice, x.(int))
}

func (h heap12) Pop() any {
	val := h.IntSlice[len(h.IntSlice)-1]
	h.IntSlice = h.IntSlice[:len(h.IntSlice)-1]
	return val
}

func maxSlidingWindow1(nums []int, k int) []int {
	list = nums
	hp := &heap12{make(sort.IntSlice, k)}
	for i := 0; i < k; i++ {
		hp.IntSlice[i] = i
	}
	heap.Init(hp)
	r := make([]int, 1)
	r[0] = nums[hp.IntSlice[0]]
	for i := k; i < len(nums); i++ {
		heap.Push(hp, i)
		for i-k >= hp.IntSlice[0] {
			heap.Pop(hp)
		}
		r = append(r, nums[hp.IntSlice[0]])
	}
	return r
}

/*
*
76. 最小覆盖子串
https://leetcode.cn/problems/minimum-window-substring/description/?envType=study-plan-v2&envId=top-100-liked
*/
func minWindow(s string, t string) string {
	cnt, left, mp, char, minStart, minSize := 0, 0, make(map[byte]bool), make(map[byte]int), 0, len(s)+1
	for i := 0; i < len(t); i++ {
		char[t[i]]++
		mp[t[i]] = true
	}
	for i := 0; i < len(s); i++ {
		if _, ok := mp[s[i]]; ok {
			char[s[i]]--
			if char[s[i]] >= 0 {
				cnt++
			}

		}
		for cnt == len(t) {
			if minSize > i-left+1 {
				minSize = i - left + 1
				minStart = left
			}
			char[s[left]]++
			if _, ok := mp[s[left]]; ok && char[s[left]] > 0 {
				cnt--
			}
			left++
		}
	}
	if minSize > len(s) {
		return ""
	}
	return s[minStart : minStart+minSize]
}

/*
*
56. 合并区间
https://leetcode.cn/problems/merge-intervals/?envType=study-plan-v2&envId=top-100-liked
*/
func merge(intervals [][]int) [][]int {
	if len(intervals) <= 1 {
		return intervals
	}
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0] || intervals[i][0] == intervals[j][0] && intervals[i][1] < intervals[j][1]
	})
	start, end, r := intervals[0][0], intervals[0][1], make([][]int, 0)
	for i := 1; i < len(intervals); i++ {
		if end < intervals[i][0] { // [1,2] [3,4]
			r = append(r, []int{start, end})
			start, end = intervals[i][0], intervals[i][1]
		} else if end >= intervals[i][0] && end <= intervals[i][1] { // [1,3] [2,4]
			end = intervals[i][1]
		} else if start <= intervals[i][0] && end >= intervals[i][1] { // [1,4] [2,3]
			continue
		} else {
			end = intervals[i][1]
		}
	}
	r = append(r, []int{start, end})
	return r
}

/*
189. 轮转数组
https://leetcode.cn/problems/rotate-array/?envType=study-plan-v2&envId=top-100-liked
*/
func rotate(nums []int, k int) {
	k = k % (len(nums))
	rev(nums, 0, len(nums)-1)
	rev(nums, 0, k-1)
	rev(nums, k, len(nums)-1)
	return
}

func rev(nums []int, i, j int) {
	for i < j {
		nums[i], nums[j] = nums[j], nums[i]
	}
	return
}
