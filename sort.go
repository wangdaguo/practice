package main

import (
	"fmt"
	"sort"
)

func main()  {
	//nums := &[]int{5,3,9,6}
	//quickSort(nums, 0, 3)

	//nums := []int64{8,4,5,7,10,3,6,2}
	//nums := []int64{8,4,5,7}
	//temp := make([]int64, len(nums))
	//mergeSort(nums, temp, 0, 3)
	//nums := []int64{8,4,5,7}
	//insertSort(&nums, len(nums))
	//nums := []int64{8,4,5,7}
	//bubbleSort(&nums, len(nums))
	//selectSort(&nums, len(nums))
	//r := findKthLargest(nums, 1)
	//nums := []int{1,1,1,1,2,2,3,4,4,4,4,4,6,7,7,7,7}
	//r := topKFrequent(nums, 3)
	//r := frequencySort("aabbaccbb")
	sortColors([]int{2,0,2,1,1,0})
	//fmt.Println(r)
	return
}

func quickSort(nums *[]int, l, r int)  {
	if l >= r {
		return
	}
	start, end, key := l, r, (*nums)[l]
	for start < end {
		for start < end && (*nums)[end] >= key {
			end --
		}
		(*nums)[start] = (*nums)[end]
		for start < end && (*nums)[start] <= key {
			start ++
		}
		(*nums)[end] = (*nums)[start]
	}
	(*nums)[start] = key
	quickSort(nums, l, start)
	quickSort(nums, start+1, r)
}

/**
l, r, mid
0, 3, 1 -> 0, 1, 0
0, 3, 1 -> 2, 3, 2
0, 3, 1 -> 0, 3, 1
 */
func mergeSort(nums, temp []int64, l, r int64)  {
	if l < r {
		mid := l + (r-l)/2
		mergeSort(nums, temp, l, mid)
		mergeSort(nums, temp, mid+1, r)
		mergeList(&nums, &temp, l, mid, r)
	}
}

/**
{8,4,5,7}
l, r, mid
0, 1, 0   {4,8,5,7}
2, 3, 2   {4,8,5,7}
0, 3, 1   {4,8},{5,7}
 */
func mergeList(nums, temp *[]int64, l, mid, r int64) {
	i, j, t := l, mid+1, l // 0, 1, 0; 2, 3, 2;  0, 2, 0
	for i <= mid && j <= r {
		if (*nums)[i] <= (*nums)[j] {
			(*temp)[t] = (*nums)[i]
			i++
			t++
		} else {
			(*temp)[t] = (*nums)[j]
			j++
			t++
		}
	}

	for i <= mid {
		(*temp)[t] = (*nums)[i]
		i++
		t++
	}

	for j <= r {
		(*temp)[t] = (*nums)[j]
		j++
		t++
	}

	for i := int64(0); i <= r; i++ {
		(*nums)[i] = (*temp)[i]
	}
}

func insertSort(nums *[]int64, n int)  {
	if n < 1 {
		return
	}
	for i:=0;i<n-1;i++ {
		for j:=i+1;j>0;j-- {
			if (*nums)[j-1] > (*nums)[j] {
				(*nums)[j-1], (*nums)[j] = (*nums)[j], (*nums)[j-1]
			}
		}
	}
	return
}

func bubbleSort(nums *[]int64, n int)  {
	if n < 1 {
		return
	}
	for i:=0;i<n-1;i++ {
		for j:=0;j<n-i-1;j++ {
			if (*nums)[j] > (*nums)[j+1] {
				(*nums)[j], (*nums)[j+1] = (*nums)[j+1], (*nums)[j]
			}
		}
	}
	return
}

func selectSort(nums *[]int64, n int)  {
	if n < 1 {
		return
	}
	for i:=0;i<n-1;i++ {
		for j:=i+1;j<n;j++ {
			if (*nums)[i] > (*nums)[j] {
				(*nums)[i], (*nums)[j] = (*nums)[j], (*nums)[i]
			}
		}
	}
	return
}

/**
215. 数组中的第K个最大元素
https://leetcode.cn/problems/kth-largest-element-in-an-array/
 */
func findKthLargest(nums []int64, k int) int64 {
	if k < 1 || k > len(nums) {
		return 0
	}
	start, end, target := 0, len(nums)-1, len(nums)-k
	for start < end {
		key := quickSortT(&nums, start, end)
		if key == target {
			return nums[key]
		}
		if key < target {
			start = key + 1
		} else {
			end = key - 1
		}
	}
	return nums[start]
}

func quickSort1(nums *[]int64, start, end int) int {
	i, j := start, end
	for i < j {
		for i < end && (*nums)[i] <= (*nums)[start] {
			i ++
		}
		for start < j && (*nums)[j] >= (*nums)[start] {
			j --
		}
		if i >= j {
			break
		}
		(*nums)[i], (*nums)[j] = (*nums)[j], (*nums)[i]
	}
	(*nums)[start], (*nums)[j] = (*nums)[j], (*nums)[start]
	return j
}

func quickSortT(nums *[]int64, start, end int) int {
	i, j, piov := start, end, (*nums)[start]
	for i < j {
		for i < j && (*nums)[j] >= piov {
			j --
		}
		(*nums)[i] = (*nums)[j]
		for i < j && (*nums)[i] <= piov {
			i ++
		}
		(*nums)[j] = (*nums)[i]
	}
	(*nums)[i] = piov
	return i
}

/**
347. 前 K 个高频元素
https://leetcode.cn/problems/top-k-frequent-elements/
 */
func topKFrequent(nums []int, k int) []int {
	mp := make(map[int]int)
	var maxCnt int
	for  _, v := range nums {
		mp[v] += 1
		maxCnt = max1(maxCnt, mp[v])
	}
	buckets := make(map[int][]int, maxCnt)
	for k, v := range mp {
		buckets[v] = append(buckets[v], k)
	}
	var r[]int
	var cnt int
	for i:=maxCnt; i>=0; i-- {
		if data, ok:=buckets[i]; ok {
			for _, d := range data {
				r = append(r, d)
				cnt ++
				if k == cnt {
					return r
				}
			}
		}
	}
	return r
}

func max1(i, j int) int {
	if i > j {
		return i
	}
	return j
}

/**
451. 根据字符出现频率排序
https://leetcode.cn/problems/sort-characters-by-frequency/
输入: s = "tree"
输出: "eert"
 */
func frequencySort(s string) string {

	if len(s) < 1 {
		return ""
	}

	mp := make(map[rune]int)
	var maxCnt int
	for _, charInt := range s {
		mp[charInt] += 1
		maxCnt = max1(maxCnt, mp[charInt])
	}

	buckets := make(map[int][]rune, maxCnt)
	for k, v := range mp {
		buckets[v] = append(buckets[v], k)
	}

	var r string
	for i:=maxCnt; i>=0; i-- {
		if data, ok := buckets[i]; ok {
			for _, d := range data {
				for j:=0; j<i; j++ {
					r = fmt.Sprintf("%s%s", r, string(d))
				}
			}
		}
	}
	return r
}

func frequencySort1(s string) string {

	if len(s) < 1 {
		return ""
	}

	mp := make(map[rune]int)
	for _, charInt := range s {
		mp[charInt] += 1
	}

	buckets :=  make([]rune, 0)
	for k, _ := range mp {
		buckets = append(buckets, k)
	}

	sort.Slice(buckets, func(i, j int) bool {
		return mp[buckets[i]] > mp[buckets[j]]
	})

	var r string
	for _, v := range buckets {
		for i:=0; i<mp[v]; i++ {
			r = fmt.Sprintf("%s%s", r, string(v))
		}
	}
	return r
}

func frequencySort2(s string) string {
	maxfreq := 0
	cmap := make(map[byte]int)

	for _,c := range []byte(s){
		cmap[c]++
		maxfreq = max1(maxfreq,cmap[c])
	}

	//初始化桶，并填充内容
	buckets := make([][]byte,maxfreq+1)
	for c,f := range cmap{
		buckets[f] = append(buckets[f],c)
	}

	ans := make([]byte,0,len(s))
	for i := maxfreq ; i > 0 ; i--{
		blen := len(buckets[i])
		for j:=0; j < blen; j++{
			c := buckets[i][j]
			for k := 0; k < i; k++{
				ans = append(ans,c)
			}
		}
	}
	return string(ans)
}

/**
75. 颜色分类
https://leetcode.cn/problems/sort-colors/
 */
func sortColors(nums []int)  {
	if len(nums) <= 0 {
		return
	}
	cur, p0, p2 := 0, 0, len(nums)-1
	for cur <= p2 {
		if nums[cur] == 0 {
			nums[p0], nums[cur] = nums[cur], nums[p0]
			p0 ++
			cur ++
		} else if nums[cur] == 2 {
			nums[p2], nums[cur] = nums[cur], nums[p2]
			p2 --
		} else {
			cur ++
		}
	}
}