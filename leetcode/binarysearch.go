package main

import "fmt"

/**
278. 第一个错误的版本
https://leetcode-cn.com/problems/first-bad-version/
给定 n = 5，并且 version = 4 是第一个错误的版本。

调用 isBadVersion(3) -> false
调用 isBadVersion(5) -> true
调用 isBadVersion(4) -> true

所以，4 是第一个错误的版本。 
*/
func firstBadVersion(n int) int {
	if n < 1 {
		return n
	}
	start, end := 1, n
	for start <= end {
		mid := (start + end) / 2
		if isBadVersion(mid) {
			end = mid - 1
		} else {
			start = mid + 1
		}
	}
	if isBadVersion(start) {
		return start
	}
	return start-1
}

func isBadVersion(n int) bool {
	if n >= 2 {
		return true
	}
	return false
}

/**
35. 搜索插入位置
https://leetcode-cn.com/problems/search-insert-position/
给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
输入: [1,3,5,6], 2
输出: 1
 */
func searchInsert2(nums []int, target int) int {
	if len(nums) < 1 {
		return 0
	}
	start := 0
	end := len(nums)-1
	for start <= end {
		mid := (start + end) / 2
		if nums[mid] > target {
			end = mid - 1
		} else if nums[mid] < target {
			start = mid + 1
		} else  {
			return mid
		}
	}
	if start > len(nums)-1 || nums[start] > target {
		return start
	}
	return start + 1
}

/**
33. 搜索旋转排序数组
https://leetcode-cn.com/problems/search-in-rotated-sorted-array/
输入: nums = [4,5,6,7,0,1,2], target = 0
输出: 4
 */
func search3(nums []int, target int) int {
	if len(nums) < 1 {
		return -1
	}
	start := 0
	end := len(nums)-1
	for start <= end {
		mid := (start + end) / 2
		if nums[mid] == target {
			return mid
		}
		if nums[mid] >= nums[start] {
			if nums[mid] > target && nums[start] <= target {
				end = mid - 1
			} else {
				start = mid + 1
			}
		} else {
			if nums[mid] < target && nums[end] >= target {
				start = mid + 1
			} else {
				end = mid - 1
			}
		}
	}
	return -1
}

/**
81. 搜索旋转排序数组 II
https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/
输入: nums = [2,5,6,0,0,1,2], target = 0
输出: true
 */
func search4(nums []int, target int) bool {
	if len(nums) < 1 {
		return false
	}
	start, end := 0, len(nums)-1
	for start <= end {
		mid := (start + end) / 2
		if nums[mid] == target {
			return true
		} else if nums[mid] == nums[start] {
			start ++
		} else if nums[mid] == nums[end] {
			end --
		} else if nums[mid] > nums[start] {
			if nums[start] <= target && nums[mid] > target {
				end = mid - 1
			} else {
				start = mid + 1
			}
		} else {
			if nums[mid] <= target && nums[end] > target {
				start = mid + 1
			} else {
				end = mid - 1
			}
		}
	}
	return false
}

/**
153. 寻找旋转排序数组中的最小值
https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/
输入: [3,4,5,1,2]
输出: 1
 */
func findMin3(nums []int) int {
	if len(nums) < 1 {
		return 0
	}
	start, end := 0, len(nums)-1
	for start < end {
		mid := (start+end) / 2
		if nums[mid] > nums[end] {
			start = mid + 1
		} else {
			end = mid
		}
	}
	return nums[start]
}

/**
154. 寻找旋转排序数组中的最小值 II
https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/
输入: [1,3,5]
输出: 1
输入: [2,2,2,0,1]
输出: 0
 */
func findMin4(nums []int) int {
	if len(nums) < 1 {
		return 0
	}
	start, end := 0, len(nums)-1
	for start < end {
		mid := (start+end) / 2
		if nums[mid] > nums[end] {
			start = mid + 1
		} else if nums[mid] < nums[end] {
			end = mid
		} else  {
			end --
		}
	}
	return nums[start]
}

/**]
162. 寻找峰值
https://leetcode-cn.com/problems/find-peak-element/
输入: nums = [1,2,3,1]
输出: 2
解释: 3 是峰值元素，你的函数应该返回其索引 2。
 */
func findPeakElement1(nums []int) int {
	if len(nums) < 1 {
		return -1
	}
	start, end := 0, len(nums)-1
	for start < end {
		mid := (start + end) / 2
		if nums[mid] < nums[mid+1] {
			start = mid+1
		} else  {
			end = mid
		}
	}
	return start
}

/**
374. 猜数字大小
https://leetcode-cn.com/problems/guess-number-higher-or-lower/

 */
func guessNumber(n int) int {
	if n < 1 {
		return 0
	}
	start, end := 1, n
	for start <= end {
		mid := (start+end)/2
		gus := guess(mid)
		if gus == 0 {
			return mid
		} else if gus > 0 {
			start = mid+1
		} else  {
			end = mid -1
		}
	}
	return -1
}

func guess(gusNum int) int {
	if gusNum > 6 {
		return 1
	} else if gusNum < 6 {
		return -1
	} else {
		return 0
	}
}

/**
34. 在排序数组中查找元素的第一个和最后一个位置
https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/
输入: nums = [5,7,7,8,8,10], target = 8
输出: [3,4]
 */
func extremeInsertionIndex1(nums []int, target int, left bool) int {
	start, end, middle := 0, len(nums), 0
	for start < end {
		middle = start + (end - start) / 2
		if nums[middle] > target || (left && nums[middle] == target) {
			end = middle
		} else {
			start = middle+1
		}
	}
	return start
}

func searchRange1(nums []int, target int) []int {
	r := []int{-1, -1}
	if len(nums) < 1 {
		return r
	}
	left := extremeInsertionIndex(nums, target, true)
	if left == len(nums) || nums[left] != target {
		return r
	}
	r[0] = left
	r[1] = extremeInsertionIndex(nums, target, false) - 1
	return r
}


func main()  {
	//n := firstBadVersion(3)
	//n := searchInsert2([]int{1,3,5}, 3)
	//n := search3([]int{4,5,6,7,0,1,2}, 0)
	//n := findMin3([]int{4,5,6,7,0,1,2})
	//n := findMin4([]int{2,2,2,0,1})
	//n := findPeakElement1([]int{1,2,3,1})
	//n := guessNumber(10)
	//n := searchRange1([]int{5,7,7,8,8,10}, 8)
	var n *int64
	fmt.Println(n)
}