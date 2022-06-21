package main

import (
	"fmt"
)

func main()  {

	//r := binarySearch1([]int{1,2,3,4,5,6}, 0)
	//r := mySqrtTest(8)
	//nums := []int{1,1,1,1,1,1,2,3,4,4,5,5,5,6,7,8,8,8,8}
	//target := 8
	//r := searchRange(nums, target)
	//nums := []int{1,3,5}
	//target := 5
	//r := search33(nums, target)
	//r := findMin1([]int{1,3,5})
	r := singleNonDuplicate([]int{1,1,2,2,4,4,5,5,9})
	fmt.Println(r)
}

/**
69. x 的平方根

给你一个非负整数 x ，计算并返回 x 的 算术平方根 。
由于返回类型是整数，结果只保留 整数部分 ，小数部分将被 舍去 。
注意：不允许使用任何内置指数函数和算符，例如 pow(x, 0.5) 或者 x ** 0.5 。

示例 1：
输入：x = 4
输出：2

示例 2：
输入：x = 8
输出：2
解释：8 的算术平方根是 2.82842..., 由于返回类型是整数，小数部分将被舍去。
*/
func mySqrtTest(x int) int {
	if x == 0 {
		return 0
	}
	start, end := 1, x
	for start <= end {
		middle := start + (end-start)/2
		if middle * middle  > x {
			end = middle - 1
		} else if middle * middle < x {
			start = middle + 1
		} else {
			return middle
		}
	}
	return end
}

/**
二分查找
 */
func binarySearch1(nums []int, target int) int {
	if len(nums) < 1 {
		return -1
	}
	start, end := 0, len(nums)-1
	for start <= end {
		middle := start + (end-start) / 2
		if nums[middle] == target {
			return middle
		} else if nums[middle] < target {
			start = middle + 1
		} else if nums[middle] > target {
			end = middle - 1
		}
	}
	return -1
}

/*
34. 在排序数组中查找元素的第一个和最后一个位置
给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。
如果数组中不存在目标值 target，返回 [-1, -1]。

进阶：
你可以设计并实现时间复杂度为 O(log n) 的算法解决此问题吗？
 
示例 1：
输入：nums = [5,7,7,8,8,10], target = 8
输出：[3,4]

示例 2：
输入：nums = [5,7,7,8,8,10], target = 6
输出：[-1,-1]

示例 3：
输入：nums = [], target = 0
输出：[-1,-1]

输入：nums = [5,7,7,8,8,10], target = 8
*/
func searchRange(nums []int, target int) []int {
	if len(nums) < 1 {
		return []int{-1, -1}
	}
	left := binarySearch(nums, target, true)
	right := binarySearch(nums, target, false)
	return []int{left, right}
}

func binarySearch(nums []int, target int, isFindLeft bool) int {
	if len(nums) == 0 {
		return -1
	}
	start, end, r := 0, len(nums)-1, -1
	for start <= end {
		middle := start + (end-start)/2
		if nums[middle]  > target {
			end = middle - 1
		} else if nums[middle] < target {
			start = middle + 1
		} else {
			r = middle
			if isFindLeft {
				end = middle - 1
			} else  {
				start = middle + 1
			}
		}
	}
	return r
}

/**
https://leetcode-cn.com/problems/search-in-rotated-sorted-array/
81. 搜索旋转排序数组 II
已知存在一个按非降序排列的整数数组 nums ，数组中的值不必互不相同。
在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转 ，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,4,4,5,6,6,7] 在下标 5 处经旋转后可能变为 [4,5,6,6,7,0,1,2,4,4] 。

给你 旋转后 的数组 nums 和一个整数 target ，请你编写一个函数来判断给定的目标值是否存在于数组中。如果 nums 中存在这个目标值 target ，则返回 true ，否则返回 false 。

你必须尽可能减少整个操作步骤。

示例 1：
输入：nums = [2,5,6,0,0,1,2], target = 0
输出：true

示例 2：
输入：nums = [2,5,6,0,0,1,2], target = 3
输出：false
 */
func search(nums []int, target int) bool {
	if len(nums) < 1 {
		return false
	}
	start, end := 0, len(nums)-1
	for start <= end {
		middle := start + (end - start) / 2
		if nums[middle] == target {
			return true
		} else if nums[start] == nums[middle] {
			start ++
		} else if nums[middle] <= nums[end] {
			if target > nums[middle] && target <= nums[end] {
				start = middle + 1
			} else  {
				end = middle - 1
			}
		} else  {
			if target >= nums[start] && target < nums[middle] {
				end = middle - 1
			} else  {
				start = middle + 1
			}
		}
	}
	return false
}

/**
33. 搜索旋转排序数组
https://leetcode-cn.com/problems/search-in-rotated-sorted-array/
 */
func search33(nums []int, target int) int {
	if len(nums) < 1 {
		return -1
	}
	start, end := 0, len(nums)-1
	for start <= end {
		middle := start + (end-start)/2
		if nums[middle] == target {
			return middle
		} else if nums[start] == nums[middle] {
			start ++
		} else if nums[middle] <= nums[end] {
			if target > nums[middle] && target <= nums[end] {
				start = middle + 1
			} else  {
				end = middle - 1
			}
		} else {
			if target >= nums[start] && target < nums[middle] {
				end = middle - 1
			} else {
				start = middle + 1
			}
		}
	}
	return -1
}

/**
https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/
153. 寻找旋转排序数组中的最小值
已知一个长度为 n 的数组，预先按照升序排列，经由 1 到 n 次 旋转 后，得到输入数组。例如，原数组 nums = [0,1,2,4,5,6,7] 在变化后可能得到：
若旋转 4 次，则可以得到 [4,5,6,7,0,1,2]
若旋转 7 次，则可以得到 [0,1,2,4,5,6,7]
注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。

给你一个元素值 互不相同 的数组 nums ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 最小元素 。

你必须设计一个时间复杂度为 O(log n) 的算法解决此问题。

示例 1：
输入：nums = [3,4,5,1,2]
输出：1
解释：原数组为 [1,2,3,4,5] ，旋转 3 次得到输入数组。

示例 2：
输入：nums = [4,5,6,7,0,1,2]
输出：0
解释：原数组为 [0,1,2,4,5,6,7] ，旋转 4 次得到输入数组。

示例 3：
输入：nums = [11,13,15,17]
输出：11
解释：原数组为 [11,13,15,17] ，旋转 4 次得到输入数组。
 */
func findMin(nums []int) int {
	if len(nums) < 1 {
		return 0
	}
	start, end := 0, len(nums)-1
	for start < end {
		middle := start + (end-start)/2
		// 转折点在右边
		if nums[middle] > nums[end] {
			start = middle + 1
		} else if nums[middle] < nums[start] {
			end = middle
		} else {
			end = middle - 1
		}
	}
	return nums[start]
}

/**
https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array-ii/
154. 寻找旋转排序数组中的最小值 II
 */
func findMin1(nums []int) int {
	if len(nums) < 1 {
		return 0
	}
	start, end := 0, len(nums)-1
	for start < end {
		middle := start + (end-start)/2
		if nums[middle] > nums[end] {
			start = middle + 1
		} else if nums[middle] < nums[start] {
			end = middle
		} else if nums[middle] == nums[end] {
			end --
		} else {
			end = middle - 1
		}
	}
	return nums[start]
}

/**
540. 有序数组中的单一元素
https://leetcode.cn/problems/single-element-in-a-sorted-array/

给你一个仅由整数组成的有序数组，其中每个元素都会出现两次，唯有一个数只会出现一次。
请你找出并返回只出现一次的那个数。
你设计的解决方案必须满足 O(log n) 时间复杂度和 O(1) 空间复杂度。

输入: nums = [1,1,2,3,3,4,4,8,8]
输出: 2

输入: nums =  [3,3,7,7,10,11,11]
输出: 10
 */
func singleNonDuplicate(nums []int) int {
	if len(nums) < 1 {
		return -1
	}
	if len(nums) == 1 {
		return nums[0]
	} else if len(nums) == 2 && nums[0] == nums[1] {
		return -1
	}
	start, end := 0, len(nums)-1
	for start < end {
		middle := start + (end-start)/2
		if middle-1 >= 0 && nums[middle] == nums[middle-1] {
			r := singleNonDuplicate(nums[0:middle-1])
			if r > -1 {
				return r
			}
			r = singleNonDuplicate(nums[middle+1:])
			if r > -1 {
				return r
			}
			return -1
		} else if middle + 1 <= len(nums)-1 && nums[middle] == nums[middle+1] {
			r := singleNonDuplicate(nums[0:middle])
			if r > -1 {
				return r
			}
			if middle + 2 <= len(nums)-1 {
				r = singleNonDuplicate(nums[middle+2:])
				if r > -1 {
					return r
				}
			}
			return -1
		} else {
			return nums[middle]
		}
	}
	return -1
}

/**
https://leetcode.cn/problems/median-of-two-sorted-arrays/
4. 寻找两个正序数组的中位数

给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。
算法的时间复杂度应该为 O(log (m+n)) 。

输入：nums1 = [1,3], nums2 = [2]
输出：2.00000
解释：合并数组 = [1,2,3] ，中位数 2

输入：nums1 = [1,2], nums2 = [3,4]
输出：2.50000
解释：合并数组 = [1,2,3,4] ，中位数 (2 + 3) / 2 = 2.5
 */
func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	totalLen := len(nums1) + len(nums2)
	if totalLen % 2 == 1 {
		midIndex := totalLen / 2
		return float64(getKthElement(nums1, nums2, midIndex+1))
	} else {
		minIndex1, minIndex2 := totalLen / 2 - 1, totalLen / 2
		return float64(getKthElement(nums1, nums2, minIndex1+1) + getKthElement(nums1, nums2, minIndex2+1)) / 2
	}
}

func getKthElement(nums1, nums2 []int, k int) int {
	var index1, index2 int
	for {
		if index1 == len(nums1) {
			return nums2[index2 + k - 1]
		}
		if index2 == len(nums2) {
			return nums1[index1 + k - 1]
		}
		if k == 1 {
			return min(nums1[index1], nums2[index2])
		}
		half := k / 2
		newIndex1 := min(index1+half, len(nums1)) - 1
		newIndex2 := min(index2+half, len(nums2)) - 1
		val1, val2 := nums1[newIndex1], nums2[newIndex2]
		if val1 <= val2 {
			k = k - (newIndex1 - index1 + 1)
			index1 = newIndex1 + 1
		} else {
			k = k - (newIndex2 - index2 + 1)
			index2 = newIndex2 + 1
		}
	}
	return 0
}

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}
