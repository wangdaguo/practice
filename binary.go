package main

import (
	"fmt"
)

func main()  {
	//r := mySqrtTest(1)
	nums := []int{1,1,1,1,1,1,2,3,4,4,5,5,5,6,7,8,8,8,8}
	target := 8
	r := searchRange(nums, target)
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
	start, end := 0, x
	for start < end {
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
	start, end := 0, len(nums)-1
	for start <= end {
		middle := start + (end-start)/2
		if nums[middle] < target {
			start = middle + 1
		} else if nums[middle] > target {
			end = middle - 1
		} else  {
			rleft, rright := middle, middle
			tmpL := binarySearch(nums[0:middle], target, true)
			if tmpL != -1 && tmpL < rleft {
				rleft = tmpL
			}
			tmpR := binarySearch(nums[middle+1:], target, false)
			if tmpR != -1 && tmpR + middle + 1 > rright {
				rright = tmpR + middle + 1
			}
			return []int{rleft, rright}
		}
	}
	return []int{-1, -1}
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

