package main

import "fmt"

func main()  {
	//nums := &[]int{5,3,9,6}
	//quickSort(nums, 0, 3)

	//nums := []int64{8,4,5,7,10,3,6,2}
	//nums := []int64{8,4,5,7}
	//temp := make([]int64, len(nums))
	//mergeSort(nums, temp, 0, 3)
	//nums := []int64{8,4,5,7}
	//insertSort(&nums, len(nums))
	nums := []int64{8,4,5,7}
	//bubbleSort(&nums, len(nums))
	selectSort(&nums, len(nums))
	fmt.Println(nums)
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

func findKthLargest(nums []int64, k int) int64 {
	if k < 1 || k > len(nums) {
		return 0
	}
	start, end, target := 0, len(nums)-1, len(nums)-k
	for start < end {
		key := quickSort1(nums, start, end)
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

func quickSort1(nums []int64, start, end int) int {

}
