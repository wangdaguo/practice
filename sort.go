package main

import "fmt"

func main()  {
	nums := &[]int{5,3,9,6}
	quickSort(nums, 0, 3)
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

