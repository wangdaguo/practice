package main

import "fmt"

func main()  {
	//nums := &[]int{5,3,9,6}
	//quickSort(nums, 0, 3)

	nums := []int64{8,4,5,7,1,3,6,2}
	temp := make([]int64, len(nums))
	mergeSort(nums, temp, 0, 7)
	fmt.Println(temp)
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

func mergeSort(nums, temp []int64, l, r int64)  {
	if l < r {
		mid := l + (r-l)/2
		mergeSort(nums, temp, l, mid)
		mergeSort(nums, temp, mid+1, r)
		mergeList(&nums, &temp, l, mid, r)
	}
}

func mergeList(nums, temp *[]int64, l, mid, r int64)  {
	i, j, t := l, mid+1, 0
	for i <= mid && j <= r {
		if (*nums)[i] <= (*nums)[j] {
			(*temp)[t] = (*nums)[i]
			i ++
			t ++
		} else {
			(*temp)[t] = (*nums)[j]
			j ++
			t ++
		}
	}

	for i <= mid {
		(*temp)[t] = (*nums)[i]
		i ++
		t ++
	}

	for j <= r {
		(*temp)[t] = (*nums)[j]
		j ++
		t ++
	}

	for i:=int64(0); i<r; i++ {
		(*nums)[i] = (*temp)[i]
	}
}
