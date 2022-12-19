package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"time"
)

func main()  {
	//s := []string{"a", "b", "c", "d", "e", "f", "g", "h"}
	//reverseStr1(s, 3)
	//fmt.Println(s)
	//return
	//r := GenCalendar(2022, 8)
	//fmt.Println(r)

	//date1 := &Date{
	//	Year:  2012,
	//	Month: 1,
	//	Day:   10,
	//}
	//
	//date2 := &Date{
	//	Year:  2013,
	//	Month: 1,
	//	Day:   10,
	//}
	//
	//cnt := BetweenDay(date1, date2)
	//fmt.Println(cnt)
	//
	//date3 := &Date{
	//	Year:  2022,
	//	Month: 8,
	//	Day:   19,
	//}
	//r := WeekDay(date3)
	//fmt.Println(r)
	//r := CloseToZero(3, []int{1,2,3,4,5,6,7,8,9,-1,1,2,-5})
	//fmt.Println(r)

	//nums := []int{89, 4, 76, 32, 45, 0, -90}
	////selsort(&nums)
	//shellSort(&nums)
	//fmt.Println(nums)

	//arr := []int{8,4,6,2,10}
	////InsertSort(&arr)
	//QuickSort(&arr, 0, 4)
	//fmt.Println(arr)

	//r := genknuth(5, 100, 300)

	//fmt.Println(r)

	//r := heapSort([]int{5, 3, 7, 2, 1, 10, 76, 12})
	//fmt.Println(r)

	//heap := NewHeap()
	//for _, val := range []int{5, 3, 7, 2, 1, 10} {  // 3,5
	//	heap.siftUp(val)
	//}
	//fmt.Println(heap)
	//min := heap.extractMin()
	//fmt.Println(heap, min)

	r := maxSum([]int{1, -10, 3, 4})
	fmt.Println(r)
	return
}

func maxSum(list []int) int {
	if len(list) < 1 {
		return 0
	}
	sum, tmpSum, i, j, start, end := 0, 0, 0, 0, 0, 0
	for cnt, val := range list {
		if tmpSum < 0 {
			tmpSum = val
			i, j = cnt, cnt
		} else {
			tmpSum += val
			j = cnt
		}
		if tmpSum > sum {
			sum = tmpSum
			start, end = i, j
		}
	}
	fmt.Println(start, end)
	return sum
}

type Heap struct {
	Nums []int
}

func NewHeap() *Heap {
	return &Heap{Nums:[]int{0}}
}

func (heap *Heap) siftUp(val int)  {
	if len(heap.Nums) < 1 {
		heap.Nums = append(heap.Nums, []int{0, val}...)
		return
	}
	heap.Nums = append(heap.Nums, val)
	i := len(heap.Nums)-1
	for i > 0 {
		p := i/2
		if heap.Nums[p] > heap.Nums[i] {
			heap.Nums[p], heap.Nums[i] = heap.Nums[i], heap.Nums[p]
			i = p
			continue
		}
		break
	}
	return
}

func (heap *Heap) siftDown(val int)  {
	if len(heap.Nums) < 1 {
		heap.Nums = append(heap.Nums, []int{0, val}...)
		return
	}
	i := 1
	for i < len(heap.Nums) {
		p := 2*i
		if p >= len(heap.Nums) {
			break
		}
		if p+1<len(heap.Nums) && heap.Nums[p+1] < heap.Nums[p] {
			p ++
		}
		if heap.Nums[p] < val {
			heap.Nums[p], heap.Nums[i] = heap.Nums[i], heap.Nums[p]
			i = p
			continue
		}
		break
	}
	if i < len(heap.Nums) {
		heap.Nums[i] = val
	}
	return
}

func (heap *Heap) extractMin() int {
	defer func() {
		lastVal := heap.Nums[len(heap.Nums)-1]
		heap.Nums = heap.Nums[:len(heap.Nums)-1]
		heap.siftDown(lastVal)
	}()
	return heap.Nums[1]
}

func heapSort(nums []int) []int {
	if len(nums) < 1 {
		return nums
	}
	heap := NewHeap()
	for i:=0; i<len(nums); i++ {
		heap.siftUp(nums[i])
	}
	var r []int
	for len(heap.Nums) > 1 {
		r = append(r, heap.extractMin())
	}
	return r
}

func genknuth(m, n, k int) []int {
	var r []int
	if m > n {
		return r
	}
	for i:=0; i<n; i++ {
		if genRand(k) % n <= m {
			m --
			r = append(r, i)
		} else {
			n --
		}
	}
	return r
}

func genRand(k int) int {
	rand.Seed(time.Now().UnixNano())
	return rand.Intn(k)
}

var monthDay = []int{0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}

func GenCalendar(year, month int) [][]int {
	var r [][]int
	startDate := &Date{
		Year:  year,
		Month: month,
		Day:   1,
	}
	tmpWeekDay := WeekDay(startDate)
	lastDay := monthDay[month]
	if IsRunYear(year) && month == 2 {
		lastDay = 29
	}
	j:=1
	tmpArr := []int{}
	for i:=startDate.Day; i<=lastDay; i ++ {
		if len(tmpArr) % 7 == 0 {
			tmpArr = []int{}
		}
		for ; j<=7; j++ {
			if j < tmpWeekDay {
				tmpArr = append(tmpArr, 0)
				continue
			}
			j = 8
			break
		}
		tmpArr = append(tmpArr, i)
		if len(tmpArr) % 7 == 0 || i == lastDay {
			r = append(r, tmpArr)
		}
	}
	return r
}

type Date struct {
	Year int
	Month int
	Day int
}

func IsRunYear(year int) bool {
	if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 {
		return true
	}
	return false
}

func YearDay(date *Date) (day int) {
	for i:=1; i<date.Month; i++ {
		day += monthDay[i]
	}
	if date.Month > 2 && IsRunYear(date.Year) {
		day += 1
	}
	day += date.Day
	return day
}

func BetweenDay(date1, date2 *Date) int {
	if date1.Year == date2.Year {
		return YearDay(date2) - YearDay(date1)
	}
	var totalDay int
	for i:=date1.Year; i<date2.Year; i++ {
		totalDay += 365
		if IsRunYear(i) {
			totalDay += 1
		}
	}
	return totalDay + YearDay(date2) - YearDay(date1)
}

//选取1900年作为参考，日历都是以1900为参考的，1900.1.1为星期一，其它日期相对它的天数与7取余
// 数再加上1就是对应的星期几。这里用1~7表示星期一到星期天
func WeekDay(date *Date) int {
	startDate := &Date{
		Year:  1900,
		Month: 1,
		Day:   1,
	}
	return BetweenDay(startDate, date) % 7 + 1
}

// clinic
var rule = []string{"et-ic", "al-is-tic", "s-tic", "p-tic", "-lyt-ic", "ot-ic", "an-tic",
	"n-tic", "c-tic", "at-ic", "h-nic", "n-ic", "m-ic", "l-lic", "b-lic", "-clic", "l-ic",
	"h-ic", "f-ic", "d-ic", "-bic", "a-ic", "-mac", "i-ac"}

func Getlastsamechar(word string) int {
	ruleList := make(map[string]int)
	for index, newStr := range rule {
		str := strings.Replace(newStr, "-", "", -1 )
		ruleList[str] = index
	}
	r := -1
	runeWord := []rune(word)
	for k, v := range ruleList {
		if len(runeWord) < len(k) {
			continue
		}
		rule := []rune(k)
		i := 0
		for i < len(k) {
			if runeWord[len(runeWord)-i-1] != rule[len(rule)-i-1] {
				break
			}
			i ++
		}
		if i == len(rule) {
			r = v
			break
		}
	}
	return r
}

/**
4.2
 */
func binarySearchFirst(arr []int, val int) int {
	if len(arr) < 1 {
		return -1
	}
	start, end := 0, len(arr)-1
	lessEqual := -1
	var compareCnt int
	for start <= end {
		middle := start + (end-start) / 2
		if arr[middle] > val {
			end = middle - 1
		} else if arr[middle] < val {
			start = middle + 1
		} else {
			lessEqual = middle
			end = middle-1
		}
		compareCnt ++
	}
	fmt.Println(compareCnt)
	return lessEqual
}

func binarySearchT(arr []int, start, end, val int) int {
	if len(arr) < 1 {
		return -1
	}
	for start <= end {
		middle := start + (end-start) / 2
		if arr[middle] > val {
			return binarySearchT(arr, start, middle-1, val)
		} else if arr[middle] < val {
			return binarySearchT(arr, start+1, end, val)
		} else {
			return middle
		}
	}
	return -1
}

/**
triangle = [
[2],
[3,4],
[6,5,7],
[4,1,8,3]
]
f[i][j] = min(f[i-1][j-1], f[i-1][j]) +c[i][j]
 */
func minimumTotal(triangle [][]int) int {
	if len(triangle) < 1 {
		return 0
	}
	f := make([][]int, len(triangle))
	for i:=0; i<len(triangle); i++  {
		f[i] = make([]int, len(triangle))
	}
	f[0][0] = triangle[0][0]
	for i:=1; i<len(triangle); i++ {
		f[i][0] = f[i-1][0] + triangle[i][0]
		for j:=1; j<i; j++ {
			f[i][j] = minM(f[i-1][j-1], f[i-1][j]) + triangle[i][j]
		}
		f[i][i] = f[i-1][i-1] + triangle[i][i]
	}
	ans := math.MaxInt32
	for i := 0; i < len(triangle); i++ {
		ans = minM(ans, f[len(triangle)-1][i])
	}
	return ans
}

func minM(x... int) int {
	min := x[0]
	for i:=1; i<len(x); i++ {
		if x[i] < min {
			min = x[i]
		}
	}
	return min
}

func maxM(x... int) int {
	max := x[0]
	for i:=1; i<len(x); i++ {
		if x[i] > max {
			max = x[i]
		}
	}
	return max
}

/**
https://leetcode.cn/problems/maximum-subarray/
53. 最大子数组和
 */
func maxSubArraySum(nums []int) int {
	if len(nums) < 1 {
		return 0
	}
	minStart, start := 0, 0
	var maxEnd, end int
	maxSum, sum := nums[0], nums[0]
	for i:=1; i<len(nums); i++ {
		if sum < 0 {
			sum = nums[i]
			start, end = i, i
		} else {
			sum += nums[i]
			end = i
		}
		if sum > maxSum {
			maxSum = sum
			minStart = start
			maxEnd = end
		}
	}
	fmt.Printf("start is: %v, end is: %v\n", minStart, maxEnd)
	return maxSum
}

func maxSubArray(nums []int) int {
	if len(nums) <= 0 {
		return 0
	}
	sum, maxSum := nums[0], nums[0]
	for i:=1; i<len(nums); i++ {
		if sum < 0 {
			sum = nums[i]
		} else {
			sum += nums[i]
		}
		if sum > maxSum {
			maxSum = sum
		}	
	}
	return maxSum
}

func maxSubArray1(nums []int) int {
	maxsofar, maxendinghere := nums[0], nums[0]
	for i:=1;i<len(nums);i++ {
		maxendinghere = maxM(maxendinghere+nums[i], nums[i])
		maxsofar = maxM(maxsofar, maxendinghere)
	}
	return maxsofar
}

func selsort(nums *[]int)  {
	if nums == nil || len(*nums) < 1 {
		return
	}
	for i:=0; i<len(*nums); i++ {
		for j:=i+1; j<len(*nums); j++ {
			if (*nums)[i] > (*nums)[j] {
				(*nums)[i], (*nums)[j] = (*nums)[j], (*nums)[i]
			}
		}
	}
	return
}

func shellSort(nums *[]int)  {
	if nums == nil || len(*nums) < 1 {
		return
	}
	dis := len(*nums) / 2
	for dis > 0 {
		for i:=0; i<len(*nums); i++ {
			j := i
			for j >=0 && j + dis < len(*nums) && (*nums)[j] > (*nums)[j+dis] {
				(*nums)[j], (*nums)[j+dis] = (*nums)[j+dis], (*nums)[j]
				j -= dis
			}
		}
		dis /= 2
	}
	return
}

func InsertSort(x *[]int)  {
	if len(*x) < 1 {
		return
	}
	for i:=1; i<len(*x); i++ {
		for j:=i; j>0; j-- {
			if (*x)[j] <= (*x)[j-1] {
				(*x)[j], (*x)[j-1] = (*x)[j-1], (*x)[j]
				continue
			}
			break
		}
	}
}

func QuickSort(x *[]int, l, r int)  {
	if len(*x) < 1 || l >= r {
		return
	}
	partition, start, end := (*x)[l], l, r
	for start < end {
		for start < end && (*x)[end] >= partition {
			end --
		}
		(*x)[start] = (*x)[end]
		for start < end && (*x)[start] <= partition {
			start ++
		}
		(*x)[end] = (*x)[start]
	}
	(*x)[start] = partition
	QuickSort(x, l, start)
	QuickSort(x, start+1, r)
	return
}

func findKthLargest1(nums []int, k int) int {
	//	r := findKLargest(nums, 0, 0, 1)
	return quickSelect(nums, 0, len(nums)-1, len(nums)-k)
}

func quickSelect(nums []int, i, j, k int) int {
	if k < 0 || k > len(nums) || len(nums) < 1 {
		return 0
	}
	start, end, target := i, j, k
	for start < end {
		partition := partition(&nums, start, end)
		if target == partition {
			return nums[k]
		} else if target > partition {
			start = partition + 1
		} else  {
			end = partition - 1
		}
	}
	return nums[start]
}

func partition(nums *[]int, start, end int) int {
	p := (*nums)[start]
	for start < end {
		for start < end && (*nums)[end] >= p {
			end --
		}
		(*nums)[start] = (*nums)[end]
		for start < end && (*nums)[start] <= p {
			start ++
		}
		(*nums)[end] = (*nums)[start]
	}
	(*nums)[start] = p
	return start
}
