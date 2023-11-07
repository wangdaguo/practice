package main

import (
	"fmt"
	"sort"
	"strconv"
)

func searchInsert(nums []int, target int) int {
	if len(nums) == 0 {
		return 0
	}
	start := 0
	end := len(nums) - 1
	middle := 0
	for {
		if start > end {
			middle = len(nums)
			break
		}
		middle = (start + end) / 2
		if nums[middle] >= target {
			if middle == 0 || nums[middle-1] < target {
				break
			} else {
				end = middle-1
			}
		} else {
			start = middle + 1
		}

	}
	return middle
}

func removeElement(nums []int, val int) int {
	j := 0
	for _, v := range nums {
		if v != val {
			nums[j] = v
			j ++
		}
	}
	return j
}

func removeDuplicates(nums []int) int {
	if len(nums) <= 0 {
		return 0
	}
	target := nums[:1]
	for k, v := range nums {
		if k == 0 {
			continue
		}
		if v != target[len(target)-1] {
			target = append(target, v)
		}
	}
	return len(target)
}

func removeDuplicates1(nums []int) int {
	if len(nums) <= 0 {
		return 0
	}
	target := nums[:1]
	for k, v := range nums {
		if k == 0 {
			continue
		}
		if v != target[len(target)-1] ||
			(v == target[len(target)-1] && len(target)-2<0) ||
			 v != target[len(target)-2] {
			target = append(target, v)
		}
	}
	return len(target)
}

func removeDuplicates3(nums []int) int {
	i, j := 0, 0
	m := make(map[int]int)

	for j < len(nums) {
		if m[nums[j]] >= 2 {
			j++
			continue
		}
		m[nums[j]]++
		nums[i], nums[j] = nums[j], nums[i]
		i++
		j++
	}
	return i
}

func rotateOk(nums []int, k int)  {
	if len(nums) <= 0 {
		return
	}
	k = k % len(nums)
	rotate(nums, 0, len(nums)-1)
	rotate(nums, 0, k-1)
	rotate(nums, k, len(nums)-1)
}

func rotate(nums []int, start int, end int)  {
	for start < end {
		nums[start], nums[end] = nums[end], nums[start]
		start ++
		end --
	}
}

//输入:  secret = "1807", guess = "7810"
//输出: "1A3B"
//解释: 1 公牛和 3 奶牛。公牛是 8，奶牛是 0, 1 和 7。
func getHint(secret string, guess string) string {
	var A, B, i, j int
	mapA := make(map[int]int)
	mapB := make(map[int]int)
	for i < len(secret) {
		s1 := fmt.Sprintf("%c", secret[i])
		s2 := fmt.Sprintf("%c", guess[i])
		i1, _ := strconv.Atoi(s1)
		i2, _ := strconv.Atoi(s2)

		if i1 == i2 {
			A ++
		} else {
			mapA[i1] ++
			mapB[i2] ++
		}
		i ++
	}
	for j < 10 {
		if mapA[j] < mapB[j] {
			B += mapA[j]
		} else {
			B += mapB[j]
		}
		j ++
	}
	return fmt.Sprint(A, "A", B, "B")
}

//输入:  secret = "1807", guess = "7810"
//输出: "1A3B"
//解释: 1 公牛和 3 奶牛。公牛是 8，奶牛是 0, 1 和 7。
func getHint1(secret string, guess string) string {
	bull := 0
	cows := 0
	buckets := [10]int{}
	for i := 0; i < len(secret); i++ {
		if secret[i] == guess[i] {
			bull++
			continue
		}
		buckets[secret[i]-'0']++
		buckets[guess[i]-'0']--
	}
	for i := 0; i < 10; i++ {
		if buckets[i] > 0 {
			cows += buckets[i]
		}
	}
	cows = len(secret) - bull - cows
	return fmt.Sprintf("%vA%vB", bull, cows)
}

func canCompleteCircuit(gas []int, cost []int) int {
	startAddr := []int{}
	for k, v := range gas {
		if v >= cost[k] {
			startAddr = append(startAddr, k)
		}
	}
	leftGass := 0
	for _, vv := range startAddr {
		v := vv
		leftGass = 0
		for {
			leftGass += gas[v]
			leftGass = leftGass - cost[v]
			if leftGass < 0 {
				break
			}
			v++
			v = v % len(gas)
			if leftGass == 0 && v != vv {
				break
			}
			if leftGass >= 0 && v == vv {
				return vv
			}
		}
	}
	return -1
}

func canCompleteCircuit1(gas []int, cost []int) int {
	total, sum, start := 0, 0, 0
	for i:=0; i<len(gas); i++ {
		total += gas[i] - cost[i]
		sum += gas[i] - cost[i]
		if sum < 0 {
			sum = 0
			start = (i + 1) % len(gas)
		}
	}
	if total < 0 {
		return -1
	}
	return start;
}

func generate(numRows int) [][]int {
	if numRows == 0 {
		return [][]int{}
	}
	triangle := [][]int{{1}}
	for rowNum:=1; rowNum<numRows; rowNum++ {
		row := make([]int, rowNum+1)
		row[0] = 1
		preRow := triangle[rowNum-1]
		for j:=1; j<rowNum; j++ {
			leftUp := preRow[j-1]
			rightUp := preRow[j]
			row[j] = leftUp + rightUp
		}
		row[rowNum] = 1
		triangle = append(triangle, row)
	}
	return triangle
}

func getRow(rowIndex int) []int {
	ret := []int{1}
	if rowIndex == 0 {
		return ret
	}
	for i:=1; i<rowIndex; i++ {
		ret = append([]int{1}, ret...)
		for j:=1; j<i; j++ {
			ret[j] = ret[j] + ret[j+1]
		}
	}
	return ret
}

//[3,3,2]
//3
func majorityElement(nums []int) int {
	var target, count int
	for _, v := range nums {
		if count == 0 {
			target = v
			count ++
		} else {
			if target == v {
				count ++
			} else {
				count --
			}
		}
	}
	return target
}

func majorityElement1(nums []int) []int {
	if len(nums) == 0 {
		return []int{}
	}
	res := []int{}
	candidateA := nums[0];
	candidateB := nums[0];
	countA := 0;
	countB := 0;
	for _, v := range nums {
		if v == candidateA {
			countA ++
			continue
		}
		if v == candidateB {
			countB ++
			continue
		}
		if countA == 0 {
			candidateA = v
			countA ++
			continue
		}
		if countB == 0 {
			candidateB = v
			countB ++
			continue
		}
		countA --
		countB --
	}
	countA, countB = 0, 0
	for _, v := range nums{
		if v == candidateA {
			countA ++
			continue
		}
		if v == candidateB {
			countB ++
			continue
		}
	}
	if countA > len(nums) / 3 {
		res = append(res, candidateA)
	}
	if countB > len(nums) / 3 {
		res = append(res, candidateB)
	}
	return res
}

//0,1,3,5,6
func hIndex(citations []int) int {
	sort.Ints(citations)
	index := 0
	for i:=len(citations)-1; i>0; i-- {
		if citations[i] > index {
			index ++
		}
	}
	return index
}

func hIndex1(citations []int) int {
	mp := make(map[int]int, len(citations)+1)
	for _, v := range citations {
		if v > len(citations) {
			mp[len(citations)] ++
		} else {
			mp[v] ++
		}
	}

	for i:=len(citations); i>0; i-- {
		if mp[i] >= i {
			return i
		} else {
			mp[i-1] += mp[i]
		}
	}
	return 0
}

func containsDuplicate(nums []int) bool {
	mp := make(map[int]int, len(nums))
	for _, v := range nums{
		if _, ok := mp[v]; ok {
			return true
		} else  {
			mp[v] = 1
		}
	}
	return false
}

func containsNearbyDuplicate(nums []int, k int) bool {
	mp := make(map[int]int)
	for i, v := range nums{
		if m, ok := mp[v];ok {
			if i-m <= k {
				return true
			}
		}
		mp[v] = i
	}
	return false
}

func containsNearbyAlmostDuplicate(nums []int, k, t int) bool {
	if t < 0 {
		return false
	}
	d := make(map[int]int)
	w := t + 1
	for i:=0; i<len(nums); i++ {
		m := getID(nums[i], w)
		if _, ok := d[m]; ok {
			return true
		}
		if _, ok := d[m-1]; ok && abs(nums[i]-d[m-1]) < w {
			return true
		}
		if _, ok := d[m+1]; ok && abs(nums[i]-d[m+1]) < w {
			return true
		}
		d[m] = nums[i]
		if i >= k {
			delete(d, getID(nums[i-k], w))
		}
	}
	return false
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func getID(x, y int) int {
	if x < 0 {
		return x/y - 1
	}
	return x/y
}

func findMin(nums []int) int {
	if len(nums) <= 0 {
		return 0
	}
	start := 0
	end := len(nums) - 1
	var middle int
	for start < end {
		middle = (start + end) / 2
		if nums[middle] > nums[end] {
			start = middle + 1
		} else {
			end = middle
		}
	}
	return nums[start]
}

func findMin1(nums []int) int {
	if len(nums) <= 0 {
		return 0
	}
	start := 0
	end := len(nums) - 1
	var middle int
	for start < end {
		middle = (start + end) / 2
		if nums[middle] > nums[end] {
			start = middle + 1
		} else if nums[middle] < nums[end] {
			end = middle
		} else {
			end --
		}
	}
	return nums[start]
}

func canJump(nums []int) bool {
	memo := make([]int, len(nums)-1)
	for i:=0; i<len(nums); i++ {
		memo[i] = 3
	}
	memo[len(nums)-1] = 1
	for i:=len(nums)-2; i>=0; i-- {
		furthestJump := min(i+nums[i], len(nums)-1)
		for j:=i+1; j<=furthestJump; j++ {
			if memo[j] == 1 {
				memo[i] = 1
				break
			}
		}
	}
	return memo[0] == 1
}

func min(x, y int) int {
	if x > y {
		return y
	}
	return x
}

func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

func canJump1(nums []int) bool {
	end := len(nums)-1
	for i:=end; i>=0; i-- {
		if i + nums[i] >= end {
			end = i
		}
	}
	return end == 0
}

func maxProfit(prices []int) int {
	if len(prices) <= 0 {
		return 0
	}
	minPrice, maxProfit := int(^uint(0) >> 1), 0
	for _, price := range prices {
		if price < minPrice {
			minPrice = price
		} else if price - minPrice > maxProfit {
			maxProfit = price - minPrice
		}
	}
	return maxProfit
}

func maxProfit1(prices []int) int {
	maxProfit := 0
	for i:=1; i<len(prices); i++ {
		if prices[i] > prices[i-1] {
			maxProfit += prices[i] - prices[i-1]
		}
	}
	return maxProfit
}

/**
dp_hold[i] = max(dp_hold[i - 1], dp_free[i - 1] - price[i])
dp_cd[i] = dp_hold[i - 1] + price[i]
dp_free[i] = max(dp_cd[i - 1], dp_free[i - 1])
 */
func maxProfit2(prices []int) int {
	hold, coolDown, free := ^int(^uint(0) >> 1), 0, 0
	for _, v := range prices {
		preCoolDown := coolDown
		coolDown = hold + v
		hold = max(hold, free-v)
		free = max(preCoolDown, free)
	}
	return max(free, coolDown)
}

func maxArea(height []int) int {
	if len(height) <= 0 {
		return 0
	}
	i, j, area := 0, len(height) - 1, 0
	for i < j {
		area = max(area, (j - i) * min(height[i], height[j]))
		if height[i] < height[j] {
			i ++
		} else {
			j --
		}
	}
	return area
}

func increasingTriplet(nums []int) bool {
	one, two := int(^uint(0) >> 1), int(^uint(0) >> 1)
	for _, v := range nums {
		if v <= one {
			one = v
		} else if v <= two {
			two = v
		} else {
			return true
		}
	}
	return false
}

func increasingTriplet1(nums []int) bool {
	ret := make([]int, len(nums))
	for i:=0; i<len(nums); i++ {
		count := 1
		for j:=0; j<len(ret); j ++ {
			if nums[i] > nums[j] && count <= ret[j] {
				count = ret[j] + 1
			}
		}
		ret = append(ret, count)
		if count > 3 {
			return  true
		}
	}
	return false
}

func longestConsecutive(nums []int) int {
	if len(nums) <= 0 {
		return 0
	}
	mp := make(map[int]int)
	for _, v := range nums {
		mp[v] = 1
	}
	longestStreak := 0
	for k := range mp {
		if _, ok := mp[k-1]; !ok {
			currentNum := k
			currentStreak := 1
			for {
				if _, ok := mp[currentNum+1]; ok {
					currentNum ++
					currentStreak ++
				} else {
					break
				}
			}
			longestStreak = max(longestStreak, currentStreak)
		}
	}
	return longestStreak
}

func findDuplicate(nums []int) int {
	slow := nums[0]
	fast := nums[nums[0]]

	for slow != fast {
		slow = nums[slow]
		fast = nums[nums[fast]]
	}

	ptr := 0
	for slow != ptr {
		slow = nums[slow]
		ptr = nums[ptr]
	}

	return slow
}

type arr [][]int

func (a arr) Less(i, j int) bool {
	if a[i][0] == a[j][0] {
		return a[i][1] < a[j][1]
	}
	return a[i][0] < a[j][0]
}

func (a arr) Swap(i,j int) {
	a[i], a[j] = a[j], a[i]
}

func (a arr) Len() int {
	return len(a)
}

func merge(intervals [][]int) [][]int {
	if len(intervals) <= 0{
		return [][]int{}
	}
	a := arr(intervals)
	sort.Sort(a)
	var r [][]int
	for _, v := range a {
		if len(r) == 0 || r[len(r)-1][1] < v[0] {
			r = append(r, v)
		} else {
			r[len(r)-1][1] = max(r[len(r)-1][1], v[1])
		}
	}
	return r
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

func minSubArrayLen(s int, nums []int) int {
	if len(nums) <= 0 {
		return 0
	}
	left, sum, ans := 0, 0, int(^uint(0) >> 1)
	for i:=0; i<len(nums); i++ {
		sum += nums[i]
		for sum >= s {
			ans = min(ans, i+1-left)
			left ++
			sum -= nums[left]
		}
	}
	if ans != int(^uint(0) >> 1) {
		return ans
	}
	return 0
}

func productExceptSelf(nums []int) []int {
	res, k := make([]int, len(nums)), 1
	for i:=0; i<len(nums); i++ {
		res[i] = k
		k *= nums[i]
	}
	k = 1
	for i:=len(nums)-1; i>=0; i-- {
		res[i] *= k
		k *= nums[i]
	}
	return res
}

func maxProduct(nums []int) int {
	if len(nums) <= 0 {
		return 0
	}
	r, imax, imin :=  ^int(^uint(0) >> 1), 1, 1
	for i:=0; i<len(nums); i++ {
		if nums[i] < 0 {
			imax, imin = imin, imax
		}
		imax = max(imax*nums[i], nums[i])
		imin = min(imin*nums[i], nums[i])

		r = max(imax, r)
	}
	return r
}

func robTest(nums []int) int {
	if len(nums) < 1 {
		return 0
	}
	dp := make([]int, len(nums)+1)
	dp[0] = 0
	//第一个最大值为 max(dp[i-1], dp[i-2]+nums[0])，所以为nums[0]
	dp[1] = nums[0]
	for i:=2; i<len(nums); i++ {
		dp[i] = max(dp[i-1], dp[i-2]+nums[i-1])
	}
	return dp[len(dp)-1]
}

func rob(nums []int) int {
	if len(nums) == 1 {
		return nums[0]
	}
	dp := make([]int, len(nums)+1)
	dp[0] = 0
	dp[1] = nums[0]
	for i:=2; i<=len(nums); i++ {
		dp[i] = max(dp[i-1], dp[i-2]+nums[i-1])
	}
	return dp[len(nums)]
}

func minPathSum(grid [][]int) int {
	if len(grid) <= 0 {
		return 0
	}
	return calculate(grid, 0, 0)
}

func calculate(grid [][]int, i, j int) int {
	if i == len(grid)-1 && j == len(grid[0])-1 {
		return grid[i][j]
	}
	if i == len(grid)-1 {
		return grid[i][j] + calculate(grid, i, j+1)
	}
	if j == len(grid[0])-1 {
		return grid[i][j] + calculate(grid, i+1, j)
	}
	return grid[i][j] + min(calculate(grid, i+1, j), calculate(grid, i, j+1))
}

func minPathSum2(grid [][]int) int {
	if len(grid) <= 0 {
		return 0
	}
	return minPathSum4(grid, len(grid)-1, len(grid[0])-1)
}

func minPathSum4(grid [][]int, i, j int) int {
	for i:=0; i<len(grid); i++ {
		for j:=0; j<len(grid[0]); j ++ {
			if i == 0 && j == 0 {
				continue
			} else if i == 0 {
				grid[i][j] = grid[i][j] + grid[i][j-1]
			} else if j == 0 {
				grid[i][j] = grid[i][j] + grid[i-1][j]
			} else {
				grid[i][j] = grid[i][j] + min(grid[i][j-1], grid[i-1][j])
			}
		}
	}
	return grid[len(grid)-1][len(grid[0])-1]
}

//0,0 => i,j
func calculate3(grid [][]int, i, j int) int {
	if i == len(grid) && j == len(grid[0]) {
		return grid[i][j]
	}
	if i == len(grid) {
		return grid[i][j] + calculate2(grid, i, j-1)
	} else  if i == len(grid[0]) {
		return grid[i][j] + calculate2(grid, i-1, j)
	}
	return grid[i][j] + min(calculate2(grid, i, j-1), calculate2(grid, i-1, j))
}

//i, j => 0,0
func calculate2(grid [][]int, i, j int) int {
	if i == 0 && j == 0 {
		return grid[i][j]
	}
	if i == 0 {
		return grid[i][j] + calculate2(grid, i, j-1)
	}
	if j == 0 {
		return grid[i][j] + calculate2(grid, i-1, j)
	}
	return grid[i][j] + min(calculate2(grid, i-1, j), calculate2(grid, i, j-1))
}

func hanoi(n, p1, p2, p3 int) {
	if n == 1 {
		fmt.Printf("%v from %v to %v\n", n, p1, p3)
		return
	}
	hanoi(n-1, p1, p3, p2)
	fmt.Printf("%v from %v to %v\n", n, p1, p3)
	hanoi(n-1, p2, p1, p3)
}

func Perm(nums []int, k, m int)  {
	if k == m-1 {
		fmt.Println(nums)
	}
	for i:=k; i<m; i++ {
		nums[i], nums[k] = nums[k], nums[i]
		Perm(nums, k+1, m)
		nums[k], nums[i] = nums[i], nums[k]
	}
}

func sumNum(day, count int) int {
	if day % 2 == 0 {
		return sumEvenDay(day, count)
	} else {
		return sumOddDay(day, count)
	}
}

func sumEvenDay(day, count int) int {
	if count == day {
		return 1
	}
	return (sumOddDay(day, count+1)+1) * 2
}

func sumOddDay(day, count int) int {
	if count == day {
		return 1
	}
	return (sumOddDay(day, count+1)+3) * 2
}

type Node struct {
	Val int
	Next *Node
}

func reverseList(list *Node) *Node {
	if list == nil || list.Next == nil {
		return list
	}
	node := reverseList(list.Next)
	list.Next.Next = list
	list.Next = nil
	return node
}

/**
输入: [0,1,2,4,5,7]
输出: ["0->2","4->5","7"]
解释: 0,1,2 可组成一个连续的区间; 4,5 可组成一个连续的区间。
 */
func summaryRanges(nums []int) []string {
	r := []string{}
	if len(nums) <= 0 {
		return r
	}
	nums = append(nums, int(^uint(0) >> 1))
	start, end, distance := nums[0], nums[0], 1
	for i:=1; i<len(nums); i++ {
		if nums[i] == start + distance {
			end = nums[i]
			distance ++
		} else {
			var s string
			if distance == 1 {
				s = fmt.Sprintf("%v", start)
			} else  {
				s = fmt.Sprintf("%v->%v", start, end)
			}
			r = append(r, s)
			start, end = nums[i], nums[i]
			distance = 1
		}
	}
	return r
}

/**
给定一个包含红色、白色和蓝色，一共 n 个元素的数组，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。
此题中，我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。
注意:
不能使用代码库中的排序函数来解决这道题。
示例:
输入: [2,0,2,1,1,0]
输出: [0,0,1,1,2,2]
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

/**
给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
示例:
输入: [0,1,0,3,12]
输出: [1,3,12,0,0]
说明:
必须在原数组上操作，不能拷贝额外的数组。
尽量减少操作次数。
 */
func moveZeroes(nums []int)  {
	lastNonZeroFoundAt := 0
	for cur:=0; cur<len(nums); cur++ {
		if nums[cur] != 0 {
			nums[lastNonZeroFoundAt], nums[cur] = nums[cur], nums[lastNonZeroFoundAt]
			lastNonZeroFoundAt ++
		}
	}
}

/**
摆动排序 II
 */
func wiggleMaxLength(nums []int) int {
	if len(nums) < 2 {
		return len(nums)
	}
	return 1 + max(cal(nums, 0, true), cal(nums, 0, false))
}

func cal(nums []int, index int, isUp bool) int {
	maxLen := 0
	for i:=index+1; i<len(nums); i++ {
		if (isUp && nums[index] < nums[i]) || (!isUp && nums[index] > nums[i]) {
			maxLen = max(maxLen, 1 + cal(nums, i, !isUp))
		}
	}
	return maxLen
}

func wiggleMaxLength1(nums []int) int {
	if len(nums) < 2 {
		return len(nums)
	}
	up, down := make([]int, len(nums)), make([]int, len(nums))
	up[0], down[0] = 1, 1
	for i:=1; i<len(nums); i++ {
		if nums[i] > nums[i - 1] {
			up[i] = down[i - 1] + 1;
			down[i] = down[i - 1];
		} else if nums[i] < nums[i - 1] {
			down[i] = up[i - 1] + 1;
			up[i] = up[i - 1];
		} else {
			down[i] = down[i - 1];
			up[i] = up[i - 1];
		}
	}
	return max(up[len(nums)-1], up[len(nums)-1])
}

/**
摆动序列
 */
func wiggleMaxLength2(nums []int) int {
	if len(nums) < 2 {
		return len(nums)
	}
	up, down := 1, 1
	for i:=1; i<len(nums); i++ {
		if nums[i] > nums[i - 1] {
			up = down + 1
		} else if nums[i] < nums[i - 1] {
			down = up + 1
		}
	}
	return max(up, down)
}

/**

 */
func wiggleSort(nums []int)  {
	if len(nums) <= 0 {
		return
	}
	sort.Ints(nums)
	middle := len(nums) / 2
	if len(nums) % 2 == 0 {
		middle -= 1
	}
	end := len(nums) - 1
	r := []int{}
	for i:=middle; i>=0; i-- {
		r = append(r, nums[i])
		if end > middle {
			r = append(r, nums[end])
			end --
		}
	}
	for i:=0; i<len(nums); i++ {
		nums[i] = r[i]
	}
	return
}

func searchInsert1(nums []int, target int) int {
	if len(nums) <= 0 {
		return 0
	}
	start, end, middle := 0, len(nums)-1, 0
	for start <= end {
		middle = (start+end) / 2
		if nums[middle] >= target {
			if middle == 0 || nums[middle-1] < target {
				break
			} else {
				end = middle - 1
			}
		} else {
			start = middle + 1
		}
	}
	if start > end {
		middle = len(nums)
	}
	return middle
}

/**
旋转数组找到旋转点
 */
func search1(nums []int) int {
	if len(nums) < 1 {
		return -1
	}
	if nums[0] < nums[len(nums)-1] {
		return nums[0]
	}
	start, end, middle := 0, len(nums)-1, 0
	for start < end {
		middle = start + (end - start) / 2
		if nums[middle] < nums[end] {
			end = middle
		} else {
			start = middle + 1
		}
	}
	return nums[start]
}

/**
33. 搜索旋转排序数组
 */
func search(nums []int, target int) int {
	start, end, middle := 0, len(nums)-1, 0
	for start <= end {
		middle = start + (end - start) / 2
		if nums[middle] == target {
			return middle
		} else if nums[middle] >= nums[start] {
			if target >= nums[start] && target < nums[middle] {
				end = middle - 1
			} else {
				start = middle + 1
			}
		} else {
			if target > nums[middle] && target <= nums[end] {
				start = middle + 1
			} else {
				end = middle - 1
			}
		}
 	}
 	return -1
}

/**
81.搜索旋转排序数组 II
 */
func search2(nums []int, target int) bool {
	start, end, middle := 0, len(nums)-1, 0
	for start <= end {
		middle = start + (end - start) / 2
		if nums[middle] == target {
			return true
		} else if nums[middle] == nums[start] {
			start ++
		} else if nums[middle] == nums[end] {
			end --
		}else if nums[middle] < nums[end] {
			if target > nums[middle] && target <= nums[end] {
				start = middle + 1
			} else {
				end = middle - 1
			}
		} else {
			if target >= nums[start] && target <= nums[middle] {
				end = middle - 1
			} else {
				start = middle + 1
			}
		}
	}
	return false
}

/**
寻找峰值
 */
func findPeakElement(nums []int) int {
	if len(nums) <= 1 {
		return 0
	}
	start, end, middle := 0, len(nums)-1, 0
	for start < end {
		middle = start + (end - start) / 2
		if nums[middle] > nums[middle+1] {
			end = middle
		} else  {
			start = middle + 1
		}
	}
	return start
}

func extremeInsertionIndex(nums []int, target int, left bool) int {
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

/**
在排序数组中查找元素的第一个和最后一个位置
 */
func searchRange(nums []int, target int) []int {
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

/**
两数组的交集
 */
func intersection(nums1 []int, nums2 []int) []int {
	mp1 := make(map[int]int, len(nums1))
	for _, v := range nums1 {
		mp1[v] ++
	}
	var r []int
	for _, v := range nums2 {
		if m, ok := mp1[v]; ok && m > 0 {
			r = append(r, v)
			mp1[v] --
		}
	}
	return r
}

/**
300. 最长上升子序列
*/
func lengthOfLIS(nums []int) int {
	if len(nums) < 1 {
		return 0
	}
	tails := make([]int, len(nums))
	res := 0
	for _, num := range nums {
		i, j := 0, res
		for i < j {
			m := (i + j) / 2
			if tails[m] < num {
				i = m + 1
			} else {
				j = m
			}
		}
		tails[i] = num
		if j == res {
			res ++
		}
	}
	return res
}

func fib(n int) int {
	if n == 0 {
		return 0
	}
	if n == 1 {
		return 1
	}
	x, y, r := 0, 1, 0
	for i:=2; i<n; i++ {
		r = x + y
		x = y
		y = r
	}
	return r
}

//n=5，w={2,2,6,5,4}，v={6,3,5,4,6}，Cap=10
func mostVal(cap int, w, v []int) int {
	dp := make([][]int, len(w)+1)
	inDp := make([]int, cap+1)
	for i:=0; i<len(w)+1; i++ {
		for j:=0; j<=cap; j++ {
			inDp[j] = 0
		}
		dp[i] = inDp
	}
	for i:=1; i<=len(w); i++ {
		for j:=1; j<=cap; j++ {
			dp[i][j] = dp[i][j-1]
			if j > w[i-1] {
				inDp[i] = max(dp[i-1][j], dp[i-1][cap-w[i-1]]+v[i-1])
			}
		}
	}

	return dp[len(dp)-1][cap-1]
}

func minSum(arr [][]int) int {
	for i:=0; i<len(arr); i++ {
		for j:=0; j<len(arr[0]); j++ {
			if i == 0 && j == 0 {
				arr[i][j] = arr[i][j]
			} else if i == 0 {
				arr[i][j] = arr[i][j-1] + arr[i][j]
			} else if j == 0 {
				arr[i][j] = arr[i-1][j] + arr[i][j]
			} else {
				arr[i][j] = min(arr[i-1][j], arr[i][j-1]) + arr[i][j]
			}
		}
	}
	return arr[len(arr)-1][len(arr[0])-1]
}

func maxLenSubString(str1, str2 string) string {
	byte1 := []byte(str1)
	byte2 := []byte(str2)
	fmt.Println(byte1, byte2)
	return ""
}

func main1() {
	//nums := []int{10,9,2,5,3,7,101,18}
	//str1 := "1A2C3D4B56"
	//str2 := "B1D23CA45B6A"
	//r := maxLenSubString(str1, str2)
	//fmt.Println(r)
	//return
	//w := []int{2,2,6,5,4}
	//v := []int{6,3,5,4,6}
	//cap := 10
	//r := mostVal(cap, w, v)

	//nums := [][]int{{1,3},{2,6},{8,10},{15,18}}
	//r := merge(nums)
	//fmt.Println(r)

	//nums := [][]int{{1,3,1},{1,5,1},{4,2,1}}
	//r := minPathSum2(nums)
	//fmt.Println(r)

	//node3 := &Node{
	//	3,
	//	nil,
	//}
	//node2 := &Node{
	//	2,
	//	node3,
	//}
	//head := &Node{
	//	1,
	//	node2,
	//}
	////p(head)
	//r := reverseList(head)
	//p(r)
}
