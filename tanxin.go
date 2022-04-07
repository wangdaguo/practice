package main

import (
	"fmt"
	"sort"
)

func main1()  {
	//cnt := findContentChild([]int64{6,5}, []int64{1,2,3})
	// := candy([]int{1,3,4,5,2})
	//r := eraseOverlapIntervals([][]int{{1,2},{2,4},{1,3}})
	//r := canPlaceFlowers([]int{1,0,0,0,0,0,1}, 2)
	//r := maxProfit1([]int{7,1,5,3,6,4})
	r := checkPossibility([]int{7,1,5,3,6,4})
	fmt.Println(r)
}
/**
455
输入：
孩子饥饿度
饼干大小

输出：
最多有多少个孩子可以吃饱
 */
func findContentChild(children, cookies []int64) int {
	childrenArr := Arr(children)
	childrenArr.Sort()
	cookiesArr := Arr(cookies)
	cookiesArr.Sort()
	var child, cookie int
	for child < childrenArr.Len() && cookie < childrenArr.Len() {
		if childrenArr[child] > cookiesArr[cookie] {
			cookie ++
		} else {
			child ++
		}
	}
	return child
}

type Arr []int64

func (a Arr) Len() int {
	return len(a)
}

func (a Arr) Less(i, j int) bool {
	return a[i] < a[j]
}

func (a Arr) Swap(i, j int) {
	a[i], a[j] = a[j], a[i]
}

func (a Arr) Sort()  {
	sort.Sort(a)
}

/**
135
老师想给孩子们分发糖果，有 N 个孩子站成了一条直线，老师会根据每个孩子的表现，预先给他们评分。
你需要按照以下要求，帮助老师给这些孩子分发糖果：

每个孩子至少分配到 1 个糖果。
评分更高的孩子必须比他两侧的邻位孩子获得更多的糖果。
那么这样下来，老师至少需要准备多少颗糖果呢？

输入：[1,0,2]
输出：5
解释：你可以分别给这三个孩子分发 2、1、2 颗糖果。
 */
func candy(children []int) int {
	if len(children) < 1 {
		return 0
	}
	if len(children) == 1 {
		return 1
	}
	r := make([]int, len(children))
	for k, _ := range children {
		r[k] = 1
	}
	for k:=1; k<len(children); k++ {
		if children[k] > children[k-1] {
			r[k] = r[k-1] + 1
		}
	}
	for k:=len(children)-1; k>0; k-- {
		if children[k-1] > children[k] {
			r[k-1] = max(r[k] + 1, r[k-1])
		}
	}
	sum := 0
	for _, v := range r {
		sum += v
	}
	return sum
}

func max(i, j int) int {
	if i > j {
		return i
	}
	return j
}

/**
435
给定一个区间的集合，找到需要移除区间的最小数量，使剩余区间互不重叠。
注意:
可以认为区间的终点总是大于它的起点。
区间 [1,2] 和 [2,3] 的边界相互“接触”，但没有相互重叠。

输入: [ [1,2], [2,3], [3,4], [1,3] ]
输出: 1
解释: 移除 [1,3] 后，剩下的区间没有重叠。
 */

func eraseOverlapIntervals(intervals [][]int) int {
	if len(intervals) < 1 {
		return 0
	}
	list := ArrayList(intervals)
	list.Sort()

	pre := list[0][len(list[0])-1]
	var total int
	for i:=1; i<len(list); i++ {
		if list[i][0] < pre {
			total ++
		} else {
			pre = list[i][len(list[i])-1]
		}
	}
	return total
}

type ArrayList [][]int

func (a ArrayList) Len() int {
	return len(a)
}

func (a ArrayList) Less(i, j int) bool {
	return a[i][len(a[i])-1] < a[j][len(a[j])-1]
}

func (a ArrayList) Swap(i, j int) {
	a[i], a[j] = a[j], a[i]
}

func (a ArrayList) Sort()  {
	sort.Sort(a)
}


/**
605
假设有一个很长的花坛，一部分地块种植了花，另一部分却没有。可是，花不能种植在相邻的地块上，它们会争夺水源，两者都会死去。
给你一个整数数组  flowerbed 表示花坛，由若干 0 和 1 组成，其中 0 表示没种植花，1 表示种植了花。另有一个数 n ，能否在不打破种植规则的情况下种入 n 朵花？能则返回 true ，不能则返回 false。

输入：flowerbed = [1,0,0,0,1], n = 1
输出：true

输入：flowerbed = [1,0,0,0,1], n = 2
输出：false

[1,0,0,0,0,0,1]
输入：flowerbed = [1,0,0,0,0,1], n = 2
输出：false
 */
func canPlaceFlowers(flowerbed []int, n int) bool {
	count := 0
	m := len(flowerbed)
	prev := -1
	for i := 0; i < m; i++ {
		if flowerbed[i] == 1 {
			if prev < 0 {
				count += i / 2
			} else {
				count += (i - prev - 2) / 2
			}
			if count >= n {
				return true
			}
			prev = i
		}
	}
	if prev < 0 {
		count += (m + 1) / 2
	} else {
		count += (m - prev - 1) / 2
	}
	return count >= n
}

/**
452
在二维空间中有许多球形的气球。对于每个气球，提供的输入是水平方向上，气球直径的开始和结束坐标。由于它是水平的，所以纵坐标并不重要，因此只要知道开始和结束的横坐标就足够了。开始坐标总是小于结束坐标。

一支弓箭可以沿着 x 轴从不同点完全垂直地射出。在坐标 x 处射出一支箭，若有一个气球的直径的开始和结束坐标为 xstart，xend， 且满足  xstart ≤ x ≤ xend，则该气球会被引爆。可以射出的弓箭的数量没有限制。 弓箭一旦被射出之后，可以无限地前进。我们想找到使得所有气球全部被引爆，所需的弓箭的最小数量。

给你一个数组 points ，其中 points [i] = [xstart,xend] ，返回引爆所有气球所必须射出的最小弓箭数。

输入：points = [[10,16],[2,8],[1,6],[7,12]]
输出：2
解释：对于该样例，x = 6 可以射爆 [2,8],[1,6] 两个气球，以及 x = 11 射爆另外两个气球
 */


/**
763. 划分字母区间
字符串 S 由小写字母组成。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。返回一个表示每个字符串片段的长度的列表。
输入：S = "ababcbacadefegdehijhklij"
输出：[9,7,8]
解释：
划分结果为 "ababcbaca", "defegde", "hijhklij"。
每个字母最多出现在一个片段中。
像 "ababcbacadefegde", "hijhklij" 的划分是错误的，因为划分的片段数较少。
*/
func partitionLabels(s string) []int {
	lastPos := [26]int{}
	for i, c := range s {
		lastPos[c-'a'] = i
	}
	var start, end int
	var partition []int
	for i, c := range s {
		if lastPos[c-'a'] > end {
			end = lastPos[c-'a']
		}
		if i == end {
			partition = append(partition, end-start+1)
			start = end + 1
		}
	}
	return partition
}

/**
给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。
你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。
返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回0

输入：[7,1,5,3,6,4]
输出：5
解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。
 */
func maxProfit1(prices []int) int {
	if len(prices) < 1 {
		return 0
	}
	var start, max int
	for i:=1; i<len(prices); i++ {
		if prices[i] < prices[start] {
			start = i
		} else if prices[i] - prices[start] > max {
			max = prices[i] - prices[start]
		}
	}
	return max
}

func maxProfit11(prices []int) int {
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

/**
122. 买卖股票的最佳时机 II
给定一个数组 prices ，其中 prices[i] 是一支给定股票第 i 天的价格。
设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。
注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
输入: prices = [7,1,5,3,6,4]
输出: 7
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 。
 */
func maxProfit(prices []int) int {
	if len(prices) <= 0 {
		return 0
	}
	var sum int
	for i:=1; i<len(prices); i++ {
		if prices[i] > prices[i-1] {
			sum += prices[i] - prices[i-1]
		}
	}
	return sum
}

/**
406. 根据身高重建队列
假设有打乱顺序的一群人站成一个队列，数组 people 表示队列中一些人的属性（不一定按顺序）。每个 people[i] = [hi, ki] 表示第 i 个人的身高为 hi ，前面 正好 有 ki 个身高大于或等于 hi 的人。
请你重新构造并返回输入数组 people 所表示的队列。返回的队列应该格式化为数组 queue ，其中 queue[j] = [hj, kj] 是队列中第 j 个人的属性（queue[0] 是排在队列前面的人）。

输入：people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]
输出：[[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]


[4,4],[5,2],[5,0],[6,1],[7,1],[7,0]
*/
func reconstructQueue(people [][]int) [][]int {
	sort.Slice(people, func(i, j int) bool {
		return people[i][0] < people[j][0] || people[i][0] == people[j][0] && people[i][1] > people[j][1]
	})
	ans := make([][]int, len(people))
	for _, person := range people {
		space := person[1] + 1
		for i := range ans {
			if ans[i] == nil {
				space --
				if space == 0 {
					ans[i] = person
					break
				}
			}
		}
	}
	return ans
}

/**
665. 非递减数列
给你一个长度为 n 的整数数组，请你判断在 最多 改变 1 个元素的情况下，该数组能否变成一个非递减数列。
我们是这样定义一个非递减数列的： 对于数组中任意的 i (0 <= i <= n-2)，总满足 nums[i] <= nums[i + 1]。

输入: nums = [4,2,3]
输出: true
解释: 你可以通过把第一个4变成1来使得它成为一个非递减数列。
 */
func checkPossibility(nums []int) bool {
	for i := 0; i < len(nums)-1; i++ {
		x, y := nums[i], nums[i+1]
		if x > y {
			nums[i] = y
			if sort.IntsAreSorted(nums) {
				return true
			}
			nums[i] = x // 复原
			nums[i+1] = x
			return sort.IntsAreSorted(nums)
		}
	}
	return true
}

func checkPossibility1(nums []int) bool {
	var cnt int
	for i:=0; i<len(nums)-1; i++ {
		x, y := nums[i], nums[i+1]
		if x > y {
			cnt ++
			if cnt > 1 {
				return false
			}
			if i > 0 && y < nums[i-1] {
				nums[i+1] = x
			}
		}
	}
	return true
}
