package main

import (
	"fmt"
	"math/rand"
	"sort"
	"strconv"
	"strings"
	"time"
)

func main() {
	//r := gcd2(10, 5)
	//r := lcm(10, 5)
	//r := convertToBase7(10)
	//r := trailingZeroes(5)
	//r := addStrings("11", "123")
	//r := isPowerOfThree(3)
	//s := Constructor1([]int{1,2,3})
	//fmt.Println(s.Nums)
	//fmt.Println(s.Shuffle())
	//fmt.Println(s.Reset())

	//s := Constructor2([]int{3,1,2,4})
	//fmt.Println(s.pre)
	//node3 := &ListNode{
	//	Val:  3,
	//	Next: nil,
	//}
	//node2 := &ListNode{
	//	Val:  2,
	//	Next: node3,
	//}
	//node1 := &ListNode{
	//	Val:  1,
	//	Next: node2,
	//}
	//s := Constructor(node1)
	//fmt.Println(s.GetRandom())
	//r := convertToTitle(701)
	r := addBinary("11", "1")
	fmt.Println(r)
}

func gcd2(a, b int) int  {
	if b == 0 {
		return a
	}
	return gcd2(b, a%b)
}

func lcm(a, b int) int {
	return a*b / gcd2(a, b)
}

/**
204. 计数质数
https://leetcode.cn/problems/count-primes/description/
*/
func countPrimes(n int) int {
	isPrime := make([]bool, n)
	for i := range isPrime {
		isPrime[i] = true
	}
	var cnt int
	for i:=2; i<n; i++ {
		if isPrime[i] {
			cnt ++
			for j:=i*2; j<n; j+=i {
				isPrime[j] = false
			}
		}
	}
	return cnt
}

/**
暴力法
 */
func countPrimes1(n int) int {
	var cnt  int
	for i:=2; i<n; i++ {
		if isPrime(i) {
			cnt ++
		}
	}
	return cnt
}

func isPrime(x int) bool {
	for i:=2; i*i<=x; i++ {
		if x % i == 0 {
			return false
		}
	}
	return true
}

/**
504. 七进制数
https://leetcode.cn/problems/base-7/
 */
func convertToBase7(num int) string {
	symbol := ""
	if num < 0 {
		symbol = "-"
	}
	n, list := abs(num), []string{}
	if n == 0 {
		list = []string{"0"}
	}
	for n > 0 {
		t := n % 7
		list = append([]string{strconv.Itoa(t)}, list...)
		n = n / 7
	}
	if symbol != "" {
		list = append([]string{symbol}, list...)
	}
	return strings.Join(list, "")
}

func abs(num int) int {
	if num < 0 {
		return -num
	}
	return num
}

/**
172. 阶乘后的零
https://leetcode.cn/problems/factorial-trailing-zeroes/
 */
func trailingZeroes(n int) int {
	if n == 0 {
		return 0
	}
	return n/5 + trailingZeroes(n/5)
}

/**
415. 字符串相加
https://leetcode.cn/problems/add-strings/
 */
func addStrings(num1 string, num2 string) string {
	if len(num1) < 1 {
		return num2
	}
	if len(num2) < 1 {
		return num1
	}
	cb, cnt, r, s := 0, len(num1), []string{}, 0
	if len(num2) > len(num1) {
		cnt = len(num2)
	}
	for i:=0; i<cnt; i++ {
		index1, index2 := len(num1)-i-1, len(num2)-i-1
		if index1 >= 0 && index2 >= 0 {
			s = int(num1[index1]-'0') + int(num2[index2]-'0') + cb
		} else if index1 >= 0 {
			s = int(num1[index1]-'0') + cb
		} else {
			s = int(num2[index2]-'0') + cb
		}
		if s > 9 {
			cb = 1
		} else {
			cb = 0
		}
		r = append([]string{strconv.Itoa(s%10)}, r...)
		s = 0
	}
	if cb > 0 {
		r = append([]string{strconv.Itoa(cb)}, r...)
	}
	return strings.Join(r, "")
}

/**
326. 3 的幂
https://leetcode.cn/problems/power-of-three/
 */
func isPowerOfThree(n int) bool {
	if n < 1 {
		return false
	}
	if n == 1 {
		return true
	}
	res := 3
	for res < n {
		res *= 3
	}
	if res == n {
		return true
	}
	return false
}

/**
384. 打乱数组
https://leetcode.cn/problems/shuffle-an-array/
 */
type Solution1 struct {
	Nums []int
}

func Constructor1(nums []int) Solution1 {
	s := Solution1{
		Nums:        nums,
	}
	return s
}

func (s *Solution1) Reset() []int {
	return s.Nums
}

func (s *Solution1) Shuffle() []int {
	if len(s.Nums) < 1 {
		return []int{}
	}
	r := make([]int, len(s.Nums))
	for k, v := range s.Nums {
		r[k] = v
	}
	rand.Seed(time.Now().UnixNano()) // unix 时间戳，秒
	for i:=0; i<len(s.Nums); i++ {
		index := rand.Intn(len(s.Nums))
		r[i], r[index] = r[index], r[i]
	}
	return r
}

/**
528. 按权重随机选择
https://leetcode.cn/problems/random-pick-with-weight/
 */
type Solution2 struct {
	pre []int
}

/**
[3,1,2,4]
 */
func Constructor2(w []int) Solution2 {
	for i:=1; i<len(w); i++ {
		w[i] += w[i-1]
	}
	return Solution2{w}
}

func (s *Solution2) PickIndex() int {
	i := rand.Intn(s.pre[len(s.pre)-1])+1
	return sort.SearchInts(s.pre, i)
}

/**
382. 链表随机节点
https://leetcode.cn/problems/linked-list-random-node
 */
type ListNode struct {
		Val  int
	Next *ListNode
}

type Solution struct {
	head *ListNode
	len int
}

func Constructor(head *ListNode) Solution {
	s := Solution{
		head: head,
	}
	node, len := head, 0
	for node != nil {
		len ++
		node = node.Next
	}
	s.len = len
	return s
}

func (s *Solution) GetRandom() int {
	rand.Seed(time.Now().UnixNano())
	i := rand.Intn(s.len) + 1
	r, node := 0, s.head
	for i > 0 {
		r = node.Val
		node = node.Next
		i --
	}
	return r
}

/**
168. Excel表列名称
https://leetcode.cn/problems/excel-sheet-column-title/
 */
func convertToTitle(columnNumber int) string {
	var r string
	for columnNumber > 0 {
		s := columnNumber % 26
		if s == 0 {
			s = 26
			columnNumber --
		}
		r = string('A' + s - 1) + r
		columnNumber = columnNumber / 26
	}
	return r
}

/**
67. 二进制求和
https://leetcode.cn/problems/add-binary
 */
func addBinary(a string, b string) string {
	if len(a) < 1 {
		return b
	}
	if len(b) < 1 {
		return a
	}
	cnt, cb, sum, r := len(a), 0, 0, ""
	if len(b) > len(a) {
		 cnt = len(b)
	}
	for i:=0; i<cnt; i++ {
		if i < len(a) && i < len(b) {
			sum = int(a[len(a)-i-1]-'0') + int(b[len(b)-i-1]-'0') + cb
		} else if i < len(a) {
			sum = int(a[len(a)-i-1]-'0') + cb
		} else if i < len(b) {
			sum = int(b[len(b)-i-1]-'0') + cb
		}
		r = fmt.Sprintf("%d%s", sum%2, r)
		cb = sum/2
		sum = 0
	}
	if cb > 0 {
		r = fmt.Sprintf("%d%s", cb, r)
	}
	return r
}

/**
238. 除自身以外数组的乘积
https://leetcode.cn/problems/product-of-array-except-self/
 */
func productExceptSelf(nums []int) []int {
	ans, R := make([]int, len(nums)), 1
	for i:=1; i<len(nums); i++ {
		ans[i] = ans[i-1] * nums[i-1]
	}
	for i:=len(nums)-1; i>=0; i-- {
		ans[i] *= R
		R *= nums[i]
	}
	return ans
}

/**
462. 最小操作次数使数组元素相等 II
https://leetcode.cn/problems/minimum-moves-to-equal-array-elements-ii/
 */
func minMoves2(nums []int) int {
	sort.Ints(nums)
	x, r := nums[len(nums)/2], 0
	for _, num := range nums {
		r += abs(x-num)
	}
	return r
}

func abs(num int) int {
	if num < 0 {
		return -num
	}
	return num
}
