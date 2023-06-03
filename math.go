package main

import (
	"fmt"
	"strconv"
	"strings"
)

func main() {
	//r := gcd2(10, 5)
	//r := lcm(10, 5)
	//r := convertToBase7(10)
	//r := trailingZeroes(5)
	r := addStrings("11", "123")
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