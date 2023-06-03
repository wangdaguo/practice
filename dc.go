package main

import (
	"fmt"
	"strconv"
)

func main() {
	r := diffWaysToCompute("2-1-1")
	fmt.Println(r)
}

/**
241. 为运算表达式设计优先级
https://leetcode.cn/problems/different-ways-to-add-parentheses
 */
func diffWaysToCompute(expression string) []int {
	if isDigit(expression) {
		v, _ := strconv.Atoi(expression)
		return []int{v}
	}

	var r []int
	for index, c := range expression {
		tmpC := string(c)
		if tmpC == "+" || tmpC == "-" || tmpC == "*" {
			left := diffWaysToCompute(expression[:index])
			right := diffWaysToCompute(expression[index+1:])
			for _, leftNum := range left {
				for _, rightNum := range right {
					var tmpR int
					if tmpC == "+" {
						tmpR = leftNum + rightNum
					} else if tmpC == "-" {
						tmpR = leftNum - rightNum
					} else {
						tmpR = leftNum * rightNum
					}
					r = append(r, tmpR)
				}
			}
		}
	}
	return r
}

func isDigit(s string) bool {
	_, err := strconv.Atoi(s)
	if err != nil {
		return false
	}
	return true
}

/**
932. 漂亮数组
https://leetcode.cn/problems/beautiful-array
 */
func beautifulArray(n int) []int {

}