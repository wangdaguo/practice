package main

import (
	"fmt"
	"strings"
)

func main()  {
	return
}

/**
tmp = s[0]
s[0] = s[i]; s[i] = s[2i]
*/
func reverseStr(s []string, m int) {

	if len(s) < 1 || m < 1 {
		return
	}
	for i:=0; i<gcd(m, len(s)); i++ {
		tmp := s[i]
		cnt := 1
		for i+(cnt*m)%len(s) != i {
			s[i+((cnt-1)*m)%len(s)] = s[i+(cnt*m)%len(s)]
			cnt ++
		}
		s[i+((cnt-1)*m)%len(s)] = tmp
	}
	return
}

func gcd(i, j int) int {
	if i == 0 || j == 0 {
		return 0
	}
	for i != j {
		if i > j {
			i -= j
		} else {
			j -= i
		}
	}
	return i
}

func reverseStr1(s []string, m int) {
	if len(s) < 1 || m < 1 {
		return
	}
	reverse(s, 0, m-1)
	reverse(s, m, len(s)-1)
	reverse(s, 0, len(s)-1)
	return
}

func reverse(s []string, i, j int) {
	if len(s) < 1 || i >= j {
		return
	}
	left, right := i, j
	for left < right {
		s[left], s[right] = s[right], s[left]
		left ++
		right --
	}
	return
}

type TreeNode1 struct {
	Val   int
	Child []*TreeNode1
}

const sum int = 0

func main11() {
	length := 0
	_, err := fmt.Scan(&length)
	if length <= 0 || err != nil {
		return
	}
	nums := make([]int, 0)
	for j := 0; j < length; j++ {
		x := 0
		_, err := fmt.Scan(&x)
		if err != nil {
			return
		}
		nums = append(nums, x)
	}

	maps := make(map[int]*TreeNode1)
	for i, v := range nums {
		node := &TreeNode1{
			Val: v,
			Child: make([]*TreeNode1, 0),
		}
		maps[i] = node
	}

	for i := 0; i < length - 1; i++ {
		number := make([]int, 0)
		for j := 0; j < 2; j++ {
			x:=0
			fmt.Scan(&x)
			number = append(number, x)
		}

		parent := maps[0]
		child := maps[1]
		parent.Child = append(parent.Child, child)
	}



	result := treeSum(maps[0])
	fmt.Println(result)
}

func treeSum(root *TreeNode1) int {
	if root == nil {
		return 0
	}
	reverse1(root, 0, root.Val, 0)
	return sum
}

func reverse1(root *TreeNode1, depth int, value int, sum int) {
	if root == nil {
		return
	}
	if len(root.Child) == 0 && depth >= 1 {
		sum += value
	}

	for _, child := range root.Child {
		reverse1(child, depth+1, value^root.Val, sum)
	}
}

func maintt() {
	length := 0
	target := 0
	_, err := fmt.Scan(&length, &target)
	if length <= 0 || err != nil {
		return
	}
	nums := make([]int, 0)
	for j := 0; j < length; j++ {
		x := 0
		_, err := fmt.Scan(&x)
		if err != nil {
			return
		}
		nums = append(nums, x)
	}

	// 数值对应在数组中的位置
	maps := make(map[int][]int)
	for i, v := range nums {
		maps[v] = append(maps[v], i)
	}

	result := 0
	for i, v := range nums {
		if _, ok := maps[v-target]; ok {
			result = max(result, maxValue(i, maps[v-target]))
		}
		if _, ok := maps[v+target]; ok {
			result = max(result, maxValue(i, maps[v+target]))
		}
	}
	fmt.Println(result)
}

func max(i, j int) int {
	if i > j {
		return i
	}
	return j
}

func maxValue(i int, nums []int) int {
	result := 0
	for _, v := range nums {
		result = max(result, abs(i - v))
	}
	return result
}

func abs(value int) int {
	if value < 0 {
		return -value
	}
	return value
}

// 规则1 映射 规则2
var RuleMap map[string]map[string]bool
// 姓 映射 名
var NameMap map[string]map[string]bool

func maint1() {
	n := 0
	m := 0
	_, err := fmt.Scan(&n, &m)
	if err != nil {
		return
	}

	RuleMap = make(map[string]map[string]bool)
	for i := 0; i < n; i ++ {
		a := ""
		b := ""
		_, err = fmt.Scan(&a, &b)
		if err != nil {
			return
		}
		SetRule(a, b)
	}
	var resp int64 = 0
	NameMap = make(map[string]map[string]bool)
	for j := 0; j < m; j ++ {
		name := ""
		_, err = fmt.Scan(&name)
		if err != nil {
			return
		}
		resp += FindLover(name)
	}
	fmt.Println(resp)
}

func FindLover(name string) (resp int64) {
	var firstBuild strings.Builder
	var lastName string
	for i := 0; i < len(name); i ++ {
		if 65 <= name[i] && name[i] <= 90 {
			firstBuild.WriteString(string(name[i]))
		} else {
			rs := []rune(name)
			lastName = string(rs[i:])
			break
		}
	}
	firstName := firstBuild.String()
	if _, ok := NameMap[firstName]; !ok {
		NameMap[firstName] = make(map[string]bool)
	}
	NameMap[firstName][lastName] = true

	if rule, ok := RuleMap[lastName]; ok {
		for ln, _ := range rule {
			if b, okay := NameMap[firstName][ln]; okay && b && lastName != ln {
				resp ++
				//NameMap[firstName][lastName] = false
				//NameMap[firstName][ln] = false
			}
		}
	}
	return resp
}

func SetRule(a, b string) {
	if _, ok := RuleMap[a]; !ok {
		RuleMap[a] = make(map[string]bool)
	}
	RuleMap[a][b] = true
	if _, ok := RuleMap[b]; !ok {
		RuleMap[b] = make(map[string]bool)
	}
	RuleMap[b][a] = true
	return
}

func CloseToZero(m int, x []int) int {
	if len(x) < 1 || len(x) <= m {
		return -1
	}
	dp := make([]int, len(x))
	dp[0] = x[0]
	for i:=1; i<len(x); i++ {
		dp[i] = dp[i-1] + x[i]
	}
	var closeTo, start = abs(dp[m]), 0
	for i:=m+1; i<len(x); i++ {
		if abs(dp[i] - dp[i-m-1]) < closeTo {
			closeTo = dp[i] - dp[i-m-1]
			start = i-m
		}
	}
	return start
}
