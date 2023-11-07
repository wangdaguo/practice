package main

import (
	"bytes"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

/**
28. 实现 strStr()
给定一个 haystack 字符串和一个 needle 字符串，在 haystack 字符串中找出 needle 字符串出现的第一个位置 (从0开始)。如果不存在，则返回  -1。
输入: haystack = "hello", needle = "ll"
输出: 2
 */
func strStr(haystack string, needle string) int {
	if haystack == "" && needle == "" || len(needle) < 1 {
		return 0
	}
	if len(haystack) < 1 || len(needle) > len(haystack) {
		return -1
	}
	var i, start, end, pos, tmp int
	var flag bool
	for i < len(haystack) {
		if haystack[i] == needle[end] {
			end ++
			i ++
			if end == len(needle) {
				return start
			}
			continue
		}
		if start + len(needle) < len(haystack) {
			tmp = start + len(needle)
			for j:=len(needle)-1; j>=0; j-- {
				if haystack[tmp] == needle[j] {
					flag = true
					pos = j
					break
				}
			}
			if flag {
				start = start + len(needle) - pos
			} else {
				start = start + len(needle) + 1
			}
			i = start
			end = 0
		} else {
			return -1
		}
	}
	return -1
}

/**
14 最长公共前缀子串
 */
func longestCommonPrefix(strs []string) string {
	if len(strs) < 1 {
		return  ""
	}
	for i:=0; i<len(strs[0]); i++ {
		c := strs[0][i:i+1]
		for j:=1; j<len(strs); j++ {
			if i == len(strs[j]) || c != strs[j][i:i+1] {
				return strs[0][0:i]
			}
		}
	}
	return strs[0]
}

/**
58.最后一个单词的长度
 */
func lengthOfLastWord(s string) int {
	s = strings.Trim(s, " ")
	if len(s) < 1 {
		return 0
	}
	var cnt int
	for i:=len(s)-1; i>=0; i-- {
		if s[i:i+1] == " " {
			break
		}
		cnt ++
	}
	return cnt
}

/**
387 第一个不重复的字符串的位置
 */
func firstUniqChar(s string) int {
	var cnt [26]int
	for _, c := range s {
		cnt[c-'a']++
	}
	for i, c := range s {
		if cnt[c-'a'] == 1 {
			return i
		}
	}
	return -1
}

/**
383. 赎金信
canConstruct("a", "b") -> false
canConstruct("aa", "ab") -> false
canConstruct("aa", "aab") -> true
 */
func canConstruct(ransomNote string, magazine string) bool {
	counts := make(map[byte]int)
	for i:=0; i<len(ransomNote); i++ {
		counts[ransomNote[i]] ++
	}
	for b, c := range counts {
		if strings.Count(magazine, string(b)) < c {
			return false
		}
	}
	return true
}

/*
344. 反转字符串
输入：["h","e","l","l","o"]
输出：["o","l","l","e","h"]
*/
func reverseString(s []byte)  {
	i, j := 0, len(s)-1
	for i < j {
		tmp := s[i]
		s[i] = s[j]
		s[j] = tmp
		i ++
		j --
	}
}

/*
151. 翻转字符串里的单词
输入: "the sky is blue"
输出: "blue is sky the"
*/
func reverseWords(s string) string {
	s = strings.Trim(s, " ")
	if len(s) < 1 {
		return s
	}
	srcBytes := []byte(s)
	bytesSlice := bytes.Split(srcBytes, []byte{' '})
	var dstBytes []byte
	for i:=len(bytesSlice)-1; i>=0; i-- {
		if len(bytesSlice[i]) == 0 {
			continue
		}
		dstBytes = append(dstBytes, bytesSlice[i]...)
		if i > 0 {
			dstBytes = append(dstBytes, ' ')
		}
	}
	return string(dstBytes)
}

/*
345. 反转字符串中的元音字母
输入: "leetcode"
输出: "leotcede"
 */
func reverseVowels(s string) string {
	if len(s) < 1 {
		return s
	}
	i, j := 0, len(s) - 1
	byteSlice := []byte(s)
	for i < j {
		if !isOrigin(s[i]) {
			i ++
			continue
		}
		if !isOrigin(s[j]) {
			j --
			continue
		}
		byteSlice[i], byteSlice[j] = byteSlice[j], byteSlice[i]
		i ++
		j --
	}
	return string(byteSlice)
}

func isOrigin(c byte) bool {
	if c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' ||
		c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U' {
		return true
	}
	return false
}

/*
290. 单词规律
输入: pattern = "abba", str = "dog cat cat dog"
输出: true
 */
func wordPattern(pattern string, str string) bool {
	byteSlice := []byte(pattern)
	stringSlice := strings.Split(str, " ")
	if len(byteSlice) != len(stringSlice) {
		return false
	}
	patMap := make(map[byte]int)
	strMap := make(map[string]int)
	for _, v := range byteSlice {
		patMap[v] ++
	}
	for _, v := range stringSlice {
		strMap[v] ++
	}
	for i:=0; i<len(stringSlice); i++ {
		if strMap[stringSlice[i]] != patMap[byteSlice[i]] {
			return false
		}
	}
	return true
}

/*
242. 有效的字母异位词
输入: s = "anagram", t = "nagaram"
输出: true
 */
func isAnagram(s string, t string) bool {
	if len(s) != len(t) {
		return false
	}
	var cnt [26]int
	for _, v := range s {
		cnt[v-'a'] ++
	}
	for _, v := range t {
		cnt[v-'a'] --
		if cnt[v-'a'] < 0 {
			return false
		}
	}
	return true
}

/*
49. 字母异位词分组
输入: ["eat", "tea", "tan", "ate", "nat", "bat"],
输出:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]
 */
func groupAnagrams(strs []string) [][]string {
	r := [][]string{}
	if len(strs) < 1 {
		return r
	}
	mark := make(map[[26]int]int)
	for _, str := range strs {
		key := [26]int{}
		for _, c := range str {
			key[c-'a'] ++
		}
		i, ok := mark[key]
		if ok {
			r[i] = append(r[i], str)
		} else {
			r = append(r, []string{str})
			mark[key] = len(mark)
		}
	}
	return r
}

func groupAnagrams1(strs []string) [][]string {
	dic := map[string][]string{}

	for _, str := range strs {
		bl := ByteList(str)
		bl.Sort()
		if slice, ok := dic[string(bl)]; ok {
			dic[string(bl)] = append(slice, str)
		} else {
			dic[string(bl)] = []string{str}
		}
	}

	var res [][]string
	for _, slice := range dic {
		res = append(res, slice)
	}

	return res
}

type ByteList []byte

func (p ByteList) Len() int           { return len(p) }
func (p ByteList) Less(i, j int) bool { return p[i] < p[j] }
func (p ByteList) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
// Sort is a convenience method.
func (p ByteList) Sort() { sort.Sort(p) }

/*
179. 最大数
输入: [3,30,34,5,9]
输出: 9534330
 */
func largestNumber(nums []int) string {
	if len(nums) < 1 {
		return ""
	}
	sSlice := []string{}
	for _, v := range nums {
		sSlice = append(sSlice, strconv.Itoa(v))
	}
	//sort.Sort(sort.Reverse(sort.StringSlice(sSlice)))
	s := StrSlice(sSlice)
	s.Sort()
	if s[0] == "0" {
		return "0"
	}
	var r string
	for _, v := range s {
		r += v
	}
	return r
}

type StrSlice []string

func (s StrSlice) Len() int {
	return len(s)
}

func (s StrSlice) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s StrSlice) Less(i, j int) bool {
	s1 := s[i] + s[j]
	s2 := s[j] + s[i]
	return strings.Compare(s1, s2) > 0
}

func (s StrSlice) Sort() {
	sort.Sort(s)
}

/*
6. Z 字形变换
 */
func convert(s string, numRows int) string {
	if numRows == 1 {
		return s
	}
	rows := make([]string, min(len(s), numRows))
	curRow := 0
	goDown := false
	for _, v := range s {
		rows[curRow] += string(v)
		if curRow == 0 || curRow == min(len(s), numRows) - 1 {
			goDown = !goDown
		}
		if goDown {
			curRow ++
		} else {
			curRow --
		}
	}
	var r string
	for _, str := range rows {
		r += str
	}
	return r
}

/*
168. Excel表列名称
    1 -> A
    2 -> B
    3 -> C
    ...
    26 -> Z
    27 -> AA
    28 -> AB
    ...
 */
func convertToTitle(n int) string {
	var s string
	for n > 0 {
		c := n % 26
		if c == 0 {
			c = 26
			n --
		}
		s = string('A'+c-1)+s
		n /= 26
	}
	return s
}

/*
171. Excel表列序号
   A -> 1
    B -> 2
    C -> 3
    ...
    Z -> 26
    AA -> 27
    AB -> 28
    ...
 */
func titleToNumber(s string) int {
	var j, r int
	for i:=len(s)-1; i>=0; i-- {
		a := int(math.Pow(float64(26), float64(j)))
		b := int(s[i]-'A'+1)
		tmp := a * b
		r += tmp
		j ++
	}
	return r
}

/*
13. 罗马数字转整数
 */
func romanToInt(s string) int {
	if len(s) < 1 {
		return 0
	}
	mp := map[byte]int{
		'I':1,
		'V':5,
		'X':10,
		'L':50,
		'C':100,
		'D':500,
		'M':1000,
	}
	var r int
	for k, _ := range s {
		t := mp[s[k]]
		if k == len(s)-1 || mp[s[k+1]] <= t {
			r += t
		} else {
			r -= t
		}
	}
	return r
}

/*
12. 整数转罗马数字
 */
func intToRoman(num int) string {
	store := []int{1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1}
	strs := []string{"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"}
	var r string
	var i int
	for num > 0 {
		cnt := num / store[i]
		for cnt > 0 {
			r += strs[i]
			cnt --
		}
		num %= store[i]
		i ++
	}
	return r
}

func intToRoman1(num int) string {
	var runes []rune
	var i int
	for num != 0 {
		switch {
		case num >= 1000:
			i = num / 1000
			num -= i * 1000
			for ; i > 0; i-- {
				runes = append(runes, 'M')
			}
		case num >= 900:
			num -= 900
			runes = append(runes, 'C', 'M')
		case num >= 500:
			num -= 500
			runes = append(runes, 'D')
		case num >= 400:
			num -= 400
			runes = append(runes, 'C', 'D')
		case num >= 100:
			i = num / 100
			num -= i * 100
			for ; i > 0; i-- {
				runes = append(runes, 'C')
			}
		case num >= 90:
			num -= 90
			runes = append(runes, 'X', 'C')
		case num >= 50:
			num -= 50
			runes = append(runes, 'L')
		case num >= 40:
			num -= 40
			runes = append(runes, 'X', 'L')
		case num >= 10:
			i = num / 10
			num -= i * 10
			for ; i > 0; i-- {
				runes = append(runes, 'X')
			}
		case num >= 9:
			num -= 9
			runes = append(runes, 'I', 'X')
		case num >= 5:
			num -= 5
			runes = append(runes, 'V')
		case num >= 4:
			num -= 4
			runes = append(runes, 'I', 'V')
		case num >= 1:
			for ; num > 0; num-- {
				runes = append(runes, 'I')
			}
		}
	}
	return string(runes)
}

/*
3. 无重复字符的最长子串
输入: "abcabcbb"
输出: 3
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
 */
func lengthOfLongestSubstring(s string) int {
	if len(s) < 1 {
		return 0
	}
	var maxLen, j int
	var window [128]int
	for i:=0; i<len(s); i++ {
		if window[s[i]] > j {
			j = window[s[i]]
		} else if i - j + 1 > maxLen {
			maxLen = i - j + 1
		}
		window[s[i]] = i + 1
	}
	return maxLen
}

/*
395. 至少有K个重复字符的最长子串
输入:
s = "ababbc", k = 2
输出:
5
最长子串为 "ababb" ，其中 'a' 重复了 2 次， 'b' 重复了 3 次。
*/
func longestSubstring(s string, k int) int {
	if len(s) < 1 || k > len(s) {
		return 0
	}
	if k < 2 {
		return len(s)
	}
	return count(s, k, 0, len(s)-1)
}

func count(s string, k int, p1, p2 int) int {
	if p2 - p1 + 1 < k {
		return 0
	}
	var times [26]int
	for i:=p1; i<=p2; i++ {
		times[s[i]-'a'] ++
	}
	for p2-p1+1 >= k && times[s[p1]-'a'] < k {
		p1 ++
	}
	for p2-p1+1 >= k && times[s[p2]-'a'] < k {
		p2 --
	}
	if p2 - p1 + 1 < k {
		return 0
	}
	for i:=p1; i<p2; i++ {
		if times[s[i]-'a'] < k {
			return max(count(s, k, p1, i-1), count(s, k, i+1, p2))
		}
	}
	return p2-p1+1
}

/**
125. 验证回文串
输入: "A man, a plan, a canal: Panama"
输出: true
*/
func isPalindrome(s string) bool {
	s = strings.Trim(s, " ")
	if len(s) < 1 {
		return true
	}
	s = strings.ToLower(s)
	i, j := 0, len(s)-1
	for i<=j {
		if (s[i] >= '0' && s[i] <= '9' || s[i] >= 'a' && s[i] <= 'z') &&
			(s[j] >= '0' && s[j] <= '9' || s[j] >= 'a' && s[j] <= 'z') {
			if s[i] != s[j] {
				return false
			}
			i ++
			j --
		} else if s[i] >= '0' && s[i] <= '9' || s[i] >= 'a' && s[i] <= 'z' {
			j --
		} else {
			i ++
		}
	}
	return true
}

type T1 struct {
	Name string
	ID   int64
	Type int64
}

type T2 struct {
	ID   int64
	Type *int64
}

func m1() {
	sourceList := []*T1{
		{Name: "name1", ID: 1, Type: 1},
		{Name: "name2", ID: 2, Type: 2},
		{Name: "name3", ID: 3, Type: 3},
	}
	fmt.Println(sourceList)
	retMap := map[int64]T2{}
	for _, v := range sourceList {
		temp := T2{
			ID:   v.ID,
			Type: &v.Type,
		}
		retMap[v.ID] = temp
	}

	fmt.Printf("retMap %+v\n", retMap)
	for _, retMapV := range retMap {
		fmt.Println(*retMapV.Type)
	}
}

/*
5. 最长回文子串
输入: "babad"
输出: "bab"
注意: "aba" 也是一个有效答案。
 */
func longestPalindrome(s string) string {
	if len(s) < 1 {
		return s
	}
	var reverseStr string
	for i:=len(s)-1; i>=0; i-- {
		reverseStr += string(s[i])
	}
	var maxLen, maxEnd int
	a := make([]int, len(s))
	for i:=0; i<len(s); i++ {
		for j:=0; j<len(reverseStr); j++ {
			if s[i] == reverseStr[j] {
				if i == 0 || j == 0 {
					a[j] = 1
				} else {
					a[j] = a[j-1] + 1
				}
			} else  {
				a[j] = 0
			}
			if a[j] > maxLen {
				beforeRev := len(s) - j - 1
				if beforeRev + a[j] + 1 == i {
					maxLen = a[j]
					maxEnd = i
				}

			}
		}
	}
	return s[maxEnd-maxLen-1:maxEnd+1]
}

func longestPalindrome1(s string) string {
	if len(s) < 1 {
		return s
	}
	var start, end int
	for i:=0; i<len(s); i++ {
		len1 := expandAroundCenter(s, i, i)
		len2 := expandAroundCenter(s, i, i+1)
		maxLen := max(len1, len2)
		if maxLen > end-start {
			start = i - (maxLen-1) / 2
			end = i + (maxLen) / 2
		}
	}
	return s[start:end+1]
}

func expandAroundCenter(s string, start, end int) int {
	for start >= 0 && end < len(s) && s[start] == s[end] {
		start --
		end ++
	}
	return end-start-1
}

/*
9. 回文数
输入: -121
输出: false
解释: 从左向右读, 为 -121 。 从右向左读, 为 121- 。因此它不是一个回文数。
*/
func isPalindromeInt(x int) bool {
	if x < 0 {
		return false
	}
	new, tmp := 0, x
	for tmp > 0 {
		new = new*10 + tmp%10
		tmp /= 10
	}
	if new == x {
		return true
	}
	return false
}

/*
131. 分割回文串
给定一个字符串 s，将 s 分割成一些子串，使每个子串都是回文串。返回 s 所有可能的分割方案。
输入: "aab"
输出:
[
  ["aa","b"],
  ["a","a","b"]
]
 */
func partition(s string) [][]string {
	r := [][]string{}
	if len(s) < 1 {
		return r
	}
	stacek := []string{}
	backTrace(s, 0, len(s), &stacek, &r)
	return r
}

func backTrace(s string, start, length int, stack *[]string, res *[][]string)  {
	if start == length {
		collected := make([]string, len(*stack))
		copy(collected, *stack)
		*res = append(*res, collected)
		return
	}
	for i:=start; i<length; i++ {
		if !checkPalindrome(s, start, i) {
			continue
		}
		*stack = append(*stack, s[start:i+1])
		backTrace(s, i+1, length, stack, res)
		*stack = (*stack)[:len(*stack)-1]
	}
}

func checkPalindrome(s string, start, end int) bool {
	for start < end {
		if s[start] != s[end] {
			return false
		}
		start ++
		end --
	}
	return true
}

/*
20. 有效的括号
输入: "()[]{}"
输出: true
 */
func isValid(s string) bool {
	if len(s) < 1 {
		return true
	}
	var stack []string
	for _, v := range s {
		if v == '{' || v == '[' || v == '(' {
			stack = append(stack, string(v))
		} else {
			if len(stack) < 1 {
				return false
			}
			for len(stack) > 0 {
				p := stack[len(stack)-1:]
				stack = stack[:len(stack)-1]
				if v == '}' && p[0] != "{" {
					return false
				} else if v == ']' && p[0] != "[" {
					return false
				} else if  v == ')' && p[0] != "(" {
					return false
				} else {
					break
				}
			}
		}
	}
	if len(stack) > 0 {
		return false
	}
	return true
}

/*
22. 括号生成
给出 n 代表生成括号的对数，请你写出一个函数，使其能够生成所有可能的并且有效的括号组合。
例如，给出 n = 3，生成结果为：
[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]
*/
func generateParenthesis(n int) []string {
	r := []string{}
	if n == 0 {
		return r
	}
	genImpl(n, n, "", &r)
	return r
}

func genImpl(left, right int, s string, r *[]string)  {
	if left == 0 && right ==0 {
		*r = append(*r, s)
		return
	}
	if left > 0 {
		genImpl(left-1, right, s+"(", r)
	}
	if right > 0 && right > left {
		genImpl(left, right-1, s+")", r)
	}
}

/*
241. 为运算表达式设计优先级
https://leetcode-cn.com/problems/different-ways-to-add-parentheses/
*/
func diffWaysToCompute(input string) []int {
	// 如果是数字，直接返回
	if isDigit(input) {
		tmp, _ := strconv.Atoi(input)
		return []int{tmp}
	}

	// 空切片
	var res []int
	// 遍历字符串
	for index, c := range input {
		tmpC := string(c)
		if tmpC == "+" || tmpC == "-" || tmpC == "*" {
			// 如果是运算符，则计算左右两边的算式
			left := diffWaysToCompute(input[:index])
			right := diffWaysToCompute(input[index+1:])

			for _, leftNum := range left {
				for _, rightNum := range right {
					var addNum int
					if tmpC == "+" {
						addNum = leftNum + rightNum
					} else if tmpC == "-" {
						addNum = leftNum - rightNum
					} else {
						addNum = leftNum * rightNum
					}
					res = append(res, addNum)
				}
			}
		}
	}

	return res
}
// 判断是否为全数字
func isDigit(input string) bool {
	_, err := strconv.Atoi(input)
	if err != nil {
		return false
	}
	return true
}

/*
392. 判断子序列
示例 1:
s = "abc", t = "ahbgdc"
返回 true.
 */
func isSubsequence(s string, t string) bool {
	if len(s) > len(t) {
		return false
	}
	if len(s) < 1 {
		return true
	}
	var i, j int
	for i<len(s) && j<len(t) {
		if s[i] == t[j] {
			i ++
			j ++
		} else {
			j ++
		}
		if i == len(s) {
			return true
		}
	}
	return false
}

/*
187. 重复的DNA序列
https://leetcode-cn.com/problems/repeated-dna-sequences/
 */
func findRepeatedDnaSequences(s string) []string {
	if len(s) < 10 {
		return []string{}
	}
	r := []string{}
	mp := make(map[string]int)
	for i:=0; i<=len(s)-10; i++ {
		if v, _ := mp[s[i:i+10]]; v == 1 {
			r = append(r, s[i:i+10])
		}
		mp[s[i:i+10]] ++
	}
	return r
}

func reverse(x int) int {
	if x == 0 {
		return 0
	}
	symbol := true
	if x < 0 {
		symbol = false
		x = -x
	}
	var r int
	for x > 0 {
		r = r*10 + x%10
		if r > 2147483647 {
			return 0
		}
		x /= 10
	}
	if !symbol {
		return -r
	}
	return r
}

/*
165. 比较版本号
https://leetcode-cn.com/problems/compare-version-numbers/
 */
func compareVersion(version1 string, version2 string) int {
	if len(version1) == 0 && len(version2) == 0 {
		return 0
	}
	v1Slice := strings.Split(version1, ".")
	v2Slice := strings.Split(version2, ".")
	var i, j int
	for i<len(v1Slice) || j<len(v2Slice) {
		v1, v2 := 0, 0
		if i < len(v1Slice) {
			v1, _ = strconv.Atoi(v1Slice[i])
		}
		if j < len(v2Slice) {
			v2, _ = strconv.Atoi(v2Slice[j])
		}
		if v1 > v2 {
			return 1
		} else if v1 < v2 {
			return -1
		}
		i ++
		j ++
	}
	return 0
}

/*
66. 加一
https://leetcode-cn.com/problems/plus-one/
 */
func plusOne(digits []int) []int {
	if len(digits) < 1 {
		return []int{1}
	}
	r := []int{}
	i, carry, cur := len(digits) - 1, 0, 0
	for i >= 0 {
		if i == len(digits) - 1 {
			cur = digits[i] + 1
		} else {
			cur = digits[i] + carry
		}
		r = append([]int{cur % 10}, r...)
		carry = cur / 10
		i --
	}
	if carry > 0 {
		r = append([]int{1}, r...)
	}
	return r
}

/*
8. 字符串转换整数 (atoi)
https://leetcode-cn.com/problems/string-to-integer-atoi/
 */
func myAtoi(str string) int {
	str = strings.Trim(str, " ")
	if len(str) < 1 {
		return 0
	}
	symbol := true
	i, r := 0, 0
	if str[0] == '-' {
		symbol = false
		i ++
	} else if str[0] == '+' {
		i ++
	} else if str[0] < '0' || str[0] > '9' {
		return 0
	}
	for ; i<len(str); i++ {
		if str[i] >= '0' && str[i] <= '9' {
			tmp, _ := strconv.Atoi(string(str[i]))
			r = r * 10 + tmp
			if r > math.MaxInt32 {
				if symbol {
					return math.MaxInt32
				}
				return math.MinInt32
			}
		} else {
			break
		}
	}
	if !symbol {
		r *= -1
	}
	return r
}

/*
258. 各位相加
https://leetcode-cn.com/problems/add-digits/
*/
func addDigits(num int) int {
	if num <= 0 {
		return 0
	}
	r := 0
	for num > 0 {
		r += num % 10
		num /= 10
		if num <= 0 {
			if r > 9 {
				num = r
				r = 0
			} else {
				break
			}
		}
	}
	return r
}

/*
67. 二进制求和
输入: a = "1010", b = "1011"
输出: "10101"
https://leetcode-cn.com/problems/add-binary/
 */
func addBinary(a string, b string) string {
	if len(a) < 1 {
		return b
	}
	if len(b) < 1 {
		return a
	}
	i, j, cur, carry := len(a)-1, len(b)-1, 0, 0
	var r string
	for i >= 0 || j >= 0 {
		a1, b1 := 0, 0
		if i >= 0 {
			a1, _ = strconv.Atoi(string(a[i]))
		}
		if j >= 0 {
			b1, _ = strconv.Atoi(string(b[j]))
		}
		cur = (a1 + b1 + carry) % 2
		carry = (a1 + b1 + carry) / 2
		r = strconv.Itoa(cur) + r
		i --
		j --
	}
	if carry > 0 {
		r = "1" + r
	}
	return r
}

/*
43. 字符串相乘
输入: num1 = "2", num2 = "3"
输出: "6"
https://leetcode-cn.com/problems/multiply-strings/
 */
func multiply(num1 string, num2 string) string {
	if len(num1) < 1 || len(num2) < 1{
		return "0"
	}
	if num1 == "0" || num2 == "0" {
		return "0"
	}
	r, carry := "", 0
	for i:=len(num2)-1; i>=0; i-- {
		tmp := ""
		for j:=0; j<len(num2)-i-1; j++ {
			tmp = tmp + "0"
		}
		n2, _ := strconv.Atoi(string(num2[i]))
		for j:=len(num1)-1; j>=0||carry>0; j-- {
			var n1 int
			if j >= 0 {
				n1, _ = strconv.Atoi(string(num1[j]))
			}
			sum := (n1 * n2 + carry) % 10
			tmp = strconv.Itoa(sum) + tmp
			carry = (n1 * n2 + carry) / 10
		}
		r = addStrings(r, tmp)
	}
	return r
}

func addStrings(a string, b string) string {
	if len(a) < 1 {
		return b
	}
	if len(b) < 1 {
		return a
	}
	i, j, cur, carry := len(a)-1, len(b)-1, 0, 0
	var r string
	for i >= 0 || j >= 0 {
		a1, b1 := 0, 0
		if i >= 0 {
			a1, _ = strconv.Atoi(string(a[i]))
		}
		if j >= 0 {
			b1, _ = strconv.Atoi(string(b[j]))
		}
		cur = (a1 + b1 + carry) % 10
		carry = (a1 + b1 + carry) / 10
		r = strconv.Itoa(cur) + r
		i--
		j--
	}
	if carry > 0 {
		r = "1" + r
	}
	return r
}

func multiply1(num1 string, num2 string) string {
	if num1 == "0" || num2 == "0" {
		return "0"
	}
	res := make([]byte, len(num1)+len(num2))
	for i:=len(num1)-1; i>=0; i-- {
		n1 := num1[i] - '0'
		for j:=len(num2)-1; j>=0; j-- {
			n2 := num2[j] - '0'
			tmp := n1*n2 + res[i+j+1]
			res[i+j+1] = tmp % 10
			res[i+j] += tmp / 10
		}
	}
	status := false
	var r string
	for i:=0; i<len(res); i++ {
		if status == false && res[i] != 0 {
			status = true
		}
		if status == true {
			r += strconv.Itoa(int(res[i]))
		}
	}
	return r
}

/*
29. 两数相除
输入: dividend = 10, divisor = 3
输出: 3
https://leetcode-cn.com/problems/divide-two-integers/
 */
func divide(dividend int, divisor int) int {
	if dividend == 0 {
		return 0
	}
	if dividend == -2147483648 && divisor == -1 {
		return 2147483647
	}
	sign := true
	if dividend > 0 && divisor < 0 || (dividend < 0 && divisor > 0) {
		sign = false
	}
	var r int
	a, b := dividend, divisor
	if dividend < 0 {
		a = -dividend
	}
	if divisor < 0 {
		b = -divisor
	}
	for a >= b {
		var shift int
		for a >= (b << shift) {
			shift ++
		}
		a -= b << (shift-1)
		r += 1 << (shift-1)
	}
	if !sign {
		return -r
	}
	return r
}

/*
69. x 的平方根
输入: 8
输出: 2
说明: 8 的平方根是 2.82842...,
     由于返回类型是整数，小数部分将被舍去。
https://leetcode-cn.com/problems/sqrtx/
*/
func mySqrt(x int) int {
	if x <= 0 {
		return 0
	}
	start, end, m := 0, x/2+1, 0
	for start < end {
		m = (start + end + 1) / 2
		if m*m > x {
			end = m-1
		} else {
			start = m
		}
	}
	return start
}

/*
50. Pow(x, n)
输入: 2.00000, 10
输出: 1024.00000

输入: 2.00000, -2
输出: 0.25000
解释: 2-2 = 1/22 = 1/4 = 0.25
https://leetcode-cn.com/problems/powx-n/
 */
func myPow(x float64, n int) float64 {
	if n == 0 {
		return float64(1)
	}
	if x == float64(0) {
		return float64(0)
	}
	if n == 1 {
		return x
	}
	if n == -1 {
		return 1 / x
	}
	half := myPow(x, n/2)
	rest := myPow(x, n%2)
	return half*half*rest
}

/*
367. 有效的完全平方数
输入：16
输出：True
https://leetcode-cn.com/problems/valid-perfect-square/
*/
func isPerfectSquare(num int) bool {
	if num == 0 {
		return true
	}
	start, end := 0, num/2
	r := false
	for start < end {
		m := (start+end) / 2
		if m*m == num {
			r = true
			break
		} else if m*m < num {
			start = m+1
		} else {
			end = m-1
		}
	}
	return r
}

/*
204. 计数质数
https://leetcode-cn.com/problems/count-primes/
 */
func countPrimes(n int) int {
	isPrim := make([]bool, n)
	for i:=0; i<len(isPrim); i++ {
		isPrim[i] = true
	}
	for i:=2; i*i<n; i++ {
		if isPrim[i] {
			for j:=i*i; j<n ;j+=i {
				isPrim[j] = false
			}
		}
	}
	cnt := 0
	for i:=2; i<n; i++ {
		if isPrim[i] {
			cnt ++
		}
	}
	return cnt
}

/*
1. 两数之和
给定 nums = [2, 7, 11, 15], target = 9

因为 nums[0] + nums[1] = 2 + 7 = 9
所以返回 [0, 1]
https://leetcode-cn.com/problems/two-sum/
 */
func twoSum1(nums []int, target int) []int {
	mp := make(map[int]int, len(nums))
	for k, v := range nums {
		if i, ok := mp[target-v]; ok {
			return []int{i, k}
		} else {
			mp[v] = k
		}
	}
	return []int{}
}

/*
167. 两数之和 II - 输入有序数组
输入: numbers = [2, 7, 11, 15], target = 9
输出: [1,2]
解释: 2 与 7 之和等于目标数 9 。因此 index1 = 1, index2 = 2 。
https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted/
*/
func twoSum(numbers []int, target int) []int {
	i, j := 0, len(numbers)-1
	for i<j {
		if numbers[i] + numbers[j] == target {
			return []int{i+1, j+1}
		} else if numbers[i] + numbers[j] > target {
			j --
		} else {
			i ++
		}
	}
	return []int{}
}

/*
15. 三数之和
https://leetcode-cn.com/problems/3sum/
 */
func threeSum(nums []int) [][]int {
	if len(nums) < 3 {
		return [][]int{}
	}
	sort.Ints(nums)
	r := [][]int{}
	for i:=0; i<len(nums); i++ {
		if nums[i] > 0 {
			break
		}
		if i>0 && nums[i] == nums[i-1] {
			continue
		}
		L, R := i+1, len(nums)-1
		for L < R  {
			sum := nums[i] + nums[L] + nums[R]
			if sum == 0 {
				tmp := []int{}
				tmp = append(tmp, nums[i], nums[L], nums[R])
				r = append(r, tmp)
				for L < R && nums[L] == nums[L+1] {
					L ++
				}
				for L < R && nums[R] == nums[R-1] {
					R --
				}
				L ++
				R --
			} else if sum > 0 {
				R --
			} else  {
				L ++
			}
		}
	}
	return r
}

/*
16. 最接近的三数之和
https://leetcode-cn.com/problems/3sum-closest/
 */
func threeSumClosest(nums []int, target int) int {
	if len(nums) < 1 {
		return 0
	}
	sort.Ints(nums)
	ans := nums[0] + nums[1] + nums[2]
	if ans == target {
		return ans
	}
	for i:=0; i<len(nums); i++ {
		L, R := i+1, len(nums)-1
		for L < R {
			sum := nums[i] + nums[L] + nums[R]
			if math.Abs(float64(target)-float64(sum)) < math.Abs(float64(target)-float64(ans)) {
				ans = sum
			}
			if target == sum {
				return sum
			} else if sum < target {
				L ++
			} else {
				R --
			}
		}
	}
	return ans
}

/*
18. 四数之和
https://leetcode-cn.com/problems/4sum/
*/
func fourSum(nums []int, target int) [][]int {
	if len(nums) < 4 {
		return [][]int{}
	}
	sort.Ints(nums)
	r := [][]int{}
	for i:=0; i<len(nums)-3; i++ {
		if i>0 && nums[i] == nums[i-1] {
			continue
		}
		if nums[i] + nums[i+1] + nums[i+2] + nums[i+3] > target {
			break
		}
		if nums[i] + nums[len(nums)-1] + nums[len(nums)-2] + nums[len(nums)-3] < target {
			continue
		}
		for j:=i+1; j<len(nums)-2; j ++ {
			if j>i+1 && nums[j] == nums[j-1] {
				continue
			}
			L, R := j+1, len(nums)-1
			if nums[i] + nums[j] + nums[L] + nums[L+1] > target {
				break
			}
			if nums[i] + nums[j] + nums[R] + nums[R-1] < target {
				continue
			}
			for L < R {
				if nums[i] + nums[j] + nums[L] + nums[R] == target {
					tmp := []int{}
					tmp = append(tmp, nums[i], nums[j], nums[L], nums[R])
					r = append(r, tmp)
					for L<R && nums[L] == nums[L+1] {
						L ++
					}
					for L<R && nums[R] == nums[R-1] {
						R --
					}
					L ++
					R --
				} else if nums[i] + nums[j] + nums[L] + nums[R] < target {
					L ++
				} else {
					R --
				}
			}
		}
	}
	return r
}

func main2() {
	//haystack := "a"
	//needle := ""
	//r := strStr(haystack, needle)
	//strs := []string{"flower","flow","flight"}
	//r := longestCommonPrefix(strs)
	//r := reverseWords("the  sky  is  blue")
	//r := reverseVowels("leetcode")
	//r := wordPattern("abba", "dog cat cat gg")
	//r := isAnagram("anagram", "nagaras")
	//r := groupAnagrams([]string{"eat", "tea", "tan", "ate", "nat", "bat"})
	//r := largestNumber([]int{0})
	//r := convert("AB", 1)
	//r := convertToTitle(701)
	//r := titleToNumber("ZY") //28
	//r := intToRoman1(30)
	//r := lengthOfLongestSubstring("abcabcbb")
	//r := longestSubstring("weitong", 2)
	//r := isPalindrome("A man, a plan, a canal: Panama")
	//r := longestPalindrome1("babad")
	//r := isPalindromeInt(10)
	//r := partition("aa")
	//r := isValid("{[]]}")
	//r := generateParenthesis(3)
	//r := diffWaysToCompute("2*3-4*5")
	//r := isSubsequence("abcd", "ahbgdc")
	//r := findRepeatedDnaSequences("AAAAAAAAAAA")
	//r := reverse(-123)
	//r := compareVersion("1.01.2", "1.001.1")
	//r := plusOne([]int{1,2,9})
	//r := myAtoi("+1")
	//r := addDigits(38)
	//r := addBinary("11", "1")
	//r := multiply1("123", "456")
	//r := mySqrt(4)
	//r := myPow(  0.44894, -5)
	//r := isPerfectSquare(16)
	//r := countPrimes(10)
	//r := twoSum([]int{2, 7, 11, 15}, 9)
	//r := threeSum([]int{-1, 0, 1, 2, -1, -4})
	r := fourSum([]int{1, 0, -1, 0, -2, 2}, 0)
	fmt.Println(r)
}