package main

import (
	"math"
)

func main()  {

	//r := twoSum([]int{2,7,11,15}, 9)
	//r := minWindow("ADOBECODEBANC", "ABC")
	//r := judgeSquareSum(1)
	//r := validPalindrome("abba")
	//fmt.Println(r)
	return

}

/**
167. 两数之和 II - 输入有序数组
给定一个已按照 升序排列  的整数数组 numbers ，请你从数组中找出两个数满足相加之和等于目标数 target 。
函数应该以长度为 2 的整数数组的形式返回这两个数的下标值。numbers 的下标 从 1 开始计数 ，所以答案数组应当满足 1 <= answer[0] < answer[1] <= numbers.length 。
你可以假设每个输入只对应唯一的答案，而且你不可以重复使用相同的元素。

输入：numbers = [2,7,11,15], target = 9
输出：[1,2]
解释：2 与 7 之和等于目标数 9 。因此 index1 = 1, index2 = 2 。
 */
func twoSum(numbers []int, target int) []int {
	head, tail := 0, len(numbers)-1
	for head < tail {
		if numbers[head] + numbers[tail] < target {
			head ++
		} else if numbers[head] + numbers[tail] > target {
			tail --
		} else {
			return []int{head+1, tail+1}
		}
	}
	return []int{head+1, head+1}
}

/**
88. 合并两个有序数组
给你两个有序整数数组 nums1 和 nums2，请你将 nums2 合并到 nums1 中，使 nums1 成为一个有序数组。
初始化 nums1 和 nums2 的元素数量分别为 m 和 n 。你可以假设 nums1 的空间大小等于 m + n，这样它就有足够的空间保存来自 nums2 的元素。

输入：nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
输出：[1,2,2,3,5,6]
 */
func merge(nums1 []int, m int, nums2 []int, n int)  {
	pos := len(nums1)-1
	for m > 0 && n > 0 {
		if nums1[m-1] > nums2[n-1] {
			nums1[pos] = nums1[m-1]
			pos --
			m --
		} else {
			nums1[pos] = nums2[n-1]
			pos --
			n --
		}
	}
	for m > 0 {
		nums1[pos] = nums1[m-1]
		pos --
		m --
	}
	for n > 0 {
		nums1[pos] = nums2[n-1]
		pos --
		n --
	}
	return
}

/**
给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。注意，pos 仅仅是用于标识环的情况，并不会作为参数传递到函数中。
说明：不允许修改给定的链表。
 */
type ListNode struct {
	Val  int
	Next *ListNode
}
func detectCycle(head *ListNode) *ListNode {
	if head == nil {
		return head
	}
	slow, fast := head, head
	for {
		if fast == nil || fast.Next == nil {
			return nil
		}
		slow = slow.Next
		fast = fast.Next.Next
		if slow == fast {
			break
		}
	}
	fast = head
	for fast != slow {
		fast = fast.Next
		slow = slow.Next
	}
	return fast
}

/**
76. 最小覆盖子串
给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。
注意：
对于 t 中重复字符，我们寻找的子字符串中该字符数量必须不少于 t 中该字符数量。
如果 s 中存在这样的子串，我们保证它是唯一的答案。
示例 1：

输入：s = "ADOBECODEBANC", t = "ABC"
输出："BANC"
示例 2：

输入：s = "a", t = "a"
输出："a"
 */
func minWindow(s string, t string) string {
	if len(s) == 0 || len(t) == 0 {
		return ""
	}
	mp := make(map[byte]bool)
	chars := make(map[byte]int)
	for i:=0; i<len(t); i++ {
		mp[t[i]] = true
		chars[t[i]] ++
	}
	var cnt, start, minStart int
	minSize := len(s) + 1
	for i:=0; i<len(s); i++ {
		if ok, _ := mp[s[i]]; ok {
			chars[s[i]] --
			if chars[s[i]] >= 0 {
				cnt ++
			}
		}
		for cnt == len(t) {
			if i-start+1 < minSize {
				minStart = start
				minSize = i-start+1
			}
			chars[s[start]] ++
			if ok, _ := mp[s[start]]; ok && chars[s[start]] > 0 {
				cnt --
			}
			start ++
		}
	}
	if minSize > len(s)  {
		return ""
	}
	return s[minStart:minSize+minStart]
}

/**
633. 平方数之和
给定一个非负整数 c ，你要判断是否存在两个整数 a 和 b，使得 a2 + b2 = c 。

输入：c = 5
输出：true
解释：1 * 1 + 2 * 2 = 5

输入：c = 3
输出：false
*/
func judgeSquareSum(c int) bool {
	start, end := 0, int(math.Sqrt(float64(c)))
	for start <= end {
		sum := start * start + end * end
		if sum < c {
			start ++
		} else if sum > c {
			end --
		} else if sum == c {
			return true
		}
	}
	return false
}

/**
680. 验证回文字符串 Ⅱ
给定一个非空字符串 s，最多删除一个字符。判断是否能成为回文字符串。

示例 1:
输入: s = "aba"
输出: true

示例 3:
输入: s = "abc"
输出: false

"aguokepatgbnvfqmgmlcupuufxoohdfpgjdmysgvhmvffcnqxjjxqncffvmhvgsymdjgpfdhooxfuupuculmgmqfvnbgtapekouga"
 */
func validPalindrome(s string) bool {
	start, end := 0, len(s)-1
	for start < end {
		if s[start] != s[end] {
			return isPalindrome(s[start+1:end+1]) || isPalindrome(s[start:end])
		}
		start ++
		end --
	}
	return true
}

func isPalindrome(s string)bool{
	start, end := 0, len(s)-1
	for start < end {
		if s[start] != s[end] {
			return false
		}
		start ++
		end --
	}
	return tru