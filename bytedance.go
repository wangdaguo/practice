package main

import (
   "fmt"
   "sort"
   "strings"
)
 
func main() {
   //node5 := &ListNode{
   //   Val:   5,
   //   Next:  nil,
   //}
   //node4 := &ListNode{
   //   Val:   4,
   //   Next: node5,
   //}
   //node3 := &ListNode{
   //   Val:   3,
   //   Next: node4,
   //}
   //node2 := &ListNode{
   //   Val:   2,
   //   Next: node3,
   //}
   //head := &ListNode{
   //   Val:   1,
   //   Next: node2,
   //}
   //PrintList(head)
   //r := reverseKGroup(head, 2)
   //r := isPalindrome("a$b12*1&ba")

   //a := []int{1,2}
   //fmt.Println(a[1:len(a)])

   //r := maxSlidingWindow2([]int{9,9,2,3,1,3,6,7}, 3)
   //r := longestCommonPrefix([]string{"flower","flower","flower","flower"})
   //r := threeSum([]int{-1,0,1,2,-1,-4})
   r := selectionSort([]int{-1,0,1,2,-1,-4})
   fmt.Println(r)
}

func selectionSort(arr []int) []int {
   if len(arr) < 1 {
      return arr
   }
   for i:=0; i<len(arr); i++ {
      min := i
      for j:=i+1; j<len(arr); j++ {
         if arr[j] < arr[min] {
            min = j
         }
      }
      arr[i], arr[min] = arr[min], arr[i]
   }
   return arr
}

func bubbleSort(arr []int) []int {
   if len(arr) < 1 {
      return arr
   }
   for i:=0; i<len(arr); i++ {
      for j:=0; j<len(arr); j++ {
         if arr[i] < arr[j] {
            arr[i], arr[j] = arr[j], arr[i]
         }
      }
   }
   return arr
}

func threeSum(nums []int) [][]int {
   r := make([][]int, 0)
   if len(nums) < 3 {
      return r
   }
   sort.Slice(nums, func(i, j int) bool {
      if nums[i] < nums[j] {
         return true
      }
      return false
   })
   if nums[0] > 0 || nums[len(nums)-1] < 0 {
      return r
   }
   for i := range nums {
      if nums[i] > 0 {
         return r
      }
      if i > 0 && nums[i] == nums[i-1] {
         continue
      }
      left, right := i+1, len(nums)-1
      for left < right {
         if nums[i] + nums[left] + nums[right] < 0 {
            left ++
         } else if nums[i] + nums[left] + nums[right] > 0 {
            right --
         } else {
            r = append(r, []int{nums[i], nums[left], nums[right]})
            for left < right && nums[left+1] == nums[left] {
               left ++
            }
            for left < right && nums[right-1] == nums[right] {
               right --
            }
            left ++
            right --
         }
      }
   }
   return r
}

func intersect(nums1 []int, nums2 []int) []int {
   sort.Slice(nums1, func(i, j int) bool {
      if nums1[i] < nums1[j] {
         return true
      }
      return false
   })
   sort.Slice(nums2, func(i, j int) bool {
      if nums2[i] < nums2[j] {
         return true
      }
      return false
   })
   var start1, start2 int
   var r []int
   for start1 < len(nums1) && start2 < len(nums2) {
      if nums1[start1] == nums2[start2] {
         r = append(r, nums1[start1])
         start1 ++
         start2 ++
      } else if nums1[start1] < nums2[start2] {
         start1 ++
      } else {
         start2 ++
      }
   }
   return r
}

func longestCommonPrefix(strs []string) string {
   if len(strs) < 1 {
      return ""
   }
   if len(strs) == 1 {
      return strs[0]
   }
   firStr, ret := strs[0], []byte{}
   for i:=0; i<len(firStr); i++ {
      var cnt int
      for j:=1; j<len(strs); j++ {
         if len(strs[j])-1 >= i && firStr[i] == strs[j][i] {
            cnt ++
         } else {
            return string(ret)
         }
      }
      if cnt == len(strs)-1 {
         ret = append(ret, firStr[i])
      }
   }
   return string(ret)
}

func maxSlidingWindow2(nums []int, k int) []int {
   ret := make([]int,0)
   if len(nums) == 0 {
      return ret
   }
   var queue []int
   for i := range nums {
      for i > 0 && (len(queue) > 0) && nums[i] > queue[len(queue)-1] {
         //将比当前元素小的元素祭天
         queue = queue[:len(queue)-1]
      }
      //将当前元素放入queue中
      queue = append(queue, nums[i])

      // 超过滑动窗口的长度数字出列
      if i >= k && nums[i-k] == queue[0] {
         queue = queue[1:]
      }
      if i >= k-1 {
         //放入结果数组
         ret = append(ret, queue[0])
      }
   }
   return ret
}

func maxSlidingWindow(nums []int, k int) []int {
   if len(nums) < 1 {
      return []int{}
   }
   r, index := []int{}, 0
   for index < len(nums) {
      m := nums[index]
      if index > len(nums)-k {
         break
      }
      for j:=index+1; j<index+k; j++  {
         if m < nums[j] {
            m = nums[j]
         }
      }
      r = append(r, m)
      index ++
   }
   return r
}

func Reverse(s []byte) {
   if len(s) < 1 {
      return
   }
   left, right := 0, len(s)-1
   for left < right {
      s[left], s[right] = s[right], s[left]
      left ++
      right --
   }
}

func firstUniqueChar(s string) int {
   if len(s) < 1 {
      return -1
   }
   var arr [26]int
   for i, v := range s {
      arr[v-'a'] = i
   }
   for i, v := range s {
      if arr[v-'a'] == i {
         return i
      }
   }
   return -1
}

func isPalindrome(s string) bool {
   if len(s) < 1 {
      return true
   }
   s = strings.ToLower(s)
   start, end := 0, len(s) - 1
   for start < end {
      if !((s[start] >= 'a' && s[start] <= 'z') ||  (s[start] >= '0' && s[start] <= '9')) {
         start ++
         continue
      }
      if !((s[end] >= 'a' && s[end] <= 'z') ||  (s[end] >= '0' && s[end] <= '9')) {
         end --
         continue
      }
      if s[start] != s[end] {
         return false
      }
      start ++
      end --
   }
   return true
}

func PrintList(head *ListNode)  {
   h := head
   var arr []int
   for h != nil {
      arr = append(arr, h.Val)
      h = h.Next
   }
   fmt.Println(arr)
}

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
type ListNode struct {
   Val  int
   Next *ListNode
}

func reverseKGroup(head *ListNode, k int) *ListNode {
   if k <= 1 {
      return head
   }
   h, cur, i, j, m := head, head, 0, 0, 0
   for h != nil {
      i ++
      h = h.Next
   }
   h = head
   cnt := i / k
   preEnd, curEnd, newHead := head, head, head
   var pre, next *ListNode
   for {
      if j == cnt {
         break
      }
      next = cur.Next
      cur.Next = pre
      pre = cur
      cur = next
      m ++
      if k == 1 {
         j++
      } else if m%k == 0 {
         if m == k {
            newHead = pre
            preEnd.Next = cur
            curEnd = cur
         } else {
            preEnd.Next = pre
            curEnd.Next = cur
            preEnd = curEnd
            curEnd = cur
         }
         j++
      }
   }
   return newHead
}
