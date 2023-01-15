package main

import (
   "fmt"
   "sort"
   "strings"
)
 
func main() {
   //node5 := &ListNode{
   //  Val:   3,
   //  Next:  nil,
   //}
   //node4 := &ListNode{
   //  Val:   5,
   //  Next: node5,
   //}
   //node3 := &ListNode{
   //  Val:   2,
   //  Next: node4,
   //}
   //node2 := &ListNode{
   //  Val:   1,
   //  Next: node3,
   //}
   //head := &ListNode{
   //  Val:   4,
   //  Next: node2,
   //}
   //PrintList(head)
   //r := reverseKGroup(head, 2)
   //r := isPalindrome("a$b12*1&ba")

   //a := []int{1,2}
   //fmt.Println(a[1:len(a)])

   //r := maxSlidingWindow2([]int{9,9,2,3,1,3,6,7}, 3)
   //r := longestCommonPrefix([]string{"flower","flower","flower","flower"})
   //r := threeSum([]int{-1,0,1,2,-1,-4})
   //r := selectionSort([]int{-1,0,1,2,-1,-4})
   //r := sortList(head)
   //PrintList(head)
   //r := []int{-1,0,1,2,-1,-4}
   //tmp := make([]int, len(r))
   //bubbleSort21(&r, 6)
   //a := []int{3,2,1,5,6,4}
   //r := findKthLargestB(a, 2)
   //r := maxSumB([]int{1, -10, 3, 4})
   r := permuteB([]int{1,2,3})
   fmt.Println(r)
}

func permuteB(nums []int) [][]int {
   r := make([][]int, 0)
   if len(nums) < 1 {
      return r
   }
   mp := make(map[int]bool)
   backTraceB(nums, []int{}, &r, mp)
   return r
}

func backTraceB(nums, arr []int, r *[][]int, mp map[int]bool) {
   if len(arr) == len(nums) {
      tmp := make([]int, 0)
      tmp = append(tmp, arr...)
      *r = append(*r, tmp)
      return
   }
   for i:=0; i<len(nums); i++ {
      if _, ok := mp[nums[i]]; ok {
         continue
      }
      mp[nums[i]] = true
      arr = append(arr, nums[i])
      backTraceB(nums, arr, r, mp)
      delete(mp, nums[i])
      arr = arr[:len(arr)-1]
   }
}

func maxSumB(list []int) int {
   if len(list) < 1 {
      return 0
   }
   max, sumMax, i, j, start, end := 0, 0, 0, 0, 0, 0
   for k, v := range list {
      if max < 0 {
         max = v
         i, j = k, k
      } else {
         max += v
         j = k
      }
      if max > sumMax {
         sumMax = max
         start, end = i, j
      }
   }
   fmt.Println(start, end)
   return sumMax
}

func findKthLargestB(nums []int, target int) int {
   if target < 1 || target > len(nums) {
      return 0
   }
   l, r, k := 0, len(nums)-1, len(nums)-target
   for l < r {
      p := GetPartitionV(nums, l, r)
      if k == p {
         return nums[p]
      } else if p > k {
         r = p-1
      } else  {
         l = p + 1
      }
   }
   return nums[l]
}

func GetPartitionV(nums []int, l int, r int) int {
   p := nums[l]
   for l < r {
      for l < r && nums[r] >= p {
         r --
      }
      nums[l] = nums[r]
      for l < r && nums[l] <= p {
         l ++
      }
      nums[r] = nums[l]
   }
   nums[l] = p
   return l
}

func bubbleSort21(nums *[]int, n int)  {
   if n < 1 {
      return
   }
   for i:=0;i<n-1;i++ {
      for j:=0;j<n-i-1;j++ {
         if (*nums)[j] > (*nums)[j+1] {
            (*nums)[j], (*nums)[j+1] = (*nums)[j+1], (*nums)[j]
         }
      }
   }
   return
}

func mergeSortB(nums, tmp []int, l, r int)  {
   if l < r {
      mid := l + (r-l)/2
      mergeSortB(nums, tmp, l, mid)
      mergeSortB(nums, tmp, mid+1, r)
      mergeListB(nums, tmp, l, mid, r)
   }
}

func mergeListB(nums []int, tmp []int, l, mid, r int) {
   i, j, t := l, mid+1, l
   for i <= mid && j <= r {
      if nums[i] < nums[j] {
         tmp[t] = nums[i]
         t ++
         i ++
      } else {
         tmp[t] = nums[j]
         t ++
         j ++
      }
   }
   for i<=mid {
      tmp[t] = nums[i]
      t ++
      i ++
   }
   for j<=r {
      tmp[t] = nums[j]
      t ++
      j ++
   }
   for i := 0; i <= r; i++ {
      nums[i] = tmp[i]
   }
}


func quickSortB(nums []int, l, r int) {
   if l >= r {
      return
   }
   start, end, partition := l, r, nums[l]
   for start < end {
      for start < end && nums[end] >= partition {
         end --
      }
      nums[start] =  nums[end]
      for start < end && nums[start] <= partition {
         start ++
      }
      nums[end] = nums[start]
   }
   nums[start] = partition
   quickSortB(nums, l, start)
   quickSortB(nums, start+1, r)
}

/**
https://leetcode-cn.com/problems/search-in-rotated-sorted-array/
81. 搜索旋转排序数组 II
 */
func searchRotationArray(nums []int, target int) int {
   if len(nums) < 1 {
      return -1
   }
   start,  end := 0,len(nums)-1
   for start <= end {
      middle := start + (end - start) / 2
      if nums[middle] == target {
         return middle
      } else if nums[start] == nums[middle] {
         start ++
      } else if nums[middle] <= nums[end] {
         if target > nums[middle] && target <= nums[end] {
            start = middle + 1
         } else {
            end = middle - 1
         }
      } else {
         if target >= nums[start] && target < nums[middle] {
            end = middle - 1
         } else {
            start = middle+1
         }
      }
   }
   return -1
}

type ListNode struct {
   Val  int
   Next *ListNode
}

func sortList(head *ListNode) *ListNode {
   return sort11(head, nil)
}

func sort11(head, tail *ListNode) *ListNode {
   if head == nil {
      return head
   }
   if head.Next == tail {
      head.Next = nil
      return head
   }
   fast, slow := head, head
   for fast != tail {
      fast = fast.Next
      slow = slow.Next
      if fast != tail {
         fast = fast.Next
      }
   }
   middle := slow
   return mergeLinkList(sort11(head, middle), sort11(middle, tail))
}

func mergeLinkList(head1, head2 *ListNode) *ListNode {
   NewHead := &ListNode{}
   h1, h2, dummyHead := head1, head2, NewHead
   for h1 != nil && h2 != nil {
      if h1.Val > h2.Val {
         dummyHead.Next = h2
         h2 = h2.Next
      } else {
         dummyHead.Next = h1
         h1 = h1.Next
      }
      dummyHead = dummyHead.Next
   }
   if h1 != nil {
      dummyHead.Next = h1
   } else if h2 != nil {
      dummyHead.Next = h2
   }
   return NewHead.Next
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

func bubbleSort1(arr []int) []int {
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
