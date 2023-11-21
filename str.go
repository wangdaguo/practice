package main

func main() {

}

/*
*
6. N 字形变换
https://leetcode.cn/problems/zigzag-conversion/description/
*/
func convert(s string, numRows int) string {
	if numRows == 1 {
		return s
	}
	curRow, goDown, list := 0, false, make([]string, min(numRows, len(s)))
	for _, v := range s {
		list[curRow] += string(v)
		if curRow == 0 || curRow == min(numRows, len(s))-1 {
			goDown = !goDown
		}
		if goDown {
			curRow++
		} else {
			curRow--
		}
	}
	var r string
	for _, v := range list {
		r += v
	}
	return r
}

/*
28. 找出字符串中第一个匹配项的下标
https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/?
*/
//func strStr(haystack string, needle string) int {
//	if len(needle) > len(haystack) {
//		return -1
//	}
//	if len(needle) == len(haystack) && haystack != needle {
//		return -1
//	}
//	idx, i, j := -1, 0, 0
//	for i < len(haystack) {
//	}
//
//	//for k := 0; k < len(haystack); k++ {
//	//	if haystack[k] == needle[j] {
//	//		j++
//	//		if j == len(needle)-1 {
//	//			return idx
//	//		}
//	//		if idx == -1 {
//	//			idx = k
//	//			i = k
//	//		}
//	//		continue
//	//	} else {
//	//		j = 0
//	//		i++
//	//		idx = -1
//	//	}
//	//}
//	//return idx
//}
