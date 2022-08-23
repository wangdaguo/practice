package main

import (
	"fmt"
	"strconv"
)

func main()  {

	//r := MaxZero([]int{25, 30, 125, 64})
	//fmt.Println(r)
	//return

	/**
	4
	13
	123
	24
	22
	 */

	//r := GetOtherEven("44")
	//fmt.Println(r)
	//return


}

func GetOtherEven(s string) string {
	input := s
	var firstPosVal, lastPosVal int64
	var middleValArr []int64
	lastPosVal, _ = strconv.ParseInt(string(input[len(input)-1]), 10, 64)
	firstPosVal, _ = strconv.ParseInt(string(input[0]), 10, 64)
	for i:=1;i<len(input)-1; i++ {
		c, _ := strconv.ParseInt(string(input[i]), 10, 64)
		middleValArr = append(middleValArr, c)
	}
	if middleValArr == nil {
		if firstPosVal == lastPosVal || lastPosVal == 0 || firstPosVal == 0 {
			return "-1"
		}
		if (lastPosVal*10 + firstPosVal) % 2 == 0 {
			return genValByRev(lastPosVal, firstPosVal, middleValArr)
		}
		return "-1"
	}
	// 如果末尾不为偶数
	if lastPosVal % 2 != 0 {
		if firstPosVal % 2 == 0 {
			return genValByRev(lastPosVal, firstPosVal, middleValArr)
		}
		if getEvenIndex(middleValArr, -1) >= 0 {
			middleValArr[getEvenIndex(middleValArr, -1)], lastPosVal = lastPosVal, middleValArr[getEvenIndex(middleValArr, -1)]
			return genValByRev(firstPosVal, lastPosVal, middleValArr)
		}
		return "-1"
	} else {  // 尾号是偶数  22332
		// 中间能换，则中间换
		i, j := getMiddleValNotEqualTwoIndex(middleValArr)
		if i >= 0 {
			middleValArr[i], middleValArr[j] = middleValArr[j], middleValArr[i]
			return genValBySeq(firstPosVal, lastPosVal, middleValArr)
		} else {
			// 中间跟首位换
			if firstPosVal != middleValArr[0] && middleValArr[0] != 0 {
				middleValArr[0], firstPosVal = firstPosVal, middleValArr[0]
				return genValByRev(firstPosVal, lastPosVal, middleValArr)
			}
			// 首位为偶数，首位与末尾换
			if firstPosVal % 2 == 0 && firstPosVal != lastPosVal && lastPosVal != 0 {
				return genValByRev(lastPosVal, firstPosVal, middleValArr)
			}
			if getEvenIndex(middleValArr, lastPosVal) >= 0 {
				middleValArr[getEvenIndex(middleValArr, lastPosVal)], lastPosVal = lastPosVal, middleValArr[getEvenIndex(middleValArr, lastPosVal)]
				return genValByRev(firstPosVal, lastPosVal, middleValArr)
			}
			return "-1"
		}
	}
}

func genValByRev(firstPos, lastPos int64, middleVal []int64) string {
	var r string
	r = fmt.Sprintf("%s%s", r, strconv.FormatInt(firstPos, 10))
	if len(middleVal) > 0 {
		for i:=len(middleVal)-1;i>=0;i-- {
			r = fmt.Sprintf("%s%s", r, strconv.FormatInt(middleVal[i], 10))
		}
	}
	r = fmt.Sprintf("%s%s", r, strconv.FormatInt(lastPos, 10))
	return r
}

func genValBySeq(firstPos, lastPos int64, middleVal []int64) string {
	var r string
	r = fmt.Sprintf("%s%s", r, strconv.FormatInt(firstPos, 10))
	if len(middleVal) > 0 {
		for i:=0;i<=len(middleVal)-1;i++ {
			r = fmt.Sprintf("%s%s", r, strconv.FormatInt(middleVal[i], 10))
		}
	}
	r = fmt.Sprintf("%s%s", r, strconv.FormatInt(lastPos, 10))
	return r
}

func getEvenIndex(middleVal []int64, exclude int64) int64 {
	for k, v := range middleVal {
		if v % 2 == 0 && v != exclude {
			return int64(k)
		}
	}
	return -1
}

// 2 323 2
func getMiddleValNotEqualTwoIndex(middleVal []int64) (int64, int64) {
	if len(middleVal) < 2 {
		return -1, -1
	}
	i, j := int64(0), int64(len(middleVal)-1)
	for i < j {
		if middleVal[i] != middleVal[j] {
			return i, j
		}
		j --
	}
	return -1, -1
}

func MaxZero(arr []int) int {
	if len(arr) < 1 {
		return 0
	}
	if len(arr) == 1 {
		return getZeroCnt(arr[0])
	}
	if len(arr) == 2 {
		return maxVal(getZeroCnt(arr[0]), getZeroCnt(arr[1]), getZeroCnt(arr[0]*arr[1]))
	}
	return maxVal(MaxZero(arr[0:len(arr)-2])*arr[len(arr)-1], MaxZero(arr[0:len(arr)-1]))
}

func getZeroCnt(r int) int {
	cnt := 0
	for r > 0 {
		if r > 0 && r % 10 == 0 {
			cnt ++
		}
		r = r/10

	}
	return cnt
}

func maxVal(args... int) int {
	var max = -1
	for _, arg := range args {
		if arg > max {
			max = arg
		}
	}
	return max
}
