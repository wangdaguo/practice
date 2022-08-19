package main

import "fmt"

func main()  {
	//s := []string{"a", "b", "c", "d", "e", "f", "g", "h"}
	//reverseStr1(s, 3)
	//fmt.Println(s)
	//return

	r := GenCalendar(2022, 8)
	fmt.Println(r)
	return
}

type Date struct {
	Year int
	Month int
	Day int
}

var monthDay = []int{0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}

func IsRunYear(year int) bool {
	return (year % 4 ==0 && year % 100 != 0) || year % 400 == 0
}

func YearDay(date *Date) int {
	totalDay := 0
	for i:=1; i<date.Month; i++ {
		totalDay += monthDay[i]
	}
	if date.Month > 2 && IsRunYear(date.Year) {
		totalDay += 1
	}
	return totalDay
}

func BetweenDay(date1, date2 *Date) int {
	if date1.Year == date2.Year {
		return YearDay(date2) - YearDay(date1)
	}
	cnt := 0
	for i:=date1.Year; i<date2.Year; i++ {
		cnt += 365
		if IsRunYear(i) {
			cnt += 1
		}
	}
	return cnt + YearDay(date2) - YearDay(date1)
}

func WeekDay(date *Date) int {
	startDate := &Date{
		Year:  1900,
		Month: 1,
		Day:   1,
	}
	return BetweenDay(startDate, date) % 7 + 1
}

func GenCalendar(year, month int) [][]int {
	var r [][]int
	startDate := &Date{
		Year:  year,
		Month: month,
		Day:   1,
	}
	startDayWeekend := WeekDay(startDate)
	lastDay := monthDay[month]
	if IsRunYear(year) && month == 2 {
		lastDay = 29
	}
	for i:=startDate.Day; i<=lastDay; i ++ {
		tmpWeekDay := startDayWeekend
		tmpArr := []int{}
		for j:=1; j<=7; j++ {
			if j < tmpWeekDay {
				tmpArr = append(tmpArr, 0)
				continue
			}
			tmpArr = append(tmpArr, tmpWeekDay)
			tmpWeekDay ++
		}
		tmpWeekDay = 1
		r = append(r, tmpArr)
	}
	return r
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

