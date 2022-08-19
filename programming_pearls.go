package main

import "fmt"

func main()  {
	//s := []string{"a", "b", "c", "d", "e", "f", "g", "h"}
	//reverseStr1(s, 3)
	//fmt.Println(s)
	//return

	date1 := &Date{
		Year:  2012,
		Month: 1,
		Day:   10,
	}

	date2 := &Date{
		Year:  2013,
		Month: 1,
		Day:   10,
	}

	cnt := BetweenDay(date1, date2)
	fmt.Println(cnt)

	date3 := &Date{
		Year:  2022,
		Month: 8,
		Day:   19,
	}
	r := WeekDay(date3)
	fmt.Println(r)

	rr := GenCalendar(2022, 8)
	fmt.Println(rr)
	return
}

var monthDay = []int{0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}

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

type Date struct {
	Year int
	Month int
	Day int
}

func IsRunYear(year int) bool {
	if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 {
		return true
	}
	return false
}

func YearDay(date *Date) (day int) {
	arr := []int{0,31,28,31,30,31,30,31,31,30,31,30,31}
	for i:=1; i<date.Month; i++ {
		day += arr[i]
	}
	if date.Month > 2 && IsRunYear(date.Year) {
		day += 1
	}
	day += date.Day
	return day
}

func BetweenDay(date1, date2 *Date) int {
	if date1.Year == date2.Year {
		return YearDay(date2) - YearDay(date1)
	}
	var totalDay int
	for i:=date1.Year; i<date2.Year; i++ {
		totalDay += 365
		if IsRunYear(i) {
			totalDay += 1
		}
	}
	return totalDay + YearDay(date2) - YearDay(date1)
}

//选取1900年作为参考，日历都是以1900为参考的，1900.1.1为星期一，其它日期相对它的天数与7取余
// 数再加上1就是对应的星期几。这里用1~7表示星期一到星期天
func WeekDay(date *Date) int {
	startDate := &Date{
		Year:  1900,
		Month: 1,
		Day:   1,
	}
	return BetweenDay(startDate, date) % 7 + 1
}


