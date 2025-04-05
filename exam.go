package main

import (
	"bytes"
	"fmt"
	"golang.org/x/sync/errgroup"
	"io/ioutil"
	"net/http"
	"sync"
	"time"
)

func main() {
	//ch := NewSimpleChannel(5)
	//
	//go func() {
	//	ch.Send("Hello")
	//	ch.Send("World")
	//	ch.Send("wang")
	//	ch.Send("xin")
	//	ch.Send("guo")
	//}()
	//
	//fmt.Println(ch.Receive()) // Hello
	//fmt.Println(ch.Receive()) // Hello
	//fmt.Println(ch.Receive()) // wang
	//fmt.Println(ch.Receive()) // xin
	//fmt.Println(ch.Receive()) // guo
	//fmt.Println(ch.Receive()) // Hello
	//fmt.Println(ch.Receive()) // World
	//TestSelect1()
	//producer()
	//abc()
}

func reqTest() {
	//url := "http://dbox.dtest.didichuxing.com/food/toolapi/rpc-proxy/call-rpc"
	////postData := []byte(`{"envName":"sim220","serviceName":"order","interfaceName":"GetOrderV3","requestBody":"{\n\t\"traceInfo\": null,\n\t\"commonInfo\": {\n\t\t\"cityID\": 55000002\n\t},\n\t\"businessInfo\": {\n\t\t\"orderID\": 5764640390401557776,\n\t\t\"role\": \"B\",\n\t\t\"bizLine\": \"FOOD\"\n\t},\n\t\"params\": {\n\t\t\"optionals\": {\n\t\t\t\"containOrderBase\": \"CONTAIN\",\n\t\t\t\"containOrderItems\": \"CONTAIN\",\n\t\t\t\"containOrderStatus\": \"CONTAIN\",\n\t\t\t\"containOrderExt\": \"CONTAIN\"\n\t\t}\n\t}\n}","userName":"huru"}`)
	//postData := []byte(`{"envName":"sim220","serviceName":"order","interfaceName":"GetOrderV3","requestBody":"{\n\t\"traceInfo\": null,\n\t\"commonInfo\": {\n\t\t\"cityID\": 55000002\n\t},\n\t\"businessInfo\": {\n\t\t\"orderID\": 5764640390401557776,\n\t\t\"role\": \"B\",\n\t\t\"bizLine\": \"FOOD\"\n\t},\n\t\"params\": {\n\t\t\"optionals\": {\n\t\t\t\"containOrderBase\": \"CONTAIN\",\n\t\t\t\"containOrderItems\": \"CONTAIN\",\n\t\t\t\"containOrderStatus\": \"CONTAIN\",\n\t\t\t\"containOrderExt\": \"CONTAIN\"\n\t\t},\n\"additionalExtRoles\":[\"B\",\"C\"]\n\t}\n}","userName":"huru"}`)
	//req, _ := http.NewRequest("POST", url, bytes.NewBuffer(postData))
	//req.Header.Set("Content-Type", "application/json")
	//req.Header.Set("Cookie", "_OMGID=a4f48421-a857-4363-b22c-4f5bfbb341c5; _kylin_username=huru; _kylin_username_zh=%E8%83%A1%E8%8C%B9; _ga_56G6NHCDPF=GS1.1.1725592742.19.0.1725592742.0.0.0; _gcl_au=1.1.1063979983.1726804940; _ga=GA1.1.10093887.1695713498; busiType=1; engine_sso_username=huru; engine_sso_user=eyJhY2NvdW50IjoiaHVydSIsImF2YXRhciI6IiIsImVtYWlsIjoiaHVydUBkaWRpZ2xvYmFsLmNvbSIsImlkIjozMTI5LCJqb2JOdW1iZXIiOiJEMjk4ODciLCJsZWFkZXJBY2NvdW50IjoiZXRoYW5saXpodW8iLCJsZWFkZXJJZCI6MCwibWVudXMiOiJbXCJkYXNoYm9hcmRcIixcIm9ubGluZVJlY29yZFwiLFwib25saW5lUmVjb3JkMlwiLFwiYmFzaWNDaGVja0luZGV4XCIsXCJvbmxpbmVOb3RpZnlJbmRleFwiLFwid2Vla09uZHV0eVwiLFwid2Vla09uZHV0eUluZGV4XCIsXCJiaXpTeXN0ZW1BbGFybVwiLFwiaG9saWRheVNjaGVkdWxpbmdcIixcImJpelN5c3RlbVwiLFwiYml6U3lzdGVtSW5kZXhcIixcIm1vdWRsZU1haW50YWluZXJJbmRleFwiLFwiZnhsaFwiLFwiYWxhcm1SZXNwb25zZVwiLFwiYXJteVJ1bGVzXCIsXCJkZXBsb3lQZWFrQ2hlY2tcIixcImNoZWNrTGlzdFN0YXRpc3RpY1wiLFwicG9ydGFsXCIsXCJ1c2VyXCIsXCJ1c2VySW5kZXhcIixcInRlYW1cIixcInRlbmFudFwiLFwic3RyZXNzVGVzdFwiLFwic3RyZXNzVGVzdEluZGV4XCIsXCJ1c2VyTG9nXCIsXCJyb2xlXCIsXCJyb2xlXCIsXCJ1c2VyTG9nXCIsXCJwdnV2XCIsXCJtb25pdG9yXCIsXCJwbGFuTWFya2V0SW5kZXhcIixcInRhc2tNYW5hZ2VyXCIsXCJ0YXNrTWFuYWdlckluZGV4XCIsXCJ3ZWVrbHlcIixcInNlcnZpY2VNYW5hZ2VtZW50XCJdIiwibmlja05hbWUiOiJIdSBSdSIsInJvbGVOYW1lIjoiZGVmYXVsdFJvbGUiLCJ0ZWFtSWQiOjAsInRlYW1OYW1lIjoiRmluYW5jZSBcdTAwMjYgT3JkZXIiLCJ0ZW5hbnROYW1lIjoiVHJhbnNhY3Rpb24gUGxhdGZvcm0gVGVjaCJ9; _ga_P8RN8E8C91=GS1.1.1728832966.13.0.1728832966.0.0.0; _clck=1x5abgo%7C2%7Cfpz%7C0%7C1610; cto_bundle=mZgmel9JZld4V3ZCWHFRRllWU1BCY29HNk1hdiUyQko1VklXYW4yRHdDSGtObFczUFd3dzJod1piNWNTNDJWYmoyeklNT1lISW8xZGh5cnQ1dmlUQWVqNFFuJTJGNXRnUnhPQUZGU21EcGhqcWlNcUNOTnJtU3AwaVdWY0U0Z254VXdvQ0hUQ0pya3diU3pZcEx2cU9WJTJGeFBoYU9IWURHNGEwcHJxQzFRWjdyQzB4QmRuSU9KbkpERmI3ZnRoZ21TY0p1bmhiSmVidnY1QnNZTXY5MTAxWm5wSmc0ZE9BJTNEJTNE; _ga_VVCWHC0G6L=GS1.1.1728832966.16.1.1728832977.0.0.0; engine_sso_ticket=409b0b54867e74e6c9c937f9c727e17a0002101090000; _kylin_ticket=a8a86da138377f9020f2733dc0b91fb80002319000")

	eg := errgroup.Group{}
	for i := 0; i < 5; i++ {
		eg.Go(func() error {
			url := "http://dbox.dtest.didichuxing.com/food/toolapi/rpc-proxy/call-rpc"
			//postData := []byte(`{"envName":"sim220","serviceName":"order","interfaceName":"GetOrderV3","requestBody":"{\n\t\"traceInfo\": null,\n\t\"commonInfo\": {\n\t\t\"cityID\": 55000002\n\t},\n\t\"businessInfo\": {\n\t\t\"orderID\": 5764640390401557776,\n\t\t\"role\": \"B\",\n\t\t\"bizLine\": \"FOOD\"\n\t},\n\t\"params\": {\n\t\t\"optionals\": {\n\t\t\t\"containOrderBase\": \"CONTAIN\",\n\t\t\t\"containOrderItems\": \"CONTAIN\",\n\t\t\t\"containOrderStatus\": \"CONTAIN\",\n\t\t\t\"containOrderExt\": \"CONTAIN\"\n\t\t}\n\t}\n}","userName":"huru"}`)
			postData := []byte(`{"envName":"sim220","serviceName":"order","interfaceName":"GetOrderV3","requestBody":"{\n\t\"traceInfo\": null,\n\t\"commonInfo\": {\n\t\t\"cityID\": 55000002\n\t},\n\t\"businessInfo\": {\n\t\t\"orderID\": 5764640390401557776,\n\t\t\"role\": \"B\",\n\t\t\"bizLine\": \"FOOD\"\n\t},\n\t\"params\": {\n\t\t\"optionals\": {\n\t\t\t\"containOrderBase\": \"CONTAIN\",\n\t\t\t\"containOrderItems\": \"CONTAIN\",\n\t\t\t\"containOrderStatus\": \"CONTAIN\",\n\t\t\t\"containOrderExt\": \"CONTAIN\"\n\t\t},\n\"additionalExtRoles\":[\"B\",\"C\"]\n\t}\n}","userName":"huru"}`)
			req, _ := http.NewRequest("POST", url, bytes.NewBuffer(postData))
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("Cookie", "_OMGID=a4f48421-a857-4363-b22c-4f5bfbb341c5; _kylin_username=huru; _kylin_username_zh=%E8%83%A1%E8%8C%B9; _ga_56G6NHCDPF=GS1.1.1725592742.19.0.1725592742.0.0.0; _gcl_au=1.1.1063979983.1726804940; _ga=GA1.1.10093887.1695713498; busiType=1; engine_sso_username=huru; engine_sso_user=eyJhY2NvdW50IjoiaHVydSIsImF2YXRhciI6IiIsImVtYWlsIjoiaHVydUBkaWRpZ2xvYmFsLmNvbSIsImlkIjozMTI5LCJqb2JOdW1iZXIiOiJEMjk4ODciLCJsZWFkZXJBY2NvdW50IjoiZXRoYW5saXpodW8iLCJsZWFkZXJJZCI6MCwibWVudXMiOiJbXCJkYXNoYm9hcmRcIixcIm9ubGluZVJlY29yZFwiLFwib25saW5lUmVjb3JkMlwiLFwiYmFzaWNDaGVja0luZGV4XCIsXCJvbmxpbmVOb3RpZnlJbmRleFwiLFwid2Vla09uZHV0eVwiLFwid2Vla09uZHV0eUluZGV4XCIsXCJiaXpTeXN0ZW1BbGFybVwiLFwiaG9saWRheVNjaGVkdWxpbmdcIixcImJpelN5c3RlbVwiLFwiYml6U3lzdGVtSW5kZXhcIixcIm1vdWRsZU1haW50YWluZXJJbmRleFwiLFwiZnhsaFwiLFwiYWxhcm1SZXNwb25zZVwiLFwiYXJteVJ1bGVzXCIsXCJkZXBsb3lQZWFrQ2hlY2tcIixcImNoZWNrTGlzdFN0YXRpc3RpY1wiLFwicG9ydGFsXCIsXCJ1c2VyXCIsXCJ1c2VySW5kZXhcIixcInRlYW1cIixcInRlbmFudFwiLFwic3RyZXNzVGVzdFwiLFwic3RyZXNzVGVzdEluZGV4XCIsXCJ1c2VyTG9nXCIsXCJyb2xlXCIsXCJyb2xlXCIsXCJ1c2VyTG9nXCIsXCJwdnV2XCIsXCJtb25pdG9yXCIsXCJwbGFuTWFya2V0SW5kZXhcIixcInRhc2tNYW5hZ2VyXCIsXCJ0YXNrTWFuYWdlckluZGV4XCIsXCJ3ZWVrbHlcIixcInNlcnZpY2VNYW5hZ2VtZW50XCJdIiwibmlja05hbWUiOiJIdSBSdSIsInJvbGVOYW1lIjoiZGVmYXVsdFJvbGUiLCJ0ZWFtSWQiOjAsInRlYW1OYW1lIjoiRmluYW5jZSBcdTAwMjYgT3JkZXIiLCJ0ZW5hbnROYW1lIjoiVHJhbnNhY3Rpb24gUGxhdGZvcm0gVGVjaCJ9; _ga_P8RN8E8C91=GS1.1.1728832966.13.0.1728832966.0.0.0; _clck=1x5abgo%7C2%7Cfpz%7C0%7C1610; cto_bundle=mZgmel9JZld4V3ZCWHFRRllWU1BCY29HNk1hdiUyQko1VklXYW4yRHdDSGtObFczUFd3dzJod1piNWNTNDJWYmoyeklNT1lISW8xZGh5cnQ1dmlUQWVqNFFuJTJGNXRnUnhPQUZGU21EcGhqcWlNcUNOTnJtU3AwaVdWY0U0Z254VXdvQ0hUQ0pya3diU3pZcEx2cU9WJTJGeFBoYU9IWURHNGEwcHJxQzFRWjdyQzB4QmRuSU9KbkpERmI3ZnRoZ21TY0p1bmhiSmVidnY1QnNZTXY5MTAxWm5wSmc0ZE9BJTNEJTNE; _ga_VVCWHC0G6L=GS1.1.1728832966.16.1.1728832977.0.0.0; engine_sso_ticket=409b0b54867e74e6c9c937f9c727e17a0002101090000; _kylin_ticket=a8a86da138377f9020f2733dc0b91fb80002319000")

			client := &http.Client{}
			resp, err := client.Do(req)
			defer func(resp *http.Response) {
				if resp == nil {
					return
				}
				err := resp.Body.Close()
				if err != nil {
					fmt.Printf("client.Do have err %v", err)
				}
			}(resp)
			if err != nil {
				fmt.Printf("err is %v\n", err)
				return err
			}
			body, err := ioutil.ReadAll(resp.Body)
			if err != nil {
				fmt.Printf("ioutil ReadAll err %v\n", err)
				return err
			}
			fmt.Println(string(body))
			return nil
		})
	}
	if err := eg.Wait(); err != nil {
		fmt.Printf("eg wait have err %v", err)
	}
	return
}

func abc() {
	c, flag1, flag2, flag3 := make(chan string), make(chan int), make(chan int), make(chan int)
	go func() {
		for i := 0; i < 10; i++ {
			<-flag1
			c <- "A"
			flag2 <- 1
		}
		close(flag2)
	}()
	go func() {
		for i := 0; i < 10; i++ {
			<-flag2
			c <- "B"
			flag3 <- 1
		}
		close(flag3)
	}()
	go func() {
		for i := 0; i < 10; i++ {
			<-flag3
			c <- "C"
			if i == 9 {
				close(flag1)
			} else {
				flag1 <- 1
			}
		}
		close(c)
	}()
	flag1 <- 1
	//for v := range c {
	//	fmt.Println(v)
	//}
	for {
		v, ok := <-c
		if !ok {
			break
		}
		fmt.Println(v)
	}
}

func consumer(data, end chan int) {
	for {
		select {
		case n := <-data:
			fmt.Println(n)
		case n := <-end:
			fmt.Printf("value is %v, end\n", n)
			return
		}
	}
}

func producer() {
	data, end := make(chan int), make(chan int)
	go consumer(data, end)
	for i := 0; i < 100; i++ {
		data <- i
	}
	close(data)
	time.Sleep(1 * time.Millisecond)
	end <- 0
	close(end)
	return
}

func TestSelect1() {
	start := time.Now()
	c := make(chan interface{})
	c1 := make(chan interface{})

	go func() {
		for i := 0; i < 10; i++ {
			c <- "wang_" + string(i)
			time.Sleep(300 * time.Millisecond)
		}
	}()
	go func() {
		time.Sleep(1000 * time.Millisecond)
		close(c1)
	}()

	fmt.Println("Blocking on read...")
	tick := time.Tick(300 * time.Millisecond)
	for {
		select {
		case v := <-tick:
			fmt.Printf("v is %v tick in.\n", v)
		case v := <-c:
			fmt.Printf("v is %v Unblocked %v later.\n", v, time.Since(start))
		case v := <-c1:
			fmt.Printf("v is %v; will break loop.\n", v)
			goto end
		}
	}
end:
	fmt.Println("break and go end\n")
}

type SimpleChannel struct {
	mu     sync.Mutex
	cond   *sync.Cond
	buffer []interface{}
	cap    int
}

// NewSimpleChannel 创建一个新的 SimpleChannel
func NewSimpleChannel(capacity int) *SimpleChannel {
	ch := &SimpleChannel{
		cap:    capacity,
		buffer: make([]interface{}, 0, capacity),
		mu:     sync.Mutex{},
	}
	ch.cond = sync.NewCond(&ch.mu)
	return ch
}

// Send 发送数据到通道
func (ch *SimpleChannel) Send(value interface{}) {
	ch.cond.L.Lock()
	defer ch.cond.L.Unlock()

	for len(ch.buffer) >= ch.cap {
		ch.cond.Wait() // 等待缓冲区有空间
	}

	ch.buffer = append(ch.buffer, value)
	ch.cond.Signal() // 唤醒等待接收的 goroutine
}

// Receive 从通道接收数据
func (ch *SimpleChannel) Receive() interface{} {
	ch.cond.L.Lock()
	defer ch.cond.L.Unlock()

	for len(ch.buffer) == 0 {
		ch.cond.Wait() // 等待数据可用
	}

	value := ch.buffer[0]
	ch.buffer = ch.buffer[1:]
	// ch.cond.Signal() // 唤醒等待发送的 goroutine
	ch.cond.Broadcast()
	return value
}

type Pool struct {
	queue chan int
	wg    *sync.WaitGroup
}

func NewPool(size int) *Pool {
	if size <= 0 {
		size = 1
	}

	return &Pool{
		queue: make(chan int, size),
		wg:    &sync.WaitGroup{},
	}
}

func (p *Pool) Add(value int) {
	p.queue <- value
	p.wg.Add(value)
}

func (p *Pool) Done() {
	<-p.queue
	p.wg.Done()
}

func (p *Pool) Wait() {
	p.wg.Wait()
}