package main

import (
	"fmt"
	"golang.org/x/sync/errgroup"
	"runtime"
	"sync"
	"unsafe"
)

func main() {
	//runtime.GOMAXPROCS(runtime.NumCPU() - 1)
	//TestSyncMap()
	//lb, receiveCh, sendCh := NewLogBuild(), make(chan struct{}), make(chan struct{})
	//go readG(lb, receiveCh, sendCh)
	//writeG(lb, receiveCh, sendCh)

	//bb := Load()
	var aa interface{}

	cc, err := aa.(readOnly)
	fmt.Printf("cc=%v, err=%v\n", cc, err)

	key := "kkkkkkk"
	e, ok := cc.m[key]
	fmt.Printf("e=%v, ok=%v\n", e, ok)

	var m map[string]interface{} = nil
	dd, oook := m["vvv"]
	fmt.Printf("dd=%v, oook=%v\n", dd, oook)
}

type readOnly struct {
	m       map[interface{}]*entry
	amended bool // true if the dirty map contains some key not in m.
}

type entry struct {
	p unsafe.Pointer // *interface{}
}

func Load() (val interface{}) {
	return nil
}

func writeG(lb *extLog, receiveCh chan struct{}, sendCh chan struct{}) string {
	res := "logBuild"
	receiveCh <- struct{}{}
	<-sendCh
	for _, key := range lb.sortKeys {
		if v, ok := lb.extInfo.Load(key); ok {
			res += fmt.Sprintf("||%v=%+v", key, v)
		}
	}
	fmt.Println(res)
	return res
}

func readG(lb *extLog, receiveCh <-chan struct{}, sendCh chan<- struct{}) {
	<-receiveCh
	eg := errgroup.Group{}
	for i := 0; i < 100; i++ {
		if i == 7 {
			sendCh <- struct{}{}
		}
		t := i
		eg.Go(func() error {
			defer RecoverPanic()
			lb.Set(fmt.Sprintf("%s%d", "k_", t), fmt.Sprintf("%s%d", "v_", t)).
				Set(fmt.Sprintf("%s%d", "kk_", t), fmt.Sprintf("%s%d", "vv_", t))
			return nil
		})
	}
	_ = eg.Wait()
}

func RecoverPanic() {
	if err := recover(); err != nil {
		buf := make([]byte, 1<<16)
		stackSize := runtime.Stack(buf, false)
		fmt.Println(err, string(buf[0:stackSize]))
	}
	return
}

func TestSyncMap() {
	eg := errgroup.Group{}
	l := NewLogBuild()
	l.Set("111", "111").Set("222", "222").GetTag("TestSyncMap_in")
	defer func() {
		l.Set("err", nil).GetTag("not have err")
		fmt.Println("end")

	}()
	for i := 0; i < runtime.NumCPU()-1; i++ {
		eg.Go(func() error {
			defer RecoverPanic()
			for {
				_ = l.Set("w", "w").Set("x", "x").GetTag("errgroup_do")
			}
			return nil
		})

	}
	_ = eg.Wait()
	fmt.Println(1122)
	return
}

type extLog struct {
	extInfo  sync.Map
	sortKeys []string
}

func NewLogBuild() (res *extLog) {
	res = &extLog{}
	res.extInfo = sync.Map{}
	return res
}

// 日志信息
func (p *extLog) Get() (res string) {
	res = "logBuild"
	var a *string
	for _, key := range p.sortKeys {
		if v, ok := p.extInfo.Load(a); ok {
			res += fmt.Sprintf("||%v=%+v", key, v)
		}
	}
	//fmt.Println(res)
	return
}

func (p *extLog) GetTag(dTag string) string {
	return p.Get()
}

func (p *extLog) Set(key string, value interface{}) *extLog {
	p.extInfo.Store(key, value)
	p.sortKeys = append(p.sortKeys, key)
	//fmt.Printf("sortKeys len is: %d, cap is: %d, addr is %p, val is %p\n", len(p.sortKeys), cap(p.sortKeys), &p.sortKeys, p.sortKeys)
	return p
}
