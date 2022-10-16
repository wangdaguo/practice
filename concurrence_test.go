package main

import (
	"context"
	"fmt"
	"golang.org/x/sync/errgroup"
	"golang.org/x/sync/semaphore"
	"time"
)

func main()  {
	abc()
	//ccc()
	return
}


func abc() {

	//logg := log.New(os.Stdout, "", log.Ltime)
	// 设置一个上下文，并设置对应的超时时间
	//ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(3*time.Millisecond))
	ctx, cancel := context.WithCancel(context.Background())
	defer func() {
		time.Sleep(time.Millisecond*10)
		cancel()
		time.Sleep(time.Millisecond*100)
	}()

	sem := semaphore.NewWeighted(4)
	eg, ctx := errgroup.WithContext(ctx)

	for i:=0; i<5; i++ {
		err := sem.Acquire(ctx, 1)
		if err != nil {
			fmt.Printf("sem.Acquire called, err is %+v", ctx.Err())
			return
		}
		cnt := i
		eg.Go(func() (gerr error) {
			defer sem.Release(1)
			select {
			case <-ctx.Done():
				fmt.Printf("ctx.Done() called, err is %+v", ctx.Err())
				return
			case <-time.After(1*time.Millisecond):
				fmt.Printf("%v time.After() called\n", cnt)
			}
			time.Sleep(time.Second*2)
			return
		})
	}
	if err:=eg.Wait(); err!=nil {
		fmt.Printf("eg.Wait() have err, err is %+v", ctx.Err())
		return
	}
	fmt.Printf("func() exec end!\n")

	//go func() {
	//	for {
	//		select {
	//		case <-ctx.Done():
	//			//logg.Printf("son go is end !")
	//			//fmt.Println("ctx.Done() called !")
	//			fmt.Printf("ctx.Done() called, err is %+v", ctx.Err())
	//			return
	//		case <-time.After(4*time.Millisecond):
	//			fmt.Printf("time.After() called\n")
	//		//default:
	//		//	fmt.Println("go default !")
	//		}
	//	}
	//}()
	//time.Sleep(time.Millisecond*5)
}

func run(ctx context.Context, id int) {
	for {
		select {
		case <-ctx.Done():
			fmt.Printf("任务%v结束退出\n", id)
			return
		default:
			fmt.Printf("任务%v正在运行中\n", id)
			time.Sleep(time.Second * 1)
		}
	}
}

func ccc() {
	//管理启动的协程
	ctx, cancel := context.WithCancel(context.Background())

	// 开启多个goroutine，传入ctx
	go run(ctx, 1)
	go run(ctx, 2)

	// 运行一段时间后停止
	time.Sleep(time.Second * 3)
	fmt.Println("停止任务1")
	cancel() // 使用context的cancel函数停止goroutine

	// 为了检测监控过是否停止，如果没有监控输出，表示停止
	time.Sleep(time.Second * 1)
	return
}
