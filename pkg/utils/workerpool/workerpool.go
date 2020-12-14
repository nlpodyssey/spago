// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package workerpool

import (
	"context"
	"os"
	"os/signal"
	"sync"
	"syscall"
)

type WorkerPool struct {
	size       int
	ingestChan chan interface{}
	jobsChan   chan interface{}
}

type WorkerFunc func(workerID int, jobData interface{})

// New returns a new WorkerPool ready-to-use.
func New(size int) *WorkerPool {
	return &WorkerPool{
		size:       size,
		ingestChan: make(chan interface{}, 1),
		jobsChan:   make(chan interface{}, size),
	}
}

func (wp *WorkerPool) Run(workerFunc WorkerFunc) {
	ctx, ctxCancelFunc := context.WithCancel(context.Background())

	wg := &sync.WaitGroup{}
	wg.Add(wp.size)

	go wp.runConsumer(ctx)

	for workerID := 0; workerID < wp.size; workerID++ {
		go wp.runWorker(workerID, wg, workerFunc)
	}

	wp.blockUntilSignal()

	ctxCancelFunc()
	wg.Wait()
}

func (wp *WorkerPool) PublishJobData(jobData interface{}) {
	wp.ingestChan <- jobData
}

func (wp *WorkerPool) runWorker(workerID int, wg *sync.WaitGroup, wFunc WorkerFunc) {
	defer wg.Done()
	for jobData := range wp.jobsChan {
		wFunc(workerID, jobData)
	}
}

func (wp *WorkerPool) runConsumer(ctx context.Context) {
	for {
		select {
		case jobData := <-wp.ingestChan:
			wp.jobsChan <- jobData
		case <-ctx.Done():
			close(wp.jobsChan)
			return
		}
	}
}

func (wp *WorkerPool) blockUntilSignal() {
	termChan := make(chan os.Signal)
	signal.Notify(termChan, syscall.SIGINT, syscall.SIGTERM)
	<-termChan
}
