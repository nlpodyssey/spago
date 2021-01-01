// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package processingqueue

// ProcessingQueue is a simple utility type for limiting the number of goroutines in execution.
// This is convenient especially in the context of heavy concurrent computations.
//
// ProcessingQueue is implemented as a queue of slots, where each slot represents a busy concurrent job.
// The queue is initialized with a size of your choice (see New) and is initially empty.
// Each time Run is called, the function first waits for one job-slot to be available; then it marks
// one free slot as busy and simply calls the given callback. Once the callback is fully executed (even
// in case of panics), it frees a busy slot, making it available for further usages.
//
// Run is suitable for being called from a goroutine. Go simply calls Run directly as a goroutine.
type ProcessingQueue chan emptyStruct

type emptyStruct struct{}

// New returns a new ProcessingQueue initialized with the given size.
// It panics if size is lower than 1.
func New(size int) ProcessingQueue {
	if size < 1 {
		panic("processingqueue: ProcessingQueue size must be greater than zero")
	}
	return make(ProcessingQueue, size)
}

// Run waits for a free job-slot to be available in the queue, than marks one slot as busy
// and calls f, eventually releasing the slot.
func (pq ProcessingQueue) Run(f func()) {
	pq <- emptyStruct{}
	defer func() { <-pq }()
	f()
}

// Go simply executes Run as a goroutine.
func (pq ProcessingQueue) Go(f func()) {
	go pq.Run(f)
}

// Size returns the size of the ProcessingQueue.
func (pq ProcessingQueue) Size() int {
	return cap(pq)
}
