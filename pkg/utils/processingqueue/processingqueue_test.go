// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package processingqueue

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestNew(t *testing.T) {
	t.Run("it panics if size is lower than 1", func(t *testing.T) {
		assert.Panics(t, func() { New(0) })
		assert.Panics(t, func() { New(-1) })
	})
}

func TestProcessingQueue_Run(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode")
	}

	workerSleepingDuration := 300 * time.Millisecond

	var mutex sync.Mutex // avoid data races in this test
	ej := make(executedJobs, 0, 4)
	pq := New(3)

	for i := 0; i < 4; i++ {
		// Be sure to run the goroutines sequentially
		time.Sleep(workerSleepingDuration / 50)

		go func(index int) {
			pq.Run(func() {
				mutex.Lock()
				item := &executedJob{
					index:     index,
					completed: false,
				}
				ej = append(ej, item)
				mutex.Unlock()

				time.Sleep(workerSleepingDuration)

				mutex.Lock()
				item.completed = true
				mutex.Unlock()
			})
		}(i)
	}

	// After waiting half of a job's duration, we should
	// be in the middle of the processing of the first
	// 3 job data items: 0, 1, and 2.
	time.Sleep(workerSleepingDuration / 2)

	mutex.Lock()

	assert.Equal(t, 3, len(ej))

	ej.AssertIncludesJobIndex(t, 0)
	ej.AssertIncludesJobIndex(t, 1)
	ej.AssertIncludesJobIndex(t, 2)

	ej.AssertCompletedCount(t, 0)

	mutex.Unlock()

	// Wait for the first 3 jobs to be completed, and
	// the 4th to be started.
	time.Sleep(workerSleepingDuration)

	mutex.Lock()

	assert.Equal(t, 4, len(ej))

	ej.AssertIncludesJobIndex(t, 3)
	ej.AssertCompletedCount(t, 3)

	mutex.Unlock()

	// Wait again, and be sure all jobs completed.
	time.Sleep(workerSleepingDuration)

	mutex.Lock()
	ej.AssertCompletedCount(t, 4)
	mutex.Unlock()
}

func TestProcessingQueue_Go(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode")
	}

	workerSleepingDuration := 300 * time.Millisecond

	var mutex sync.Mutex // avoid data races in this test
	var wg sync.WaitGroup
	ej := make(executedJobs, 0, 4)
	pq := New(3)

	for i := 0; i < 4; i++ {
		wg.Add(1)
		// Be sure to run the goroutines sequentially
		time.Sleep(workerSleepingDuration / 50)

		index := i
		pq.Go(func() {
			defer wg.Done()
			mutex.Lock()
			item := &executedJob{
				index:     index,
				completed: false,
			}
			ej = append(ej, item)
			mutex.Unlock()

			time.Sleep(workerSleepingDuration)

			mutex.Lock()
			item.completed = true
			mutex.Unlock()
		})
	}

	// After waiting half of a job's duration, we should
	// be in the middle of the processing of the first
	// 3 job data items: 0, 1, and 2.
	time.Sleep(workerSleepingDuration / 2)

	mutex.Lock()

	assert.Equal(t, 3, len(ej))

	ej.AssertIncludesJobIndex(t, 0)
	ej.AssertIncludesJobIndex(t, 1)
	ej.AssertIncludesJobIndex(t, 2)

	ej.AssertCompletedCount(t, 0)

	mutex.Unlock()

	// Wait for the first 3 jobs to be completed, and
	// the 4th to be started.
	time.Sleep(workerSleepingDuration)

	mutex.Lock()

	assert.Equal(t, 4, len(ej))

	ej.AssertIncludesJobIndex(t, 3)
	ej.AssertCompletedCount(t, 3)

	mutex.Unlock()

	// Wait again, and be sure all jobs completed.
	wg.Wait()
	time.Sleep(100)

	mutex.Lock()
	ej.AssertCompletedCount(t, 4)
	mutex.Unlock()
}

func TestProcessingQueue_Run_panic(t *testing.T) {
	pq := New(2)
	jobsCount := 0

	var mutex sync.Mutex // avoid data races in this test
	var wg sync.WaitGroup

	for i := 0; i < 4; i++ {
		wg.Add(1)

		go func() {
			defer func() {
				recover()
				wg.Done()
			}()
			pq.Run(func() {
				mutex.Lock()
				jobsCount++
				mutex.Unlock()
				panic("something bad happened")
			})
		}()
	}
	wg.Wait()
	assert.Equal(t, 4, jobsCount, "all jobs were executed")
}

func TestProcessingQueue_Size(t *testing.T) {
	for i := 1; i < 4; i++ {
		size := i
		t.Run(fmt.Sprintf("size %d", size), func(t *testing.T) {
			pq := New(size)
			assert.Equal(t, size, pq.Size())
		})
	}
}

type executedJob struct {
	index     int
	completed bool
}

type executedJobs []*executedJob

func (ej executedJobs) String() string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("<%d jobs: [", len(ej)))
	for _, item := range ej {
		sb.WriteString(fmt.Sprintf(" %+v", *item))
	}
	sb.WriteString(" ]>")

	return sb.String()
}

func (ej executedJobs) AssertIncludesJobIndex(t *testing.T, index int) {
	t.Helper()
	for _, item := range ej {
		if item.index == index {
			return
		}
	}
	t.Errorf("expected %s to contain index:%d", ej, index)
}

func (ej executedJobs) AssertCompletedCount(t *testing.T, expectedCount int) {
	t.Helper()
	if c := ej.CountCompleted(); c != expectedCount {
		t.Errorf("exptected %d completed jobs, actual %d: %s", expectedCount, c, ej)
	}
}

func (ej executedJobs) CountCompleted() int {
	n := 0
	for _, item := range ej {
		if item.completed {
			n++
		}
	}
	return n
}
