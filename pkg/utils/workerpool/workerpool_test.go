// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package workerpool

import (
	"fmt"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

func TestWorkerPool(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode")
	}

	var mutex sync.Mutex // avoid data races in this test

	workerSleepingDuration := 300 * time.Millisecond
	//mainProcessSleepingDuration := workerSleepingDuration * 2

	ej := make(ExecutedJobs, 0, 3)
	runCompleted := false

	// Create a new WorkerPool and make it run async
	wp := New(3)
	go func() {
		wp.Run(func(workerID int, jobData interface{}) {
			mutex.Lock()
			item := &ExecutedJob{
				workerID:  workerID,
				jobData:   jobData.(string),
				completed: false,
			}
			ej = append(ej, item)
			mutex.Unlock()

			time.Sleep(workerSleepingDuration)

			mutex.Lock()
			item.completed = true
			mutex.Unlock()
		})

		mutex.Lock()
		runCompleted = true
		mutex.Unlock()
	}()

	// Publish 4 job data items to be processed, that it
	// one more than the total concurrent capacity
	wp.PublishJobData("foo")
	wp.PublishJobData("bar")
	wp.PublishJobData("baz")
	wp.PublishJobData("qux")

	// After waiting half of a job's duration, we should
	// be in the middle of the processing of the first
	// 3 job data items: "foo", "bar", and "baz".
	// "qux"
	time.Sleep(workerSleepingDuration / 2)

	mutex.Lock()

	if runCompleted {
		t.Errorf("exptected Run() execution not to be completed")
	}

	ej.AssertLen(t, 3)

	ej.AssertIncludesJobData(t, "foo")
	ej.AssertIncludesJobData(t, "bar")
	ej.AssertIncludesJobData(t, "foo")

	ej.AssertIncludesWorkerID(t, 0)
	ej.AssertIncludesWorkerID(t, 1)
	ej.AssertIncludesWorkerID(t, 2)

	ej.AssertCompletedCount(t, 0)

	mutex.Unlock()

	// Wait for the first 3 jobs to be completed, and
	// the 4th to be started.
	time.Sleep(workerSleepingDuration)

	mutex.Lock()

	if runCompleted {
		t.Errorf("exptected Run() execution not to be completed")
	}

	ej.AssertLen(t, 4)
	ej.AssertIncludesJobData(t, "qux")
	ej.AssertCompletedCount(t, 3)

	if ej[3].completed {
		t.Errorf("exptected last job not to be completed: %s", ej)
	}

	mutex.Unlock()

	// Send signal, and give it some time
	err := syscall.Kill(syscall.Getpid(), syscall.SIGINT)
	if err != nil {
		t.Fatalf("kill signal error: %v", err)
	}

	time.Sleep(workerSleepingDuration / 3)

	// This job should never be executed
	wp.PublishJobData("xyzzy")

	// The last job should still be running (same assertions from above)
	mutex.Lock()

	if runCompleted {
		t.Errorf("exptected Run() execution not to be completed")
	}

	ej.AssertLen(t, 4)
	ej.AssertIncludesJobData(t, "qux")
	ej.AssertCompletedCount(t, 3)

	if ej[3].completed {
		t.Errorf("exptected last job not to be completed: %s", ej)
	}

	mutex.Unlock()

	// Wait even more and expect graceful termination to be completed
	time.Sleep(workerSleepingDuration)

	mutex.Lock()

	if !runCompleted {
		t.Errorf("exptected Run() execution to be completed")
	}

	ej.AssertLen(t, 4)
	ej.AssertCompletedCount(t, 4)

	mutex.Unlock()
}

type ExecutedJob struct {
	workerID  int
	jobData   string
	completed bool
}

type ExecutedJobs []*ExecutedJob

func (ej ExecutedJobs) String() string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("<%d jobs: [", len(ej)))
	for _, item := range ej {
		sb.WriteString(fmt.Sprintf(" %+v", *item))
	}
	sb.WriteString(" ]>")

	return sb.String()
}

func (ej ExecutedJobs) AssertIncludesJobData(t *testing.T, jobData string) {
	t.Helper()
	for _, item := range ej {
		if item.jobData == jobData {
			return
		}
	}
	t.Errorf("expected %s to contain jobData:%s", ej, jobData)
}

func (ej ExecutedJobs) AssertIncludesWorkerID(t *testing.T, workerID int) {
	t.Helper()
	for _, item := range ej {
		if item.workerID == workerID {
			return
		}
	}
	t.Errorf("expected %s to contain workerID:%d", ej, workerID)
}

func (ej ExecutedJobs) AssertLen(t *testing.T, expectedLen int) {
	t.Helper()
	if len(ej) != expectedLen {
		t.Errorf("expected %d jobs, actual %d: %s", expectedLen, len(ej), ej)
	}
}

func (ej ExecutedJobs) AssertCompletedCount(t *testing.T, expectedCount int) {
	t.Helper()
	if c := ej.CountCompleted(); c != expectedCount {
		t.Errorf("exptected %d completed jobs, actual %d: %s", expectedCount, c, ej)
	}
}

func (ej ExecutedJobs) CountCompleted() int {
	n := 0
	for _, item := range ej {
		if item.completed {
			n++
		}
	}
	return n
}
