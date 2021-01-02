// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package processingqueue_test

import (
	"fmt"
	"github.com/nlpodyssey/spago/pkg/utils/processingqueue"
	"sync"
	"time"
)

func ExampleProcessingQueue_Run() {
	pq := processingqueue.New(2)

	var wg sync.WaitGroup
	wg.Add(4)
	for i := 0; i < 4; i++ {
		time.Sleep(100 * time.Millisecond)

		go func(i int) {
			fmt.Printf("Before %d\n", i)
			// Do something computationally light...

			pq.Run(func() {
				fmt.Printf("Processing %d\n", i)
				// Do something computationally heavy...
				time.Sleep(500 * time.Millisecond)
				fmt.Printf("Processed %d\n", i)
			})

			fmt.Printf("After %d\n", i)
			// Do something computationally light...

			wg.Done()
		}(i)
	}
	wg.Wait()

	// Output:
	// Before 0
	// Processing 0
	// Before 1
	// Processing 1
	// Before 2
	// Before 3
	// Processed 0
	// After 0
	// Processing 2
	// Processed 1
	// After 1
	// Processing 3
	// Processed 2
	// After 2
	// Processed 3
	// After 3
}

func ExampleProcessingQueue_Go() {
	pq := processingqueue.New(2)

	var wg sync.WaitGroup
	wg.Add(4)
	for i := 0; i < 4; i++ {
		time.Sleep(100 * time.Millisecond)

		ii := i
		pq.Go(func() {
			fmt.Printf("Processing %d\n", ii)
			// Do something computationally heavy...
			time.Sleep(500 * time.Millisecond)
			fmt.Printf("Processed %d\n", ii)
			wg.Done()
		})
	}
	wg.Wait()

	// Output:
	// Processing 0
	// Processing 1
	// Processed 0
	// Processing 2
	// Processed 1
	// Processing 3
	// Processed 2
	// Processed 3
}
