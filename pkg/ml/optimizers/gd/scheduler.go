// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gd

// EpochScheduler is implemented by any value that has the IncEpoch method.
type EpochScheduler interface {
	// IncEpoch beats the occurrence of a new epoch.
	IncEpoch()
}

// BatchScheduler is implemented by any value that has the IncBatch method.
type BatchScheduler interface {
	// IncBatch beats the occurrence of a new batch.
	IncBatch()
}

// ExampleScheduler is implemented by any value that has the IncExample method.
type ExampleScheduler interface {
	// IncExample beats the occurrence of a new example.
	IncExample()
}
