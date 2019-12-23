// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gd

type EpochScheduler interface {
	// Beat the occurrence of a new epoch
	IncEpoch()
}

type BatchScheduler interface {
	// Beat the occurrence of a new batch
	IncBatch()
}

type ExampleScheduler interface {
	// Beat the occurrence of a new example
	IncExample()
}
