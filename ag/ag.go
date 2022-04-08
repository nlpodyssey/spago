// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import "sync"

var ongoingComputations = sync.WaitGroup{}

// WaitForOngoingComputations wait until all forward computations are complete.
func WaitForOngoingComputations() {
	ongoingComputations.Wait()
}
