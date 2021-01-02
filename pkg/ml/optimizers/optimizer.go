// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package optimizers

// Optimizer is implemented by any value that has the Optimize method.
type Optimizer interface {
	Optimize()
}
