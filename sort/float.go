// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sort

// Float is a type constraint satisfied by any floating-point type.
type Float interface {
	~float32 | ~float64
}
