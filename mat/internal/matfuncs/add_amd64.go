// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && gc && !purego

package matfuncs

// Add32 adds x1 and x2 element-wise, storing the result in y (32 bits).
func Add32(x1, x2, y []float32) {
	if hasAVX {
		AddAVX32(x1, x2, y)
		return
	}
	AddSSE32(x1, x2, y)
}

// Add64 adds x1 and x2 element-wise, storing the result in y (64 bits).
func Add64(x1, x2, y []float64) {
	if hasAVX {
		AddAVX64(x1, x2, y)
		return
	}
	AddSSE64(x1, x2, y)
}
