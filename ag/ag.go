// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

var (
	// debug is a global variable that indicates if the program is in debugging mode or not.
	// In debugging mode the operators wait for the forward goroutine to finish.
	debug = false
)

// SetDebugMode enables or disables the debugging mode.
// In debugging mode the operators wait for the forward goroutine to finish.
func SetDebugMode(d bool) {
	debug = d
}
