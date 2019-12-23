// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package decay

type Function interface {
	Decay(lr float64, t int) float64
}
