// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package utils

func PanicIfErr(err error) {
	if err != nil {
		panic(err)
	}
}
