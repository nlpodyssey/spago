// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package charlm

func splitByRune(str string) []string {
	out := make([]string, 0)
	for _, item := range str {
		out = append(out, string(item))
	}
	return out
}
