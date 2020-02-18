// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fofe

import (
	"saientist.dev/spago/pkg/mat"
	"sort"
)

type item struct {
	id     int
	offset int
}

func Decode(alpha float64, z *mat.Sparse) []int {
	if alpha <= 0 || alpha > 0.5 {
		panic("fofe: alpha doesn't satisfy 0 < alpha ≤ 0.5")
	}

	var buf []item
	z.DoNonZero(func(i, _ int, v float64) {
		for _, k := range offsets(alpha, v) {
			buf = append(buf, item{id: i, offset: k})
		}
	})

	sort.Slice(buf, func(i, j int) bool {
		return buf[i].offset > buf[j].offset
	})

	seq := make([]int, len(buf))
	for i, value := range buf {
		seq[i] = value.id
	}

	return seq
}

func offsets(base float64, v float64) []int {
	const limit = 400 // arbitrary limit
	var lst []int
	n := v
	i := 0
	for n != 0.0 && n < limit {
		if n >= 1.0 {
			lst = append(lst, i)
			n = (n - 1.0) / base
		} else {
			n = n / base
		}
		i++
	}
	return lst
}
