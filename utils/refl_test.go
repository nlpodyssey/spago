// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package utils

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestReverseInPlace(t *testing.T) {
	t.Parallel()
	examples := []struct{ s, expected []int }{
		{nil, nil},
		{[]int{}, []int{}},
		{[]int{1}, []int{1}},
		{[]int{1, 2}, []int{2, 1}},
		{[]int{1, 2, 3}, []int{3, 2, 1}},
		{[]int{1, 2, 3, 4}, []int{4, 3, 2, 1}},
		{[]int{1, 2, 3, 4, 5}, []int{5, 4, 3, 2, 1}},
	}
	for _, ex := range examples {
		t.Run(fmt.Sprintf("%v", ex.s), func(t *testing.T) {
			s := make([]int, len(ex.s))
			copy(s, ex.s)
			ReverseInPlace(s)
			if len(ex.expected) == 0 {
				assert.Empty(t, s)
			} else {
				assert.Equal(t, ex.expected, s)
			}
		})
	}
}
