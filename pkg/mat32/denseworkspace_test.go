// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat32

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestGet(t *testing.T) {
	run := func(size, expCap int) {
		t.Run(fmt.Sprintf("size %d", size), func(t *testing.T) {
			slice := GetDenseWorkspace(size, 1).data
			assertLenCap(t, slice, size, expCap)
		})
	}
	run(0, 1)
	run(1, 1)
	run(2, 2)
	run(3, 4)
	run(5, 8)
	run(10, 16)
	run(30, 32)
	run(100, 128)
	run(1000, 1024)
}

func TestGetAndRelease(t *testing.T) {
	a1 := GetDenseWorkspace(5, 1)
	b1 := GetDenseWorkspace(10, 1)

	assertLenCap(t, a1.data, 5, 8)
	assertLenCap(t, b1.data, 10, 16)

	a1.data[0] = 42
	b1.data[0] = 24

	ReleaseDense(a1)
	ReleaseDense(b1)

	a2 := GetDenseWorkspace(6, 1)
	b2 := GetDenseWorkspace(9, 1)

	x := GetDenseWorkspace(6, 1)
	y := GetDenseWorkspace(9, 1)

	assertLenCap(t, a2.data, 6, 8)
	assertLenCap(t, b2.data, 9, 16)
	assertLenCap(t, x.data, 6, 8)
	assertLenCap(t, y.data, 9, 16)

	if a2.data[0] != 42 {
		t.Errorf("a1 and a2 should share the same slice data")
	}
	if b2.data[0] != 24 {
		t.Errorf("b1 and b2 should share the same slice data")
	}
	if x.data[0] != 0 {
		t.Errorf("slice data of `x` should be blank")
	}
	if y.data[0] != 0 {
		t.Errorf("slice data of `y` should be blank")
	}
}

func TestGetDenseWorkspace(t *testing.T) {
	d := GetDenseWorkspace(2, 3)

	assert.Equal(t, 2, d.Rows())
	assert.Equal(t, 3, d.Columns())
	assert.Equal(t, []Float{0, 0, 0, 0, 0, 0}, d.Data())

	d.SetData([]Float{1, 2, 3, 4, 5, 6})
	ReleaseDense(d)
	d = GetDenseWorkspace(2, 3)
	assert.Equal(t, []Float{1, 2, 3, 4, 5, 6}, d.Data(), "possible dirty data is not zeroed")
	ReleaseDense(d)
}

func TestGetEmptyDenseWorkspace(t *testing.T) {
	d := GetEmptyDenseWorkspace(2, 3)

	assert.Equal(t, 2, d.Rows())
	assert.Equal(t, 3, d.Columns())
	assert.Equal(t, []Float{0, 0, 0, 0, 0, 0}, d.Data())

	d.SetData([]Float{1, 2, 3, 4, 5, 6})
	ReleaseDense(d)
	d = GetEmptyDenseWorkspace(2, 3)
	assert.Equal(t, []Float{0, 0, 0, 0, 0, 0}, d.Data(), "possible dirty data is zeroed")
	ReleaseDense(d)
}

func TestReleaseDense(t *testing.T) {
	t.Run("it panics if the matrix does not come from the workspace", func(t *testing.T) {
		d := NewEmptyDense(3, 4)
		defer ReleaseDense(d)
		view := d.View(4, 3)
		assert.Panics(t, func() { ReleaseDense(view) })
	})
}

func assertLenCap(t *testing.T, slice []Float, l, c int) {
	if len(slice) != l {
		t.Errorf("expected len %d, actual %d", l, len(slice))
	}
	if cap(slice) != c {
		t.Errorf("expected cap %d, actual %d", c, cap(slice))
	}
}
