// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestWrapper_StopGrad(t *testing.T) {
	t.Run("float32", testWrapperStopGrad[float32])
	t.Run("float64", testWrapperStopGrad[float64])
}

func testWrapperStopGrad[T mat.DType](t *testing.T) {
	s := NewScalar[T](42)

	result := StopGrad[T](s)
	assert.IsType(t, &Wrapper[T]{}, result)
	w := result.(*Wrapper[T])

	assert.Same(t, s, w.Node)
}
