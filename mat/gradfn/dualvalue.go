// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"github.com/nlpodyssey/spago/mat"
	"reflect"
)

// DualValue is implemented by any value that implements automatic differentiation features.
type DualValue interface {
	// Value returns the value of the operand.
	Value() mat.Matrix
	// AccGrad accumulate the gradients gx to the operands.
	AccGrad(gx mat.Matrix)
	// RequiresGrad returns true if the operand requires gradients.
	RequiresGrad() bool
}

func isNil[O DualValue](o O) bool {
	if any(o) == nil {
		return true
	}
	v := reflect.ValueOf(o)
	return v.Kind() == reflect.Pointer && v.IsNil()
}
