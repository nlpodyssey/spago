// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"fmt"
	"github.com/nlpodyssey/spago/mat"
	"sync"
)

var (
	operatorPoolFloat32 = &sync.Pool{
		New: func() any { return new(Operator[float32]) },
	}
	operatorPoolFloat64 = &sync.Pool{
		New: func() any { return new(Operator[float64]) },
	}
)

func getOperatorPool[T mat.DType]() *sync.Pool {
	switch any(T(0)).(type) {
	case float32:
		return operatorPoolFloat32
	case float64:
		return operatorPoolFloat64
	default:
		panic(fmt.Sprintf("ag: no operator pool for type %T", T(0)))
	}
}
