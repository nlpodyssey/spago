// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestTimeStep(t *testing.T) {
	t.Run("value implementing TimeStepper", func(t *testing.T) {
		assert.Equal(t, 42, TimeStep(timeStepperType(42)))
	})
	t.Run("value not implementing TimeStepper", func(t *testing.T) {
		assert.Equal(t, -1, TimeStep(nil))
	})
}

func TestSetTimeStep(t *testing.T) {
	t.Run("value implementing TimeStepSetter", func(t *testing.T) {
		funcCalls := 0
		tss := timeStepSetterType(func(v int) {
			funcCalls++
			assert.Equal(t, 42, v)
		})
		SetTimeStep(tss, 42)
		assert.Equal(t, funcCalls, 1)
	})
	t.Run("value not implementing TimeStepSetter", func(t *testing.T) {
		assert.Panics(t, func() {
			SetTimeStep(nil, 42)
		})
	})
}

type timeStepperType int

func (t timeStepperType) TimeStep() int { return int(t) }

type timeStepSetterType func(int)

func (t timeStepSetterType) SetTimeStep(v int) { t(v) }
