// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestApply(t *testing.T) {
	for _, tt := range traversalTests {
		t.Run(tt.name, func(t *testing.T) {
			var actual []collectedModel
			Apply(tt.model, func(m Model) {
				actual = append(actual, collectedModel{model: m})
			})
			assert.Equal(t, tt.expectedModels, actual)
		})
	}
}

func TestForEachParam(t *testing.T) {
	for _, tt := range traversalTests {
		t.Run(tt.name, func(t *testing.T) {
			var actual []collectedParam
			ForEachParam(tt.model, func(p *Param) {
				actual = append(actual, collectedParam{param: p})
			})
			assert.Equal(t, tt.expectedParams, actual)
		})
	}
}

func TestForEachParamStrict(t *testing.T) {
	for _, tt := range traversalTests {
		t.Run(tt.name, func(t *testing.T) {
			var actual []collectedParam
			ForEachParamStrict(tt.model, func(p *Param) {
				actual = append(actual, collectedParam{param: p})
			})
			assert.Equal(t, tt.expectedParamsStrict, actual)
		})
	}
}
