// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package global

import (
	"fmt"
	"os"
	"strconv"
)

// This is used to regulate the use of approximate math functions over the default implementations.
var mathOptimizationLevel = 0

const minMathOptimizationLevel = 0
const maxMathOptimizationLevel = 2
const defaultMathOptimizationLevel = 0

var ballast []byte

func init() {
	// Create a large virtual heap allocation of 10 GiB to reduce GC activity.
	// https://blog.twitch.tv/go-memory-ballast-how-i-learnt-to-stop-worrying-and-love-the-heap-26c2462549a2
	ballast = make([]byte, 10<<30)

	strOptLevel := os.Getenv("OPTIMIZATION_LEVEL")
	if strOptLevel == "" {
		SetMathOptimizationLevel(defaultMathOptimizationLevel)
	} else {
		if i, err := strconv.Atoi(strOptLevel); err == nil {
			SetMathOptimizationLevel(i)
		} else {
			panic(fmt.Sprintf("global: optimization level must be a number in the range [%d-%d]",
				minMathOptimizationLevel, maxMathOptimizationLevel))
		}
	}
}

// SetMathOptimizationLevel the global optimization level to i.
// It returns the previous level.
func SetMathOptimizationLevel(i int) int {
	if !(i >= minMathOptimizationLevel && i <= maxMathOptimizationLevel) {
		panic(fmt.Sprintf("global: optimization level must be in the range [%d-%d], found %d",
			minMathOptimizationLevel, maxMathOptimizationLevel, i))
	}
	prev := mathOptimizationLevel
	mathOptimizationLevel = i
	return prev
}

// MathOptimizationLevel returns the global optimization
func MathOptimizationLevel() int {
	return mathOptimizationLevel
}
