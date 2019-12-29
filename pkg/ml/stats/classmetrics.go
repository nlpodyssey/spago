// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package stats

import "math"

type ClassMetrics struct {
	TruePos  int // The number of true positive results (correctly marked as positive)
	FalsePos int // The number of false positive results (that should have been negative)
	FalseNeg int // The number of false negative results (that should have been positive)
}

func NewMetricCounter() *ClassMetrics {
	return &ClassMetrics{
		TruePos:  0,
		FalsePos: 0,
		FalseNeg: 0,
	}
}

// Reset sets all the counters to zero.
func (c *ClassMetrics) Reset() {
	c.TruePos = 0
	c.FalsePos = 0
	c.FalseNeg = 0
}

func (c *ClassMetrics) IncTruePos() {
	c.TruePos++
}

func (c *ClassMetrics) IncFalsePos() {
	c.FalsePos++
}

func (c *ClassMetrics) IncFalseNeg() {
	c.FalseNeg++
}

// ExpectedPos returns the sum of true positive and false negative
func (c *ClassMetrics) ExpectedPos() int {
	return c.TruePos + c.FalseNeg
}

// Precision returns the precision metric, calculated as true positive / (true positive + false positive).
func (c *ClassMetrics) Precision() float64 {
	return zeroIfNaN(float64(c.TruePos) / float64(c.TruePos+c.FalsePos))
}

// Precision returns the precision metric, calculated as true positive / (true positive + false negative).
func (c *ClassMetrics) Recall() float64 {
	return zeroIfNaN(float64(c.TruePos) / float64(c.TruePos+c.FalseNeg))
}

// F1Score returns the a measure of a accuracy, calculated as 2 * (precision * recall / (precision + recall))
func (c *ClassMetrics) F1Score() float64 {
	return zeroIfNaN(2.0 * ((c.Precision() * c.Recall()) / (c.Precision() + c.Recall())))
}

func zeroIfNaN(value float64) float64 {
	if value == math.NaN() {
		return 0.0
	} else {
		return value
	}
}
