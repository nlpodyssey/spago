// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package stats

import "math"

type ClassMetrics struct {
	TruePos  int // The number of true positive results (correctly marked as positive)
    TrueNeg  int // The number of true negative results (correctly marked as negative)
	FalsePos int // The number of false positive results (that should have been negative)
	FalseNeg int // The number of false negative results (that should have been positive)
}

func NewMetricCounter() *ClassMetrics {
	return &ClassMetrics{
		TruePos:  0,
        TrueNeg:  0,
		FalsePos: 0,
		FalseNeg: 0,
	}
}

// Reset sets all the counters to zero.
func (c *ClassMetrics) Reset() {
	c.TruePos = 0
    c.TrueNeg = 0
	c.FalsePos = 0
	c.FalseNeg = 0
}

func (c *ClassMetrics) IncTruePos() {
	c.TruePos++
}

func (c *ClassMetrics) IncTrueNeg() {
	c.TrueNeg++
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

// Recall returns the recall (true positive rate) metric, calculated as true positive / (true positive + false negative).
func (c *ClassMetrics) Recall() float64 {
	return zeroIfNaN(float64(c.TruePos) / float64(c.TruePos+c.FalseNeg))
}

// F1Score returns the harmonic mean of precision and recall, calculated as 2 * (precision * recall / (precision + recall))
func (c *ClassMetrics) F1Score() float64 {
	return zeroIfNaN(2.0 * ((c.Precision() * c.Recall()) / (c.Precision() + c.Recall())))
}

// Specificity returns the specificity (selectivity, true negative rate) metric, calculated as true negative / (true negative + false positive).
func (c *ClassMetrics) Specificity() float64 {
    return zeroIfNaN(float64(c.TrueNeg) / float64(c.TrueNeg+c.FalsePos))
}

// Accuracy returns the accuracy metric, calculated as (true positive + true negative) / (TP + TN + FP + FN).
func (c *ClassMetrics) Accuracy() float64 {
    numerator := float64(c.TruePos) + float64(c.TrueNeg)
    return zeroIfNaN(numerator / (numerator + float64(c.FalseNeg+c.FalsePos)))
}

func zeroIfNaN(value float64) float64 {
	if value == math.NaN() {
		return 0.0
	} else {
		return value
	}
}
