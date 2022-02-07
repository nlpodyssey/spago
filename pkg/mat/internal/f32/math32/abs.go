package math32

// Abs returns the absolute value of x.
//
// Special cases are:
//	Abs(±Inf) = +Inf
//	Abs(NaN) = NaN
func Abs(x float32) float32 {
	return Float32frombits(Float32bits(x) &^ (1 << 31))
}
