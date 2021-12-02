package gbst

import "math"

// greatest common divisor (GCD) via Euclidean algorithm
func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// lcm finds the Least Common Multiple (LCM) via GCD
func lcm(a, b int, integers ...int) int {
	result := a * b / gcd(a, b)
	for i := 0; i < len(integers); i++ {
		result = lcm(result, integers[i])
	}
	return result
}

// nextDivisibleLength finds the greater value next to sequence length value, divisible by multiple
func nextDivisibleLength(seqLen, multiple int) int {
	return int(math.Ceil(float64(seqLen)/float64(multiple))) * multiple
}
