package tasks

import "github.com/nlpodyssey/spago/pkg/mat"

// ClassConfidencePair is a JSON-serializable pair of Class and Confidence.
type ClassConfidencePair[T mat.DType] struct {
	Class      string `json:"class"`
	Confidence T      `json:"confidence"`
}

// ClassifyResponse is a JSON-serializable structure which holds server
// classification response data.
type ClassifyResponse[T mat.DType] struct {
	Class        string                   `json:"class"`
	Confidence   T                        `json:"confidence"`
	Distribution []ClassConfidencePair[T] `json:"distribution"`

	// Took is the number of milliseconds it took the server to execute the request.
	Took int64 `json:"took"`
}
