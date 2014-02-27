package common

import "math"

// UniToler is a type for checking the convergence of a variable with either
// relative or absolute convergence values.
type UniToler struct {
	hist   []float64
	last   int  // Index of the last value added
	filled bool // Has the history been populated fully at least once

	absTol float64
	relTol float64

	recent float64
}

// Init initializes the UniToler. relatievWindow is the stencil for comparing
// values with the relative tolerance. If the relative tolerance is negative
// it is ignorned.
func (t *UniToler) Init(absTol, relTol float64, relativeWindow int, initVal float64) {
	if relTol > 0 {
		if len(t.hist) < relativeWindow {
			t.hist = make([]float64, relativeWindow)
		} else {
			t.hist = t.hist[:relativeWindow]
		}
		t.last = 0
		t.hist[0] = initVal
	}
	t.recent = initVal
	t.relTol = relTol
	t.absTol = absTol
	t.filled = false
}

// Add adds a new value to the toler (after an iteration)
func (t *UniToler) Add(v float64) {
	t.recent = v
	if t.relTol > 0 {
		t.last++
		if t.last == len(t.hist) {
			t.last = 0
			t.filled = true
		}
		t.hist[t.last] = v
	}
}

// AbsConverged returns true if the most recent value is below the absolute tolerance
func (t *UniToler) AbsConverged() bool {
	if math.IsNaN(t.absTol) {
		return false
	}
	return t.recent < t.absTol
}

// RelConverged returns true if the absolute difference between the most recent added
// value and the value added relativeWindow times ago is less than the relative tolerance
func (t *UniToler) RelConverged() bool {
	if t.relTol < 0 {
		return false
	}
	recent := t.hist[t.last]

	prevInd := t.last + 1
	if prevInd == len(t.hist) {
		prevInd = 0
	}
	previous := t.hist[prevInd]

	return math.Abs(previous-recent) < t.relTol
}
