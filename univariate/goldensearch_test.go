package univariate

import (
	"math"
	"testing"

	"github.com/gonum/floats"
)

func TestGoldenSection(t *testing.T) {
	c := 5.0
	b := 3.0
	q := quadratic{b: b, c: c}

	// Test under normal conditions
	tol := 1e-7

	optimizer := NewGoldenSection(1, tol)
	result, err := OptimizeGradFree(q, -7, nil, optimizer)
	if err != nil {
		t.Errorf("Error optimizing", err)
	}
	if math.Abs(result.Loc-q.OptLoc()) > tol {
		t.Errorf("location doesn't match. Expected: %v, Found %v. Status %v", q.OptLoc(), result.Loc, result.Status)
	}

	result, err = OptimizeGradFree(q, -7, nil, optimizer)
	if err != nil {
		t.Errorf("Error optimizing", err)
	}
	if math.Abs(result.Loc-q.OptLoc()) > tol {
		t.Errorf("location doesn't match. Expected: %v, Found %v. Status %v", q.OptLoc(), result.Loc, result.Status)
	}

	// Should return where we started
	largeInit := 13.0
	result, err = OptimizeGradFree(q, largeInit, nil, optimizer)
	if err != nil {
		t.Errorf("Error optimizing", err)
	}
	if !floats.EqualWithinAbsOrRel(result.Loc, largeInit, tol, tol) {
		t.Errorf("location doesn't match. Expected: %v, Found %v. Status %v", largeInit, result.Loc, result.Status)
	}

	optimizer = NewGoldenSection(-1, tol)
	result, err = OptimizeGradFree(q, 14, nil, optimizer)
	if err != nil {
		t.Errorf("Error optimizing", err)
	}
	if !floats.EqualWithinAbsOrRel(result.Loc, q.OptLoc(), tol, tol) {
		t.Errorf("location doesn't match. Expected: %v, Found %v. Status %v", q.OptLoc(), result.Loc, result.Status)
	}
}
