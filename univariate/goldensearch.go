package univariate

import (
	"errors"
	"math"

	"github.com/btracey/opt/common"
	"github.com/gonum/floats"
)

// TODO: Need to improve numerical behavior when REALLY close to optimum

const resphi = 2 - math.Phi

// GoldenSection performs a golden section search in order to find the optimum. Assumes
// there is an optimum in the direction of the initial step from the initial
// location. User must set the initial step
// Will attempt to find the nearest minimum to the initial location (not guaranteed)
type GoldenSection struct {
	InitialStep float64 // lists the initial step
	// MaxStep puts a contraint on how far away the bisection can look.
	// It must have the same sign as InitialStep. If MaxStep is zero, it is
	// assumed that there is no maximum

	//MaxStep float64
	//MaxObj  float64 // Objective at the maximum value

	Tol float64 // How close can the bounds get before returning

	initLoc float64

	initialStepPos bool

	currentStep float64
	closerToMin bool

	//maxLoc float64
	maxStep float64
	maxObj  float64

	minStep float64
	minObj  float64

	middleStep float64
	middleObj  float64

	bestObj float64
	bestLoc float64

	f Objective
}

func NewGoldenSection(initialStep, tol float64) *GoldenSection {
	return &GoldenSection{
		InitialStep: initialStep,
		Tol:         tol,
	}
}

func (b *GoldenSection) Init(f Objective, initLoc, initObj float64) error {
	// Set the initial step if it hasn't been
	if b.InitialStep == 0 {
		return errors.New("initial step not set")
	}
	b.initLoc = initLoc

	b.initialStepPos = b.InitialStep > 0

	if b.initialStepPos && (b.maxStep < 0) {
		return errors.New("sign of MaxStep does not match InitialStep")
	}
	if !b.initialStepPos && (b.maxStep > 0) {
		return errors.New("sign of MaxStep does not match InitialStep")
	}
	if b.initialStepPos {
		b.maxStep = math.Inf(1)
	} else {
		b.maxStep = math.Inf(-1)
	}
	b.maxObj = math.Inf(1)

	b.currentStep = b.InitialStep

	b.minStep = 0
	b.minObj = initObj

	b.middleStep = math.Inf(1)
	b.middleObj = math.Inf(1)

	b.bestObj = initObj
	b.bestLoc = initLoc

	b.f = f

	return nil
}

func (b *GoldenSection) Status() common.Status {

	if floats.EqualWithinAbsOrRel(b.maxStep+b.initLoc, b.minStep+b.initLoc, b.Tol, b.Tol) {
		return common.BoundsConverged
	}
	//if math.Abs(b.maxStep-b.minStep) < b.Tol {
	//	return common.BoundsConverged
	//}
	// Check if we're so close that got equal
	if !math.IsInf(b.middleStep, 0) && (b.middleStep == b.maxStep || b.middleStep == b.minStep) {
		return common.BoundsConverged
	}
	return common.Continue
}

// Next returns the next location that will be evaluated
func (b *GoldenSection) Next() float64 {
	return b.currentStep
}

func (b *GoldenSection) Iterate() (loc, obj float64, nFunEvals int, err error) {
	// TODO: Add the max stuff back in with a better way of avoiding unnecssary function evaluations

	// Evaluate at the new point
	loc = b.initLoc + b.currentStep
	newObj := b.f.Obj(loc)

	// Update the values
	if math.IsInf(b.maxStep, 0) {
		// Still looking for a bound on the function minimum
		if newObj >= b.minObj {
			// Have found a bound on the minimum location
			b.maxStep = b.currentStep
			b.maxObj = newObj
			if math.IsInf(b.middleStep, 0) {
				b.currentStep, b.closerToMin = goldenNoMiddleStep(b.minStep, b.maxStep)
			} else {
				b.currentStep, b.closerToMin = goldenNewStep(b.minStep, b.maxStep, b.middleStep)
			}
		} else {
			// Haven't found a bound, so keep going until a bound is found,
			// but keep track of best point for when we have bounded it
			if newObj < b.middleObj {
				b.middleObj = newObj
				b.middleStep = b.currentStep
			}

			b.currentStep *= 2
		}
		return loc, newObj, 1, nil
	}
	if math.IsInf(b.middleStep, 0) {
		// Have bounded the minimum, but still looking for a middle location
		switch {
		case newObj < b.minObj && newObj < b.maxObj:
			// Is between the two objectives, so establish this as the middle point
			b.middleStep = b.currentStep
			b.middleObj = newObj

			b.currentStep, b.closerToMin = goldenNewStep(b.minStep, b.maxStep, b.middleStep)
		default:
			b.resetMax(newObj)
		}
		return loc, newObj, 1, nil
	}

	if newObj > b.minObj && newObj > b.maxObj {
		b.resetMax(newObj)

		return loc, newObj, 1, nil
	}

	if newObj < b.middleObj {
		if b.closerToMin {
			b.maxStep = b.middleStep
			b.maxObj = b.middleObj
		} else {
			b.minStep = b.middleStep
			b.minObj = b.middleObj
		}
		b.middleStep = b.currentStep
		b.middleObj = newObj

	} else {
		if b.closerToMin {
			b.minStep = b.currentStep
			b.minObj = newObj
		} else {
			b.maxStep = b.currentStep
			b.maxObj = newObj
		}
	}

	b.currentStep, b.closerToMin = goldenNewStep(b.minStep, b.maxStep, b.middleStep)
	return loc, newObj, 1, nil
}

func (b *GoldenSection) resetMax(newObj float64) {
	// The new found point is higher in value than the ends, which means
	// there must be more than one local minima of the function
	// (we are operating under the assumption there is a minima). Try
	// to find the minimum close to the initial point
	b.maxStep = b.currentStep
	b.maxObj = newObj
	b.currentStep, b.closerToMin = goldenNoMiddleStep(b.minStep, b.maxStep)
	b.middleStep = math.Inf(1)
}

//  returns new step and if it's closer to the min
func goldenNoMiddleStep(min, max float64) (float64, bool) {
	newLoc := min + resphi*(max-min)
	return newLoc, true
}

// returns new step and if it's closer to the min
func goldenNewStep(min, max, middle float64) (float64, bool) {

	var newLoc float64
	if max > 0 {
		if (max - middle) > (middle - min) {
			newLoc = middle + resphi*(max-middle)
		} else {
			newLoc = middle - resphi*(middle-min)
		}
		return newLoc, (middle - min) > (newLoc - min)
	}
	if (max - middle) < (middle - min) {
		newLoc = middle + resphi*(max-middle)
	} else {
		newLoc = middle - resphi*(middle-min)
	}
	return newLoc, (middle - min) < (newLoc - min)
}

func (g *GoldenSection) Result() {
	return
}
