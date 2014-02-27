package univariate

import (
	"errors"
	"math"

	"github.com/btracey/opt/common"
	"github.com/gonum/floats"
)

// Bisection performs bisection on the gradient in order to find the minimum
type Bisection struct {
	initLoc     float64
	currStep    float64
	posInitGrad bool

	f ObjGrader

	InitialStepMag float64

	minStep float64
	minObj  float64
	minGrad float64

	maxStep float64
	maxObj  float64
	maxGrad float64
}

func (b *Bisection) Init(f ObjGrader, initLoc, initObj, initGrad float64) error {
	if b.InitialStepMag == 0 {
		return errors.New("bisection: initial step is zero")
	}

	b.initLoc = initLoc
	b.f = f

	b.minStep = 0
	b.minObj = initObj
	b.minGrad = initGrad

	b.maxStep = math.Inf(1)
	b.maxObj = math.Inf(1)
	b.maxGrad = math.Inf(1)

	// need to swap signs because moving in the opposite direction
	b.currStep = b.InitialStepMag
	if initGrad > 0 {
		b.posInitGrad = true
		b.currStep *= -1
		b.minGrad *= -1
	}
	return nil
}

func (b *Bisection) Iterate() (loc, obj, grad float64, nFunEvals int, err error) {
	var realgrad float64
	loc = b.initLoc + b.currStep
	obj, realgrad = b.f.ObjGrad(loc)

	grad = realgrad
	if b.posInitGrad {
		grad = -realgrad
	}
	/*
		fmt.Println("minstep = ", b.minStep, " minobj = ", b.minObj, " mingrad = ", b.minGrad)
		fmt.Println("maxstep = ", b.maxStep, " maxobj = ", b.maxObj, " maxgrad = ", b.maxGrad)
		fmt.Println("currstp = ", b.currStep, " newobj = ", obj, " newgrad = ", grad)
	*/
	if math.IsInf(b.maxStep, 1) || b.maxGrad < 0 {
		// Have not yet fully bounded the minimum
		switch {
		case grad > 0:
			// Found a change in sign of the derivative, so now have bounded the derivative
			b.maxStep = b.currStep
			b.maxObj = obj
			b.maxGrad = grad
			b.currStep = (b.minStep + b.maxStep) / 2
			return loc, obj, realgrad, 1, nil
		case floats.EqualWithinAbsOrRel(b.minObj, obj, 1e-15, 1e-15):
			// Too close to compare the objectives directly (within floating point error)
			// assume that if the derivative is negative should move more in that direction
			fallthrough
		case obj < b.minObj:
			// Still haven't found an upper bound, but decrease in function value
			// so go more in that direction
			b.minStep = b.currStep
			b.minObj = obj
			b.minGrad = grad
			if math.IsInf(b.maxStep, 1) {
				b.currStep *= 2
			} else {
				b.currStep = (b.minStep + b.maxStep) / 2
			}
			return loc, obj, realgrad, 1, nil
		default:
			// Increase in function value, but the gradient is still negative.
			// Means we must have skipped over a local minimum, so set this point
			// as the new maximum
			b.maxStep = b.currStep
			b.maxObj = obj
			b.maxGrad = grad
			b.currStep = (b.minStep + b.maxStep) / 2
			return loc, obj, realgrad, 1, nil
		}
	}
	// Have already bounded the minimum, now need to find the point
	if grad < 0 {
		b.minStep = b.currStep
		b.minObj = obj
		b.minGrad = grad
	} else {
		b.maxStep = b.currStep
		b.maxObj = obj
		b.maxGrad = grad
	}
	b.currStep = (b.minStep + b.maxStep) / 2
	return loc, obj, realgrad, 1, nil
}

func (b *Bisection) SetInitStep(f float64) {
	b.InitialStepMag = f
}

func (b *Bisection) Status() common.Status { return common.Continue }

func (b *Bisection) Result() {}
