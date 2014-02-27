package linesearch

import (
	"errors"
	"math"

	"github.com/btracey/opt/common"
	"github.com/btracey/opt/univariate"
)

/*
Copyright (c) 2001, 2002 Enthought, Inc.
All rights reserved.

Copyright (c) 2003-2012 SciPy Developers.
All rights reserved.

Copyright (c) 2014 Brendan Tracey
All rights reserved

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of Enthought nor the names of the SciPy Developers
     may be used to endorse or promote products derived from this software
     without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// TODO: Better comment

//     This subroutine computes a safeguarded step for a search
//     procedure and updates an interval that contains a step that
//     satisfies a sufficient decrease and a curvature condition.
//
//     The parameter stx contains the step with the least function
//     value. If brackt is set to .true. then a minimizer has
//     been bracketed in an interval with endpoints stx and sty.
//     The parameter stp contains the current step.
//     The subroutine assumes that if brackt is set to .true. then
//
//           min(stx,sty) < stp < max(stx,sty),
//
//     and that the derivative at stx is negative in the direction
//     of the step.

const (
	xtrapl = 1.1
	xtrapu = 4.0
)

// Dcsrch implements the minpack2 dsrch algorithm from scipy
type Dcstep struct {
	fun            univariate.ObjGrader
	InitialStepMag float64
	MinStep        float64
	MaxStep        float64

	initLoc float64

	stmin, stmax   float64
	stpmin, stpmax float64

	stx, fx, gx float64
	sty, fy, gy float64
	stp, fp, gp float64

	bracket bool

	width  float64
	width1 float64

	stage        int
	finit, ginit float64
	ftest        float64
	gtest        float64

	wolfeGradConst float64
	wolfeFunConst  float64
}

func NewDcstep() *Dcstep {
	return &Dcstep{
		InitialStepMag: 1.0,
		MinStep:        1e-8,
		MaxStep:        50,
	}
}

func (d *Dcstep) SetWolfeConditions(fConst, gConst float64, isStrong bool) error {
	if !isStrong {
		return errors.New("must have strong wolfe conditions")
	}
	d.wolfeFunConst = fConst
	d.wolfeGradConst = gConst
	return nil
}

func (d *Dcstep) SetInitStep(f float64) {
	d.InitialStepMag = f
}

func (d *Dcstep) Status() common.Status { return common.Continue }

func (d *Dcstep) Result() {}

func (d *Dcstep) Init(f univariate.ObjGrader, initLoc, initObj, initGrad float64) error {
	if d.InitialStepMag == 0 {
		return errors.New("bisection: initial step is zero")
	}

	d.initLoc = initLoc

	d.finit = initObj
	d.ginit = initGrad
	d.gtest = d.wolfeFunConst * d.ginit

	// TODO: Better variable names
	d.fun = f
	d.stage = 1

	d.stpmin = d.MinStep
	d.stpmax = d.MaxStep

	d.width = d.stpmax - d.stpmin
	d.width1 = d.width / 0.5

	d.stpmin = d.MinStep
	d.stpmax = d.MaxStep

	d.stx = 0
	d.fx = initObj
	d.gx = initGrad
	d.sty = 0
	d.fy = initObj
	d.gy = initGrad

	d.bracket = false
	d.stmin = 0
	d.stmax = d.InitialStepMag + xtrapu*d.InitialStepMag

	d.stp = d.InitialStepMag
	if d.stp < 0 {
		return errors.New("Initial step direction is negative")
	}

	return nil
}

func (d *Dcstep) Iterate() (loc, obj, grad float64, nFunEvals int, err error) {
	newLoc := d.initLoc + d.stp
	f, g := d.fun.ObjGrad(newLoc)

	d.ftest = d.finit + d.stp*d.gtest
	if d.stage == 1 && f <= d.ftest && g >= 0 {
		d.stage = 2
	}

	if d.stage == 1 && f < d.fx && f > d.ftest {
		fm := f - d.stp*d.gtest
		fxm := d.fx - d.stx*d.gtest
		fym := d.fy - d.sty*d.gtest
		gm := g - d.gtest
		gxm := d.gx - d.gtest
		gym := d.gy - d.gtest

		d.stx, fxm, gxm, d.sty, fym, gym, d.stp, d.bracket =
			dcstep(d.stx, fxm, gxm, d.sty, fym, gym, d.stp, fm, gm, d.bracket, d.stpmin, d.stpmax)

		d.fx = fxm + d.stx*d.gtest
		d.fy = fym + d.sty*d.gtest
		d.gx = gxm + d.gtest
		d.gy = gym + d.gtest
	} else {

		//d.stx, d.fx, d.gx, d.sty, d.fy, d.gy, d.stp, d.bracket =
		//	dcstep(d.stx, d.fx, d.gx, d.sty, d.fy, d.gy, d.stp, f, g, d.bracket, d.stpmin, d.stpmax)
		d.stx, d.fx, d.gx, d.sty, d.fy, d.gy, d.stp, d.bracket =
			dcstep(d.stx, d.fx, d.gx, d.sty, d.fy, d.gy, d.stp, f, g, d.bracket, d.stpmin, d.stpmax)
	}

	// Decide if a bisection is needed
	if d.bracket {
		if math.Abs(d.sty-d.stx) >= 0.66*d.width1 {
			d.stp = d.stx + 0.5*(d.sty-d.stx)
		}
		d.width1 = d.width
		d.width = math.Abs(d.sty - d.stx)
	}

	// Set the minimum and maximum steps allowed
	if d.bracket {
		d.stmin = math.Min(d.stx, d.sty)
		d.stmax = math.Max(d.stx, d.sty)
	} else {
		d.stmin = d.stp + xtrapl*(d.stp-d.stx)
		d.stmax = d.stp + xtrapu*(d.stp-d.stx)
	}

	// Force the step to be within the bounds
	d.stp = math.Max(d.stp, d.stpmin)
	d.stp = math.Min(d.stp, d.stpmax)

	// TODO: Add status condition

	// TODO: replace variable names

	return newLoc, f, g, 1, nil
}

func dcstep(stx, fx, dx, sty, fy, dy, stp, fp, dp float64, bracket bool, stpmin, stpmax float64) (
	rstx, rfx, rdx, rsty, rfy, rdy, rstp float64, rbrak bool) {

	sgnd := dp * (dx / math.Abs(dx))
	var stpf float64

	if fp > fx {
		//  First case: A higher function value. The minimum is bracketed.
		//  If the cubic step is closer to stx than the quadratic step, the
		//  cubic step is taken, otherwise the average of the cubic and
		//  quadratic steps is taken.

		theta := 3.0*(fx-fp)/(stp-stx) + dx + dp
		s := math.Max(math.Abs(theta), math.Abs(dx))
		s = math.Max(s, math.Abs(dp))
		tmp := (theta/s)*(theta/s) - (dx/s)*(dp/s)
		gamma := s * math.Sqrt(tmp)
		if stp < stx {
			gamma = -gamma
		}
		p := (gamma - dx) + theta
		q := ((gamma - dx) + gamma) + dp
		r := p / q
		stpc := stx + r*(stp-stx)
		stpq := stx + ((dx/((fx-fp)/(stp-stx)+dx))/2.0)*(stp-stx)
		if math.Abs(stpc-stx) < math.Abs(stpq-stx) {
			stpf = stpc
		} else {
			stpf = stpc + (stpq-stpc)/2.0
		}
		bracket = true
	} else if sgnd < 0 {
		// Second case: A lower function value and derivatives of opposite
		// sign. The minimum is bracketed. If the cubic step is farther from
		// stp than the secant step, the cubic step is taken, otherwise the
		// secant step is taken.
		theta := 3.0*(fx-fp)/(stp-stx) + dx + dp
		s := math.Max(math.Abs(theta), math.Abs(dx))
		s = math.Max(s, math.Abs(dp))
		tmp := (theta/s)*(theta/s) - (dx/s)*(dp/s)
		gamma := s * math.Sqrt(tmp)
		if stp > stx {
			gamma = -gamma
		}
		p := (gamma - dp) + theta
		q := ((gamma - dp) + gamma) + dx
		r := p / q
		stpc := stp + r*(stx-stp)
		stpq := stp + (dp/(dp-dx))*(stx-stp)
		if math.Abs(stpc-stp) > math.Abs(stpq-stp) {
			stpf = stpc
		} else {
			stpf = stpq
		}
		bracket = true
	} else if math.Abs(dp) < math.Abs(dx) {
		// Third case: A lower function value, derivatives of the same sign,
		// and the magnitude of the derivative decreases.

		// The cubic step is computed only if the cubic tends to infinity
		// in the direction of the step or if the minimum of the cubic
		// is beyond stp. Otherwise the cubic step is defined to be the
		// secant step.

		theta := 3.0*(fx-fp)/(stp-stx) + dx + dp
		s := math.Max(math.Abs(theta), math.Abs(dx))
		s = math.Max(s, math.Abs(dp))

		// The case gamma = 0 only arises if the cubic does not tend
		// to infinity in the direction of the step.
		tmp := (theta/s)*(theta/s) - (dx/s)*(dp/s)
		tmp = math.Max(0, tmp)
		gamma := s * math.Sqrt(tmp)
		if stp > stx {
			gamma = -gamma
		}
		p := (gamma - dp) + theta
		q := (gamma + (dx - dp)) + gamma
		r := p / q

		var stpc float64
		if r < 0 && gamma != 0 {
			stpc = stp + r*(stx-stp)
		} else if stp > stx {
			stpc = stpmax
		} else {
			stpc = stpmin
		}
		stpq := stp + (dp/(dp-dx))*(stx-stp)
		if bracket {
			// A minimizer has been bracketed. If the cubic step is
			// closer to stp than the secant step, the cubic step is
			// taken, otherwise the secant step is taken.
			if math.Abs(stpc-stp) < math.Abs(stpq-stp) {
				stpf = stpc
			} else {
				stpf = stpq
			}
			if stp > stx {
				stpf = math.Min(stpf, stp+0.66*(sty-stp))
			} else {
				stpf = math.Max(stpf, stp+0.66*(sty-stp))
			}
		} else {
			// A minimizer has not been bracketed. If the cubic step is
			// farther from stp than the secant step, the cubic step is
			// taken, otherwise the secant step is taken.
			if math.Abs(stpc-stp) > math.Abs(stpq-stp) {
				stpf = stpc
			} else {
				stpf = stpq
			}
			stpf = math.Min(stpmax, stpf)
			stpf = math.Max(stpmin, stpf)
		}
	} else {
		// Fourth case: A lower function value, derivatives of the same sign,
		// and the magnitude of the derivative does not decrease. If the
		// minimum is not bracketed, the step is either stpmin or stpmax,
		// otherwise the cubic step is taken.
		if bracket {
			theta := 3.0*(fp-fy)/(sty-stp) + dy + dp
			s := math.Max(math.Abs(theta), math.Abs(dx))
			s = math.Max(s, math.Abs(dp))
			tmp := (theta/s)*(theta/s) - (dy/s)*(dp/s)
			gamma := s * math.Sqrt(tmp)
			if stp < sty {
				gamma = -gamma
			}
			p := (gamma - dp) + theta
			q := ((gamma - dp) + gamma) + dy
			r := p / q
			stpc := stp + r*(sty-stp)
			stpf = stpc
		} else if stp > stx {
			stpf = stpmax
		} else {
			stpf = stpmin
		}
	}

	// TODO: Something is different between this and the scipy implementation.
	// need to figure out if it is and if that's bad

	// Update the interval which contains a minimizer.
	if fp > fx {
		sty = stp
		fy = fp
		dy = dp
	} else {
		if sgnd < 0 {
			sty = stx
			fy = fx
			dy = dx
		}
		stx = stp
		fx = fp
		dx = dp
	}

	stp = stpf
	return stx, fx, dx, sty, fy, dy, stp, bracket
}
