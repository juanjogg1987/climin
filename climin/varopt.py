# -*- coding: utf-8 -*-

"""This module provides an implementation of Variational Optimisation with momentum."""

from __future__ import absolute_import

import warnings

from .base import Minimizer

import numpy as np


class VarOpt(Minimizer):
    """Variational Optimisation. (VarOpt).

    VarOpt is a method for the optimization of stochastic objective functions
    following a natural gradient approach.
    
    Variational optimisation method proposes to minimise a possibly non-convex function $g(\theta)$, where
    $\theta$ represents the set of parameters that minimise the function, by introducing exploration in the
    parameter space of a variational (or exploratory) distribution $q(\theta)=\mathcal{N}(\theta|\mu,\sigma^2)$
    by bounding the function $g(\theta)$ as follows:

	\begin{align}
	\tilde{\mathcal{L}}=\mathbb{E}_{q(\theta)}[g(\theta)]+\mathbb{D}_{KL}\big(q(\theta)||p(\theta)\big),
	\end{align}

    where $\mathbb{D}_{KL}(\cdot||\cdot)$ is a Kullback-Leibler (KL) divergence and $p(\theta)=\mathcal{N}(\theta|0,\lambda^{-1})$
    is a prior penalization distribution with precision parameter $\lambda$.
        
    This algorithm was part of the work [J. Giraldo and M. A. Alvarez (2020): 'A Fully Natural Gradient Scheme for Improving Inference of the
    Heterogeneous Multi-Output Gaussian Process Model', https://arxiv.org/abs/1911.10225] for implementing the updates in Eq. (24) and (25). 
    
    An alternative way to implement such updates (24) and (25) consists on following a similar implementation of the Adam algorithm [Kingma, 
    Diederik, and Jimmy Ba. "Adam: A Method for Stochastic Optimization.", arXiv preprint arXiv:1412.6980 (2014)],
    we follow the work in [M. E. Khan et al. (2018) 'Fast and Scalable Bayesian Deep Learning by Weight-Perturbation in Adam', 
    https://arxiv.org/pdf/1806.04854.pdf] where the idea is to estimate the first two moments with exponentially decaying 
    running averages. Additionally, these estimates are bias corrected.

    Let $\hat{\nabla} \mathcal{L}(\theta_{t})$ be the derivative of the loss with respect to the parameters
    at time step $t$. In its basic form, given a step rate $\alpha$, decay terms $\beta_1$ and $\beta_2$ for
    the first and second moment estimates respectively, and an prior precision $\lambda$. We initialise the
    following quantities:

    .. math::
       m_0 & \\leftarrow 0 \\\\
       s_0 & \\leftarrow 1000 \text{by default 1000 or assigned by user with parameter s_ini} \\\\
       t & \\leftarrow 0 \\\\
       \lambda & \\leftarrow \text{by default 1e-8 or assigned by user with parameter prior_lambda} \\\\
 
    and perform the following updates:

    .. math::
       
       \begin{align}
	\sigma_{t} & \leftarrow {1 \over  \sqrt{{s}_{t} + \lambda}} \\\\
	\theta_{t} & \sim \mathcal{N}(\mu_{t},\sigma^2_{t}) \quad \text{take a sample for } \theta_{t} \\\\
	g_{t}   & \leftarrow \hat{\nabla} \mathcal{L}(\theta_{t}) \\\\
	m_{t+1}   & \leftarrow \beta_1 \cdot (g_t+\lambda\mu_t) + (1 - \beta_1) \cdot m_{t} \\\\
	s_{t+1}   &\leftarrow  \beta_2 \cdot g_{t}^2 + (1 - \beta_2) \cdot s_{t}  \\\\
	\hat{m}_{t+1}  &\leftarrow {m_{t+1} \over (1 - (1 - \beta_1)^t)} \\\\
	\hat{s}_{t+1}  &\leftarrow {s_{t+1} \over (1 - (1 - \beta_2)^t)} \\\\
	\mu_{t+1}   &\leftarrow \mu_{t} - \alpha {\hat{m}_{t+1} \over (\sqrt{\hat{s}_{t+1}} + \lambda)}\\\\
	t     & \leftarrow t + 1 \\\\
	\end{align}
       

    As suggested in the original paper, the last three steps are optimized for
    efficieny by using:

    .. math::
        \\alpha_t  &\\leftarrow \\alpha {\\sqrt{(1 - (1 - \\beta_2)^t)} \\over (1 - (1 - \\beta_1)^t)} \\\\
        \\theta_t   &\\leftarrow \\theta_{t-1} - \\alpha_t {m_t \\over (\\sqrt{v_t} + \\epsilon)}

    The quantities in the algorithm and their corresponding attributes in the
    optimizer object are as follows.

    ======================= =================== ===========================================================
    Symbol                  Attribute           Meaning
    ======================= =================== ===========================================================
    :math:`t`               ``n_iter``          Number of iterations, starting at 0.
    :math:`m_t`             ``est_mom_1_b``     Biased estimate of first moment.
    :math:`s_t`             ``est_mom_2_b``     Biased estimate of second moment.
    :math:`\hat{m}_t`       ``est_mom_1``       Unbiased estimate of first moment.
    :math:`\hat{s}_t`       ``est_mom_2``       Unbiased estimate of second moment.
    :math:`\alpha`          ``step_rate``       Step rate parameter.
    :math:`\beta_1`         ``decay_mom1``      Exponential decay parameter for first moment estimate.
    :math:`\beta_2`         ``decay_mom2``      Exponential decay parameter for second moment estimate.
    :math:`\lambda`         ``prior_lambda``    Precision of the Guassian prior over the set to optimise.
    ======================= =================== ===========================================================

    .. note::
       The use of decay parameters :math:`\\beta_1` and :math:`\\beta_2` differs
       from the definition in the original paper [adam2014]_:
       With :math:`\\beta^{\\ast}_i` referring to the parameters as defined in
       the paper, we use :math:`\\beta_i` with :math:`\\beta_i = 1 - \\beta^{\\ast}_i`

    """

    state_fields = 'n_iter step_rate decay_mom1 decay_mom2 step prior_lambda est_mom1_b est_mom2_b'.split()

    def __init__(self, wrt, fprime, step_rate=.0002,
                 s_ini=1000,
                 decay_mom1=0.1,
                 decay_mom2=0.001,
                 momentum=0,
                 prior_lambda=1e-8, args=None):
        """Create a VarOpt object.

        Parameters
        ----------

        wrt : array_like
            Array that represents the solution. Will be operated upon in
            place.  ``fprime`` should accept this array as a first argument.

        fprime : callable
            Callable that given a solution vector as first parameter and *args
            and **kwargs drawn from the iterations ``args`` returns a
            search direction, such as a gradient.

        step_rate : scalar or array_like, optional [default: 1]
            Value to multiply steps with before they are applied to the
            parameter vector.

        s_ini : float, optional [default: 1000] this parameter is inverse to the std deviation sig
             of the variational posterior distribution q(\mu,\sig^2). This parameter is associated
             to the initial state of exploration of the variational distribution q(\mu,\sig^2). Big 
             values indicate little exploration at the beginning of optimisation and small values mean
             the contrary. In the practice we noticed that if the input dimensionality of a supervised
             Gaussian Process model is P<=3 then s_ini should be in the interval (1000-10000). For P>3
             its initialisation can be s_ini<=100.  
        
        decay_mom1 : float, optional, [default: 0.1]
            Decay parameter for the exponential moving average estimate of the
            first moment.

        decay_mom2 : float, optional, [default: 0.001]
            Decay parameter for the exponential moving average estimate of the
            second moment.

        momentum : float or array_like, optional [default: 0]
            Momentum to use during optimization. Can be specified analogously
            (but independent of) step rate.

        prior_lambda : float, optional, [default: 1e-8]
            This is the Precision of the Guassian prior over the set to optimise.
            Before taking the square root of the running averages, this lambda
            is added.

        args : iterable
            Iterator over arguments which ``fprime`` will be called with.
        """
        if not 0 < decay_mom1 <= 1:
            raise ValueError('decay_mom1 has to lie in (0, 1]')
        if not 0 < decay_mom2 <= 1:
            raise ValueError('decay_mom2 has to lie in (0, 1]')
        if not (1 - decay_mom1 * 2) / (1 - decay_mom2) ** 0.5 < 1:
            warnings.warn("constraint from convergence analysis for adam not "
                          "satisfied; check original paper to see if you "
                          "really want to do this.")

        super(VarOpt, self).__init__(wrt, args=args)

        self.fprime = fprime
        self.step_rate = step_rate
        self.decay_mom1 = decay_mom1
        self.decay_mom2 = decay_mom2
        self.prior_lambda = prior_lambda
        self.momentum = momentum
        
        self.est_mom1_b = 0
        self.est_mom2_b = s_ini 

        print('Using Variational Optimisation with s_ini:', self.est_mom2_b)
        
        self.step = 0

    def _iterate(self):
        for args, kwargs in self.args:
            m = self.momentum
            dm1 = self.decay_mom1
            dm2 = self.decay_mom2
            lamb1 = self.prior_lambda
            t = self.n_iter + 1

            step_m1 = self.step
            step1 = step_m1 * m
            self.wrt -= step1

            est_mom1_b_m1 = self.est_mom1_b
            est_mom2_b_m1 = self.est_mom2_b

            e_noise = np.random.randn(self.wrt.shape[0])
            sig = (est_mom2_b_m1+lamb1)**(-0.5)
            sig_noise = sig * e_noise   #
            gradient = self.fprime(self.wrt + sig_noise, *args, **kwargs)
            self.est_mom1_b = dm1 * (gradient + lamb1 * self.wrt) + (1 - dm1) * est_mom1_b_m1
            
            self.est_mom2_b = dm2 * gradient ** 2 + (1 - dm2) * est_mom2_b_m1

            step_t = self.step_rate * (1 - (1 - dm2) ** t) ** 0.5 / \
                     (1 - (1 - dm1) ** t)
            step2 = step_t * self.est_mom1_b / (self.est_mom2_b ** 0.5 + lamb1)

            self.wrt -= step2
            self.step = step1 + step2

            self.n_iter += 1

            yield {
                'n_iter': self.n_iter,
                'gradient': gradient,
                'args': args,
                'kwargs': kwargs,
            }
