# Bayesian Optimisation using Gaussian Processes
# Stephen Dann <stephen.dann@stfc.ac.uk>

import numpy as np
import scipy.stats as spstat
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

class AugmentedGaussianProcess:
    """A pair of Gaussian processes: one for the samples and another for the error.

    Arguments:
     * sample_kernel: the kernel used for the sample predictor
     * error_kernel: the kernel used for the error predictor; if not specified,
       defaults to the sample kernel plus a white noise term.

    Keyword-only arguments:
     * fit_white_noise: if True, add a white noise term to the kernel and include
       the white noise term in the sample error rather than the model error.
    
    Attributes:
     * submodel_samples: the sample predictor (replaced by each call to fit())
     * submodel_errors: the error predictor

    Note that direct access to the submodels doesn't include the corrections
    applied by fit_white_noise.
    """

    def __init__(self, sample_kernel, error_kernel=None, *, fit_white_noise=False,efficiency_factor=1):
        if fit_white_noise:
            sample_kernel = sample_kernel + kernels.WhiteKernel()

        if error_kernel is None:
            if fit_white_noise:
                error_kernel = sample_kernel
            else:
                error_kernel = sample_kernel + kernels.WhiteKernel()

        self.fit_white_noise = fit_white_noise
        self.efficiency_factor= efficiency_factor
        self.sample_kernel = sample_kernel
        self.submodel_samples = GaussianProcessRegressor(self.sample_kernel)
        self.submodel_errors = GaussianProcessRegressor(error_kernel)

    def fit(self, X, Y, Y_err):
        """Fit the model to a set of data with errors."""

        self.submodel_samples = GaussianProcessRegressor(self.sample_kernel, alpha=Y_err**2)

        self.submodel_samples.fit(X, Y)
        self.submodel_errors.fit(X, Y_err)

    def predict(self, X, return_std=False, return_efficiency=False):
        """Predict the mean, possibly also the standard error and sampling efficiency.

        If return_std is False, returns the predicted mean.
        If return_std is True, also returns the standard error of the prediction.
        If return_efficiency is also True, also returns the sampling
        efficicency, defined as the portion of the total sampling error
        attributable to the model uncertainty.
        """

        if return_std:
            mean, std = self.submodel_samples.predict(X, return_std=True)
            sigma = self.predict_sample_error(X)
            if self.fit_white_noise:
                white_noise_level = self.submodel_samples.kernel_.k2.noise_level
                var =std**2 - white_noise_level
                var = np.clip(var,0,None) # added bodge to prevent spurious results
                #std = np.sqrt(std**2 - white_noise_level)
                std = np.sqrt(var)
            if return_efficiency:
                efficiency = 1 - self.efficiency_factor*sigma / np.sqrt(sigma**2 + std**2)
                return mean, std, efficiency
            else:
                return mean, std
        else:
            return self.submodel_samples.predict(X)

    def predict_sample_error(self, X):
        """Predict the sample error."""

        sigma = self.submodel_errors.predict(X)
        if self.fit_white_noise:
            white_noise_level = self.submodel_samples.kernel_.k2.noise_level
            sigma = np.sqrt(sigma**2 + white_noise_level)
        return sigma

class AcquisitionFunctionUCB:
    """The upper confidence bound acquisition function.

    This class is callable: simply call the constructed object with the points
    at which the function should be evaluated.

    Arguments:
     * model: the AugmentedGaussianProcess model to use
     * kappa: the multiple of the standard error to add to or subtract from the mean
     * invert: if True, subtracts the standard error from the mean and applies
       the cutoff when the mean is too big
     * use_efficiency: if True, multiplies the standard error by the sampling efficiency
     * mean_cutoff: if set, any points with a predicted mean less than this will return NaN
     * efficiency_cutoff: if set, any points with an efficiency less than this will return NaN
    """

    def __init__(self, model, kappa, invert=False,
            use_efficiency=False, mean_cutoff=None, efficiency_cutoff=None):
        self.model = model
        self.kappa = kappa
        self.invert = invert
        self.use_efficiency = use_efficiency
        self.mean_cutoff = mean_cutoff
        self.efficiency_cutoff = efficiency_cutoff

    def __call__(self, x):
        if self.use_efficiency:
            mean, std, efficiency = self.model.predict(x,
                    return_std=True, return_efficiency=True)
        else:
            mean, std = self.model.predict(x, return_std=True)
            efficiency = 1

        if self.invert:
            ucb = mean - std * self.kappa * efficiency
            if self.mean_cutoff is not None:
                ucb[mean > self.mean_cutoff] = np.nan
        else:
            ucb = mean + std * self.kappa * efficiency
            if self.mean_cutoff is not None:
                ucb[mean < self.mean_cutoff] = np.nan

        if self.use_efficiency and self.efficiency_cutoff is not None:
            ucb[efficiency < self.efficiency_cutoff] = np.nan

        return ucb

class AcquisitionFunctionEI:
    """The expected improvement acquisition function.

    Note that since the best value so far is passed into the constructor,
    you'll need to construct a new object every time this changes. Or simply
    modify the best_val attribute.

    This class is callable: simply call the constructed object with the points
    at which the function should be evaluated.

    Arguments:
     * model: the AugmentedGaussianProcess model to use
     * best_val: the value which we're trying to improve on
     * invert: if True, smaller values are considered improved. Still returns positive values
     * use_efficiency: if True, multiplies the result by the sampling efficiency
     * mean_cutoff: if set, any points with a predicted mean less than this will return NaN
     * efficiency_cutoff: if set, any points with an efficiency less than this will return NaN
    """

    def __init__(self, model, best_val, invert=False,
            use_efficiency=False, mean_cutoff=None, efficiency_cutoff=None):
        self.model = model
        self.best_val = best_val
        self.invert = invert
        self.use_efficiency = use_efficiency
        self.mean_cutoff = mean_cutoff
        self.efficiency_cutoff = efficiency_cutoff
        

    def __call__(self, x):
        if self.use_efficiency:
            mean, std, efficiency = self.model.predict(x,
                    return_std=True, return_efficiency=True)
        else:
            mean, std = self.model.predict(x, return_std=True)
            efficiency = 1

        if self.invert:
            diff = self.best_val - mean
        else:
            diff = mean - self.best_val

        ei = diff * spstat.norm.cdf(diff / std) + std * spstat.norm.pdf(diff / std)
        ei = ei * efficiency

        if self.mean_cutoff is not None:
            if self.invert:
                ei[mean > self.mean_cutoff] = np.nan
            else:
                ei[mean < self.mean_cutoff] = np.nan

        if self.use_efficiency and self.efficiency_cutoff is not None:
            ei[efficiency < self.efficiency_cutoff] = np.nan

        return ei

class BasicOptimiser:
    """A multidimensional optimiser that seems to work well in practice.

    Maximises the measured values, whatever they are. Requires that the
    configuration space is roughly isotropic.

    Arguments:
    * n_dims: the number of dimensions over which to optimise
    * mean_cutoff: if set, points with predicted means less than this won't be sampled
    
    Keyword-only arguments:
    * kernel: the kernel to use for the GP model
    * sample_scale: how far to look for a new point to sample. If the
      configuration space is not unit-sized, you need to set this.
    * maximise_effort: how much computational effort to spend trying to
      maximise the acquisition function. Defaults to 100. Note that as more
      samples are added the process will slow down.
    * bounds: an optional list of n_dims tuples; each tuple is a pair of
      lower_bound and upper_bound, either of which may be None.
    * scale: an optional 1D array-like giving scale factors for the different
      dimensions. This can be used to compensate for an anisotropic parameter
      space. Each value passed into tell is divided by the scale, and each
      value provided by ask or optimum is multiplied by the scale. The bounds
      (if any) apply to the input (unscaled) values.

    Additional keyword arguments are passed to AugmentedGaussianProcess.
    """

    def __init__(self, n_dims, mean_cutoff=None, *,
            kernel=None, sample_scale=1, maximise_effort=100, bounds=None,
            scale=None, use_efficiency=True, **kwargs):
        self.n_dims = n_dims
        if kernel is None:
            kernel = 1.0 * kernels.RBF([1.0] * n_dims)
        self.model = AugmentedGaussianProcess(kernel, **kwargs)
        self.x_samples = []
        self.y_samples = []
        self.y_err_samples = []
        self.mean_cutoff = mean_cutoff
        self.sample_scale = sample_scale
        self.maximise_effort = maximise_effort
        self.use_efficiency = use_efficiency
        
        if bounds is None:
            bounds = [None]*n_dims
        if scale is not None:
            def scale_bound(b, s):
                if b is None:
                    return
                else:
                    return b / s

            def scale_bounds(b, s):
                if b is None:
                    return
                else:
                    return tuple(scale_bound(b1, s) for b1 in b)

            bounds = [scale_bounds(b, s) for b, s in zip(bounds, scale)]
        self.bounds = bounds
        if scale is None:
            scale = [1] * n_dims
        self.scale = np.asarray(scale)
        self.dirty = False
        
        self.Thresh = AcquisitionFunctionUCB(self.model, 2, invert=True)

    def tell(self, x, y, y_error):
        """Provide a sample to the optimiser.

        This doesn't have to match an earlier call to ask(), and in fact must
        be called at least once before ask().
        """

        self.x_samples.append(x)
        self.y_samples.append(y)
        self.y_err_samples.append(y_error)
        self.dirty = True

    def _fit(self):
        if self.dirty:
            self.dirty = False
            self.model.fit(np.asarray(self.x_samples) / self.scale,
                    np.asarray(self.y_samples), np.asarray(self.y_err_samples))
            _, best_val = self._maximise(self.model.predict)
            self.Acq = AcquisitionFunctionEI(self.model, best_val, use_efficiency=self.use_efficiency,
                    mean_cutoff=self.mean_cutoff)
    
    def random_vector(self):
        """Randomly and uniformly sample a vector on the surface of the unit hypersphere"""

        amplitude = 2
        while not 1e-3 < amplitude < 1:
            vec = np.random.uniform(-1, 1, self.n_dims)
            amplitude = np.sqrt(np.sum(vec**2))
        
        return vec / amplitude
    
    def _maximise(self, F):
        # Build up samples from lines through existing samples
        x = []
        for _ in range(self.maximise_effort):
            idx = np.random.choice(np.arange(len(self.x_samples)))
            x0 = self.x_samples[idx] / self.scale
            x_diff = self.random_vector()
            x_diff_amount = np.linspace(-2, 2, 200).reshape(-1, 1) * self.sample_scale
            x.append(x0 + x_diff * x_diff_amount)
        x = np.concatenate(x)
        if self.bounds is not None:
            for i in range(self.n_dims):
                bounds = self.bounds[i]
                if bounds is not None:
                    x[:, i] = np.clip(x[:, i], bounds[0], bounds[1])
        y = F(x)
        idx = np.nanargmax(y)
        return x[idx], y[idx]
    
    def ask(self, ei_cutoff, *, return_ei=False):
        """Returns the next point to sample, or None if the process has converged.

        Requires a convergence threshold to be passed. Smaller values mean the
        convergence will take place more slowly. Values approximately 1e-3
        times the true optimum seem to work well.

        This function is time-consuming to call and may return a different
        value each time, even without an intervening call to tell().
        """
        
        self._fit()

        max_pos, max_val = self._maximise(self.Acq)
        max_pos *= self.scale

        if max_val < ei_cutoff:
            max_pos = None

        if return_ei:
            return max_pos, max_val
        else:
            return max_pos

    def optimum(self):
        """Returns the best position found so far and an estimate of the mean there."""

        self._fit()

        best_pos, _ = self._maximise(self.Thresh)

        best_val = self.model.predict(best_pos.reshape(1, -1))[0]

        best_pos *= self.scale

        return best_pos, best_val


class BasicOptimiser_discrete:
    """A multidimensional optimiser that seems to work well in practice.

    Maximises the measured values, whatever they are. Requires that the
    configuration space is roughly isotropic.

    Arguments:
    * n_dims: the number of dimensions over which to optimise
    * mean_cutoff: if set, points with predicted means less than this won't be sampled
    
    Keyword-only arguments:
    * kernel: the kernel to use for the GP model
    * sample_scale: how far to look for a new point to sample. If the
      configuration space is not unit-sized, you need to set this.
    * maximise_effort: how much computational effort to spend trying to
      maximise the acquisition function. Defaults to 100. Note that as more
      samples are added the process will slow down.
    * bounds: an optional list of n_dims tuples; each tuple is a pair of
      lower_bound and upper_bound, either of which may be None.
    * scale: an optional 1D array-like giving scale factors for the different
      dimensions. This can be used to compensate for an anisotropic parameter
      space. Each value passed into tell is divided by the scale, and each
      value provided by ask or optimum is multiplied by the scale. The bounds
      (if any) apply to the input (unscaled) values.

    Additional keyword arguments are passed to AugmentedGaussianProcess.
    """

    def __init__(self, n_dims, mean_cutoff=None, *,
            kernel=None, sample_scale=1, maximise_effort=100, bounds=None,
            scale=None, use_efficiency=True, **kwargs):
        self.n_dims = n_dims
        if kernel is None:
            kernel = 1.0 * kernels.RBF([1.0] * n_dims)
        self.model = AugmentedGaussianProcess(kernel, **kwargs)
        self.x_samples = []
        self.y_samples = []
        self.y_err_samples = []
        self.mean_cutoff = mean_cutoff
        self.sample_scale = sample_scale
        self.maximise_effort = maximise_effort
        self.use_efficiency = use_efficiency
        
        if bounds is None:
            bounds = [None]*n_dims
        if scale is not None:
            def scale_bound(b, s):
                if b is None:
                    return
                else:
                    return b / s

            def scale_bounds(b, s):
                if b is None:
                    return
                else:
                    return tuple(scale_bound(b1, s) for b1 in b)

            bounds = [scale_bounds(b, s) for b, s in zip(bounds, scale)]
        self.bounds = bounds
        if scale is None:
            scale = [1] * n_dims
        self.scale = np.asarray(scale)
        self.dirty = False
        
        self.Thresh = AcquisitionFunctionUCB(self.model, 2, invert=True)

    def tell(self, x, y, y_error):
        """Provide a sample to the optimiser.

        This doesn't have to match an earlier call to ask(), and in fact must
        be called at least once before ask().
        """

        self.x_samples.append(x)
        self.y_samples.append(y)
        self.y_err_samples.append(y_error)
        self.dirty = True

    def _fit(self):
        if self.dirty:
            self.dirty = False
            self.model.fit(np.asarray(self.x_samples) / self.scale,
                    np.asarray(self.y_samples), np.asarray(self.y_err_samples))
            _, best_val = self._maximise(self.model.predict)
            self.Acq = AcquisitionFunctionEI(self.model, best_val, use_efficiency=self.use_efficiency,
                    mean_cutoff=self.mean_cutoff)
    
    def random_vector(self):
        """Randomly and uniformly sample a vector on the surface of the unit hypersphere"""

        amplitude = 2
        while not 1e-3 < amplitude < 1:
            vec = np.random.uniform(-1, 1, self.n_dims)
            amplitude = np.sqrt(np.sum(vec**2))
        
        return vec / amplitude
    
    def _maximise(self, F):
        # Build up samples from random selection of integers
        x = []
        
        for nD in range(self.n_dims):
            bounds = self.bounds[nD]
            m = np.min(bounds)
            r = np.max(bounds) - m
            x.append((np.random.rand(self.maximise_effort)*(r+1) + m-0.5).astype(int).reshape(-1,1))

        x = np.array(x).reshape(-1,self.n_dims)
        x = x.tolist()
        for x_s in self.x_samples:
            x2 = [xi for xi in x if not xi==x_s.tolist()]
            x = x2
        y = F(x)
        idx = np.nanargmax(y)
        return x[idx], y[idx]
    
    def ask(self, ei_cutoff, *, return_ei=False):
        """Returns the next point to sample, or None if the process has converged.

        Requires a convergence threshold to be passed. Smaller values mean the
        convergence will take place more slowly. Values approximately 1e-3
        times the true optimum seem to work well.

        This function is time-consuming to call and may return a different
        value each time, even without an intervening call to tell().
        """
        
        self._fit()

        max_pos, max_val = self._maximise(self.Acq)
        max_pos *= self.scale

        if max_val < ei_cutoff:
            max_pos = None

        if return_ei:
            return max_pos, max_val
        else:
            return max_pos

    def optimum(self):
        """Returns the best position found so far and an estimate of the mean there."""

        self._fit()

        best_pos, _ = self._maximise(self.Thresh)

        best_val = self.model.predict(np.array(best_pos).reshape(1, -1))[0]

        best_pos *= self.scale

        return best_pos, best_val
