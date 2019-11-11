import pymc3 as pm
import numpy as np
import theano.tensor as tt
# Matplotlib and seaborn for visualization
import matplotlib.pyplot as plt
import scipy
from statsmodels.stats.outliers_influence import variance_inflation_factor

"""
IMPORTANT NOTES:
1) The variance of the parameters was previously samples from a half normal. This made is slower and harder to see convergence
INSTEAD, when fixing sigma /sd = 0.5, the convergence was faster and better.
"""





class bayesian_linear_model:
    def __init__(self, param_samples = 2000, chains=None):
        self.param_samples = param_samples
        self.chains = chains
        self.full_param_trace = None
        self.linear_params_values = None
        self.beta_params = None
        self.alpha_param = None

    #==========================================
    def set_normal_distr_params(self,num_chains=3,num_last_samples = None):
        """

        :param list_np_arrays:
        :return:
        """
        if num_last_samples == None:
            np_2d_array = self.full_param_trace["betas"]
        else:
            total_num_samples = self.full_param_trace["betas"].shape[0]
            single_chain_size = int(total_num_samples/num_chains)
            np_2d_array = self.full_param_trace["betas"][0:single_chain_size,:]
            for i in range(2,num_chains+1):
                np_2d_array = np.vstack((np_2d_array,self.full_param_trace["betas"][(i-1)*single_chain_size:i*single_chain_size,:]))
        #end else

        mean = np.mean(np_2d_array, axis=0)
        variance = np.var(np_2d_array, axis=0)
        cov_matx = np.diag(variance)
        self.beta_params = np.random.multivariate_normal(mean, cov_matx, self.param_samples)
        np_1d_array = self.full_param_trace["alpha"]
        mean = np.mean(np_1d_array)
        sd_dev = np.std(np_1d_array)
        self.alpha_param = np.random.normal(mean, sd_dev, self.param_samples)

    #===================================================================

    def learn_bayesian_linear_model(self,
                                    encoded_plans,
                                    prior_weights,
                                    number_of_dimensions,
                                    sd = 1,
                                    sampling_count=2000,
                                    num_chains = 3,
                                    bias_preference = 0.0):

        #the encoded plans contains a list of [<encoding>,<rating>]
        input_dataset = np.array([x[0] for x in encoded_plans],dtype=np.float)
        output_dataset = np.array([x[1]  for x in encoded_plans],dtype=np.float)

        # try:
        #     vif = [variance_inflation_factor(input_dataset, i) for i in range(input_dataset.shape[1])]
        #     print(vif)
        #     print("number of entries in vif should match num discovered features")
        # except:
        #     print("Error during VIF Variance Inflation Factor computation")

        #TODO USE SAME MODEL AND TEST ON DUMMY DATA WITH CLEARLY KNOWN FUNCTION
        # maybe it is ok that it does not converge, but works with metropolis sampling. Expected? in early stages
        bias_preference = tt.constant(bias_preference)
        #todo Make bias A  learnable parameter
        with pm.Model() as linear_model:
            # Intercept
            # alpha = pm.Normal('alpha', mu=0.0, sd=sd)
            alpha = pm.Deterministic('alpha', bias_preference)
            cov = np.diag(np.full((number_of_dimensions,), sd)) #for both mu and beta (slope)
            #todo add support to have the covariance of unknown features to be much larger ? SD = 1.0 is enough !!
            # Slope
            prior_weights = np.random.rand(number_of_dimensions)
            betas = pm.MvNormal('betas', mu=prior_weights, cov=cov, shape=(number_of_dimensions,))
            # Standard deviation
            sigma = pm.HalfNormal('sigma', sd=sd)
            # sigma = sd #unfair knowledge
            # Estimate of mean
            mean = alpha + tt.dot(input_dataset, betas)
            # Observed values
            Y_obs = pm.Normal('Y_obs', mu=mean, sd=sigma, observed=output_dataset)
            # Sampler
            step = pm.NUTS()
            # step = pm.Metropolis()
            # step = pm.HamiltonianMC()
            # Posterior distribution
            linear_params_trace= pm.sample(sampling_count, step,chains=num_chains,cores=num_chains)

            #todo NOTE do not add tuning if deterministic. Fails spectacularly, not it's intended use.
            #todo note: may consider making mu and cov as parameters sampled from distributions too
            # mu = pm.MvNormal('mu', mu=prior_weights, cov=cov, shape=(number_of_dimensions,))
        #end with
        # todo look into the aplha values that were sampled, because they didn't appear in the plot
        self.full_param_trace = linear_params_trace # we only take the last 2000, and assume it is after sufficient mixing and good values.
        self.linear_params_values = linear_params_trace[-2000:] # we only take the last 2000, and assume it is after sufficient mixing and good values.
        self.set_normal_distr_params(num_chains=num_chains,num_last_samples=None)
    #---end function learn_linear_model

    #===================================================================
    # def OLD_learn_bayesian_linear_model(self,
    #                                 encoded_plans,
    #                                 prior_weights,
    #                                 number_of_dimensions,
    #                                 sd = 1,
    #                                 sampling_count=2000,
    #                                 num_chains = 2,
    #                                 bias_preference = 0.0):
    #
    #     #the encoded plans contains a list of [<encoding>,<rating>]
    #     input_dataset = np.array([x[0] for x in encoded_plans],dtype=np.float)
    #     output_dataset = np.array([x[1]  for x in encoded_plans],dtype=np.float)
    #
    #     #TODO USE SAME MODEL AND TEST ON DUMMY DATA WITH CLEARLY KNOWN FUNCTION
    #     # maybe it is ok that it does not converge, but works with metropolis sampling. Expected? in early stages
    #     bias_preference = tt.constant(bias_preference)
    #     with pm.Model() as linear_model:
    #         # Intercept
    #         # alpha = pm.Normal('alpha', mu=0.5, sd=sd)
    #         alpha = pm.Deterministic('alpha', bias_preference)
    #
    #         cov = np.diag(np.full((number_of_dimensions,), sd)) #for both mu and beta (slope)
    #         #todo note: may consider making mu and cov as parameters sampled from distributions too
    #         # mu = pm.MvNormal('mu', mu=prior_weights, cov=cov, shape=(number_of_dimensions,))
    #
    #         # Slope
    #         prior_weights = np.random.rand(number_of_dimensions)
    #         betas = pm.MvNormal('betas', mu=prior_weights, cov=cov, shape=(number_of_dimensions,))
    #
    #         # Standard deviation
    #
    #         sigma = pm.HalfNormal('sigma', sd=sd)
    #         # sigma = sd #seems to work better
    #
    #         # Estimate of mean
    #         mean = alpha + tt.dot(input_dataset, betas)
    #
    #         # Observed values
    #         Y_obs = pm.Normal('Y_obs', mu=mean, sd=sigma, observed=output_dataset)
    #
    #         # Sampler
    #         step = pm.NUTS()
    #         # step = pm.Metropolis()
    #         # step = pm.HamiltonianMC()
    #
    #         # Posterior distribution
    #         linear_params_trace= pm.sample(sampling_count, step,chains=num_chains,cores=num_chains) #todo NOTE do not add tuning if deterministic. Fails spectacularly, not it's intended use.
    #     #end with
    #     # todo look into the aplha values that were sampled, because they didn't appear in the plot
    #     self.full_param_trace = linear_params_trace # we only take the last 2000, and assume it is after sufficient mixing and good values.
    #     self.linear_params_values = linear_params_trace[-2000:] # we only take the last 2000, and assume it is after sufficient mixing and good values.
    #

    #---end function learn_linear_model

    #========================================================================

    def plot_learnt_parameters(self):
        """
        :summary: plot the parameters
        :return:
        """
        pm.traceplot(self.full_param_trace, figsize=(12, 12))
        plt.show()

    # ========================================================================

    def get_outputs_and_kernelDensityEstimate(self,input_encoded_plan, num_samples = 500):
        """
        :summary :
        :param input_encoded_plan:
        :return:
        """
        # all_outputs = self.linear_params_values["alpha"] + np.matmul(self.linear_params_values["betas"],np.array(input_encoded_plan))
        all_outputs = self.alpha_param + np.matmul(self.beta_params,np.array(input_encoded_plan))

        #the intercept alpha is known to be 0.

        if num_samples > all_outputs.shape[0]:
            num_samples = all_outputs.shape[0]
        #todo NOTE this was changed to sample the last n parameter points
        outputs = np.random.choice(all_outputs,num_samples, replace=False)
        return outputs
        # try:
        #     kernel = scipy.stats.gaussian_kde(outputs)
        # except:
        #     #an error can be thrown when the covariance matrix computed is a singular matrix.
        #     #if the encodings is all zeros (no features), then the output could be all 0 as well, leading to problems
        #     # print(" Singular covariance matrix error, the outputs are")
        #     # print(outputs)
        #     singular_value = outputs[0]
        #     #then density is 1.0 for y = default bias value. Which is the value we want to output when there are no features
        #     current_bias = np.mean(self.alpha_param)
        #     kernel = lambda x: np.array([1.0*int(y == current_bias) for y in x])
        # return outputs,kernel


#---end class bayesian linear model



