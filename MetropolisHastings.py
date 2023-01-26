from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from scipy.io import savemat
from scipy.stats import norm as normal
import time as time
import corner as corner

class MetropolisHastings:
    '''
    Authors (Please put your authorage here if you make changes):
    Michael-David Johnson
    '''
    def __init__(self, initial_parameters, model, data, prior_function,
                   proposal_distribution,
                   likelihood_function, measurement_error, epochs, 
                   burn_in=False, return_MAP=False, adaptive=False, adaptive_delay=False,
                   extra_conditions = False, log_level = False, logs = False,
                   targeted_acceptance_rate = 0.2, adaptive_multiplier = 1.02, external_model = True,
                   optimizer = False):
        '''
        Init function for a general Metropolis Hastings sampler. Scalable for 
        n parameter estimation.
        Inputs:
        initial_parameters: Starting points of parameters of interest (list)
        model: The underlying model of which the parameters use. Set to False if the likelihood
               function already contains an underlying model (function)
        data: The measurement data to compare the model with.
        prior_function: A list of any given prior function, alongside it's means and stds
        proposal_distribution: The distribution of which to move the parameters, alongside it's means and stds
        likelihood_function: The likelihood function used to make the posterior calculation.
        measurement_error: The assumed error between the model and the data.
        epochs: Number of iterations
        burn_in: int or False, automatically throw away the first int samples.
        return_MAP: Return the Maximum a posteriori estimation.
        adaptive: If True, the proposal distributions means and stds are dictated by an adaptive scheme. 
        extra_conditions: Takes a function as argument in order to throw away non-representative / non-physical samples
        log_level: Set to avoid prints. 
        logs: Set if the user has used log priors / log likelihoods
        targeted_acceptance_rate: User defined acceptance rate for the chain
        adaptive_multiplier: Affects the severity of the adaptive width.  
        optimizer: Decides whether to remove the random number generation for the posterior updater.
        '''

        self.initial_parameters = np.array(initial_parameters)
        self.model = model
        self.data = data
        self.prior_function = prior_function[0]
        self.prior_means = prior_function[1]
        self.prior_stds = prior_function[2]
        self.proposal_distribution = proposal_distribution[0]
        self.proposal_move_means = proposal_distribution[1]
        self.proposal_move_stds = proposal_distribution[2]
        self.likelihood_function = likelihood_function
        self.measurement_error = measurement_error
        self.epochs = epochs
        self.burn_in = burn_in
        self.return_MAP = return_MAP
        self.adaptive = adaptive
        self.extra_conditions = extra_conditions
        self.MAP = [0,0]
        self.parameter_store = []
        self.success = 1
        self.fail = 0
        self.acceptance = 1
        self.log_level = log_level
        self.logs = logs #Flag to tell if using log likelihood / log prior
        self.targeted_acceptance_rate = targeted_acceptance_rate
        self.adaptive_delay = adaptive_delay
        self.adaptive_multiplier = adaptive_multiplier
        self.optimizer = optimizer
        self.acceptance_array = []
        self.gamma_array = []
        self.posterior_array = []
        self.cholc_array = []
        
    def run(self):
        if self.adaptive == 'Gradient':
            self.run_adaptive_grad()
        elif self.adaptive == 'AM':
            self.run_adaptive_AM()
        else:
            self.run_normal()
  
    def run_adaptive_AM(self):
        '''
        Based from "An adaptive Metropolis algorithm. Haario et. al. 2001.
        Need some citations for why we can deal with the cholesky, as well as the constants
        '''

        self.gamma = 2.38**2 / np.size(self.initial_parameters) #Looks like slightly smaller sd from Haario.
        self.gamma_array.append(self.gamma)
        self.cholC = np.diag(self.proposal_move_stds) * self.gamma #Using user proposal stds for intial CholC
        self.cholc_array.append(self.cholC)
        nvars = np.size(self.initial_parameters)
        for _ in range(self.burn_in):
            initial_posterior = self.generate_posterior(self.initial_parameters)
            proposed_position = self.initial_parameters + (np.matmul(self.cholC.T, np.random.randn(nvars)))
            if not self.extra_conditions == False:
                for cri in self.extra_conditions:
                    while cri(proposed_position) == False:
                        proposed_position = self.initial_parameters + (np.matmul(self.cholC.T,
                                                                         np.random.randn(nvars)))
            
                        '''
                        There is some discussion about the validity of using this 'rerolling'
                        based approach for when the initial parameter is not stored into the array.
                        Originally, I would reroll the proposed_position for positions of theta
                        outside of the valid subspace without storing the initial position in the storage.
                        This was analagous to only being able to explore the subspace with a markov chain.
                        The issue is, this behaviour can potentially skew the proposal distribution, making 
                        it non-Markovian. 
                        By appending to the parameter store the initial_parameters, this is analagous to 
                        our criteria being 0 ie either the likelihood function or the prior is 0. This appears
                        to make the most sense for now.
                        '''
                        self.fail = self.fail + 1
                        self.parameter_store.append(self.initial_parameters)
            self.logic(proposed_position, initial_posterior)

        #First initial cholesky covariance matrix
        self.covariance_updater(0)
        count = 1 
        for _ in range(self.epochs - self.burn_in):
            if count % self.adaptive_delay == 0:
                self.covariance_updater((self.burn_in + count) - self.adaptive_delay)
      
            initial_posterior = self.generate_posterior(self.initial_parameters)
            proposed_position = self.initial_parameters + (np.matmul(self.cholC.T, np.random.randn(nvars)))
      
            if not self.extra_conditions == False:
                for cri in self.extra_conditions:
                    while cri(proposed_position) == False:
                        proposed_position = self.initial_parameters + (np.matmul(self.cholC.T,
                                                                     np.random.randn(nvars)))
                        self.parameter_store.append(self.initial_parameters)
                        self.fail = self.fail + 1
                        print("I'm stuck!")
                        print(proposed_position)
                        clear_output(wait=True)
            self.logic(proposed_position, initial_posterior)
            count = count + 1
            if self.log_level == 1:
                print(f'Cholesky width: {self.cholC}')
  
    def covariance_updater(self, indicies):
        '''
        Function that updates the cholesky factor, while slighty changing gamma 
        which should point the acceptance rate to the user defined acceptance rate.
        '''
        self.gamma = self.gamma * (self.adaptive_multiplier ** (self.acceptance - self.targeted_acceptance_rate))
        self.gamma_array.append(self.gamma)
        epsilon = 1e-10 #Very small number from Haario.
        array_slice = np.array(self.parameter_store)[indicies:]
        covariance = np.cov(array_slice.T)
        #This should be the correct definiton from Haario.
        self.cholC = self.gamma * np.linalg.cholesky(covariance + epsilon * np.eye(np.size(self.initial_parameters)))
        self.cholc_array.append(self.cholC)
        if self.log_level == 2:
            print(array_slice)
            print(f'Gamma: {self.gamma}')
            print(f'Cov: {covariance}')
            print(f'Cholesky: {self.cholC}')
            clear_output(wait=True)
            time.sleep(5)

    def run_adaptive_grad(self):
        '''
        Based from https://arxiv.org/pdf/1911.01373.pdf, under the special case of 
        the proposal distribution being a normal distribution. Although this
        could be modified for more proposal distributions
        '''
    
    def run_normal(self):
        '''
        The main Metropolis-Hastings logic, this will use all the information given 
        from init to generate samples from the posterior.
        '''
        for _ in range(self.epochs):
            initial_posterior = self.generate_posterior(self.initial_parameters)
            proposed_position = self.initial_parameters + self.move()

            if not self.extra_conditions == False:
                for cri in self.extra_conditions:
                    while cri(proposed_position) == False:
                        proposed_position = np.array(self.initial_parameters) + self.move()

            self.logic(proposed_position, initial_posterior)
        print("Running has finished")
        
    def _sort_ax(self, ax, x_label = '', y_label = '', title = '', legend = True, **legend_args):
        # utility function for plots
        ax.grid('on')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title, loc = 'left') #, fontdict = {'fontweight' : 'demi'})
        if legend:
            ax.legend(**legend_args)

    def plot_traces(self, parameter_indexes = [], parameter_names = [], 
                        ax = None, max_rows = 3, title = 'Parameter evolution',
                        show_markers = None,
                        return_fig = False):
        '''
        Return a trace plot for each parameter of interest or a specific list of parameter indices
        Parameters
        ----------
        self : MetropolisHastings Object
            Should have run before attempting plot
        parameter_indexes : list, optional
            List of parameters to plot (labelled by index). The default is [].
            If empty plots all
        parameter_names : list, optional
            Names of parameters to plot (in order of indexes). The default is [].
            If empty labels by index
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            Axis to add graph to. The default is None.
            If None creates new figure and axis.
            Cannot be used if number of parameters to plot is > 1
        max_rows : int, optional
            Maximum number of rows in a figure before creating columns. 
            The default is 3.
        title : string, optional
            Title for plot.
        show_markers : bool, optional
            Defines whether to show markers on scatter plot
            If None only shows for small number of epochs.
        return_fig : bool, optional
            Switch for whether fig, axs are returned. The default is False.
        Raises
        ------
        ValueError
        Returns
        -------
        fig, axs if return_fig
        None, None if not return_fig
        '''
        if type(parameter_indexes) == int:
            parameter_indexes = [parameter_indexes]
            if type(parameter_names) == str:
                parameter_names = [parameter_names]  
        elif len(parameter_indexes) == 0:
            parameter_indexes = range(len(self.initial_parameters))
        n = len(parameter_indexes)
        
        if len(parameter_names) == 0:
            parameter_names = ['param ' + str(j) for j in parameter_indexes]
        elif len(parameter_names) != len(parameter_indexes):
            raise ValueError("Invalid parameter_names passed to plot_traces: should be same length as parameter list (parameter_indexes)")
        
        if any(parameter_indexes) >= len(self.initial_parameters):
            raise ValueError("Invalid parameter_index passed to plot_traces - index greater than number of parameters (%d)" % len(self.initial_parameters))
        
        # sort out axes
        num_cols, num_rows, fig = 1, 1, None
        if (n > 1) and ax:
            raise ValueError("Cannot use existing axis if plotting more than one parameter")
        if not ax:
            if n > 1:
                if n > max_rows:
                    num_cols = int(np.ceil(n/max_rows))
                    num_rows = max_rows
                else:
                    num_cols = 1
                    num_rows = n
            fig, ax = plt.subplots(num_rows, num_cols, sharex = True)
          
        axs = np.array(ax).flatten() # axs numbers from L-R, T-B
        
        # plot parameter store values
        parameter_store_by_index = np.array(self.parameter_store).T
        
        if ((show_markers is None) and (self.epochs < 200)) or (show_markers == True):
            marker = 'x'
        else:
            marker = None
        
        j = -1
        for p in parameter_indexes:
            j += 1
            axs[j].plot(range(self.burn_in+1), parameter_store_by_index[p][:self.burn_in+1], 
                        c = 'gray',
                        marker = marker, markersize = 5,
                        label = 'pre burn-in')
            axs[j].plot(range(self.burn_in, len(parameter_store_by_index[p])), parameter_store_by_index[p][self.burn_in:], 
                        c = 'C' + str(j), 
                        marker = marker, markersize = 5,
                        label = parameter_names[j])
            y_lim = axs[j].get_ylim()
            if self.burn_in > 0:
                axs[j].plot([self.burn_in, self.burn_in], y_lim,
                        c = 'k', lw = 0.8, ls = '--',
                        label = 'burn-in cutoff')
            #axs[j].annotate('burn in', xy = [self.burn_in*1.05, y_lim[0] + 0.9 * (y_lim[1]-y_lim[0])])
            axs[j].set_ylim(y_lim)
            if j >= (len(parameter_indexes) - num_cols):
                self._sort_ax(axs[j], x_label = 'epoch', y_label = 'value',
                        title = parameter_names[j], legend = False)
            else:
                self._sort_ax(axs[j], y_label = 'value',
                        title = parameter_names[j], legend = False)
        
        if fig:
            fig.suptitle(title)
            fig.tight_layout()
        
        if return_fig:
            return fig, axs
        else:
            return None, None

  
    def plot_hists(self, parameter_indexes = [], parameter_names = [], 
                        n_bins = None,
                        ax = None, max_rows = 3, title = 'Posterior Distributions of Parameters',
                        return_fig = False):
        '''
        Return the histograms for each parameter of interest or for a specific list of parameter indices
        Parameters
        ----------
        self : MetropolisHastings Object
            Should have run before attempting plot
        parameter_indexes : list, optional
            List of parameters to plot (labelled by index). The default is [].
            If empty plots all
        parameter_names : list, optional
            Names of parameters to plot (in order of indexes). The default is [].
            If empty labels by index
        n_bins : int, optional
            Number of bins for histogram.
            If None uses epochs * 0.2
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            Axis to add graph to. The default is None.
            If None creates new figure and axis.
            Cannot be used if number of parameters to plot is > 1
        max_rows : int, optional
            Maximum number of rows in a figure before creating columns. 
            The default is 3.
        title : string, optional
            Title for plot.
        return_fig : bool, optional
            Switch for whether fig, axs are returned. The default is False.
        Raises
        ------
        ValueError
        Returns
        -------
        fig, axs if return_fig
        None, None if not return_fig
        '''
        
        # input validation/sorting
        
        if type(parameter_indexes) == int:
            parameter_indexes = [parameter_indexes]
            if type(parameter_names) == str:
                parameter_names = [parameter_names]  
        elif len(parameter_indexes) == 0:
            parameter_indexes = range(len(self.initial_parameters))
        n = len(parameter_indexes)
        
        if len(parameter_names) == 0:
            parameter_names = ['param ' + str(j) for j in parameter_indexes]
        elif len(parameter_names) != len(parameter_indexes):
            raise ValueError("Invalid parameter_names passed to plot_hists: should be same length as parameter list (parameter_indexes)")
        
        if any(parameter_indexes) >= len(self.initial_parameters):
            raise ValueError("Invalid parameter_index passed to plot_hists - index greater than number of parameters (%d)" % len(self.initial_parameters))
        
        if not n_bins:
            n_bins = int(self.epochs * 0.2)
        
        # sort out axes
        num_cols, num_rows, fig = 1, 1, None
        if (n > 1) and ax:
            raise ValueError("Cannot use existing axis if plotting more than one parameter")
        if not ax:
            if n > 1:
                if n > max_rows:
                    num_cols = int(np.ceil(n/max_rows))
                    num_rows = max_rows
                else:
                    num_cols = 1
                    num_rows = n
            fig, ax = plt.subplots(num_rows, num_cols)
          
        axs = np.array(ax).flatten() # axs numbers from L-R, T-B
        
        # plot parameter store values
        parameter_store_by_index = np.array(self.parameter_store).T
        
        j = -1
        for p in parameter_indexes:
            j += 1
            axs[j].hist(parameter_store_by_index[p], density = True, bins = n_bins, 
                        color = 'C' + str(j), alpha = 0.8, edgecolor = 'C' + str(j))
            self._sort_ax(axs[j], x_label = 'value', 
                    y_label = 'count', title = parameter_names[j], legend = False)
        
        if fig:
            fig.suptitle(title)
            fig.tight_layout()
        
        if return_fig:
            return fig, axs
        else:
            return None, None


    def plot_corner(self, i = None, n_bins = None, grid = False, 
                    show_ylabel = False, hist_same_scale = False,
                    return_fig = False):
        '''
        plots covariances between each histogram
        uses module corner.py
        citation:
         @article{corner,
          doi = {10.21105/joss.00024},
          url = {https://doi.org/10.21105/joss.00024},
          year  = {2016},
          month = {jun},
          publisher = {The Open Journal},
          volume = {1},
          number = {2},
          pages = {24},
          author = {Daniel Foreman-Mackey},
          title = {corner.py: Scatterplot matrices in Python},
          journal = {The Journal of Open Source Software}
        }
        '''

        if not n_bins:
            n_bins = self.epochs/10
            
        parameter_store_by_index = pd.DataFrame(self.parameter_store)
    
        fig = corner.corner(parameter_store_by_index, bins = n_bins, 
                            show_titles = False, plot_contours = False,
                            fill_contours = True)
        
        n = len(self.initial_parameters)
        axs = np.array(fig.axes).reshape((n,n))
        if grid:
            for ax in axs.flatten():
                ax.grid()
                
        if show_ylabel:
            axs[0][0].set_ylabel('count')                
            for i in range(n):
                axs[i][i].set_yscale('linear')
                if hist_same_scale and (i != 0):
                    axs[i][i].set_yticklabels([])        
                
        if hist_same_scale:
            min_ylim = 1000
            max_ylim = 0
            for i in range(n):
                ylim = axs[i][i].get_ylim()
                if ylim[0] < min_ylim:
                    min_ylim = ylim[0]
                if ylim[1] > max_ylim:
                    max_ylim = ylim[1]
            for i in range(n):
                axs[i][i].set_ylim(min_ylim, max_ylim)    

        if show_ylabel and not hist_same_scale:            
            fig.subplots_adjust(left=0.125, bottom=0.1, right=0.95, top=0.95, wspace=0.2, hspace=0.075)
        elif show_ylabel:
            fig.subplots_adjust(left=0.125, bottom=0.1, right=0.95, top=0.95, wspace=0.075, hspace=0.075)
        
        if return_fig:
            return fig
        return None
  
    def move(self):
        '''
        Moves as per the proposal distribution.
        '''
        move = [] 
        for i in range(len(self.proposal_move_means)):
            move.append(self.proposal_distribution(self.proposal_move_means[i],
                                                 self.proposal_move_stds[i]))
        return move
  
    def generate_posterior(self,position):
        '''
        Generates the posterior at a given position, given the data and the measurement error.
        '''
        if self.model == False or self.model == None:
            if self.measurement_error == False:
                likelihood = self.likelihood_function(position[0:-1], self.data,
                                                      position[-1])
            
            else:
                likelihood = self.likelihood_function(position, self.data,
                                                 self.measurement_error) 
        else:
            if self.measurement_error == False:
                likelihood = self.likelihood_function(self.model(position[0:-1]), self.data,
                                                   position[-1])
                
            else:
                likelihood = self.likelihood_function(self.model(position), self.data,
                                                  self.measurement_error)
        prior = self.prior_function(position, self.prior_means, self.prior_stds)
        if self.log_level == 1:
            print(f"Likelihood: {likelihood}")
            print(f"Prior: {prior}")
        if self.logs: #If given the log likelihood and the log of the prior
            return likelihood + prior
        else:
            return likelihood * prior
  
    def logic(self, proposed_position, initial_posterior):
        '''Pretty much all the adaptive sampling end up using the same logic for
           acceptance and rejection, so having the logic separate appears to make sense.
        '''
        proposed_posterior = self.generate_posterior(proposed_position)
        if not self.optimizer:
            if self.return_MAP:
                if -np.log(proposed_posterior) < -np.log(initial_posterior):
                    self.MAP = [-np.log(proposed_posterior), proposed_position]
      
            if not self.logs:
                criteria = np.exp( np.log(proposed_posterior) - np.log(initial_posterior)) #This is sometimes giving NaNs...
                monte_carlo = np.random.uniform()
      
            if self.logs:
                criteria = proposed_posterior - initial_posterior
                monte_carlo = np.log(np.random.uniform())
        else:
            if np.abs(proposed_posterior) > 1e16:
                criteria = 0
                
            elif proposed_posterior >= initial_posterior:
                criteria = 1
            else:
                criteria = 0.005 #May need to be changed
                
            monte_carlo = np.random.uniform()
        
        if self.log_level == 1:
            print(f"{proposed_position}")
            print(f"{initial_posterior}")
            print(f"{monte_carlo}")
            print(f"{criteria}")
                
        if monte_carlo > criteria:
            self.initial_parameters = self.initial_parameters
            self.parameter_store.append(self.initial_parameters)
            self.posterior_array.append(initial_posterior)
            self.fail = self.fail + 1
      
        elif monte_carlo < criteria:
            self.initial_parameters = proposed_position
            self.parameter_store.append(self.initial_parameters)
            self.posterior_array.append(proposed_posterior)
            
            self.success = self.success + 1 

        else:
            self.initial_parameters = self.initial_parameters
            self.parameter_store.append(self.initial_parameters)
            self.posterior_array.append(initial_posterior)
            self.fail = self.fail + 1
      
        self.acceptance = self.success / (self.success + self.fail)
        self.acceptance_array.append(self.acceptance)
        
        if self.log_level == 1:
            clear_output(wait=True)
            print(f"Parameters: {self.initial_parameters}")
            print(f"Acceptance: {self.acceptance}")
            print(f"Criteria: {criteria}")
            print(f"Monte-Carlo: {monte_carlo}")
      
    def save(self, file_name = None, full = False):        
        if not file_name:
            file_name = 'MetropolisHastings_' + pd.to_datetime('today').strftime('%Y%m%d-%H%M%S') + '.csv'
        
        export_df = pd.DataFrame(self.parameter_store, 
                                 columns = ['param %d' % i for i in range(len(self.initial_parameters))])
        export_df.to_csv(file_name)
        
        if full:
            dicty = {'Posterior':self.posterior_array,
                    'Params':self.parameter_store,
                    'Gamma':self.gamma_array,
                    'Acceptancs':self.acceptance_array}
            savemat(f"{file_name}_full.mat",dicty)
        
        if self.optimizer:
            savemat(f"{file_name}_Optimizer_files.mat",{'Posterior':self.posterior_array})
        print("Saved to %s" % file_name)
        
    def load_parameter_store(self, file_name):
        import_df = pd.read_csv(file_name, index_col = 0)
        
        self.parameter_store = import_df.values
        self.epochs = len(import_df)
        self.initial_parameters = self.parameter_store[0]
        
        print("Loaded")

    #Getters and setters, these are probably never going to be used, but it 
    #could be useful i.e. changing prior std through the Metropolis-Hastings.
    #Allowing for the adaptive scheme. 

    def set_initial_parameters(self, initial_parameters):
        self.initial_parameters = initial_parameters

    def get_initial_parameters(self):
        return self.initial_parameters

    def set_model(self, model):
        self.model = model
    
    def get_model(self):
        return self.model

    def set_data(self, data):
        self.data = data
    
    def get_data(self):
        return self.data

    def set_prior_function(self, prior_function):
        self.prior_function = prior_function

    def get_prior_function(self):
        return self.prior_function

    def set_prior_means(self, prior_means):
        self.prior_means = prior_means
    
    def get_prior_means(self):
        return self.prior_means

    def set_prior_stds(self, prior_stds):
        self.prior_stds = prior_stds
    
    def get_prior_stds(self):
        return self.prior_stds

    def set_proposal_distribution(self, proposal_distribution):
        self.proposal_distribution = proposal_distribution
    
    def get_proposal_distribution(self):
        return self.proposal_distribution

    def set_proposal_move_means(self, proposal_move_means):
        self.proposal_move_means = proposal_move_means

    def get_proposal_move_means(self):
        return self.proposal_move_means

    def set_proposal_move_stds(self, proposal_move_stds):
        self.proposal_move_stds = proposal_move_stds

    def get_proposal_move_stds(self):
        return self.proposal_move_stds

    def set_likelihood_function(self, likelihood_function):
        self.likelihood_function = likelihood_function
    
    def get_likelihood_function(self):
        return self.likelihood_function

    def set_measurement_error(self, measurement_error):
        self.measurement_error = measurement_error

    def get_measurement_error(self):
        return self.measurement_error

    def set_epochs(self, epochs):
        self.epochs = epochs

    def get_epochs(self):
        return self.epochs

    def set_burn_in(self, burn_in):
        self.burn_in = burn_in

    def get_burn_in(self):
        return self.burn_in

    def set_return_MAP(self, return_MAP):
        self.return_MAP = return_MAP

    def get_return_MAP(self):
        return self.return_MAP

    def set_adaptive(self, adaptive):
        self.adaptive = adaptive 

    def get_adaptive(self):
        return self.adaptive

    def set_extra_conditions(self, extra_conditions):
        self.extra_conditions = extra_conditions

    def get_extra_conditions(self):
        return self.extra_conditions

    def set_MAP(self, MAP):
        self.MAP = MAP
  
    def get_MAP(self):
        return self.MAP
  
    def set_parameter_score(self, parameter_score):
        self.parameter_score = parameter_score
  
    def get_parameter_score(self):
        return self.parameter_score

    def set_success(self, success):
        self.success = success
  
    def get_success(self):
        return self.success
  
    def set_fail(self, fail):
        self.fail = fail

    def get_fail(self):
        return self.fail
  
    def set_acceptance(self, acceptance):
        self.acceptance = acceptance
  
    def get_acceptance(self):
        return self.acceptance

    def set_log_level(self, log_level):
        self.log_level = log_level

    def get_log_level(self):
        return self.log_level
  
    def set_logs(self,logs):
        self.logs = logs

    def get_logs(self):
        return self.logs

    def set_targeted_acceptance_rate(self, targeted_acceptance_rate):
        self.targeted_acceptance_rate = targeted_acceptance_rate

    def get_targeted_acceptance_rate(self):
        return self.targeted_acceptance_rate

    def set_adaptive_delay(self, adaptive_delay):
        self.adaptive_delay = adaptive_delay

    def get_adaptive_delay(self):
        return self.adaptive_delay

    def set_adaptive_multiplier(self, adaptive_multiplier):
        self.adaptive_multiplier = adaptive_multiplier
    
    def get_adaptive_multiplier(self):
        return self.adaptive_multiplier
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    def get_optimizer(self):
        return self.optimizer
