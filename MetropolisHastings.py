from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp

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
               targeted_acceptance_rate = 0.2, adaptive_multiplier = 1.02):
    '''
    Init function for a general Metropolis Hastings sampler. Scalable for 
    n parameter estimation.

    Inputs:
    initial_parameters: Starting points of parameters of interest (list)
    model: The underlying model of which the parameters use. (function)
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
    
    #Need getters / setters
    self.logs = logs #Flag to tell if using log likelihood / log prior
    self.targeted_acceptance_rate = targeted_acceptance_rate
    self.adaptive_delay = adaptive_delay
    self.adaptive_multiplier = adaptive_multiplier

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
    self.cholC = np.diag(self.proposal_move_stds) * self.gamma #Using user proposal stds for intial CholC
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
    epsilon = 1e-10 #Very small number from Haario.
    array_slice = np.array(self.parameter_store)[indicies:]
    covariance = np.cov(array_slice.T)
    #This should be the correct definiton from Haario.
    self.cholC = self.gamma * np.linalg.cholesky(covariance + epsilon * np.eye(np.size(self.initial_parameters)))
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
                        label = parameter_names[j])
            axs[j].plot(range(self.burn_in, len(parameter_store_by_index[p])), parameter_store_by_index[p][self.burn_in:], 
                        c = 'C' + str(j), 
                        marker = marker, markersize = 5,
                        label = parameter_names[j])
            y_lim = axs[j].get_ylim()
            axs[j].plot([self.burn_in, self.burn_in], y_lim,
                        c = 'k', lw = 0.8, ls = '--',
                        label = 'burn in')
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

  def plot_corner(self, i = None):
    '''
    plots covariances between each histogram

    uses module corner.py
    '''
  
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

    if self.return_MAP:
      if -np.log(proposed_posterior) < -np.log(initial_posterior):
        self.MAP = [-np.log(proposed_posterior), proposed_position]
      
    if not self.logs:
      criteria = np.exp( np.log(proposed_posterior) - np.log(initial_posterior)) #This is sometimes giving NaNs...
      monte_carlo = np.random.uniform()
      
    if self.logs:
      criteria = proposed_posterior - initial_posterior
      monte_caelo = np.log(np.ranfom.uniform)

    if self.log_level == 1:
      print(f"{proposed_position}")
      print(f"{initial_posterior}")
      print(f"{monte_carlo}")
      print(f"{criteria}")
      

    if monte_carlo > criteria:
      self.initial_parameters = self.initial_parameters
      self.parameter_store.append(self.initial_parameters)
      self.fail = self.fail + 1
      
    elif monte_carlo < criteria:
      self.initial_parameters = proposed_position
      self.parameter_store.append(self.initial_parameters)
      self.success = self.success + 1 

    else:
      self.initial_parameters = self.initial_parameters
      self.parameter_store.append(self.initial_parameters)
      self.fail = self.fail + 1
      
    self.acceptance = self.success / (self.success + self.fail)
      
    if self.log_level == 1:
      clear_output(wait=True)
      print(f"Parameters: {self.initial_parameters}")
      print(f"Acceptance: {self.acceptance}")
      print(f"Criteria: {criteria}")
      print(f"Monte-Carlo: {monte_carlo}")
      
  def save(self, file_name = None):        
        if not file_name:
            file_name = 'MetropolisHastings_' + pd.to_datetime('today').strftime('%Y%m%d-%H%M%S') + '.csv'
        
        export_df = pd.DataFrame(self.parameter_store, 
                                 columns = ['param %d' % i for i in range(len(self.initial_parameters))])
        export_df.to_csv(file_name)
        
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

  
  
if __name__ == '__main__':
    # test program
    from scipy.stats import norm as normal
    
    print("Initialising")
    def prior(position,mean,std):
      meanie = normal(mean[0],std[0]).pdf(position[0])
      stdie = normal(mean[1],std[1]).pdf(position[1])
      return meanie*stdie
    
    def data(positions):
      points = np.linspace(3,7,40)
      return normal(positions[0],positions[1]).pdf(points)
    
    def proposal(means, stds):
      return np.random.normal(means,stds)
    
    def likelihood(positions, data, error):
      
      return normal(positions[0], positions[1]).pdf(data).prod()
  
    a = MetropolisHastings([5,1],
                           data,
                           data([5,1]),
                           [prior, [4,3], [2,4]],
                           [proposal, [0,0], [0.001,.001]],
                           likelihood,
                           0,
                           epochs = 5000,
                           burn_in = 50,
                           #adaptive_delay = 100,
                           adaptive = False,
                           targeted_acceptance_rate=0.69,
                           log_level=1)
    a.run()



    a.plot_traces(show_markers = False)
    fig, axs = a.plot_hists(return_fig = True)

    # plot trace and histogram on same figure
    fig, axs = plt.subplots(1, 2)
    a.plot_traces(0, '', ax = axs[0])
    a.plot_hists(0, '', ax = axs[1])
    fig.tight_layout()
    

