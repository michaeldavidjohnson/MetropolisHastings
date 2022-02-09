class MetropolisHastings:
  def __init__(self, initial_parameters, model, data, prior_function,
               proposal_distribution,
               likelihood_function, measurement_error, epochs, 
               burn_in=False, return_MAP=False, adaptive=False,
               extra_conditions = False, log_level = False):
    
    self.initial_parameters = initial_parameters
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
    
    #Haven't had getters and setters yet 
    self.MAP = [0,0]
    self.parameter_store = []
    self.success = 1
    self.fail = 0
    self.acceptance = 1
    self.log_level = log_level

  def run(self):
    for _ in range(self.epochs):
      initial_posterior = self.generate_posterior(self.initial_parameters)
      proposed_position = np.array(self.initial_parameters) + self.move()

      if not self.extra_conditions == False:
        for cri in self.extra_conditions:
          while not cri(proposed_position) == True:
            proposed_position = np.array(self.initial_parameters) + self.move()
      

      proposed_posterior = self.generate_posterior(proposed_position)

      if self.return_MAP:
        if -np.log(proposed_posterior) < -np.log(initial_posterior):
          self.MAP = [-np.log(proposed_posterior), proposed_position]

      criteria = np.exp( np.log(proposed_posterior) - np.log(initial_posterior))
      monte_carlo = np.random.uniform()
      if self.log_level == 1:
        print(f"{proposed_position}")
        print(f"{initial_posterior}")
        print(f"{monte_carlo}")
        print(f"{criteria}")

      if monte_carlo > criteria:
        self.initial_parameters = self.initial_parameters
        self.parameter_store.append(self.initial_parameters)
        self.fail = self.fail + 1
      
      else:
        self.initial_parameters = proposed_position
        self.parameter_store.append(self.initial_parameters)
        self.success = self.success + 1 
      
      self.acceptance = self.success / (self.success + self.fail)
      
      if self.log_level == 1:
        clear_output(wait=True)
        print(f"Parameters: {self.initial_parameters}")
        print(f"Acceptance: {self.acceptance}")
        print(f"Criteria: {criteria}")

    print("Running has finished")

  def plot_traces(self):
    return 0 

  def plot_hists(self):
    return 0
  


  def move(self):
    move = [] 
    for i in range(len(self.proposal_move_means)):
      move.append(self.proposal_distribution(self.proposal_move_means[i],
                                             self.proposal_move_stds[i]))
    return move
  
  def generate_posterior(self,position):
    likelihood = self.likelihood_function(self.model(position), self.data,
                                      self.measurement_error)
    prior = self.prior_function(position, self.prior_means, self.prior_stds)
    if self.log_level == 1:
      print(f"Likelihood: {likelihood}")
      print(f"Prior: {prior}")
    
    return likelihood * prior

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

    
