class DataSet():
    # TODO: write tooling to go to and from file to support static
    # benchmarking runs in future.
    
    def __init__(self,
        observed_covariate_data,
        oracle_covariate_data,
        observed_outcome_data,
        oracle_outcome_data):
        # Observed and oracle/transformed covariate data.
        self.observed_covariate_data = observed_covariate_data
        self.oracle_covariate_data = oracle_covariate_data

        # Generated data
        self.observed_outcome_data = observed_outcome_data
        self.oracle_outcome_data = oracle_outcome_data

    @property
    def observed_data(self):
        '''
        Assemble and return the observable data.
        '''

        return self.observed_outcome_data \
            .join(self.observed_covariate_data)

    @property
    def oracle_data(self):
        '''
        Assemble and return the non-observable/oracle data.
        '''

        return self.oracle_outcome_data.join(self.oracle_covariate_data)
