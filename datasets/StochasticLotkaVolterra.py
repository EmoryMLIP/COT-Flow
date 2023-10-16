import numpy as np

class StochasticLotkaVolterra():
    def __init__(self, T=30, dt=0.2):
        # number of parameters and observations
        self.d = 4;
        self.nobs = 9;
        # prior parameters
        self.logtheta_a = -5;
        self.logtheta_b = 2;
        # initial condition
        self.x0 = [50,100]
        # integration params
        self.T = T
        self.dt = dt
        self.tt = np.arange(0, T, step=dt)

    def sample_joint(self, N):
        theta = np.zeros((N, self.d))
        yobs  = np.zeros((N, self.nobs))
        ctr = 0
        while(ctr < N):
            # print(ctr)
            theta[ctr,:] = self.sample_prior(1)
            try:
                yobs[ctr,:] = self.sample_data(theta[ctr,:])
                ctr += 1
            except:
                continue
        return (theta, yobs)

    def sample_prior(self, N):
        # generate uniform samples
        theta_range = (self.logtheta_b - self.logtheta_a)
        logtheta = theta_range * np.random.rand(N,self.d) + self.logtheta_a
        # transform to non-uniform domain
        return np.exp(logtheta)

    def simulate(self, theta):
        # simulate the stochastic LV system (a Markov jump processs)
        # using the Gillespie algorithm
        # check dimension of theta
        assert(theta.shape == (self.d,))
        X = self.x0[0]; Y = self.x0[1]
        # declare array to store states
        n_time_steps = len(self.tt)
        XYtt = np.zeros((n_time_steps, 2))
        XYtt[0,:] = np.array([X,Y])
        # define current time and time counter
        cur_time = self.dt
        time = 0
        n_steps = 0
        max_nsteps = 10000
        # run stochastic model
        for ti in np.arange(1,n_time_steps):
            while cur_time > time:
                # compute rates
                rates = np.array([theta[0]*X*Y, theta[1]*X, theta[2]*Y, theta[3]*X*Y])
                total_rate = sum(rates)
                # determine time to next reaction and update time
                if total_rate == 0:
                    time = np.inf
                    break
                time += np.random.exponential(scale=1/total_rate)
                # sample reaction
                reaction = np.random.choice([0,1,2,3], 1, p=rates/total_rate)
                # update predator/prey counts based on reaction
                if reaction == 0:
                    X += 1
                elif reaction == 1:
                    X -= 1
                elif reaction == 2:
                    Y += 1
                elif reaction == 3:
                    Y -= 1
                # update n_steps
                n_steps += 1;
                if n_steps > max_nsteps:
                    raise ValueError('Reached maximum of n_steps')
            # store states
            XYtt[ti,:] = [X,Y]
            # update current time
            cur_time += self.dt
        return XYtt

    def sample_data(self, theta):
        # check dimensions
        assert(theta.shape == (self.d,))
        # collect time-series of population dynamics
        try:
            xy = self.simulate(theta)
        except:
            raise ValueError('Did not finish simulation')
        # extract statistics
        xt = np.zeros(self.nobs)
        xt[0:2] = np.mean(xy, axis=0)
        xt[2:4] = np.log(np.var(xy, axis=0)+1)
        xt[4]   = self.auto_corr(xy[:,0], lag=1)
        xt[5]   = self.auto_corr(xy[:,0], lag=2)
        xt[6]   = self.auto_corr(xy[:,1], lag=1)
        xt[7]   = self.auto_corr(xy[:,1], lag=2)
        xt[8]   = np.corrcoef(xy[:,0],xy[:,1])[0,1]
        return xt

    def auto_corr(self, y, lag, meany=None, stdy=None):
        if stdy == None:
            stdy = np.std(y)
        if meany == None:
            meany = np.mean(y)
        # normalize signal
        y = (y - meany)/stdy
        # compute inner product
        ac = np.sum(y[:-lag]*y[lag:])
        ac /= (len(y)-1)
        return ac

    def prior_pdf(self, theta):
        logtheta = np.log(theta)
        # check if samples are within range
        theta_range = (self.logtheta_b - self.logtheta_a);
        inside_range = (logtheta > self.logtheta_a and logtheta < self.logtheta_b)
        # compute density
        return np.prod(1./theta_range * inside_range, axis=1)

    def log_likelihood(self, xobs, theta):
        raise ValueError('Not available for this model')

    def likelihood(self, xobs, theta):
        raise ValueError('Not available for this model')

if __name__=='__main__':

    # define model
    LV = StochasticLotkaVolterra()

    # define true parameters and observation
    xtrue = np.array([[0.01,0.5,1,0.01]])
    #xtrue = LV.sample_prior(1)
    yplot = LV.simulate(xtrue[0,:])
    yobs  = LV.sample_data(xtrue[0,:])
    print(yobs)

    # plot single simulation
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(LV.tt, yplot)
    plt.xlabel('$t$')
    plt.ylabel('Observations')
    plt.show()

    # generate data
    x, y = LV.sample_joint(100)
