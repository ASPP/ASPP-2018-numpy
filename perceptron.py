# -----------------------------------------------------------------------------
# Copyright (C) 2018  Nicolas P. Rougier
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np

def f(x):
    return x > 0

class Perceptron:
    ''' Perceptron class. '''

    def __init__(self, n, m):
        ''' Initialization of the perceptron with given sizes.  '''

        self.input  = np.ones(n+1)
        self.output = np.ones(m)
        self.weights= np.zeros((m,n+1))
        self.reset()

    def reset(self):
        ''' Reset weights '''

        self.weights[...] = np.random.uniform(-.5, .5, self.weights.shape)

    def propagate_forward(self, data):
        ''' Propagate data from input layer to output layer. '''

        # Set input layer (but not bias)
        self.input[1:]  = data
        self.output[...] = f(np.dot(self.weights,self.input))

        # Return output
        return self.output

    def propagate_backward(self, target, lrate=0.1):
        ''' Back propagate error related to target using lrate. '''

        error = np.atleast_2d(target-self.output)
        input = np.atleast_2d(self.input)
        self.weights += lrate*np.dot(error.T,input)

        # Return error
        return (error**2).sum()


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    np.random.seed(123)
    
    samples = np.zeros(100, dtype=[('input',  float, 2),
                                   ('output', float, 1)])

    P = np.random.uniform(0.05,0.95,(len(samples),2))
    samples["input"] = P
    stars = np.where(P[:,0]+P[:,1] < 1)
    discs = np.where(P[:,0]+P[:,1] > 1)
    samples["output"][stars] = +1
    samples["output"][discs] = 0


    network = Perceptron(2,1)
    network.reset()
    lrate = 0.05

    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(1,1,1, aspect=1, frameon=False)
    ax.scatter(P[stars,0], P[stars,1], color="red", marker="*", s=50, alpha=.5)
    ax.scatter(P[discs,0], P[discs,1], color="blue", s=25, alpha=.5)
    line, = ax.plot([], [], color="black", linewidth=2)
    ax.set_xlim(0,1)
    ax.set_xticks([])
    ax.set_ylim(0,1)
    ax.set_yticks([])
    plt.tight_layout()

    def animate(i):
        global lrate
        error = 0

        count = 0
        lrate *= 0.99
        while error == 0 and count < 10:
            n = np.random.randint(samples.size)
            network.propagate_forward( samples['input'][n] )
            error = network.propagate_backward( samples['output'][n], lrate )
            count += 1

        c,a,b = network.weights[0]
        x0 = -2
        x1 = +2
        if a != 0:
            y0 = (-c -b*x0)/a
            y1 = (-c -b*x1)/a
        else:
            y0 = 0
            y1 = 1
            
        line.set_xdata([x0,x1])
        line.set_ydata([y0,y1])
        
        return line,

    anim = animation.FuncAnimation(fig, animate, np.arange(1, 300))
    #Writer = animation.writers['ffmpeg']
    #writer = Writer(fps=30,
    #                metadata=dict(artist='Nicolas P. Rougier'), bitrate=1800)
    # anim.save('perceptron.mp4', writer=writer)
    plt.show()
