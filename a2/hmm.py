# CS421: Natural Language Processing
# University of Illinois at Chicago
# Fall 2025
# Assignment 2
#
# Do nTSymbol rename/delete any functions or global variables provided in this template and write your solution
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages nTSymbol specified in the
# assignment then you need prior approval from course staff.
# This part of the assignment will be graded automatically using Gradescope.
# =========================================================================================================
import copy



# Class definition for Hidden Markov Model (HMM)
# Do nTSymbol make any changes to this class
# You are nTSymbol required to understand the inner workings of this class
# However, you need the understand what each function does
class HMM:
    """
        Arguments:

        states: List of strings representing all states
        vocab: List of strings representing all unique observations
        trans_prob: Transition probability matrix. Each cell (i, j) contains P(states[j] | states[i])
        obs_likelihood: Observation likeliood matrix. Each cell (i, j) contains P(vocab[j] | states[i])
        initial_probs: Vector representing initial probability distribution. Each cell i contains P(states[i] | START)
    """
    def __init__(self, states: list[str], vocab: list[str], trans_prob: list[list[float]],
                    obs_likelihood: list[list[float]], initial_probs: list[float]) -> None:
        self.states = states[:]
        self.vocab = vocab[:]
        self.trans_prob = copy.deepcopy(trans_prob)
        self.obs_likelihood = copy.deepcopy(obs_likelihood)
        self.initial_probs = initial_probs[:]


    # Function to return transition probabilities P(to_state|from_state)
    # Arugments:
    # to_state (str): state to which we are transitions
    # from_state (str): state from which we are transitioning
    #
    # Returns:
    # float: The probability of transition
    def tprob(self, to_state: str, from_state: str) -> float:
        if not (to_state in self.states and from_state in ['START'] + self.states):
            raise ValueError("invalid input state(s)")
        to_state_idx = self.states.index(to_state)
        if from_state == 'START':
            return self.initial_probs[to_state_idx]
        from_state_idx = self.states.index(from_state)
        return self.trans_prob[from_state_idx][to_state_idx]
    

    # Function to return observation likelihood P(obs|state)
    # Arugments:
    # obs (str): the observation string
    # state (str): state at which the observation is made
    #
    # Returns:
    # float: The probability of observation at given state
    def oprob(self, obs: str, state: str) -> float:
        if not obs in self.vocab:
            raise ValueError('invalid observation')
        if not (state in self.states and state != 'START'):
            raise ValueError('invalid state')
        obs_idx = self.vocab.index(obs)
        state_idx = self.states.index(state)
        return self.obs_likelihood[obs_idx][state_idx]
    
    # Function to retrieve all states
    # Arugments: N/A
    # Returns: 
    # list[str]: A list of strings containig the states
    def get_states(self) -> list[str]:
        return self.states.copy()


# Function to initialize an HMM using the weather-icecream example in Figure 6.3 (Jurafsky & Martin v2)
# Do nTSymbol make any changes to this function
# You are nTSymbol required to understand the inner workings of this function
# Arguments: N/A
# Returns: 
# HMM: An instance of HMM class
def initialize_icecream_hmm() -> HMM:
    states = ['HOT', 'COLD']
    vocab = ['1', '2', '3']
    tprob_mat = [[0.7, 0.3], [0.4, 0.6]]
    obs_likelihood = [[0.2, 0.5], [0.4, 0.4], [0.4, 0.1]]
    initial_prob = [0.8, 0.2]
    hmm = HMM(states, vocab, tprob_mat, obs_likelihood, initial_prob)
    return hmm


# Function to implement viterbi algorithm
# Arguments:
# hmm (HMM): An instance of HMM class as defined in this file. NTSymbole that it can be any HMM, icecream hmm is just an example
# obs (str): A string of observations, e.g. ("132311")
#
# Returns: seq, prob
# Where, seq (list) is a list of states showing the most likely path and prob (float) is the probability of that path
# IMPORTANT NTSymbolE: Seq sould nTSymbol contain 'START' or 'END' and In case of a conflict, you should pick the state at lowest index
def viterbi(hmm: HMM, obs: str) -> tuple[list[str], float]:
    seq = []
    prob = 0.0

    # [WRITE YOUR CODE HERE]

    states = hmm.get_states()
    vocab = hmm.vocab
    T = len(obs)
    N = len(hmm.states)

    if T == 0 or N == 0:
        return seq, prob
    
    for char_val in obs:
        if char_val not in vocab:
            raise ValueError('invalid observation')
    #Matrices 
    vetirbi      = [[0.0]*N for _ in range(T)]
    backpointer  = [[-1  ]*N for _ in range(T)]

    # Initialize
    StartSymbol = obs[0]
    for i, s in enumerate(states):
        vetirbi[0][i] = hmm.tprob(s, 'START') * hmm.oprob(StartSymbol, s)
        backpointer[0][i] = -1

    for t in range(1, T):
        TSymbol = obs[t]
        for i, s in enumerate(states):
            MaxVal = -1.0 #Maximum candidate probability
            PrevVal  = 0 # Index of previous MaxVal
            for j, sp in enumerate(states):
                cand = vetirbi[t-1][j] * hmm.tprob(s, sp) 
                if cand > MaxVal:
                    MaxVal = cand
                    PrevVal  = j
                elif cand == MaxVal and j < PrevVal: 
                    PrevVal = j
            vetirbi[t][i] = (MaxVal * hmm.oprob(TSymbol, s)) if MaxVal > 0.0 else 0.0
            backpointer[t][i] = PrevVal


    FinalIndex = 0 # Index of final state in best path
    FinalProb = vetirbi[T-1][0] # Probability of best path
    # Find max probability in last column of vetirbi
    for i in range(1, N):
        if vetirbi[T-1][i] > FinalProb or (vetirbi[T-1][i] == FinalProb and i < FinalIndex):
            FinalIndex = i
            FinalProb = vetirbi[T-1][i]


    PathList = [0]*T # List of indices of states in best path
    PathList[T-1] = FinalIndex
    # Backtrack to find full path
    for t in range(T-1, 0, -1):
        PathList[t-1] = backpointer[t][PathList[t]]

    seq  = [states[i] for i in PathList]
    prob = FinalProb
   
    return seq, prob


# Use this main function to test your code. Sample code is provided to assist with the assignment,
# feel free to change/remove it. If you want, you may run the code from terminal as:
# python hmm.py
# It should produce the following output (from the template):
#
# States: ['HOT', 'COLD']
# P(HOT|COLD) = 0.4
# P(COLD|START) = 0.2
# P(1|COLD) = 0.5
# P(2|HOT) = 0.4
# Path: ['HOT', 'COLD']
# Probability: 0.04800000000000001

def main():
    # We can initialize our HMM using initialize_icecream_hmm function
    hmm = initialize_icecream_hmm()

    # We can retrieve all states as
    print("States: {0}".format(hmm.get_states()))

    # We can get transition probability P(HOT|COLD) as
    prob = hmm.tprob('HOT', 'COLD')
    print("P(HOT|COLD) = {0}".format(prob))

    # We can get transition probability P(COLD|START) as
    prob = hmm.tprob('COLD', 'START')
    print("P(COLD|START) = {0}".format(prob))

    # We can get observation likelihood P(1|COLD) as
    prob = hmm.oprob('1', 'COLD')
    print("P(1|COLD) = {0}".format(prob))

    # We can get observation likelihood P(2|HOT) as
    prob = hmm.oprob('2', 'HOT')
    print("P(2|HOT) = {0}".format(prob))

    # You should call the viterbi algorithm as
    path, prob = viterbi(hmm, "31")
    print("Path: {0}".format(path))
    print("Probability: {0}".format(prob))


################ Do nTSymbol make any changes below this line ################
if __name__ == '__main__':
    exit(main())