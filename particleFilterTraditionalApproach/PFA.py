import numpy
import random
#particle filter: propose, weight, resample
class PFA(object):
    #to be read from the configuration file, for test hardcoding the values
    Samples={0: [(0, 0.2), (0, 0.2), (0, 0.2)],1: [(1, 0.4), (1, 0.4), (1, 0.4)],2: [(2, 0.4), (2, 0.4), (2, 0.4)]}
    TransitionModel={(0, 1): 0.2,(0, 0): 0.3,(0, 2): 0.1,(1, 1): 0.2,(1, 0): 0.3,(1, 2): 0.1,(2, 1): 0.2,(2, 0): 0.3,(2, 2): 0.1}
    SensorModel={(0,0):0.5,(0,1):0.2,(0,2):0.3,(1,0):0.5,(1,1):0.2,(1,2):0.3,(2,0):0.5,(2,1):0.2,(2,2):0.3}
    Prior={0:[(0,0.5)],1:[(1,0.2)],2:[(2,0.3)]}
    Weights ={}

    def particlefiltering(self,evidence=0, N=5):
        #for i in range(0,N+1):
        #    print("particle:",i)
        Samples = self.initialise(self,N)
        TransitionModel=self.TransitionModel
        SensorModel=self.SensorModel
        self.Samples=self.updateSamplesBasedOnTM(self,Samples,TransitionModel) #step 1
        print("step1:", self.Samples)
        self.Weights=self.weighSamplesOnSensorModel(self,Samples,SensorModel) #step 2
        print("step2:", self.Weights)
        self.Samples=self.weightedResamplingWithReplacement(self,N,self.Samples,self.Weights) #step 3
        print("step3:",self.Samples) #for next timeslice
        #next=[0] #set of samples for the next timestep
        #weight=[]
        #Samples={}
        #Samples=initialise(N)
        #for i in range(1,N):
            #next[i] = getValueFromTransitionModel()
            #weight[i] = getValuesFromSensorModel()
        #Samples = weightedSampleWithReplacement(N,Samples,weight)
        #return samples
        Samples = self.Samples
        return Samples

    #initialise the State for first timestep
    def initialise(self,N):
        Samples={}
        priorValues=[value[0][1] for value in self.Prior.values()]
        distinctStates=list(self.Prior.keys())
        allvalues = [value for sublist in self.Prior.values() for value in sublist]
        sortedStates=sorted(allvalues, key=lambda x: x[1], reverse=True)
        Samples=self.generateSamples(self,self.Prior,N)
        return Samples#initialise

    def getValueFromTransitionModel(self):
        return 0

    def getValuesFromSensorModel(self):
        return 0

    def getTransitionModel(self):
        #StateN,StateO
        transitionModel=[]
        transitionModelValue=0
        return transitionModelValue

    def getSensorModel(Evidence,State):
        sensorModel=[]
        sensorModelValue=0
        return sensorModelValue

    def weightedSampleWithReplacement(N,Samples,weights):
        #normalise and return the Samples
        return Samples

#other helper functions
    def generateSamplesAsList(self, Prior, N):
        #Calculating total probability
        tprob= sum(prob[0][1] for prob in Prior.values())
        print("tprob:",tprob)
        #getting the highest probability state
        highest_prob_keys = [key for key, prob in Prior.items() if prob[0][1] == max(prob[0][1] for prob in Prior.values())]
        #calculating the highest probability ratio
        highest_prob_ratio = Prior[highest_prob_keys[0]][0][1] / tprob
        #calculating number of additional samples for highest probability key
        additional_samples_highest_prob = int(N * highest_prob_ratio)
        #generating samples based on probability
        samples = []
        for key, prob in Prior.items():
            num_samples = int(N * prob[0][1])
            if key in highest_prob_keys:
                num_samples += additional_samples_highest_prob
            samples.extend([key] * num_samples)
        #shuffle for random order if necessary
        #random.shuffle(samples)
        return samples


    def generateSamplesTest(self,Prior,N):
        # Calculating total probability
        tprob = sum(prob[0][1] for prob in Prior.values())
        print("tprob:", tprob)
        # getting the highest probability state
        highest_prob_keys = [key for key, prob in Prior.items() if
                             prob[0][1] == max(prob[0][1] for prob in Prior.values())]
        # calculating the highest probability ratio
        highest_prob_ratio = Prior[highest_prob_keys[0]][0][1] / tprob
        # calculating number of additional samples for highest probability key
        additional_samples_highest_prob = int(N * highest_prob_ratio)
        # generating samples based on probability
        Samples={}
        for key, prob in Prior.items():
            num_samples = int(N * prob[0][1])
            if key in highest_prob_keys:
                num_samples += additional_samples_highest_prob
            # Generate samples for the current key
            samples_list = [(key, prob[0][1]) for _ in range(num_samples)]
            Samples[key] = samples_list
        return Samples

    def generateSamples(self, Prior, N):
        # Calculating total probability
        tprob = sum(prob[0][1] for prob in Prior.values())
        print("tprob:", tprob)
        # getting the highest probability state
        highest_prob_keys = [key for key, prob in Prior.items() if
                             prob[0][1] == max(prob[0][1] for prob in Prior.values())]
        # calculating the highest probability ratio
        highest_prob_ratio = Prior[highest_prob_keys[0]][0][1] / tprob
        # calculating number of additional samples for highest probability key
        additional_samples_highest_prob = int(N * highest_prob_ratio)
        # generating samples based on probability
        Samples = {}
        total_samples = 0
        for key, prob in Prior.items():
            num_samples = int(N * prob[0][1] / tprob)
            if key in highest_prob_keys:
                num_samples += additional_samples_highest_prob
            Samples[key] = [(key, prob[0][1] / tprob)] * num_samples
            total_samples += num_samples

        # Adjusting total samples to match N exactly
        if total_samples > N:
            excess_samples = total_samples - N
            for key in highest_prob_keys:
                excess_samples_key = min(len(Samples[key]) - int(N * Prior[key][0][1] / tprob), excess_samples)
                Samples[key] = Samples[key][:-excess_samples_key]
                excess_samples -= excess_samples_key
                if excess_samples == 0:
                    break
        elif total_samples < N:
            remaining_samples = N - total_samples
            for key in highest_prob_keys:
                remaining_samples_key = int(N * Prior[key][0][1] / tprob) - len(Samples[key])
                remaining_samples_key = min(remaining_samples_key, remaining_samples)
                Samples[key].extend([(key, prob[0][1] / tprob)] * remaining_samples_key)
                remaining_samples -= remaining_samples_key
                if remaining_samples == 0:
                    break

        return Samples

    #updating the Samples based on Transition Model, need to be changed check*****
    def updateSamplesBasedOnTM(self,Samples, TransitionModel):
        # Update Samples based on TransitionModel
        for key, samples_list in Samples.items():
            for i in range(len(samples_list)):
                for transition_key, transition_value in TransitionModel.items():
                    if transition_key[0] == key and transition_key[1] == samples_list[i][0]:
                        Samples[key][i] = (samples_list[i][0], transition_value)
        return Samples

    #weigthing the samples check*****
    def weighSamplesOnSensorModel(self,Samples, SensorModel):
        # Calculate weights in Samples format
        Weights = {}
        for key, samples_list in Samples.items():
            for sample in samples_list:
                sensor_key = (sample[0], key)  # Creating SensorModel key based on Samples format
                weight = SensorModel.get(sensor_key, 0)  # Get corresponding SensorModel value
                if key in Weights:
                    Weights[key].append((sample[0], weight))
                else:
                    Weights[key] = [(sample[0], weight)]
        print(Weights)
        return Weights

    #Weighted resample with replacement
    def weightedResamplingWithReplacement(self,N,Samples,Weights):
        resampled_weights = {}
        for key, weights_list in Weights.items():
            total_weight = sum(weight for _, weight in weights_list)
            resampled_weights[key] = []
            for particle, weight in weights_list:
                num_samples = int(weight * len(weights_list) / total_weight)
                resampled_weights[key].extend([(particle, weight)] * num_samples)
        Samples=resampled_weights
        #returns samples for next state
        return Samples


if __name__ == "__main__":
    pfa = PFA
    print(pfa.TransitionModel)
    print(pfa.TransitionModel)
    N=20
    #pfa.particlefiltering(pfa, 0, 50, 0)
    pfa.Samples=pfa.initialise(pfa,20)
    print((pfa.Samples))
    samples = [value[1] for sublist in pfa.Samples.values() for value in sublist]
    print(len(samples)) #fix this bug for N=20, getting 30 as the max value is taking 20 N; ##fixed (recheck)
    pfa.particlefiltering(pfa,0,20)



