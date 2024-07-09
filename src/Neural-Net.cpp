#include <vector>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

class Neuron;

typedef std::vector<Neuron> Layer;
struct Connection{
    double weight;
    double deltaWeight;

};
class TrainingData{
public:
    explicit TrainingData(const std::string& filename);
    bool isEOF(){ return  m_trainingDataFile.eof();}
    void getTopology(std::vector<unsigned> &topology);

    unsigned getNextInputs(std::vector<double> &inputVals);
    unsigned getTargetOutputs(std::vector<double> &targetOutputVals);
    

private:
    std::ifstream m_trainingDataFile;
};
void TrainingData::getTopology(std::vector<unsigned int> &topology) {
    std::string line;
    std::string label;
    std::getline(m_trainingDataFile, line);
    std::stringstream ss(line);
    ss>>label;
    if (this->isEOF()|| label!="topology:"){
        abort();
    }
    while (!ss.eof()){
        unsigned n;
        ss>>n;
        topology.push_back(n);
    }


}
TrainingData::TrainingData(const std::string& filename) {
       m_trainingDataFile.open(filename.c_str());
}
unsigned TrainingData::getNextInputs(std::vector<double> &inputVals) {
       inputVals.clear();
       std::string line;
       std::getline(m_trainingDataFile, line);
       std::stringstream ss(line);

       std::string label;
       ss>>label;
       if (label=="in:"){
            double oneValue;
             while (ss>>oneValue){
            inputVals.push_back(oneValue);
            }

       }
       return inputVals.size();
}
unsigned TrainingData::getTargetOutputs(std::vector<double> &targetOutputVals) {
    targetOutputVals.clear();
    std::string line;
    std::getline(m_trainingDataFile, line);
    std::stringstream ss(line);
    std::string label;
    ss>>label;
    if (label=="out:"){
        double oneValue;
        while (ss>>oneValue){
            targetOutputVals.push_back(oneValue);
        }

    }
    return targetOutputVals.size();
}
class Neuron{
public:
    explicit Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double  val){m_outputVal=val;}
    [[nodiscard]] double getOutputVal()const {return m_outputVal;}
    void feedForward( const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer) const;
private:
    static double eta;
    static double alpha;
    static double transferFunction(double  x);
    static double transferFunctionDerivative(double  x);
    double m_outputVal{};
    std::vector<Connection> m_outputWeights;
    static double randomWeight () { return (double) random()/double(RAND_MAX);};
    [[nodiscard]] double sumDOW(const Layer &nextLayer) const;
    unsigned m_myIndex;
    double m_gradient{};

};

Neuron::Neuron(unsigned int numOutputs, unsigned myIndex) {
    for (unsigned c = 0; c <numOutputs ; ++c) {
            m_outputWeights.emplace_back(Connection());
            m_outputWeights.back().weight=randomWeight();
    }
    m_myIndex=myIndex;
}

double Neuron::alpha=0.5;
double Neuron::eta=0.15;
void Neuron::feedForward(const Layer &prevLayer) {

    double sum=0;
    for (const auto & n : prevLayer) {
        sum+=n.getOutputVal()* n.m_outputWeights[m_myIndex].weight;
    }
    m_outputVal=Neuron::transferFunction(sum);

}

double Neuron::transferFunction(double x) {
     return tanh(x);
}

double Neuron::transferFunctionDerivative(double x) {
    return 1.0- pow(x, 2.0);
}

void Neuron::calcOutputGradients(double targetVal) {

    double delta=targetVal-m_outputVal;
    m_gradient=delta*Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer) {
        double dow=sumDOW(nextLayer);
        m_gradient=dow*Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::sumDOW(const Layer &nextLayer) const {
    double sum=0;
    for (unsigned n = 0; n <nextLayer.size() ; ++n) {
        sum+=m_outputWeights[n].weight*nextLayer[n].m_gradient;
    }
    return sum;
}

void Neuron::updateInputWeights(Layer &prevLayer) const {
    for (auto & neuron : prevLayer) {
        double oldDeltaWeight=neuron.m_outputWeights[m_myIndex].deltaWeight;
        double newDeltaWeight=eta* neuron.getOutputVal()*m_gradient+alpha*oldDeltaWeight;
        neuron.m_outputWeights[m_myIndex].deltaWeight=newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight+=newDeltaWeight;

    }
}


class Net{
public:
    explicit Net(const std::vector<unsigned> &topology);
    void feedForward(const std::vector<double > &inputVals);
    void backProp(const std::vector<double> &targetVals);
    void getResults(std::vector<double> &resultVals) const;
    [[nodiscard]] double getRecentAverageError( ) const{ return m_recentAverageError;}
private:
    std::vector<Layer> m_layers;
    double m_error{};
    double m_recentAverageError{};
    double m_recentAverageSmoothingFactor{};
};

Net::Net(const std::vector<unsigned> &topology){
    unsigned numLayers= topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
            m_layers.emplace_back();   //creating our layers
            unsigned numOutputs=layerNum==topology.size()-1?0: topology[layerNum+1];
            for (unsigned neuronNum = 0; neuronNum <= topology[layerNum] ; ++neuronNum) {
                m_layers.back().emplace_back(Neuron(numOutputs, neuronNum));
        }
             m_layers.back().back().setOutputVal(1.0);
    }

}

void Net::feedForward(const std::vector<double> &inputVals) {
    assert(inputVals.size()==m_layers[0].size()-1);
    for (unsigned i = 0; i < inputVals.size(); ++i) {
            m_layers[0][i].setOutputVal(inputVals[i]);
        for (unsigned layerNum= 1; layerNum < m_layers.size(); ++layerNum) {
            Layer &prevLayer=m_layers[layerNum-1];
            for (unsigned n = 0; n < m_layers[layerNum].size()-1; ++n) {
                        m_layers[layerNum][n].feedForward(prevLayer);
            }
        }

    }
}

void Net::backProp(const std::vector<double> &targetVals) {
      Layer &outputLayer=m_layers.back();
      m_error=0;
    for (unsigned n = 0; n < outputLayer.size()-1; ++n) {
        double delta= targetVals[n]-outputLayer[n].getOutputVal();
        m_error+= pow(delta,2.0);


    }
    m_error/(double )outputLayer.size()-1;
    m_error= sqrt(m_error);

    m_recentAverageError=(m_recentAverageError*m_recentAverageSmoothingFactor+m_error)/(m_recentAverageSmoothingFactor+1.0);

    for (unsigned n = 0; n <outputLayer.size() ; ++n) {
          outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    for (unsigned layerNum = m_layers.size()-2; layerNum >0; --layerNum) {
          Layer &hiddenLayer=m_layers[layerNum];
          Layer &nextLayer=m_layers[layerNum+1];

        for (auto & n : hiddenLayer) {
              n.calcHiddenGradients(nextLayer);
        }

    }

    for (unsigned layerNum = m_layers.size()-1; layerNum > 0; --layerNum) {
           Layer &layer=m_layers[layerNum];
           Layer &prevLayer=m_layers[layerNum-1];

        for (unsigned n = 0; n <layer.size()-1; ++n) {
                                                  layer[n].updateInputWeights(prevLayer);
        }
    }
}

void Net::getResults(std::vector<double> &resultVals) const {
       resultVals.clear();
    for (unsigned n = 0; n < m_layers.back().size()-1; ++n) {
         resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

void showVectorVals(const std::string& label, std::vector<double> &v){
    std::cout<<label<<" ";
    for (double i : v) {
        std::cout<<i<<" ";
    }
    std::cout<<std::endl;
}

int main(int argc, char *argv[]){
    TrainingData trainingData(argv[1]);
    std::vector<unsigned> topology;
    trainingData.getTopology(topology);
    Net myNet(topology);
    std::vector<double> inputVals, targetVals, resultVals;
    int trainingPass=0;
    while (!trainingData.isEOF()){
        ++trainingPass;
        std::cout<<std::endl<<"Pass "<<trainingPass;
        if (trainingData.getNextInputs(inputVals)!=topology[0]){
            break;
        }
        showVectorVals(": Inputs:", inputVals);
        myNet.feedForward(inputVals);

        myNet.getResults(resultVals);
        showVectorVals("Outputs:", resultVals);

        trainingData.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);

        assert(targetVals.size()==topology.back());

        myNet.backProp(targetVals);

        std::cout<<"Net recent average error: "<<myNet.getRecentAverageError()<<std::endl;
    }
    std::cout<<std::endl<<"Done"<<std::endl;



}