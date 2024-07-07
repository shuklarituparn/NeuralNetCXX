#include <vector>


class Neuron;

typedef std::vector<Neuron> Layer;

class Net{
public:
    Net(const std::vector<unsigned int> &topology); // const as it is not changing element in the topology
    void feedForward(const std::vector<double > &inputVals); // it just reads the value from input and transfers to
                                                                //input neuron, so its a const
    void backProp(const std::vector<double> &targetVals);
    void getResults(std::vector<double> &resultVals) const;  //reads the output vals and put it in a container so const will go here
                            //result vals are not const as we will store our value in them
private:
    std::vector<Layer> m_layers;  //All the neurons will be arranged in layers and all the layers are in an array
    //m_layers[LayerNum][neuronNum]  first subscript is the layer number and the second one is the neuron number
};


int main(){
    // 3 2 1 - basically the layout of our net, three layers, first with 3 neuron and so on
    std::vector<unsigned int> topology;
    Net myNet(topology);  //basically structure of our net, number of layers and neurons per layers
    std::vector<double> inputVals
    std::vector<double> targetVals;
    std::vector<double> resultVals;

    myNet.feedForward(inputVals);  //it feeds forward a bunch of input values
    myNet.backProp(targetVals);  //telling what the output supposed to have been to make it relearn
    myNet.getResults(resultVals);  //to get the result after its being trained

}