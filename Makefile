EXEC=Neural-Net.cpp
TRAININGSET=makeTrainingSets.cpp
CODEDIR=./src
BINARYDIR= bin
DATADIR=lib
.PHONY: all TrainingData NeuralNet

TrainingData:
	@echo "Enter the topology, it should be three numbers space separated like 2 4 1"; \
	read a b c; \
	echo "Entering the directory: " $(CODEDIR);\
	g++ $(CODEDIR)/$(TRAININGSET) -o $(BINARYDIR)/$@ ;\
	./$(BINARYDIR)/$@ $$a $$b $$c > $(DATADIR)/$@.txt


NeuralNet:
	@echo "Running the neural net with the topology and data given"; \
	 echo "Entering the directory: " $(CODEDIR);\
     g++ $(CODEDIR)/$(EXEC) -o $(BINARYDIR)/$@ ;\
     ./$(BINARYDIR)/$@ $(DATADIR)/TrainingData.txt > $(DATADIR)/output.txt


