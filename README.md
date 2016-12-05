# parallelmachinelearning


Parallel implementation of two machine learning algorithms using OpenMP


Commands to use for execution of Naive Bayesian code


- g++ -std=c++0x -fopenmp -O programname.cpp -o objectname


- ./objectname


Based on which dataset you are using, you may need to update the filepaths, attribute and rowcounts inside the code you are trying to execute.


For k-Means source code,

============================Serial Implementation=========================================


Running the seq.c in hydra


1. Copy the "color.txt" file in the local profile of hydra


2. compile the seq.c program


   gcc seq.c -o seq


3. Running the seq file


  ./seq color.txt
  
  
  Then you will be promoted with the number of cluster, enter the number of cluster.


///////////OUTPUT/////////////


4. The time will be returned


5. Two files will be created which consists of images membership and the final cluster centroid location 


============================Parallel Implementation=========================================


Running the para.c in hydra


1. Copy the "color.txt" file in the local profile of hydra


2. compile the seq.c program


  gcc -O -fopenmp para.c -o para


3. Running the seq file


  ./para color.txt
  
  
  Then you will be promoted with the number of cluster, enter the number of cluster.
  
  
  Then you will be promoted with the number of threads, enter the number of threads.


///////////OUTPUT/////////////


4. The time will be returned


5. Two files will be created which consists of images membership and the final cluster centroid location 

