# parallelmachinelearning


Parallel implementation of two machine learning algorithms using OpenMP


Commands to use for execution of Naive Bayesian code


- g++ -std=c++0x -fopenmp -O programname.cpp -o objectname


- ./objectname


Based on which dataset you are using, you may need to update the filepaths, class values, attribute counts and row counts inside the code you are trying to execute.

For ad data and large Data, you can use same file as training as well as testing.


Following are the classes used in each of the files:


AdData(2)={"ad.","nonad."}


- Use same file as training and testing

- Number of Lines: 327900

- Number of Columns: 1559


LargeData(2)={"yes","no"}

- Use same file as training and testing

- Number of Lines: 28000

- Number of Columns: 5

Network log data(38)={"apache2.","back.","buffer_overflow.","ftp_write.","guess_passwd.","httptunnel.","imap.","ipsweep.","land.","loadmodule.","mailbomb.","mscan.","multihop.","named.","neptune.","nmap.","normal.","perl.","phf.","pod.","portsweep.","processtable.","ps.","rootkit.","saint.","satan.","sendmail.","smurf.","snmpgetattack.","snmpguess.","sqlattack.","teardrop.","udpstorm.","warezmaster.","worm.","xlock.","xsnoop.","xterm."}


- Number of Lines in training: 311029

- Number of Lines in testing : 494021

- Number of Columns: 42

For k-Means source code,

=========Serial Implementation=======


Running the seq.c in DOZER


1. Copy the "color.txt" file in the local profile of DOZER


2. compile the seq.c program


   gcc seq.c -o seq


3. Running the seq file


   ./seq color.txt
   
   
   Then you will be promoted with the number of cluster, enter the number of cluster.


///////////OUTPUT/////////////


4. The time will be returned


5. Two files will be created which consists of images membership and the final cluster centroid location 


=======Parallel Implementation==========


Running the para.c in DOZER


1. Copy the "color.txt" file in the local profile of DOZER


2. compile the seq.c program


   gcc -O -fopenmp para.c -o para


3. Running the seq file


   ./para color.txt
  
  
  Then you will be promoted with the number of cluster, enter the number of cluster.


///////////OUTPUT/////////////


4. The time will be returned


5. Two files will be created which consists of images membership and the final cluster centroid location 
