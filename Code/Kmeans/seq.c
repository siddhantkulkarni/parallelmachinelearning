#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include <fcntl.h>
#include <unistd.h>     
#include <assert.h>
#include <sys/time.h>    
#include <sys/types.h>  
#include <sys/stat.h>
#include <string.h>
#include <time.h>
#define MAX_CHAR_PER_LINE 128

float** read_file(char *filename,int  *numImg,int  *numFea)     
{
    float **images;
    int     i, j, len;
    ssize_t numBytesRead;

	FILE *infile;
    char *line, *ret;
    int   lenghtline;

    if ((infile = fopen(filename, "r")) == NULL) 
	    {
            fprintf(stderr, "Error: filename is not present (%s)\n", filename); // prints error message if file is not read
            return NULL;
        }

        
    lenghtline = MAX_CHAR_PER_LINE; // assigning the number of data points per line
    line = (char*) malloc(lenghtline); // allocates the memory for the lenghtline variable
    assert(line != NULL);

    (*numImg) = 0;
    while (fgets(line, lenghtline, infile) != NULL) // make sure all the input is received
	{
        while (strlen(line) == lenghtline-1) 
		{
                
                len = strlen(line);     // gets the lenght of the line
                fseek(infile, -len, SEEK_CUR);

                
                lenghtline += MAX_CHAR_PER_LINE;
                line = (char*) realloc(line, lenghtline);
                assert(line != NULL);

                ret = fgets(line, lenghtline, infile);
                assert(ret != NULL);
        }

            if (strtok(line, " \t\n") != 0) // breaks the string according to the delimiter mentioned
                (*numImg)++;               // gets the number of images present in the file  
    }
    rewind(infile); // points to the begining of the file
    
        
    (*numFea) = 0; // this specifies number of points in each images
    while (fgets(line, lenghtline, infile) != NULL) 
	{
            if (strtok(line, " \t\n") != 0)  // divide the string into tokens
			{
                
                while (strtok(NULL, " ,\t\n") != NULL) (*numFea)++;// eliminate the fist element of every line as first element is the image_ID
                break; 
            }
    }
    rewind(infile);
  

        
    len = (*numImg) * (*numFea); // product of number of images and number of points in each images
    images    = (float**)malloc((*numImg) * sizeof(float*)); // assign the memory for each object based upon the number of images
    assert(images != NULL);
    images[0] = (float*) malloc(len * sizeof(float));
    assert(images[0] != NULL);
    for (i=1; i<(*numImg); i++)   // for each images in the file
	{
            images[i] = images[i-1] + (*numFea);// assign the elements to each images
    }
    i = 0;
        
    while (fgets(line,lenghtline, infile) != NULL) 
	   {
        if (strtok(line, " \t\n") == NULL) continue;
            for (j=0; j<(*numFea); j++)      // based upon number of images all the datapoints will be stored     
                images[i][j] = atof(strtok(NULL, " ,\t\n")); //reads every datapoints (coverting string to float)
            i++;
        }

        fclose(infile);
        free(line);
	
    return images;
}

static float euclidean_distance(int ele,float *image,float *clust)   
{
    int i;
    float ans=0.0;
	
    for (i=0; i<ele; i++) // for each elements in the cluster centroid and the images
        ans += (image[i]-clust[i]) * (image[i]-clust[i]); // calculating the distance between the cluster centroid and the each points in the images

    return(ans);
}
static int cluster_index(int  numClusters,int  numFea,float  *object,float **clusters)
{
    int   index, i;
    float dist, min_dist;

    
    index    = 0;
    min_dist = euclidean_distance(numFea, object, clusters[0]);// calculating the euclidean  distances for the first cluster

    for (i=1; i<numClusters; i++) // check the distances for the other clusters centroid
	{
        dist = euclidean_distance(numFea, object, clusters[i]);
        if (dist < min_dist) // compare the distance with the distance for the first cluster centroid
		{ 
            min_dist = dist;
            index    = i;  // get the index in which the image belong
        }
    }
    return(index);
}

float** kmeans(float **images,int numFea,int numImg,int numClusters,float threshold,int *cluster_id,int *loop_iterations)
{
    int      i, j, index, loop=0;
    int     *newClusterSize; 
    float    change;          
    float  **clusters;       
    float  **newClusters;    

    
    clusters  = (float**) malloc(numClusters * sizeof(float*)); // allocates the memory based upon the number of clusters
    assert(clusters != NULL);
    clusters[0] = (float*)  malloc(numClusters * numFea * sizeof(float));// allocates memeory based upon number of points in each images
    assert(clusters[0] != NULL);
    for (i=1; i<numClusters; i++)
        clusters[i] = clusters[i-1] + numFea; 

    //getting random points for the cluster center
    for (i=0; i<numClusters; i++)
        for (j=0; j<numFea; j++)
            clusters[i][j] = images[i][j]; // The first numcluster of the images will be selected as the center of the cluster

    
    for (i=0; i<numImg; i++) cluster_id[i] = -1; // assign initial cluster_id for each images

   
    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);

    newClusters    = (float**) malloc(numClusters * sizeof(float*));
    assert(newClusters != NULL);
    newClusters[0] = (float*)  calloc(numClusters * numFea, sizeof(float));
    assert(newClusters[0] != NULL);
    for (i=1; i<numClusters; i++)
        newClusters[i] = newClusters[i-1] + numFea;

    do {
        change = 0.0;  // is used to check the number of times the memebership changed
        for (i=0; i<numImg; i++) // for each images in the text file
		{
            
            index = cluster_index(numClusters, numFea, images[i],clusters); // getting index of the nearest cluster for each images


            if (cluster_id[i] != index) change += 1.0; //if the cluster_id is changes the increment the change by one

            
            cluster_id[i] = index;  //assign the index or cluster_id for each images

           
            newClusterSize[index]++;
            for (j=0; j<numFea; j++)
                newClusters[index][j] += images[i][j];// assign the image to the cluster it belong
        }

        
        for (i=0; i<numClusters; i++) {
            for (j=0; j<numFea; j++) {
                if (newClusterSize[i] > 0)
                    clusters[i][j] = newClusters[i][j] / newClusterSize[i];// finding the new location for the old cluster centroid by calculating the mean for all the elemnets in it 
                newClusters[i][j] = 0.0;   
            }
            newClusterSize[i] = 0;   
        }
            
        change /= numImg;
    } while (change > threshold && loop++ < 500);

    *loop_iterations = loop + 1;

    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

    return clusters;
}

int output(char  *filename,int  numClusters, int  numImg, int numFea, float  **clusters,int  *cluster_id) 
{
    FILE *fptr;
    int   i, j;
    char  outputfile[1024];

    
    sprintf(outputfile, "%s.cluster_centres", filename);
    fptr = fopen(outputfile, "w");
	fprintf(fptr,"=============The final cluster centroid location==============\n");
    for (i=0; i<numClusters; i++) {
        fprintf(fptr, "%d ", i);
        for (j=0; j<numFea; j++)
            fprintf(fptr, "%f ", clusters[i][j]); // The final location of the cluster centroid
        fprintf(fptr, "\n");
    }
    fclose(fptr);

    
    sprintf(outputfile, "%s.cluster_id", filename);
    fptr = fopen(outputfile, "w");
    fprintf(fptr,"|IMAGE_ID | CLUSTER_ID|\n");
    for (i=0; i<numImg; i++)
        fprintf(fptr, "|%d\t |%d\t  |\n", i, cluster_id[i]);// the image_ID along with the its respective cluster_id
    fclose(fptr);

    return 1;
}

int main(int argc, char **argv) {
		   int     i, j;
           int     numClusters, numFea, numImg;
           int    *cluster_id;    
           char   *filename;
           float **images;       
           float **clusters;      
           float   threshold;
           int     loop_iterations;
		   

    threshold        = 0.001;
    numClusters      = 0;
    filename         = NULL;
	
	filename =argv[1]; // get the filename
	printf("enter the number of clusters\n");// get the number of cluster
	scanf("%d",&numClusters);
    if (filename == 0 || numClusters <= 1) printf("please enter in this format 'seq color.txt'\n ");// if the filename and the number of cluster not given then go to incomplete function
    printf("The filename you have entered = %s\n", filename);
	printf("The number of cluster you have entered = %d\n", numClusters);
	clock_t begin = clock(); // the time begin
	
    images = read_file(filename, &numImg, &numFea); // get the data from the input file
	if (images == NULL) exit(1);
	
	cluster_id = (int*) malloc(numImg * sizeof(int)); // cluster_id is the cluster ID for each images
    assert(cluster_id != NULL);

    clusters = kmeans(images, numFea, numImg, numClusters, threshold,cluster_id, &loop_iterations); // calls the kmeans_algorithm

    free(images[0]);
    free(images);

    
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC; // specify the time spent
    
    output(filename, numClusters, numImg, numFea, clusters,cluster_id);

    free(cluster_id);
    free(clusters[0]);
    free(clusters);

    printf("========The serial Implementation of the K-Means=======\n");
    printf("Computation timing = %10.4f sec\n", time_spent);
    

    return(0);
}