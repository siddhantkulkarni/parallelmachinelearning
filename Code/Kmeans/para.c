#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <assert.h>
#include <omp.h>
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
            fprintf(stderr, "Error: file is not present (%s)\n", filename); // prints error message if file is not read
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

                len = strlen(line);     // gets the length of the line
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


    len = (*numImg) * (*numFea); // product of number of images and number of  in each images
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
            for (j=0; j<(*numFea); j++)      // based upon number of images all the data-points will be stored
                images[i][j] = atof(strtok(NULL, " ,\t\n")); //reads every data-points (converting string to float)
            i++;
        }

        fclose(infile);
        free(line);

    return images;


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



float** kmeans_omp(float **images,int  numFea,int  numImg,int  numClusters,float  threshold,int  *cluster_id)
{

    int      i, j, k, index, loop=0;
    int     *newClusterSize; // number of elements in the new clusters
    float    change;          //  specifies the number of object that changed their cluster
    float  **clusters;       // this will be the cluster centroid
    float  **newClusters;    // this will be the cluster centroid new location after finding out mean of the elements in that cluster
    double   timing;

    int      nthreads;             // specifies the number of thread
    int    **local_newClusterSize;
    float ***local_newClusters;

    nthreads = omp_get_max_threads();


    clusters    = (float**) malloc(numClusters * sizeof(float*)); // allocate the memory for the cluster centroid based upon the number of cluster
    assert(clusters != NULL);
    clusters[0] = (float*)  malloc(numClusters * numFea * sizeof(float));// for each cluster allocate the memory based upon the number of cluster and the number of points in the images
    assert(clusters[0] != NULL);
    for (i=1; i<numClusters; i++)
        clusters[i] = clusters[i-1] + numFea;


    for (i=0; i<numClusters; i++)
        for (j=0; j<numFea; j++)
            clusters[i][j] = images[i][j]; // The first K number of the images will be selected as the centroid of the cluster


    for (i=0; i<numImg; i++) cluster_id[i] = -1;// initially all the images cluster_id will be set to -1


    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);

    newClusters    = (float**) malloc(numClusters * sizeof(float*));// allocate the memory for the new cluster based upon the number of cluster
    assert(newClusters != NULL);
    newClusters[0] = (float*)  calloc(numClusters * numFea, sizeof(float));// allocate the memory for the each new cluster formed based upon the number of cluster and the number of points in the images
    assert(newClusters[0] != NULL);
    for (i=1; i<numClusters; i++)
        newClusters[i] = newClusters[i-1] + numFea;


        local_newClusterSize    = (int**) malloc(nthreads * sizeof(int*));// allocate the memory for the local_newclustersize variable
        assert(local_newClusterSize != NULL);
        local_newClusterSize[0] = (int*)  calloc(nthreads*numClusters,
                                                 sizeof(int));
        assert(local_newClusterSize[0] != NULL);
        for (i=1; i<nthreads; i++)
            local_newClusterSize[i] = local_newClusterSize[i-1]+numClusters;


        local_newClusters    =(float***)malloc(nthreads * sizeof(float**)); // allocate the memory for the local_newcluster variable
        assert(local_newClusters != NULL);
        local_newClusters[0] =(float**) malloc(nthreads * numClusters *
                                               sizeof(float*));
        assert(local_newClusters[0] != NULL);
        for (i=1; i<nthreads; i++)
            local_newClusters[i] = local_newClusters[i-1] + numClusters;
        for (i=0; i<nthreads; i++)
        {
            for (j=0; j<numClusters; j++)
            {
                local_newClusters[i][j] = (float*)calloc(numFea,
                                                         sizeof(float));
                assert(local_newClusters[i][j] != NULL);
            }
        }

    do {
        change = 0.0;
        #pragma omp parallel \
                    shared(images,clusters,cluster_id,local_newClusters,local_newClusterSize)
            {
                int tid = omp_get_thread_num();
                #pragma omp for \
                            private(i,j,index) \
                            firstprivate(numImg,numClusters,numFea) \
                            schedule(static) \
                            reduction(+:change)
                for (i=0; i<numImg; i++)    // parallel the for loop
				{

                    index = cluster_index(numClusters, numFea,images[i], clusters);// call the index function to get index for each images


                    if (cluster_id[i] != index) change += 1.0; // increment the change value when the images changes its membership


                    cluster_id[i] = index;  // The cluster_id[image_id] is equal to the index


                    local_newClusterSize[tid][index]++; // increment the local_newClusterSize if the data-point falls in that cluster
                    for (j=0; j<numFea; j++)
                        local_newClusters[tid][index][j] += images[i][j]; // add the data-point to the local_newClusters of that particular index
                }
            }

            for (i=0; i<numClusters; i++)
            {
                for (j=0; j<nthreads; j++)
                {
                    newClusterSize[i] += local_newClusterSize[j][i];// assign the value of the local_newClusterSize to the newClusterSize
                    local_newClusterSize[j][i] = 0.0;
                    for (k=0; k<numFea; k++)
                    {
                        newClusters[i][k] += local_newClusters[j][i][k]; // assign the value present in the local_newClusters to the newClusters
                        local_newClusters[j][i][k] = 0.0;
                    }
                }
            }



        for (i=0; i<numClusters; i++) {
            for (j=0; j<numFea; j++) {
                if (newClusterSize[i] > 1)
                    clusters[i][j] = newClusters[i][j] / newClusterSize[i];// calculate the mean to get the new cluster centroid location
                newClusters[i][j] = 0.0;
            }
            newClusterSize[i] = 0;
        }

        change /= numImg;
    } while (change > threshold && loop++ < 500);  // check the condition





        free(local_newClusterSize[0]);
        free(local_newClusterSize);

        for (i=0; i<nthreads; i++)
            for (j=0; j<numClusters; j++)
                free(local_newClusters[i][j]);
        free(local_newClusters[0]);
        free(local_newClusters);

    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

    return clusters;
}
int main(int argc, char **argv) {
           int     i, j, nthreads;
           int     output_timing;
           int     numClusters, numFea, numImg;
           int    *cluster_id;
           char   *filename;
           float **images;
           float **clusters;
           float   threshold;
    // default values
    nthreads          = 4;
    numClusters       = 0;
    threshold         = 0.001;
    numClusters       = 0;
    filename          = NULL;

    filename =argv[1]; // get the filename
	printf("Enter the number of clusters\n"); // get the number of clusters
	scanf("%d",&numClusters);
	if (filename == 0 || numClusters <= 1) printf("please enter in this format 'seq color.txt'\n ");// if the filename and the number of cluster print the message
    printf("The filename you have entered = %s\n", filename);
	printf("The number of cluster you have entered = %d\n", numClusters);
	


    if (nthreads > 0)
		omp_set_num_threads(nthreads); // allocate the thread


	 clock_t begin = clock(); // start the clock

    images = read_file(filename, &numImg, &numFea); // read the data-points from the text file
    if (images == NULL) exit(1);

    cluster_id = (int*) malloc(numImg * sizeof(int)); // cluster_id specifies in which index does the image belong
    assert(cluster_id != NULL);

    clusters = kmeans_omp(images, numFea, numImg, numClusters, threshold, cluster_id);// calls the parallel kmeans


    free(images[0]);
    free(images);

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC; // specify the time spent in the execution



    output(filename, numClusters, numImg, numFea, clusters, cluster_id);// output the final cluster centroid and the membership of the each images

    free(cluster_id);
    free(clusters[0]);
    free(clusters);



	printf("========The parallel Implementation of the K-Means=======\n");
    printf("Computation timing = %10.4f sec\n", time_spent);


    return(0);
}
