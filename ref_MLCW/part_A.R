# Loading required libraries
library(readxl)        # Reading Excel files
library(NbClust)       # Determining the optimal number of clusters
library(factoextra)    # Visualizing clustering results
library(cluster)       # Clustering algorithms
library(ggplot2)       # Plotting graphs
library(tidyverse)     # Data manipulation and visualization
library(rlang)         # Programming with R
library(gridExtra)     # Arranging multiple grid-based plots
library(fpc)            #  cluster validation statistics


# Reading the datasets
df <- read_excel("C:/Users/Asus/Desktop/ref_MLCW/vehicles.xlsx")

summary(df)

# Adjust the margins as needed
par(mar = c(2, 4, 4, 2)) 

# Boxplot for visualizing the distribution of the first 18 attributes
windows(width = 12, height = 8)
par(mar = c(5, 4, 4, 2) + 0.1)
boxplot(df[,1:18])

# Let's assume we are only working with the 'vehicles' dataset for clustering analysis
# If 'vehicles' dataset contains more attributes, select the first 11 attributes for consistency
dfn <- df[,1:18]
dfn

# Clear the current plotting device
dev.off()

# Open a new plotting window with specified size
windows(width = 12, height = 8) # For Windows

# Set margins
par(mar = c(4, 4, 2, 1)) # Adjust margins as needed

# Set layout
par(mfrow = c(ceiling(length(attributes) / 3), 3))

# Plot each attribute
attributes <- colnames(dfn)
for (attribute in attributes) {
  plot(dfn[[attribute]], main = attribute, xlab = "Index", ylab = "Value")
}


# Subtask 01: Scaling and outlier removal
# Scale the data to standardize the attributes
sca_data <- scale(dfn)
sca_data

# Calculate the z-score for each sample to identify outliers
z_score <- apply(sca_data, 1, function(x) max(abs(x)))
for (threshold in c(2, 2.5, 3, 3.5, 4)) {
  outlierz <- which(z_score > threshold)
  print(paste("Number of outliers detected using z-score method with threshold",
              threshold, ":", length(outlierz)))
}

# Identify outliers using z-score method
outlierz <- which(z_score > 4)
print(paste("Number of outliers detected using z-score method:", length(outlierz)))

# Remove outliers from the dataset
cleanData <- sca_data[-outlierz, ]
print(paste("Number of samples after outlier removal:", nrow(cleanData)))

# Finding the number of clusters using automated tools

# Using NbClust to determine the optimal number of clusters
nb_result <- NbClust(cleanData, distance = "euclidean", min.nc = 2, max.nc = 10, method = "kmeans")
summary(nb_result)
nb_result <- NbClust(cleanData, distance = "euclidean", min.nc = 2, max.nc = 10, method = "kmeans")
print(nb_result$Best.nc)

# Using NbClust to determine the optimal number of clusters with Manhattan distance
nb_result_manhattan <- NbClust(cleanData, distance = "manhattan", min.nc = 2, max.nc = 10, method = "kmeans")
print(nb_result_manhattan$Best.nc)

# Elbow method to visualize the optimal number of clusters
fviz_nbclust(cleanData, kmeans, method = "wss") + labs(subtitle = "Elbow method")

# Gap statistics to determine the optimal number of clusters
set.seed(123)
gap_stat <- clusGap(cleanData, FUN = kmeans, nstart = 25, K.max = 15, B = 50)
fviz_gap_stat(gap_stat)

# Silhouette method to determine the optimal number of clusters
fviz_nbclust(cleanData, kmeans, method = "silhouette") + labs(subtitle = "Silhouette method")

# Calculating k-means
k = 2  # Assume the chosen number of clusters is 2
kmeans_vehicles <- kmeans(cleanData, centers = k, nstart = 10)
print(kmeans_vehicles)

# Calculate Within-Cluster Sum of Square (WSS) and Between-Cluster Sum of Square (BSS)
wss = kmeans_vehicles$tot.withinss
bss = kmeans_vehicles$betweenss
print(wss)
print(bss)

# Silhouette plot to visualize the quality of clustering
sil <- silhouette(kmeans_vehicles$cluster, dist(cleanData))
png("silhouette_plot.png", width = 1200, height = 800)
fviz_silhouette(sil)
dev.off()


################ Subtask 02: PCA
# Calculate variance for each attribute to understand data distribution
#apply(dfn, 2, var)

# Performing Principal Component Analysis (PCA)
pca <- prcomp(cleanData, center = TRUE, scale. = TRUE)
summary(pca)

# Ensure that scaled_data is properly defined and scaled
scaled_data = apply(dfn, 2, scale)

# Calculate eigenvalues & eigenvectors for PCA
vehicles_cov <- cov(scaled_data)
vehicles_eigen <- eigen(vehicles_cov)

# Check the structure of the eigen object
str(vehicles_eigen$vectors)

# Extract the first two eigenvectors
phi <- vehicles_eigen$vectors[, 1:2]

# Invert the sign of the eigenvectors for better interpretation (optional)
phi <- -phi
row_names <- colnames(dfn)  
row.names(phi) <- row_names
colnames(phi) <- c("PC1", "PC2")
print(phi)
summary(phi)

# Calculate the cumulative proportion of variance explained by each PC
cumulative_var <- cumsum(vehicles_eigen$values) / sum(vehicles_eigen$values)
num_PCs <- which(cumulative_var >= 0.92)[1]

# Get selected PCs
selected_pcs <- vehicles_eigen$vectors[, 1:num_PCs]
print(selected_pcs)

# Calculate the percentage of PCs with cumulative score >= 92%
percentage_above_92 <- sum(cumulative_var[1:num_PCs] >= 0.92) / length(cumulative_var) * 100
cat("Percentage of PCs with cumulative score >= 92%:", percentage_above_92, "%\n")

# PCA result summary
c_score <- prcomp(selected_pcs)
sum <- summary(c_score)
print(sum)

# Finding the number of clusters for the new dataset

# Using NbClust for the PCA data to determine the optimal number of clusters
windows(width = 12, height = 8)
par(mar = c(5, 4, 4, 2) + 0.1)
nb_result <- NbClust(scaled_data, distance = "euclidean", min.nc = 2, max.nc = 10, method = "kmeans")
summary(nb_result)


# Elbow method for the PCA data
fviz_nbclust(scaled_data, kmeans, method = "wss") + labs(subtitle = "Elbow method")

# Gap statistics for the PCA data
set.seed(123)
gap_stat <- clusGap(scaled_data, FUN = kmeans, nstart = 25, K.max = 15, B = 50)
fviz_gap_stat(gap_stat)

# *Silhouette method for the PCA data
fviz_nbclust(scaled_data, kmeans, method = "silhouette") + labs(subtitle = "Silhouette method")

#* K-means analysis for the new dataset
k = 2  # Assume the chosen number of clusters is 2
kmeans_vehicles <- kmeans(scaled_data, centers = k, nstart = 10)
print(kmeans_vehicles)

#* WSS and BSS for the new dataset
wss = kmeans_vehicles$tot.withinss
bss = kmeans_vehicles$betweenss
print(wss)
print(bss)

# *Silhouette plot for the new dataset
sil <- silhouette(kmeans_vehicles$cluster, dist(scaled_data))
fviz_silhouette(sil)

# *Calinski-Harabasz Index calculation
ch_index <- numeric(9)
set.seed(20)
for (k in 2:10) {
  k_means <- kmeans(scaled_data, centers = k, nstart = 25, iter.max = 100)
  ch_index[k - 1] <- calinhara(scaled_data, k_means$cluster)
}

#* Plot Calinski-Harabasz Index
plot(2:10, ch_index, type = "b", xlab = "Number of clusters",
     ylab = "CH Index", main = "CHI vs Number of clusters")
################# 

# Perform Principal Component Analysis (PCA)
pca <- prcomp(cleanData, center = TRUE, scale. = TRUE)
summary(pca)

# Calculate the cumulative proportion of variance explained by each PC
cumulative_var <- cumsum(pca$sdev^2) / sum(pca$sdev^2)
num_PCs <- which(cumulative_var >= 0.92)[1]

# Get selected PCs
selected_pcs <- pca$x[, 1:num_PCs]

# Determine the optimal number of clusters using NbClust with Euclidean distance on PCA data
nb_result_pca <- NbClust(selected_pcs, distance = "euclidean", min.nc = 2, max.nc = 10, method = "kmeans")
optimal_clusters_pca <- nb_result_pca$Best.nc[1]  # Assume the first suggested number of clusters is the optimal

# Determine the optimal number of clusters using NbClust with Manhattan distance on PCA data
nb_result_pca_manhattan <- NbClust(selected_pcs, distance = "manhattan", min.nc = 2, max.nc = 10, method = "kmeans")
optimal_clusters_pca_manhattan <- nb_result_pca_manhattan$Best.nc[1]  # Assume the first suggested number of clusters is the optimal

# Determine the optimal number of clusters using Elbow Method for PCA data
windows(width = 12, height = 8)
par(mar = c(5, 4, 4, 2) + 0.1)
fviz_nbclust(selected_pcs, kmeans, method = "wss") + labs(subtitle = "Elbow method")

# Determine the optimal number of clusters using Gap Statistic for PCA data
set.seed(123)
gap_stat_pca <- clusGap(selected_pcs, FUN = kmeans, nstart = 25, K.max = 15, B = 50)
fviz_gap_stat(gap_stat_pca) + labs(subtitle = "Gap Statistic Method")

# Determine the optimal number of clusters using the Silhouette Method for PCA data
windows(width = 12, height = 8)
par(mar = c(5, 4, 4, 2) + 0.1)
fviz_nbclust(selected_pcs, kmeans, method = "silhouette") + labs(subtitle = "Silhouette method")


# Perform k-means clustering on the PCA data with the optimal number of clusters
kmeans_pca <- kmeans(selected_pcs, centers = optimal_clusters_pca, nstart = 10)

# Function to calculate TSS, BSS, and WSS
calculate_ss <- function(data, kmeans_result) {
  # Total Sum of Squares (TSS)
  overall_mean <- colMeans(data)
  TSS <- sum(apply(data, 1, function(row) sum((row - overall_mean)^2)))
  
  # Between-Cluster Sum of Squares (BSS)
  cluster_means <- kmeans_result$centers
  BSS <- sum(kmeans_result$size * apply(cluster_means, 1, function(mean) sum((mean - overall_mean)^2)))
  
  # Within-Cluster Sum of Squares (WSS)
  WSS <- sum(kmeans_result$withinss)
  
  return(list(TSS = TSS, BSS = BSS, WSS = WSS))
}

# Apply the function to the kmeans result on PCA data
ss_values_pca <- calculate_ss(selected_pcs, kmeans_pca)
print(ss_values_pca)

# Silhouette plot to visualize the quality of clustering on PCA data
sil_pca <- silhouette(kmeans_pca$cluster, dist(selected_pcs))
png("silhouette_plot_pca.png", width = 1200, height = 800)
fviz_silhouette(sil_pca)
dev.off()
# Silhouette plot for the new dataset
sil <- silhouette(kmeans_vehicles$cluster, dist(scaled_data))
fviz_silhouette(sil)

# *Calinski-Harabasz Index calculation
ch_index <- numeric(9)
set.seed(20)
for (k in 2:10) {
  k_means <- kmeans(scaled_data, centers = k, nstart = 25, iter.max = 100)
  ch_index[k - 1] <- calinhara(scaled_data, k_means$cluster)
}

#* Plot Calinski-Harabasz Index
plot(2:10, ch_index, type = "b", xlab = "Number of clusters",
     ylab = "CH Index", main = "CHI vs Number of clusters")

# Calculate TSS, BSS, and WSS for the PCA data
ss_values_pca <- calculate_ss(selected_pcs, kmeans_pca)
print(ss_values_pca)
