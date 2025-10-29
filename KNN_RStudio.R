  # =============================
  # Lectura del dataset
  # =============================
  ruta_dataset <- file.path("data", "iris.csv")
  dataset <- read.csv(ruta_dataset)

  cat("Primeras filas del dataset:\n")
  print(head(dataset))

  # Comprobamos si hay valores nulos
  if(any(is.na(dataset))){
    cat("¡Hay valores nulos en el dataset!\n")
  } else {
    cat("No hay valores nulos en el dataset.\n")
  }

  # Eliminamos la columna Id si existe
  dataset$Id <- NULL  

  cat("\nEstructura del dataset:\n")
  str(dataset)

  # Resumen de las variables
  summary(dataset)

  # Convertimos Species en factor (variable categórica)
  dataset$Species <- as.factor(dataset$Species)

  # =============================
  # Gráficos con ggplot2 y GGally
  # =============================
  if(!require(ggplot2)) install.packages("ggplot2")
  if(!require(GGally)) install.packages("GGally")
  library(ggplot2)
  library(GGally)

  # Pairplot: todas las variables numéricas, coloreadas por especie
  ggpairs(dataset[, 1:4],
          mapping = aes(color = dataset$Species, alpha = 0.6),
          upper = list(continuous = wrap("points", size = 2.5, alpha = 0.6))) +
    theme_minimal() +
    ggtitle("Matriz de dispersión del Iris por especie")

  # =============================
  # Heatmap de correlación
  # =============================
  dataset_num <- dataset[, 1:4]
  cor_matrix <- cor(dataset_num)
  cor_long <- as.data.frame.table(cor_matrix)

  ggplot(cor_long, aes(x = Var2, y = Var1, fill = Freq)) +
    geom_tile() +   
    geom_text(aes(label = round(Freq, 2)), size = 4) +       
    scale_fill_gradient2(low = "blue", high = "red", midpoint = 0, limit = c(-1, 1)) +
    theme_minimal() +
    ggtitle("Matriz de correlación") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    scale_y_discrete(limits = rev(levels(cor_long$Var1)))

  # =============================
  # KNN con librería
  # =============================
  if(!require(class)) install.packages("class")
  library(class)

  X <- dataset[, 1:4]
  y <- dataset$Species

  set.seed(123)
  train_index <- sample(1:nrow(dataset), 0.7 * nrow(dataset))
  X_train <- X[train_index, ]
  X_test <- X[-train_index, ]
  y_train <- y[train_index]
  y_test <- y[-train_index]

  # KNN con class
  y_pred_lib <- knn(X_train, X_test, y_train, k = 27)

  # Precisión KNN con librería
  accuracy_lib <- mean(y_pred_lib == y_test)
  cat("Precisión del KNN con librería:", round(accuracy_lib * 100, 2), "%\n")

  # =============================
  # KNN manual sin librerías
  # =============================
  euclid <- function(a, b){
    sqrt(sum((a - b)^2))
  }

  k <- 27
  y_pred_manual <- vector()
  for(i in 1:nrow(X_test)){
    distances <- apply(X_train, 1, function(row) euclid(row, X_test[i, ]))
    nearest <- order(distances)[1:k]
    labels <- y_train[nearest]
    y_pred_manual[i] <- names(sort(table(labels), decreasing = TRUE))[1]
  }

  y_pred_manual <- factor(y_pred_manual, levels = levels(y_test))
  accuracy_manual <- mean(y_pred_manual == y_test)
  cat("Precisión del KNN manual:", round(accuracy_manual * 100, 2), "%\n")

  # =============================
  # Matriz de confusión KNN con librería
  # =============================
  conf_matrix_lib <- table(Predicho = y_pred_lib, Real = y_test)
  conf_df_lib <- as.data.frame(conf_matrix_lib)

  ggplot(conf_df_lib, aes(x = Real, y = Predicho, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq), color = "white", size = 6) +
    scale_fill_gradient(low = "lightblue", high = "blue") +
    theme_minimal() +
    ggtitle("Matriz de confusión del KNN (librería)") +
    xlab("Clase real") +
    ylab("Clase predicha")

  # =============================
  # Matriz de confusión KNN manual
  # =============================
  conf_matrix_manual <- table(Predicho = y_pred_manual, Real = y_test)
  conf_df_manual <- as.data.frame(conf_matrix_manual)

  ggplot(conf_df_manual, aes(x = Real, y = Predicho, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq), color = "white", size = 6) +
    scale_fill_gradient(low = "lightblue", high = "blue") +
    theme_minimal() +
    ggtitle("Matriz de confusión del KNN (manual)") +
    xlab("Clase real") +
    ylab("Clase predicha")
