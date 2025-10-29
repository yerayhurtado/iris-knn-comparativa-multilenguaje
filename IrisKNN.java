import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Attribute;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class IrisKNN {

    public static void main(String[] args) throws Exception {
        // =============================
        // Cargar dataset
        // =============================
        String rutaDataset = "data/iris.arff";
        DataSource source = new DataSource(rutaDataset);
        Instances dataset = source.getDataSet();
        if (dataset.classIndex() == -1)
            dataset.setClassIndex(dataset.numAttributes() - 1); // Clase

        // Mostrar nombres de atributos
        System.out.println("Atributos en el dataset:");
        for (int i = 0; i < dataset.numAttributes(); i++)
            System.out.println(i + ": " + dataset.attribute(i).name());

        // =============================
        // Matriz de correlación
        // =============================
        int numAttr = dataset.numAttributes() - 1;
        double[][] corMatrix = new double[numAttr][numAttr];
        System.out.println("\n=== Matriz de correlación ===");
        for (int i = 0; i < numAttr; i++) {
            for (int j = 0; j < numAttr; j++) {
                corMatrix[i][j] = pearson(dataset, i, j);
                System.out.printf("%.2f\t", corMatrix[i][j]);
            }
            System.out.println();
        }

        // =============================
        // Dividir dataset en train/test
        // =============================
        int seed = 123;
        dataset.randomize(new Random(seed));
        int trainSize = (int) Math.round(dataset.numInstances() * 0.7);
        int testSize = dataset.numInstances() - trainSize;
        Instances train = new Instances(dataset, 0, trainSize);
        Instances test = new Instances(dataset, trainSize, testSize);

        // =============================
        // KNN con librería (IBk)
        // =============================
        int k = 27;
        IBk knn = new IBk(k);
        knn.buildClassifier(train);

        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(knn, test);

        System.out.println("\n=== KNN con Weka (IBk) ===");
        System.out.println("Precisión: " + String.format("%.2f", eval.pctCorrect()) + "%");
        printConfusionMatrix(eval.confusionMatrix(), dataset.classAttribute());

        // =============================
        // KNN manual
        // =============================
int kManual = 27;
int aciertos = 0;

for (int i = 0; i < test.numInstances(); i++) {
    Instance actual = test.instance(i);

    // Calcular distancia a todos los ejemplos de entrenamiento
    double[] distancias = new double[train.numInstances()];
    String[] clases = new String[train.numInstances()];

    for (int j = 0; j < train.numInstances(); j++) {
        distancias[j] = euclideanDistance(actual, train.instance(j));
        clases[j] = train.instance(j).stringValue(train.classIndex());
    }

    // Buscar los k vecinos más cercanos
    // (en lugar de usar Comparator, hacemos una búsqueda simple)
    int[] vecinos = new int[kManual];
    Arrays.fill(vecinos, -1);

    for (int v = 0; v < kManual; v++) {
        double minDist = Double.MAX_VALUE;
        int minIndex = -1;
        for (int j = 0; j < distancias.length; j++) {
            if (distancias[j] < minDist) {
                minDist = distancias[j];
                minIndex = j;
            }
        }
        vecinos[v] = minIndex;
        distancias[minIndex] = Double.MAX_VALUE; // marcar como usado
    }

    // Contar votos de las clases vecinas
    int votosSetosa = 0, votosVersicolor = 0, votosVirginica = 0;
    for (int v = 0; v < kManual; v++) {
        String clase = clases[vecinos[v]];
        if (clase.equals("Iris-setosa")) votosSetosa++;
        else if (clase.equals("Iris-versicolor")) votosVersicolor++;
        else if (clase.equals("Iris-virginica")) votosVirginica++;
    }

    // Determinar la clase más votada
    String prediccion;
    if (votosSetosa > votosVersicolor && votosSetosa > votosVirginica)
        prediccion = "Iris-setosa";
    else if (votosVersicolor > votosVirginica)
        prediccion = "Iris-versicolor";
    else
        prediccion = "Iris-virginica";

    // Comparar con la clase real
    String real = actual.stringValue(test.classIndex());
    if (prediccion.equals(real))
        aciertos++;
}

double precisionManual = 100.0 * aciertos / test.numInstances();
System.out.println("\n=== KNN Manual (versión sencilla) ===");
System.out.println("Precisión: " + String.format("%.2f", precisionManual) + "%");
    }


    // =============================
    // Funciones auxiliares
    // =============================
    public static double pearson(Instances data, int attr1, int attr2) {
        int n = data.numInstances();
        double mean1 = 0, mean2 = 0;
        for (int i = 0; i < n; i++) {
            mean1 += data.instance(i).value(attr1);
            mean2 += data.instance(i).value(attr2);
        }
        mean1 /= n;
        mean2 /= n;
        double num = 0, a = 0, b = 0;
        for (int i = 0; i < n; i++) {
            double diff1 = data.instance(i).value(attr1) - mean1;
            double diff2 = data.instance(i).value(attr2) - mean2;
            num += diff1 * diff2;
            a += diff1 * diff1;
            b += diff2 * diff2;
        }
        return num / Math.sqrt(a * b);
    }

    public static double euclideanDistance(Instance a, Instance b) {
        double sum = 0;
        for (int i = 0; i < a.numAttributes() - 1; i++)  
            sum += Math.pow(a.value(i) - b.value(i), 2);
        return Math.sqrt(sum);
    }

    public static void printConfusionMatrix(double[][] matrix, Attribute classAttr) {
        System.out.println("\nMatriz de confusión:");
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++)
                System.out.print((int) matrix[i][j] + "\t");
            System.out.println();
        }
        System.out.print("Clases: ");
        for (int i = 0; i < classAttr.numValues(); i++)
            System.out.print(classAttr.value(i) + " ");
        System.out.println();
    }
}
