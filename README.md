# 🌸 Análisis del Dataset Iris — Comparativa Multilenguaje (Python, R y Java)

**Autor:** Yeray Hurtado Dragón  
**Fecha:** Octubre 2025  
**Tema:** Big Data e Inteligencia Artificial  

---

## 📘 Descripción del Proyecto

Este proyecto realiza un **análisis comparativo del algoritmo K-Nearest Neighbors (KNN)** aplicado al clásico **dataset Iris**, implementado en **tres lenguajes de programación**:  
- 🐍 **Python (Google Colab)**  
- 📊 **R (Posit Cloud)**  
- ☕ **Java (Weka)**  

Cada implementación incluye:  
1. Matriz de correlación  
2. Gráfico de dispersión  
3. KNN con librerías  
4. KNN implementado manualmente  
5. Evaluación mediante matriz de confusión y precisión  

El objetivo es **comparar rendimiento, facilidad de implementación y capacidades visuales** entre lenguajes.

---

## 📂 Estructura del Repositorio

├── /python/
│ ├── knn_libreria.py
│ ├── knn_manual.py
│
├── /r/
│ ├── knn_libreria.R
│ ├── knn_manual.R
│
├── /java/
│ ├── knn_weka.java
│
├── Análisis_del_Dataset_Iris__Comparativa_Multilenguaje.pdf
└── README.md

ruby
Copiar código

---

## 🧠 Algoritmo KNN

El **K-Nearest Neighbors (KNN)** es un algoritmo de **clasificación supervisada** que asigna una clase a una instancia según la mayoría de sus *k* vecinos más cercanos, utilizando comúnmente la **distancia euclidiana**:

\[
d(p, q) = \sqrt{\sum_{i=1}^{n}(p_i - q_i)^2}
\]

### Ventajas:
- Fácil de entender e implementar.  
- Eficaz en problemas multiclase.  

### Limitaciones:
- Sensible a la escala de los datos.  
- Costoso en conjuntos grandes.

---

## 🧪 Resultados Comparativos

| Lenguaje | Precisión (librerías) | Precisión (manual) | Ventajas principales | Limitaciones |
|-----------|----------------------:|--------------------:|----------------------|---------------|
| 🐍 Python | 91.11 % | 91.11 % | Sintaxis simple, gran ecosistema ML | Requiere optimización en datasets grandes |
| 📊 R | 97.78 % | 97.78 % | Potente en análisis y visualización | Sintaxis menos intuitiva |
| ☕ Java (Weka) | 97.78 % | 97.78 % | Eficiente, entorno gráfico robusto | Limitada capacidad visual avanzada |

---

## 📈 Conclusiones

- Los tres lenguajes superan el **90 % de precisión**, mostrando la **consistencia del KNN**.  
- **R y Java (Weka)** alcanzan el mejor rendimiento (97.78 %).  
- **Python** ofrece una implementación más flexible y educativa.  
- La versión manual del algoritmo refuerza la comprensión teórica de KNN y la distancia euclidiana.  

---
