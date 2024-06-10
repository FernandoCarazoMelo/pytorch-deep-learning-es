# Apuntes de PyTorch y Redes Neuronales

<div style="text-align: center;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c6/PyTorch_logo_black.svg/2560px-PyTorch_logo_black.svg.png" style="width: 300px;">
</div>


[PyTorch](https://pytorch.org) es una librería de código abierto para el desarrollo de modelos de inteligencia artificial con redes neuronales (Deep Learning). Aplicaciones como ChatGPT se han desarrollado con esta librería, por lo que PyTorch es una herramienta imprescindible para cualquier persona interesada en desarrollar aplicaciones de AI. 

En esta web te comparto mis apuntes personales sobre PyTorch. Si tienes alguna sugerencia o corrección, no dudes en contactarme a través de [mi perfil de LinkedIn](https://www.linkedin.com/in/fernandocarazomelo/) o hacer un pull request en el [repositorio de GitHub](https://github.com/FernandoCarazoMelo/pytorch-deep-learning-es).

Para prearar este material me he basado principalmente en los siguienes recursos:

- [PyTorch](https://pytorch.org/tutorials/) documentación oficial
- [3Blue1Brown](https://www.youtube.com/c/3blue1brown) canal del YouTube divulgativo de matemáticas y ciencia
- [Learn PyTorch for Deep Learning](https://github.com/mrdbourke/pytorch-deep-learning/) curso de PyTorch de Daniel Bourke
- [Jovian](https://jovian.ai/learn/deep-learning-with-pytorch-zero-to-gans) curso de PyTorch de Jovian
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) libro de Aurélien Géron
- [Short-courses de DeepLearning.ai](https://www.deeplearning.ai/courses/) Cursos sobre procesamiento del lenguaje natural (NLP) de Andrew Ng y colaboradores

## Temario del Curso de PyTorch y Redes Neuronales


- [x] **Lección 1: Fundamentos de PyTorch y Descenso de Gradiente**

    * Conceptos básicos de PyTorch: tensores, gradientes y autograd
    * Regresión lineal y descenso de gradiente desde cero


- [x] **Lección 2: Introducción a Redes Neuronales con PyTorch**

    * Conceptos básicos: neurona, funcion de activación, función de pérdida
    * Flujo estándard de entrenamiento de una red neuronal con descenso de gradiente
    * Uso de módulos de PyTorch: `nn.Linear` y `nn.functional`


- [x] **Lección 3: Trabajo con Imágenes y Regresión Logística**

    * División de entrenamiento y validación en el conjunto de datos MNIST
    * Regresión logística, softmax y entropía cruzada (cross-entropy)
    * Entrenamiento del modelo, evaluación y predicción


- [ ] **Lección 4: Entrenamiento de Redes Neuronales Profundas en GPU**

    * Redes neuronales multicapa usando `nn.Module`
    * Funciones de activación, no linealidad y retropropagación
    * Entrenamiento de modelos más rápido usando GPUs en la nube



- [ ] **Lección 5: Clasificación de Imágenes con Redes Neuronales Convolucionales**

    * Trabajo con imágenes RGB de 3 canales
    * Convoluciones, kernels y mapas de características
    * Curva de entrenamiento, sobreajuste y subajuste


- [ ] **Lección 6: Introducción a Transformers y HuggingFace**

    * Modelos de lenguaje preentrenados y transferencia de aprendizaje
    * Clasificación de texto con BERT y GPT-2
    * Generación de texto con GPT-2 y GPT-3


- [ ] **Lección 7: Procesamiento de Lenguaje Natural Avanzado con PyTorch**

    * Tokenización, incrustación de palabras y modelos de lenguaje
    * Clasificación de sentimientos con una red neuronal recurrente
    * Entrenamiento de modelos de lenguaje desde cero