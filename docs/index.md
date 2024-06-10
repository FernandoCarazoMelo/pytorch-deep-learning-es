# Apuntes de PyTorch y Redes Neuronales

<div style="text-align: center;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c6/PyTorch_logo_black.svg/2560px-PyTorch_logo_black.svg.png" style="width: 300px;">
</div>

[PyTorch](https://pytorch.org) es una librería de código abierto para el desarrollo de modelos o aprendizaje profundo (Deep Learning) con redes neuronales. ChatGPT de OpenAI, AlphaGo de DeepMind y muchos otros sistemas de inteligencia artificial se han desarrollado utilizando PyTorch, por lo que es una herramienta clave para cualquier persona interesada en el campo de la inteligencia artificial.

En esta web te comparto mis apuntes sobre PyTorch y redes neuronales utilizando Pytorch. 

El material lo he preparado utilizando los siguienes recursos principales:

- [PyTorch](https://pytorch.org/tutorials/) documentación oficial
- [Jovian](https://jovian.ai/learn/deep-learning-with-pytorch-zero-to-gans)
- [Learn PyTorch for Deep Learning](https://github.com/mrdbourke/pytorch-deep-learning/)
- [3Blue1Brown](https://www.youtube.com/c/3blue1brown) canal del YouTube divulgativo de matemáticas y ciencia
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) libro de Aurélien Géron

### Temario del Curso de PyTorch y Redes Neuronales


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


- [ ] **Lección 6: Aumento de Datos, Regularización y Redes ResNet**

    * Añadir capas residuales con batch normalization a CNNs
    * Disminución del ritmo de aprendizaje, decaimiento del peso y más
    * Entrenamiento de un modelo de vanguardia en 5 minutos


- [ ] **Lección 7: Redes Generativas Antagónicas y Aprendizaje por Transferencia**

    * Generación de dígitos falsos y caras de anime con GANs
    * Entrenamiento de redes generadoras y discriminadoras
    * Aprendizaje por transferencia para la clasificación de imágenes