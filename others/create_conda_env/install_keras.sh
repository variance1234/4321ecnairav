#!/bin/bash

#cd ../keras

#declare -a versions=("2.2.2" "2.2.1" "2.2.0"
		     #"2.1.6" "2.1.5" "2.1.4" "2.1.3" "2.1.2" "2.1.1" "2.1.0"
		     #"2.0.9" "2.0.8" "2.0.7" "2.0.6" "2.0.5" "2.0.4" "2.0.3" "2.0.2" "2.0.1" "2.0.0")

declare -a keras_versions=("2.2.2")

#declare -a keras_versions=("2.2.2")
#declare -a tensorflow_versions=("1.10.0" "1.9.0" "1.8.0" "1.7.0" "1.6.0" )
declare -a tensorflow_versions=("1.11.0" "1.12.0" "1.13.0" "1.14.0")
#declare -a theano_versions=("1.0.2" "1.0.1" "0.9.0" "0.8.2" "0.7.0" )
#declare -a cntk_versions=("2.4" "2.3.1" "2.2" "2.1" )

for keras_version in "${keras_versions[@]}"
do

    for tensorflow_version in "${tensorflow_versions[@]}"
    do
        conda create --clone K_${keras_version}_no_backend --name K_${keras_version}_tensorflow_${tensorflow_version}

        source activate K_${keras_version}_tensorflow_${tensorflow_version}

        #install tensorflow
        #pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-${tensorflow_version}-cp36-cp36m-linux_x86_64.whl
        pip install --upgrade tensorflow-gpu==${tensorflow_version}

        source deactivate
    done

    #for cntk_version in "${cntk_versions[@]}"
    #do
    #    conda create --clone K_${keras_version}_no_backend --name K_${keras_version}_cntk_${cntk_version}

    #    source activate K_${keras_version}_cntk_${cntk_version}

    #    #install cntk
    #    pip install --upgrade https://cntk.ai/PythonWheel/GPU/cntk-${cntk_version}-cp36-cp36m-linux_x86_64.whl

    #    source deactivate
    #done

    #do 2.5.1 separately
    #conda create --clone K_${keras_version}_no_backend --name K_${keras_version}_cntk_2.5.1
    #conda activate K_${keras_version}_cntk_2.5.1
    #pip install --upgrade https://cntk.ai/PythonWheel/GPU/cntk_gpu-2.5.1-cp36-cp36m-linux_x86_64.whl
    #conda deactivate

    #for theano_version in "${theano_versions[@]}"
    #do
    #    conda create --clone K_${keras_version}_no_backend --name K_${keras_version}_theano_${theano_version}

    #    source activate K_${keras_version}_theano_${theano_version}

        #install theano
    #    conda install theano=${theano_version}

    #    source deactivate
    #done



    #git checkout $keras_version

    #conda create --clone allbackend_no_keras --name allbackend_K_$keras_version

    #source activate allbackend_K_$keras_version

    #python setup.py install
    #conda install scikit-learn
    #conda install -c conda-forge scikit-image
    #conda install -c anaconda pydot

    #source deactivate

done

#cd ../crossmodelchecking
