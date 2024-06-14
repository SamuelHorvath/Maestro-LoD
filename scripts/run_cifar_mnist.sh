LR=${LR:-"0.01"}
BS=${BS:-"256"}
GP_LAMBDA=${GP_LAMBDA:-"1e-6"}

pushd ../
python cifar_mnist_main.py \
       --model lenet \
       --batch-norm \
       --identifier test \
       --cuda \
       --decomposition \
       --progressive \
       --gp \
       --gp-lambda ${GP_LAMBDA} \
       --sampler per_layer \
       --no-full-pass \
       --lr $LR \
       --epochs 2 \
       -b $BS
popd
